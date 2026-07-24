"""Data module for pytorch lightning"""

import os

import numpy as np
import torch
import pandas as pd

from lightning.pytorch import LightningDataModule
from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_data_sampler.numpy_sample.common_types import NumpySample, TensorBatch
from ocf_data_sampler.torch_datasets.pvnet_dataset import PVNetDataset
from ocf_data_sampler.torch_datasets.utils.torch_batch_utils import batch_to_tensor
from torch.utils.data import DataLoader, Subset, Sampler, WeightedRandomSampler

def collate_fn(samples: list[NumpySample]) -> TensorBatch:
    """Convert a list of NumpySample samples to a tensor batch"""
    return batch_to_tensor(stack_np_samples_into_batch(samples))

def get_country(loc_id: pd.Series) -> np.ndarray:
    """Returns np array labelled UK or NL for this specific dataset"""
    return np.where(loc_id < 352, "UK", "NL")

def get_sampler(pvnet_dataset: PVNetDataset, 
                weight_NL: float) -> Sampler:
    """
    Args:
    * pvnet_dataset: ocf_data_sampler.torch_datasets.pvnet_dataset.PVNetDataset
    * weight_NL: Float ratio of sampling NL
    Returns:
    * Sampler"""
    # Currently only works for pvnet_dataset.complete_generation = False
    # Could replace this later e.g., user provides country in zarr file
    country = get_country(pvnet_dataset.valid_t0_and_location_ids["location_id"])
    n_UK = sum(country == "UK")
    n_NL = sum(country == "NL")
    n_total = len(country)
    weight_UK = 1.0 - weight_NL
    print(f"Sample UK with frequency {weight_UK}, and NL with frequency {weight_NL}")
    # weight_NL and weight_UK are class weights - we need weight per sample, considering different number of samples
    weight_per_sample_UK = weight_UK / n_UK
    weight_per_sample_NL = weight_NL / n_NL
    print(f"For UK, the weight per sample is {weight_per_sample_UK}, and for NL weight per sample is {weight_per_sample_NL}")
    weights_all = np.where(country == "UK", weight_per_sample_UK, weight_per_sample_NL)
    print(weights_all)
    weighted_sampler = CustomWeightedRandomSampler(weights_all, len(weights_all))
    return weighted_sampler

class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled
    https://github.com/pytorch/pytorch/issues/2576#issuecomment-831780307 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

class PVNetDataModule(LightningDataModule):
    """Base Datamodule which streams samples using a sampler from ocf-data-sampler."""

    def __init__(
        self,
        configuration: str,
        train_periods: list[tuple[None | str, None | str]],
        val_periods: list[tuple[None | str, None | str]],
        batch_size: int,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        seed: int | None = None,
        dataset_pickle_dir: str | None = None,
        use_sampler: bool = False,
        weight_NL: float | None = None,
    ):
        """Base Datamodule for streaming samples.

        Args:
            configuration: Path to ocf-data-sampler configuration file.
            train_periods: List of (start_time, end_time) tuples for the train dataset. If 
                start_time or end_time is None, it means that there is no lower/upper bound on the 
                time period.
            val_periods: List of (start_time, end_time) tuples for the validation dataset.
            batch_size: Batch size.
            num_workers: Number of workers to use in multiprocess batch loading.
            prefetch_factor: Number of batches loaded in advance by each worker.
            persistent_workers: If True, the data loader will not shut down the worker processes
                after a dataset has been consumed once. This allows to maintain the workers Dataset
                instances alive.
            pin_memory: If True, the data loader will copy Tensors into device/CUDA pinned memory
                before returning them.
            seed: Random seed used in shuffling datasets.
            dataset_pickle_dir: Directory in which the val and train set will be presaved as
                pickle objects. Setting this speeds up instantiation of multiple workers a lot.
        """
        super().__init__()

        self.configuration = configuration
        self.train_periods = train_periods
        self.val_periods = val_periods
        self.seed = seed
        self.dataset_pickle_dir = dataset_pickle_dir
        self.use_sampler = use_sampler
        if self.use_sampler:
            self.weight_NL = weight_NL

        self._common_dataloader_kwargs = dict(
            batch_size=batch_size,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )

    def setup(self, stage: str | None = None):
        """Called once to prepare the datasets."""

        # This logic runs only once at the start of training, therefore the val dataset is only
        # shuffled once
        if stage == "fit":
            # Prepare the train dataset
            self.train_dataset = self._get_dataset(self.train_periods)

            # Prepare the sampler
            if self.use_sampler:
                self.sampler = get_sampler(self.train_dataset, 
                                        weight_NL=self.weight_NL
                                        )

            # Prepare and pre-shuffle the val dataset and set seed for reproducibility
            val_dataset = self._get_dataset(self.val_periods)

            shuffled_indices = np.random.default_rng(seed=self.seed).permutation(len(val_dataset))
            self.val_dataset = Subset(val_dataset, shuffled_indices)

            if self.dataset_pickle_dir is not None:
                os.makedirs(self.dataset_pickle_dir, exist_ok=True)
                train_dataset_path = f"{self.dataset_pickle_dir}/train_dataset.pkl"
                val_dataset_path = f"{self.dataset_pickle_dir}/val_dataset.pkl"

                # For safety, these pickled datasets cannot be overwritten.
                # See: https://github.com/openclimatefix/pvnet/pull/445
                for path in [train_dataset_path, val_dataset_path]:
                    if os.path.exists(path):
                        raise FileExistsError(
                            f"The pickled dataset path '{path}' already exists. Make sure that "
                            "this can be safely deleted (i.e. not currently being used by any "
                            "training run) and delete it manually. Else change the "
                            "`dataset_pickle_dir` to a different directory."
                        )

                self.train_dataset.presave_pickle(train_dataset_path)
                self.train_dataset.presave_pickle(val_dataset_path)

    def teardown(self, stage: str | None = None) -> None:
        """Clean up the pickled datasets"""
        if self.dataset_pickle_dir is not None:
            for filename in ["val_dataset.pkl", "train_dataset.pkl"]:
                filepath = f"{self.dataset_pickle_dir}/{filename}"
                if os.path.exists(filepath):
                    os.remove(filepath)

    def _get_dataset(self, time_periods: list[tuple[str | None, str | None]]) -> PVNetDataset:
        return PVNetDataset(self.configuration, time_periods=time_periods)

    def train_dataloader(self) -> DataLoader:
        """Construct train dataloader"""
        return DataLoader(
            self.train_dataset,
            shuffle=False if self.use_sampler else True,
            sampler=self.sampler if self.use_sampler else None,
            drop_last=True,
            **self._common_dataloader_kwargs
        )

    def val_dataloader(self) -> DataLoader:
        """Construct val dataloader"""
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            drop_last=False,
            **self._common_dataloader_kwargs)
