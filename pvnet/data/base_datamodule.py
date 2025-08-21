""" Data module for pytorch lightning """

import os
from glob import glob

import torch
from lightning.pytorch import LightningDataModule
from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_data_sampler.numpy_sample.common_types import NumpySample, TensorBatch
from ocf_data_sampler.torch_datasets.sample.base import SampleBase, batch_to_tensor
from torch.utils.data import DataLoader, Dataset, Subset


def collate_fn(samples: list[NumpySample]) -> TensorBatch:
    """Convert a list of NumpySample samples to a tensor batch"""
    return batch_to_tensor(stack_np_samples_into_batch(samples))


class PresavedSamplesDataset(Dataset):
    """Dataset of pre-saved samples

    Args:
        sample_dir: Path to the directory of pre-saved samples.
        sample_class: sample class type to use for save/load/to_numpy
    """

    def __init__(self, sample_dir: str, sample_class: SampleBase):
        """Initialise PresavedSamplesDataset"""
        self.sample_paths = glob(f"{sample_dir}/*")
        self.sample_class = sample_class

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx) -> NumpySample:
        sample = self.sample_class.load(self.sample_paths[idx])
        return sample.to_numpy()


class BasePresavedDataModule(LightningDataModule):
    """Base Datamodule for loading pre-saved samples."""

    def __init__(
        self,
        sample_dir: str,
        batch_size: int = 16,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory: bool = False,
    ):
        """Base Datamodule for loading pre-saved samples

        Args:
            sample_dir: Path to the directory of pre-saved samples.
            batch_size: Batch size.
            num_workers: Number of workers to use in multiprocess batch loading.
            prefetch_factor: Number of batches loaded in advance by each worker.
            persistent_workers: If True, the data loader will not shut down the worker processes 
                after a dataset has been consumed once. This allows to maintain the workers Dataset 
                instances alive.
            pin_memory: If True, the data loader will copy Tensors into device/CUDA pinned memory 
                before returning them.
        """
        super().__init__()

        self.sample_dir = sample_dir

        self._common_dataloader_kwargs = dict(
            batch_size=batch_size,
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    def _get_premade_samples_dataset(self, subdir: str) -> Dataset:
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        """Construct train dataloader"""
        dataset = self._get_premade_samples_dataset("train")
        return DataLoader(dataset, shuffle=True, **self._common_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        """Construct val dataloader"""
        dataset = self._get_premade_samples_dataset("val")
        return DataLoader(dataset, shuffle=False, **self._common_dataloader_kwargs)


class BaseStreamedDataModule(LightningDataModule):
    """Base Datamodule which streams samples using a sampler for ocf-data-sampler."""

    def __init__(
        self,
        configuration: str,
        batch_size: int = 16,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        train_period: list[str | None] = [None, None],
        val_period: list[str | None] = [None, None],
        seed: int | None = None,
        dataset_pickle_dir: str | None = None,
    ):
        """Base Datamodule for streaming samples.

        Args:
            configuration: Path to ocf-data-sampler configuration file.
            batch_size: Batch size.
            num_workers: Number of workers to use in multiprocess batch loading.
            prefetch_factor: Number of batches loaded in advance by each worker.
            persistent_workers: If True, the data loader will not shut down the worker processes 
                after a dataset has been consumed once. This allows to maintain the workers Dataset 
                instances alive.
            pin_memory: If True, the data loader will copy Tensors into device/CUDA pinned memory 
                before returning them.
            train_period: Date range filter for train dataloader.
            val_period: Date range filter for val dataloader.
            seed: Random seed used in shuffling datasets.
            dataset_pickle_dir: Directory in which the val and train set will be presaved as
                pickle objects. Setting this speeds up instatiation of multiple workers a lot.
        """
        super().__init__()

        self.configuration = configuration
        self.train_period = train_period
        self.val_period = val_period
        self.seed = seed
        self.dataset_pickle_dir = dataset_pickle_dir

        self._common_dataloader_kwargs = dict(
            batch_size=batch_size,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    def setup(self, stage: str | None = None):
        """Called once to prepare the datasets."""

        # This logic runs only once at the start of training, therefore the val dataset is only
        # shuffled once
        if stage == "fit":

            # Prepare the train dataset
            self.train_dataset = self._get_streamed_samples_dataset(*self.train_period)

            # Prepare and pre-shuffle the val dataset and set seed for reproducibility
            val_dataset = self._get_streamed_samples_dataset(*self.val_period)

            if self.seed is not None:
                torch.manual_seed(self.seed)

            shuffled_indices = torch.randperm(len(val_dataset))
            self.val_dataset = Subset(val_dataset, shuffled_indices)
        
            if self.dataset_pickle_dir is not None:
                os.makedirs(self.dataset_pickle_dir, exist_ok=True)
                self.train_dataset.presave_pickle(f"{self.dataset_pickle_dir}/train_dataset.pkl")
                self.train_dataset.presave_pickle(f"{self.dataset_pickle_dir}/val_dataset.pkl")


    def teardown(self):
        """Clean up the pickled datasets"""
        if self.dataset_pickle_dir is not None:
            for filename in ["val_dataset.pkl", "train_dataset.pkl"]:
                filepath = f"{self.dataset_pickle_dir}/{filename}"
                if os.path.exists(filepath):
                    os.remove(filepath)


    def _get_streamed_samples_dataset(
        self,
        start_time: str | None,
        end_time: str | None
    ) -> Dataset:
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        """Construct train dataloader"""
        return DataLoader(self.train_dataset, shuffle=True, **self._common_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        """Construct val dataloader"""
        return DataLoader(self.val_dataset, shuffle=False, **self._common_dataloader_kwargs)
