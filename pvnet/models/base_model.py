"""Base model for all PVNet submodels"""

import copy
import logging
import os
import shutil
import time
from importlib.metadata import version
from pathlib import Path
from typing import Optional

import hydra
import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import ModelCard, ModelCardData, snapshot_download
from huggingface_hub.hf_api import HfApi
from ocf_data_sampler.numpy_sample.common_types import TensorBatch
from safetensors.torch import load_file, save_file
from torchmetrics.regression import ContinuousRankedProbabilityScore
from torchvision.transforms.functional import center_crop

from pvnet.utils import (
    DATA_CONFIG_NAME,
    DATAMODULE_CONFIG_NAME,
    FULL_CONFIG_NAME,
    MODEL_CARD_NAME,
    MODEL_CONFIG_NAME,
    PYTORCH_WEIGHTS_NAME,
)


def fill_config_paths_with_placeholder(
    config: dict, placeholder: str = "PLACEHOLDER"
) -> dict:
    """Modify the config in place to fill data paths with placeholder strings.

    Args:
        config: The data config
        placeholder: String placeholder for data sources
    """
    input_config = config["input_data"]

    for source in ["gsp", "satellite"]:
        if source in input_config:
            # If not empty - i.e. if used
            if input_config[source]["zarr_path"] != "":
                input_config[source]["zarr_path"] = f"{placeholder}.zarr"

    if "nwp" in input_config:
        for source in input_config["nwp"]:
            if input_config["nwp"][source]["zarr_path"] != "":
                input_config["nwp"][source]["zarr_path"] = f"{placeholder}.zarr"

    if "pv" in input_config:
        for d in input_config["pv"]["pv_files_groups"]:
            d["pv_filename"] = f"{placeholder}.netcdf"
            d["pv_metadata_filename"] = f"{placeholder}.csv"

    return config


def minimize_config_for_model(config: dict, model: "BaseModel") -> dict:
    """Strip out parts of the data config which aren't used by the model

    Args:
        config: The data config
        model: The PVNet model object
    """
    input_config = config["input_data"]

    if "nwp" in input_config:
        if not model.include_nwp:
            del input_config["nwp"]
        else:
            for nwp_source in list(input_config["nwp"].keys()):
                nwp_config = input_config["nwp"][nwp_source]

                if nwp_source not in model.nwp_encoders_dict:
                    # If not used, delete this source from the config
                    del input_config["nwp"][nwp_source]
                else:
                    # Replace the image size
                    nwp_pixel_size = model.nwp_encoders_dict[
                        nwp_source
                    ].image_size_pixels
                    nwp_config["image_size_pixels_height"] = nwp_pixel_size
                    nwp_config["image_size_pixels_width"] = nwp_pixel_size

                    # Replace the interval_end_minutes minutes
                    nwp_config["interval_end_minutes"] = (
                        nwp_config["interval_start_minutes"]
                        + (model.nwp_encoders_dict[nwp_source].sequence_length - 1)
                        * nwp_config["time_resolution_minutes"]
                    )

    if "satellite" in input_config:
        if not model.include_sat:
            del input_config["satellite"]
        else:
            sat_config = input_config["satellite"]

            # Replace the image size
            sat_pixel_size = model.sat_encoder.image_size_pixels
            sat_config["image_size_pixels_height"] = sat_pixel_size
            sat_config["image_size_pixels_width"] = sat_pixel_size

            # Replace the interval_end_minutes minutes
            sat_config["interval_end_minutes"] = (
                sat_config["interval_start_minutes"]
                + (model.sat_encoder.sequence_length - 1)
                * sat_config["time_resolution_minutes"]
            )

    if "pv" in input_config:
        if not model.include_pv:
            del input_config["pv"]

    if "gsp" in input_config:
        gsp_config = input_config["gsp"]

        # Replace the forecast minutes
        gsp_config["interval_end_minutes"] = model.forecast_minutes

    if "solar_position" in input_config:
        solar_config = input_config["solar_position"]
        solar_config["interval_end_minutes"] = model.forecast_minutes

    return config


def download_from_hf(
    repo_id: str,
    filename: str | list[str],
    revision: str,
    cache_dir: str | None,
    force_download: bool,
    max_retries: int = 5,
    wait_time: int = 10,
) -> str | list[str]:
    """Tries to download one or more files from HuggingFace up to max_retries times.

    Args:
        repo_id: HuggingFace repo ID
        filename: Name of the file(s) to download
        revision: Specific model revision
        cache_dir: Cache directory
        force_download: Whether to force a new download
        max_retries: Maximum number of retry attempts
        wait_time: Wait time (in seconds) before retrying

    Returns:
        The local file path of the downloaded file(s)
    """
    for attempt in range(1, max_retries + 1):
        try:
            save_dir = snapshot_download(
                repo_id=repo_id,
                allow_patterns=filename,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
            )

            if isinstance(filename, list):
                return [f"{save_dir}/{f}" for f in filename]
            else:
                return f"{save_dir}/{filename}"

        except Exception as e:
            if attempt == max_retries:
                raise Exception(
                    f"Failed to download {filename} from {repo_id} after {max_retries} attempts."
                ) from e
            logging.warning(
                (
                    f"Attempt {attempt}/{max_retries} failed to download {filename} "
                    f"from {repo_id}. Retrying in {wait_time} seconds..."
                )
            )
            time.sleep(wait_time)


class HuggingfaceMixin:
    """Mixin for saving and loading model to and from huggingface"""

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: str,
        cache_dir: str | None = None,
        force_download: bool = False,
        strict: bool = True,
    ) -> "BaseModel":
        """Load Pytorch pretrained weights and return the loaded model."""

        if os.path.isdir(model_id):
            print("Loading model from local directory")
            model_file = f"{model_id}/{PYTORCH_WEIGHTS_NAME}"
            config_file = f"{model_id}/{MODEL_CONFIG_NAME}"
        else:
            print("Loading model from huggingface repo")

            model_file, config_file = download_from_hf(
                repo_id=model_id,
                filename=[PYTORCH_WEIGHTS_NAME, MODEL_CONFIG_NAME],
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                max_retries=5,
                wait_time=10,
            )

        with open(config_file, "r") as f:
            model = hydra.utils.instantiate(yaml.safe_load(f))

        state_dict = load_file(model_file)
        model.load_state_dict(state_dict, strict=strict)  # type: ignore
        model.eval()  # type: ignore

        return model

    @classmethod
    def get_data_config(
        cls,
        model_id: str,
        revision: str,
        cache_dir: str | None = None,
        force_download: bool = False,
    ) -> str:
        """Load data config file."""
        if os.path.isdir(model_id):
            print("Loading data config from local directory")
            data_config_file = os.path.join(model_id, DATA_CONFIG_NAME)
        else:
            print("Loading data config from huggingface repo")
            data_config_file = download_from_hf(
                repo_id=model_id,
                filename=DATA_CONFIG_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                max_retries=5,
                wait_time=10,
            )

        return data_config_file

    def _save_model_weights(self, save_directory: str) -> None:
        """Save weights from a Pytorch model to a local directory."""
        save_file(self.state_dict(), f"{save_directory}/{PYTORCH_WEIGHTS_NAME}")

    def save_pretrained(
        self,
        save_directory: str,
        model_config: dict,
        data_config_path: str,
        wandb_repo: str,
        wandb_ids: list[str] | str,
        card_template_path: str,
        datamodule_config_path: str | None = None,
        experiment_config_path: str | None = None,
        hf_repo_id: str | None = None,
        push_to_hub: bool = False,
    ) -> None:
        """Save weights in local directory or upload to huggingface hub.

        Args:
            save_directory:
                Path to directory in which the model weights and configuration will be saved.
            model_config (`dict`):
                Model configuration specified as a key/value dictionary.
            data_config_path:
                The path to the data config.
            wandb_repo: Identifier of the repo on wandb.
            wandb_ids: Identifier(s) of the model on wandb.
            datamodule_config_path:
                The path to the datamodule config.
            experiment_config_path:
                The path to the full experimental config.
            hf_repo_id:
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to
                the folder name if not provided.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the HuggingFace Hub after saving it.

            card_template_path: Path to the HuggingFace model card template. Defaults to card in
                PVNet library if set to None.
        """

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save model weights/files
        self._save_model_weights(save_directory)

        # Save the model config and data config
        if isinstance(model_config, dict):
            with open(save_directory / MODEL_CONFIG_NAME, "w") as outfile:
                yaml.dump(
                    model_config, outfile, sort_keys=False, default_flow_style=False
                )

        # Save cleaned version of input data configuration file
        with open(data_config_path) as cfg:
            config = yaml.load(cfg, Loader=yaml.FullLoader)

        config = fill_config_paths_with_placeholder(config)
        config = minimize_config_for_model(config, self)

        with open(save_directory / DATA_CONFIG_NAME, "w") as outfile:
            yaml.dump(config, outfile, sort_keys=False, default_flow_style=False)

        # Save the datamodule config
        if datamodule_config_path is not None:
            shutil.copyfile(
                datamodule_config_path, save_directory / DATAMODULE_CONFIG_NAME
            )

        # Save the full experimental config
        if experiment_config_path is not None:
            shutil.copyfile(experiment_config_path, save_directory / FULL_CONFIG_NAME)

        card = self.create_hugging_face_model_card(
            card_template_path, wandb_repo, wandb_ids
        )

        (save_directory / MODEL_CARD_NAME).write_text(str(card))

        if push_to_hub:
            api = HfApi()

            api.upload_folder(
                repo_id=hf_repo_id,
                folder_path=save_directory,
                repo_type="model",
                commit_message=f"Upload models - {wandb_ids}",
            )

            # Print the most recent commit hash
            c = api.list_repo_commits(repo_id=hf_repo_id, repo_type="model")[0]

            message = (
                f"The latest commit is now: \n"
                f"    date: {c.created_at} \n"
                f"    commit hash: {c.commit_id}\n"
                f"    by: {c.authors}\n"
                f"    title: {c.title}\n"
            )

            print(message)

        return

    @staticmethod
    def create_hugging_face_model_card(
        card_template_path: str,
        wandb_repo: str,
        wandb_ids: list[str] | str,
    ) -> ModelCard:
        """
        Creates Hugging Face model card

        Args:
            card_template_path: Path to the HuggingFace model card template
            wandb_repo: Identifier of the repo on wandb.
            wandb_ids: Identifier(s) of the model on wandb.

        Returns:
            card: ModelCard - Hugging Face model card object
        """

        # Creating and saving model card.
        card_data = ModelCardData(language="en", license="mit", library_name="pytorch")

        if isinstance(wandb_ids, str):
            wandb_ids = [wandb_ids]

        wandb_links = ""
        for wandb_id in wandb_ids:
            link = f"https://wandb.ai/{wandb_repo}/runs/{wandb_id}"
            wandb_links += f" - [{link}]({link})\n"

        # Find package versions for OCF packages
        packages_to_display = ["pvnet", "ocf-data-sampler"]
        packages_and_versions = {
            package: version(package) for package in packages_to_display
        }

        package_versions_markdown = ""
        for package, v in packages_and_versions.items():
            package_versions_markdown += f" - {package}=={v}\n"

        return ModelCard.from_template(
            card_data,
            template_path=card_template_path,
            wandb_links=wandb_links,
            package_versions=package_versions_markdown,
        )


class BaseModel(torch.nn.Module, HuggingfaceMixin):
    """Abstract base class for PVNet submodels"""

    def __init__(
        self,
        history_minutes: int,
        forecast_minutes: int,
        output_quantiles: Optional[list[float]] = None,
        num_gmm_components: Optional[int] = None,
        target_key: str = "gsp",
        interval_minutes: int = 30,
    ):
        """Abtstract base class for PVNet submodels.

        Args:
            history_minutes (int): Length of the GSP history period in minutes
            forecast_minutes (int): Length of the GSP forecast period in minutes
            output_quantiles: A list of float (0.0, 1.0) quantiles to predict values for. If set to
                None the output is a single value.
            num_gmm_components: Number of Gaussian Mixture Model components to use for the model.
            If None, output quantiles must be set. If  both None, the output is a single value.
            target_key: The key of the target variable in the batch
            interval_minutes: The interval in minutes between each timestep in the data
        """
        super().__init__()

        self._target_key = target_key

        self.history_minutes = history_minutes
        self.forecast_minutes = forecast_minutes
        self.output_quantiles = output_quantiles
        self.num_gmm_components = num_gmm_components
        self.interval_minutes = interval_minutes

        # Number of timestemps for 30 minutely data
        self.history_len = history_minutes // interval_minutes
        self.forecast_len = (forecast_minutes) // interval_minutes

        # Store whether the model should use quantile regression or simply predict the mean
        self.use_quantile_regression = self.output_quantiles is not None

        self.use_gmm = self.num_gmm_components is not None

        # Both quantile regression and GMM cannot be used at the same time
        if self.use_quantile_regression and self.use_gmm:
            raise ValueError(
                "Cannot use quantile regression and GMM at the same time. "
                "Please set either output_quantiles or num_gmm_components to None."
            )

        # Store the number of ouput features that the model should predict for
        if self.use_quantile_regression:
            self.num_output_features = self.forecast_len * len(self.output_quantiles)
        elif self.use_gmm:
            self.num_output_features = self.forecast_len * self.num_gmm_components * 3
        else:
            self.num_output_features = self.forecast_len

        # CRPS
        self.crps_metric = ContinuousRankedProbabilityScore()

    def _adapt_batch(self, batch: TensorBatch) -> TensorBatch:
        """Slice batches into appropriate shapes for model.

        Returns a new batch dictionary with adapted data, leaving the original batch unchanged.
        We make some specific assumptions about the original batch and the derived sliced batch:
        - We are only limiting the future projections. I.e. we are never shrinking the batch from
          the left hand side of the time axis, only slicing it from the right
        - We are only shrinking the spatial crop of the satellite and NWP data

        """
        # Create a copy of the batch to avoid modifying the original
        new_batch = {key: copy.deepcopy(value) for key, value in batch.items()}

        if "gsp" in new_batch.keys():
            # Slice off the end of the GSP data
            gsp_len = self.forecast_len + self.history_len + 1
            new_batch["gsp"] = new_batch["gsp"][:, :gsp_len]
            new_batch["gsp_time_utc"] = new_batch["gsp_time_utc"][:, :gsp_len]

        if "site" in new_batch.keys():
            # Slice off the end of the site data
            site_len = self.forecast_len + self.history_len + 1
            new_batch["site"] = new_batch["site"][:, :site_len]

            # Slice all site related datetime coordinates and features
            site_time_keys = [
                "site_time_utc",
                "site_date_sin",
                "site_date_cos",
                "site_time_sin",
                "site_time_cos",
            ]

            for key in site_time_keys:
                if key in new_batch.keys():
                    new_batch[key] = new_batch[key][:, :site_len]

        if self.include_sat:
            # Slice off the end of the satellite data and spatially crop
            # Shape: batch_size, seq_length, channel, height, width
            new_batch["satellite_actual"] = center_crop(
                new_batch["satellite_actual"][:, : self.sat_sequence_len],
                output_size=self.sat_encoder.image_size_pixels,
            )

        if self.include_nwp:
            # Slice off the end of the NWP data and spatially crop
            for nwp_source in self.nwp_encoders_dict:
                # shape: batch_size, seq_len, n_chans, height, width
                new_batch["nwp"][nwp_source]["nwp"] = center_crop(
                    new_batch["nwp"][nwp_source]["nwp"],
                    output_size=self.nwp_encoders_dict[nwp_source].image_size_pixels,
                )[:, : self.nwp_encoders_dict[nwp_source].sequence_length]

        if self.include_sun:
            sun_len = self.forecast_len + self.history_len + 1
            # Slice off end of solar coords
            for s in ["solar_azimuth", "solar_elevation"]:
                if s in new_batch.keys():
                    new_batch[s] = new_batch[s][:, :sun_len]

        return new_batch

    def _parse_gmm_params(self, y_gmm):
        """
        Reshape flat output into (μ, σ, π) tensors.

        y_gmm: (batch, forecast_len * num_components * 3)

        Returns:
            mus:    (batch, forecast_len, num_components)
            sigmas: (batch, forecast_len, num_components)
            pis:    (batch, forecast_len, num_components)
        """
        bsz = y_gmm.shape[0]
        # reshape to [batch, forecast_len, num_components, 3]
        params = y_gmm.view(
            bsz,
            self.forecast_len,
            self.num_gmm_components,
            3,
        )
        mus = params[..., 0]
        # enforce positivity & stability
        sigmas = F.softplus(params[..., 1]) + 1e-3
        # softmax over components to get mixture weights
        logits = params[..., 2]
        pis = F.softmax(logits, dim=-1)
        return mus, sigmas, pis

    def _quantiles_to_prediction(self, y_quantiles: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Note:
            Implementation copied from:
                https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting
                /metrics/quantile.html#QuantileLoss.loss

        Args:
            y_quantiles: Quantile prediction of network

        Returns:
            torch.Tensor: Point prediction
        """
        # y_quantiles Shape: batch_size, seq_length, num_quantiles
        idx = self.output_quantiles.index(0.5)
        return y_quantiles[..., idx]

    def _gmm_to_prediction(self, y_gmm):
        """
        Compute the mixture’s expectation E[Y] = Σ π_i μ_i

        Args:
            y_gmm:   (batch, forecast_len * num_components * 3)

        Returns shape (batch, forecast_len)
        """
        mus, sigmas, pis = self._parse_gmm_params(y_gmm)
        # expectation over components
        y_pred = (pis * mus).sum(dim=-1)  # Here we sum over the components
        # y_pred shape: (batch, forecast_len)
        return y_pred

    def _sample_from_gmm(self, mus, sigmas, pis, n_samples=20):
        """
        Sample from Gaussian Mixture Model for each timestep and batch.

        Args:
            mus:    [batch, horizon, components]
            sigmas: [batch, horizon, components]
            pis:    [batch, horizon, components]
            n_samples: Number of samples to draw

        Returns:
            samples: [n_samples, batch, horizon]
        """
        batch, horizon, num_components = mus.shape

        # Sample component indices according to mixture weights (pis)
        categorical = torch.distributions.Categorical(pis)
        component_indices = categorical.sample(
            (n_samples,)
        )  # [n_samples, batch, horizon]

        # Gather the corresponding μ and σ for each sampled index
        mus_samples = mus.unsqueeze(0).expand(n_samples, -1, -1, -1)
        sigmas_samples = sigmas.unsqueeze(0).expand(n_samples, -1, -1, -1)

        idx = component_indices.unsqueeze(-1)
        gathered_mus = torch.gather(mus_samples, dim=3, index=idx).squeeze(-1)
        gathered_sigmas = torch.gather(sigmas_samples, dim=3, index=idx).squeeze(-1)

        # Sample from the normal distribution
        normal = torch.distributions.Normal(gathered_mus, gathered_sigmas)
        samples = normal.sample()  # [n_samples, batch, horizon]

        return samples
