"""A script to run backtest for PVNet and (optionally) the summation model

This script uses exported PVNet and PVNet summation models stored either locally or on huggingface.
See example_backtest_data_config.yaml for the expected format of the input data config yaml file.


Example command to run the backtest:
```
python backtest.py \
    --input-data-paths example_backtest_data_config.yaml \
    --output-dir /path/to/output \
    --pvnet-model-name openclimatefix-models/pvnet_uk_region \
    --pvnet-model-version xxxxxxx \
    --summation-model-name openclimatefix-models/pvnet_v2_summation \
    --summation-model-version xxxxxxx \
    --min-solar-elevation 0 \
    --start-datetime "2022-01-01 00:00" \
    --end-datetime "2022-12-31 23:30" \
    --device-name 'cuda:0' \
    --num-workers 8
```

"""

import logging
import os
import shutil

import numpy as np
import pandas as pd
import torch
import typer
import xarray as xr
import yaml
from ocf_data_sampler.torch_datasets.utils.torch_batch_utils import (
    batch_to_tensor,
    copy_batch_to_device,
)
from pvnet_summation.data.datamodule import StreamedDataset
from pvnet_summation.models.base_model import BaseModel as SummationBaseModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from pvnet.models.base_model import BaseModel as PVNetBaseModel

# ------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# FUNCTIONS

_model_mismatch_msg = (
    "The PVNet version running in this app is {}/{}. The summation model running in this app was "
    "trained on outputs from PVNet version {}/{}. Combining these models may lead to an error if "
    "the shape of PVNet output doesn't match the expected shape of the summation model. Combining "
    "may lead to unreliable results even if the shapes match."
)

def populate_config_with_data_data_filepaths(config: dict, data_paths: dict) -> dict:
    """Populate the data source filepaths in the config

    Args:
        config: The data config
        data_paths: The paths to the input data
    """

    # Replace the generation data path
    config["input_data"]["generation"]["zarr_path"] =  data_paths["generation"]

    # Replace satellite data path if using it
    if "satellite" in config["input_data"]:
        if config["input_data"]["satellite"]["zarr_path"] != "":
            config["input_data"]["satellite"]["zarr_path"] = data_paths["satellite"]

    # NWP is nested so must be treated separately
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():
            provider = nwp_config[nwp_source]["provider"]
            assert provider in data_paths["nwp"], f"Missing NWP path: {provider}"
            nwp_config[nwp_source]["zarr_path"] = data_paths["nwp"][provider]

    return config


def overwrite_config_dropouts(config: dict) -> dict:
    """Overwrite the config drouput parameters for the backtest

    Args:
        config: The data config
    """
    if "satellite" in config["input_data"]:

        satellite_config = config["input_data"]["satellite"]

        if satellite_config["zarr_path"] != "":
            satellite_config["dropout_timedeltas_minutes"] = []
            satellite_config["dropout_fraction"] = 0

    # Don't modify NWP dropout since this accounts for the expected NWP delay
    
    return config


def construct_model_data_config(
    pvnet_model_name: str, 
    pvnet_model_version: str | None, 
    input_data_paths: str, 
    output_path: str
) -> None:
    """Construct the data config for the backtest and save to a yaml file
    
    Args:
        pvnet_model_name: The name or path of the PVNet model to use in the backtest
        pvnet_model_version: The version of the PVNet model to use in the backtest
        input_data_paths: Path to yaml file containing paths to input data zarrs. 
            See example_backtest_data_config.yaml for format.
        output_path: Path to save the constructed data config yaml file to
    """
    model_data_config_path = PVNetBaseModel.get_data_config(
        model_id=pvnet_model_name,
        revision=pvnet_model_version,
    )

    with open(model_data_config_path) as file:
        model_data_config = yaml.load(file, Loader=yaml.FullLoader)

    with open(input_data_paths) as file:
        input_data_config = yaml.load(file, Loader=yaml.FullLoader)

    data_config = populate_config_with_data_data_filepaths(model_data_config, input_data_config)
    data_config = overwrite_config_dropouts(data_config)

    with open(output_path, "w") as file:
        yaml.dump(data_config, file, default_flow_style=False)


class BacktestStreamedDataset(StreamedDataset):
    """A torch dataset object used only for backtesting"""

    def _get_sample(self, t0: pd.Timestamp) -> ...:
        """Generate a concurrent PVNet sample for given init-time + augment for backtesting.

        Args:
            t0: init-time for sample
        """

        sample = super()._get_sample(t0)

        total_capacity = self.national_data.sel(time_utc=t0).capacity_mwp.item()

        sample.update(
            {
                "backtest_t0": t0,
                "backtest_national_capacity": total_capacity,
            }
        )

        return sample


class Forecaster:
    """Class for making solar forecasts for all regions and optionally the national total"""

    def __init__(
        self,
        pvnet_model_name: str,
        pvnet_model_version: str | None,
        summation_model_name: str | None,
        summation_model_version: str | None,
        min_solar_elevation: float | None,
        device: torch.device,
        
    ):
        """Initialise the forecaster"""
        self.pvnet_model_name = pvnet_model_name
        self.pvnet_model_version = pvnet_model_version
        self.summation_model_name = summation_model_name
        self.summation_model_version = summation_model_version
        self.device = device
        self.min_solar_elevation = min_solar_elevation
        
        # Load the regional model
        self.model = PVNetBaseModel.from_pretrained(
            model_id=pvnet_model_name,
            revision=pvnet_model_version,
        ).to(device).eval()

        # Load the summation model
        if summation_model_name is not None:
            self.sum_model = SummationBaseModel.from_pretrained(
                model_id=summation_model_name,
                revision=summation_model_version,
            ).to(device).eval()

            # Compare the current regional model with the one the summation model was trained on
            datamodule_path = SummationBaseModel.get_datamodule_config(
                model_id=summation_model_name,
                revision=summation_model_version,
            )
            with open(datamodule_path) as cfg:
                sum_pvnet_cfg = yaml.load(cfg, Loader=yaml.FullLoader)["pvnet_model"]

            sum_expected_reg_model = (sum_pvnet_cfg["model_id"], sum_pvnet_cfg["revision"])
            this_reg_model = (pvnet_model_name, pvnet_model_version)

            if sum_expected_reg_model != this_reg_model:
                logger.warning(_model_mismatch_msg.format(*this_reg_model, *sum_expected_reg_model))

        # These are the steps this forecast will predict for
        self.steps = pd.timedelta_range(
            start=f"{self.model.interval_minutes}min", 
            freq=f"{self.model.interval_minutes}min", 
            periods=self.model.forecast_len,
        )

    @torch.inference_mode()
    def predict(self, sample: dict) -> xr.Dataset:
        """Make predictions for the batch and store results internally"""
        
        x = copy_batch_to_device(batch_to_tensor(sample["pvnet_inputs"]), self.device)
        
        # Run batch through model
        normed_preds = self.model(x).detach().cpu().numpy()

        # Convert region results to xarray DataArray
        t0 = sample["backtest_t0"]
        location_ids = sample["pvnet_inputs"]["location_id"]

        da_normed = self.to_dataarray(
            normed_preds,
            t0,
            location_ids,
            self.model.output_quantiles,
        )

        # Multiply normalised forecasts by capacities and clip negatives
        da_abs = (
            da_normed.clip(0, None) 
            * sample["pvnet_inputs"]["capacity_mwp"][None, :, None, None].numpy()
        )

        # Apply sundown mask if specified
        if self.min_solar_elevation is not None:
            # The dataloader normalises solar elevation data to the range [0, 1]
            elevation_degrees = (sample["pvnet_inputs"]["solar_elevation"] - 0.5) * 180
            # We only need elevation mask for forecasted values, not history
            elevation_degrees = elevation_degrees[:, -normed_preds.shape[1]:]
            sun_down_masks = elevation_degrees < self.min_solar_elevation

            da_sundown_mask = self.to_dataarray(sun_down_masks, t0, location_ids, None)

            da_abs = da_abs.where(~da_sundown_mask, other=0.0)

        if self.summation_model_name is not None:
            # Make national predictions using summation model
            # - Need to add batch dimension and convert to torch tensors on device
            sample["pvnet_outputs"] = torch.tensor(normed_preds[None]).to(self.device)
            for k in ["relative_capacity", "azimuth", "elevation"]:
                sample[k] = sample[k][None].to(self.device)
            normed_national = self.sum_model(sample).detach().squeeze().cpu().numpy()

            # Convert national predictions to DataArray
            da_normed_national = self.to_dataarray(
                normed_national[np.newaxis],
                t0,
                location_ids=[0],
                output_quantiles=self.sum_model.output_quantiles,
            )

            # Multiply normalised forecasts by capacity and clip negatives
            national_capacity = sample["backtest_national_capacity"]
            da_abs_national = da_normed_national.clip(0, None) * national_capacity

            if self.min_solar_elevation is not None:
                # Apply sundown mask - All regions must be masked to mask national
                da_abs_national = da_abs_national.where(
                        ~da_sundown_mask.all(dim="location_id"), 
                        other=0.0,
                )

            # Convert to Dataset and add attrs about the models used
            da_abs = xr.concat([da_abs_national, da_abs], dim="location_id")
            
        da_abs = da_abs.to_dataset(name="hindcast")
        da_abs.attrs.update(
            {
                "pvnet_model_name": self.pvnet_model_name,
                "pvnet_model_version": self.pvnet_model_version or "none",
                "summation_model_name": self.summation_model_name or "none",
                "summation_model_version": self.summation_model_version or "none",
            }
        )

        return da_abs

    def to_dataarray(
        self,
        preds: np.ndarray,
        t0: pd.Timestamp,
        location_ids: list[int],
        output_quantiles: list[float] | None,
    ) -> xr.DataArray:
        """Put numpy array of predictions into a dataarray"""

        dims = ["init_time_utc", "location_id", "step"]
        coords = dict(
            init_time_utc=[t0],
            location_id=location_ids,
            step=self.steps,
        )

        if output_quantiles is not None:
            dims.append("quantile")
            coords["quantile"] = output_quantiles

        return xr.DataArray(data=preds[np.newaxis, ...], dims=dims, coords=coords)


def create_zarr_encoding(ds: xr.Dataset) -> dict:
    """Get the zarr encoding for the forecasts"""

    encoding = {d: {"chunks": (len(ds[d]),)} for d in ds.dims if d != "init_time_utc"}
    encoding["hindcast"] = {
        "chunks": tuple(
            1 if d=="init_time_utc" else len(ds[d]) 
            for d in ds.hindcast.dims
        ),
    }
    encoding["init_time_utc"] = {
        "dtype": "int", 
        "units": "nanoseconds since 1970-01-01", 
        "calendar": "proleptic_gregorian", 
        "chunks": (1000,),
    }

    return encoding

# ------------------------------------------------------------------
# RUN

app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def main(
    input_data_paths: str = typer.Option(
        ..., 
        help="Path to yaml file containing paths to input data zarrs. "
        "See example_backtest_data_config.yaml for format."
    ),
    output_dir: str = typer.Option(
        ..., 
        help="Working dir during backtest; final results saved to {output_dir}.zarr"
    ),
    pvnet_model_name: str = typer.Option(
        ...,
        help="PVNet model name or path"
    ),
    pvnet_model_version: str | None = typer.Option(
        ..., 
        help="PVNet model revision"
    ),
    summation_model_name: str | None = typer.Option(
        None, 
        help="Summation model name or path. Set to empty string to disable."
    ),
    summation_model_version: str | None = typer.Option(
        None, 
        help="Summation model revision"
    ),
    min_solar_elevation: float | None = typer.Option(
        None,
        help="Minimum solar elevation for forecasts. Forecasts are set to zero if solar elevation "
        "is less than this. Set parameter to None to disable."
    ),
    start_datetime: str | None = typer.Option(
        None, 
        help="Start datetime for backtest, e.g. '2022-01-01 00:00'"
    ),
    end_datetime: str | None = typer.Option(
        None, 
        help="End datetime for backtest, e.g. '2022-12-31 23:30'"
    ),
    device_name: str = typer.Option(
        None, 
        help="Device to use, e.g. 'cuda', 'cuda:0', 'cpu'. Defaults to cuda if available."
    ),
    num_workers: int = typer.Option(
        8, 
        help="Number of workers to use in dataloader. Default is 8. Set to 0 to disable "
        "multiprocessing."
    )
):
    """Run the backtest"""

    # Treat empty strings as None
    if summation_model_name == "":
        summation_model_name = None
    if summation_model_version == "":
        summation_model_version = None

    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    # Set up output dir for temporary files during backtest 
    os.makedirs(output_dir, exist_ok=False)

    model_data_config_filepath = f"{output_dir}/data_config.yaml"

    construct_model_data_config(
        pvnet_model_name=pvnet_model_name, 
        pvnet_model_version=pvnet_model_version, 
        input_data_paths=input_data_paths, 
        output_path=model_data_config_filepath
    )

    dataset = BacktestStreamedDataset(
        config_filename=model_data_config_filepath,
        time_periods=[[start_datetime, end_datetime]],
    )

    if num_workers>0:
        dataset.presave_pickle(f"{output_dir}/dataset.pkl")

    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers>0 else None,
        multiprocessing_context="forkserver" if num_workers>0 else None,
        shuffle=False,
        batch_size=None,
        sampler=None,
        batch_sampler=None,
        collate_fn=None,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        persistent_workers=False,                       
    )

    forecaster = Forecaster(
        pvnet_model_name=pvnet_model_name,
        pvnet_model_version=pvnet_model_version,
        summation_model_name=summation_model_name,
        summation_model_version=summation_model_version,
        min_solar_elevation=min_solar_elevation,
        device=device,
    )

    # Loop through the batches
    zarr_path = f"{output_dir}.zarr"
    pbar = tqdm(total=len(dataloader))
    
    for i, sample in enumerate(dataloader):
        # Make predictions for the init-time
        ds_abs_all = forecaster.predict(sample)
        
        # Save the results to zarr, appending if this is not the first forecast
        if i == 0:
            encoding = create_zarr_encoding(ds_abs_all)
            ds_abs_all.to_zarr(zarr_path, mode="w", encoding=encoding, consolidated=False)
        else:
            ds_abs_all.to_zarr(zarr_path, mode="a", append_dim="init_time_utc", consolidated=False)

        pbar.update()

    pbar.close()

    # Clean up temporary files
    shutil.rmtree(output_dir)


if __name__ == "__main__":
    app()