"""
A script to run backtest for PVNet and the summation model for UK regional and national

Use:

- This script uses exported PVNet and PVNet summation models stored either locally or on huggingface
- The save directory, model paths, the backtest time range, the input data paths, and number of 
  workers used are near the top of the script as hard-coded user variables. These should be changed.


```
python backtest_uk_gsp.py
```

"""

import logging
import os

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from ocf_data_sampler.torch_datasets.sample.base import batch_to_tensor, copy_batch_to_device
from pvnet_summation.data.datamodule import StreamedDataset
from pvnet_summation.models.base_model import BaseModel as SummationBaseModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from pvnet.models.base_model import BaseModel as PVNetBaseModel

# ------------------------------------------------------------------
# USER CONFIGURED VARIABLES
output_dir = "/home/james/tmp/test_backtest/pvnet_v2"

# Number of workers to use in the dataloader
num_workers = 16

# Location of the exported PVNet and summation model pair
pvnet_model_name: str = "openclimatefix/pvnet_uk_region"
pvnet_model_version: str | None = "ff09e4aee871fe094d3a2dabe9d9cea50e4b5485"

summation_model_name: str = "openclimatefix/pvnet_v2_summation"
summation_model_version: str | None = "d746683893330fe3380e57e65d40812daa343c8e"

# Forecasts will be made for all available init times between these
start_datetime: str | None = "2022-01-01 00:00" 
end_datetime: str | None = "2022-12-31 23:30"

# The paths to the input data for the backtest
backtest_paths = {
    "gsp": "/mnt/raphael/fast/crops/pv/pvlive_gsp_new_boundaries_2019-2025.zarr",
    "nwp": {
        "ukv": [
            "/mnt/raphael/fast/crops/nwp/ukv/UKV_v7/UKV_intermediate_version_7.1.zarr",
            "/mnt/raphael/fast/crops/nwp/ukv/UKV_v7/UKV_2021_missing.zarr",
            "/mnt/raphael/fast/crops/nwp/ukv/UKV_v7/UKV_2022.zarr",
        ], 
        "ecmwf": [
            "/mnt/raphael/fast/crops/nwp/ecmwf/uk_v3/ECMWF_2019.zarr",
            "/mnt/raphael/fast/crops/nwp/ecmwf/uk_v3/ECMWF_2020.zarr",
            "/mnt/raphael/fast/crops/nwp/ecmwf/uk_v3/ECMWF_2021.zarr",
            "/mnt/raphael/fast/crops/nwp/ecmwf/uk_v3/ECMWF_2022.zarr",
        ], 
        "cloudcasting": "/mnt/raphael/fast/cloudcasting/simvp.zarr",
    },
    "satellite": [
        "/mnt/raphael/fast/crops/sat/uk_sat_crops/v1/2019_nonhrv.zarr",
        "/mnt/raphael/fast/crops/sat/uk_sat_crops/v1/2020_nonhrv.zarr",
        "/mnt/raphael/fast/crops/sat/uk_sat_crops/v1/2021_nonhrv.zarr",
        "/mnt/raphael/fast/crops/sat/uk_sat_crops/v1/2022_nonhrv.zarr",
    ],
}

# When sun as elevation below this, the forecast is set to zero
MIN_DAY_ELEVATION = 0

# ------------------------------------------------------------------

logger = logging.getLogger(__name__)

# This will run on GPU if it exists
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# FUNCTIONS

_model_mismatch_msg = (
    "The PVNet version running in this app is {}/{}. The summation model running in this app was "
    "trained on outputs from PVNet version {}/{}. Combining these models may lead to an error if "
    "the shape of PVNet output doesn't match the expected shape of the summation model. Combining "
    "may lead to unreliable results even if the shapes match."
)

def populate_config_with_data_data_filepaths(config: dict) -> dict:
    """Populate the data source filepaths in the config

    Args:
        config: The data config
    """

    # Replace the GSP data path
    config["input_data"]["gsp"]["zarr_path"] =  backtest_paths["gsp"]

    # Replace satellite data path if using it
    if "satellite" in config["input_data"]:
        if config["input_data"]["satellite"]["zarr_path"] != "":
            config["input_data"]["satellite"]["zarr_path"] = backtest_paths["satellite"]

    # NWP is nested so much be treated separately
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():
            provider = nwp_config[nwp_source]["provider"]
            assert provider in backtest_paths["nwp"], f"Missing NWP path: {provider}"
            nwp_config[nwp_source]["zarr_path"] = backtest_paths["nwp"][provider]

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


class BacktestStreamedDataset(StreamedDataset):
    """A torch dataset object used only for backtesting"""

    def _get_sample(self, t0: pd.Timestamp) -> ...:
        """Generate a concurrent PVNet sample for given init-time + augment for backtesting.

        Args:
            t0: init-time for sample
        """

        sample = super()._get_sample(t0)

        total_capacity = self.national_gsp_data.sel(time_utc=t0).effective_capacity_mwp.item()

        sample.update(
            {
                "backtest_t0": t0,
                "backtest_national_capacity": total_capacity,
            }
        )

        return sample


class Forecaster:
    """Class for making and solar forecasts for all GB GSPs and national total"""

    def __init__(self):
        """Class for making and solar forecasts for all GB GSPs and national total
        """
        
        # Load the GSP-level model
        self.model = PVNetBaseModel.from_pretrained(
            model_id=pvnet_model_name,
            revision=pvnet_model_version,
        ).to(device).eval()

        # Load the summation model
        self.sum_model = SummationBaseModel.from_pretrained(
            model_id=summation_model_name,
            revision=summation_model_version,
        ).to(device).eval()

        # Compare the current GSP model with the one the summation model was trained on
        datamodule_path = SummationBaseModel.get_datamodule_config(
            model_id=summation_model_name,
            revision=summation_model_version,
        )
        with open(datamodule_path) as cfg:
            sum_pvnet_cfg = yaml.load(cfg, Loader=yaml.FullLoader)["pvnet_model"]

        sum_expected_gsp_model = (sum_pvnet_cfg["model_id"], sum_pvnet_cfg["revision"])
        this_gsp_model = (pvnet_model_name, pvnet_model_version)

        if sum_expected_gsp_model != this_gsp_model:
            logger.warning(_model_mismatch_msg.format(*this_gsp_model, *sum_expected_gsp_model))

        # These are the steps this forecast will predict for
        self.steps = pd.timedelta_range(
            start="30min", 
            freq="30min", 
            periods=self.model.forecast_len,
        )

    @torch.inference_mode()
    def predict(self, sample: dict) -> xr.Dataset:
        """Make predictions for the batch and store results internally"""
        
        x = copy_batch_to_device(batch_to_tensor(sample["pvnet_inputs"]), device)
        
        # Run batch through model
        normed_preds = self.model(x).detach().cpu().numpy()

        # Calculate sun mask
        # The dataloader normalises solar elevation data to the range [0, 1]
        elevation_degrees = (sample["pvnet_inputs"]["solar_elevation"] - 0.5) * 180
        # We only need elevation mask for forecasted values, not history
        elevation_degrees = elevation_degrees[:, -normed_preds.shape[1]:]
        sun_down_masks = elevation_degrees < MIN_DAY_ELEVATION

        # Convert GSP results to xarray DataArray
        t0 = sample["backtest_t0"]
        gsp_ids = sample["pvnet_inputs"]["gsp_id"]

        da_normed = self.to_dataarray(
            normed_preds,
            t0,
            gsp_ids,
            self.model.output_quantiles,
        )

        da_sundown_mask = self.to_dataarray(sun_down_masks, t0, gsp_ids, None)

        # Multiply normalised forecasts by capacities and clip negatives
        da_abs = (
            da_normed.clip(0, None) 
            * sample["pvnet_inputs"]["gsp_effective_capacity_mwp"][None, :, None, None].numpy()
        )
        
        # Apply sundown mask
        da_abs = da_abs.where(~da_sundown_mask).fillna(0.0)

        # Make national predictions using summation model
        # - Need to add batch dimension and convert to torch tensors on device
        sample["pvnet_outputs"] = torch.tensor(normed_preds[None]).to(device)
        sample["relative_capacity"] = sample["relative_capacity"][None].to(device)
        normed_national = self.sum_model(sample).detach().squeeze().cpu().numpy()

        # Convert national predictions to DataArray
        da_normed_national = self.to_dataarray(
            normed_national[np.newaxis],
            t0,
            gsp_ids=[0],
            output_quantiles=self.sum_model.output_quantiles,
        )

        # Multiply normalised forecasts by capacity and clip negatives
        national_capacity = sample["backtest_national_capacity"]
        da_abs_national = da_normed_national.clip(0, None) * national_capacity

        # Apply sundown mask - All GSPs must be masked to mask national
        da_abs_national = da_abs_national.where(~da_sundown_mask.all(dim="gsp_id")).fillna(0.0)

        # Convert to Dataset and add attrs about the models used
        ds_result = xr.concat([da_abs_national, da_abs], dim="gsp_id").to_dataset(name="hindcast")
        ds_result.attrs.update(
            {
                "pvnet_model_name": pvnet_model_name,
                "pvnet_model_version": pvnet_model_version,
                "summation_model_name": summation_model_name,
                "summation_model_version": summation_model_version,
            }
        )

        return ds_result

    def to_dataarray(
        self,
        preds: np.ndarray,
        t0: pd.Timestamp,
        gsp_ids: list[int],
        output_quantiles: list[float] | None,
    ) -> xr.DataArray:
        """Put numpy array of predictions into a dataarray"""

        dims = ["init_time_utc", "gsp_id", "step"]
        coords = dict(
            init_time_utc=[t0],
            gsp_id=gsp_ids,
            step=self.steps,
        )

        if output_quantiles is not None:
            dims.append("quantile")
            coords["quantile"] = output_quantiles

        return xr.DataArray(data=preds[np.newaxis, ...], dims=dims, coords=coords)

# ------------------------------------------------------------------
# RUN

if __name__=="__main__":

    # Set up output dir
    os.makedirs(output_dir)

    data_config_path = PVNetBaseModel.get_data_config(
        model_id=pvnet_model_name,
        revision=pvnet_model_version,
    )

    with open(data_config_path) as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)

    data_config = populate_config_with_data_data_filepaths(data_config)
    data_config = overwrite_config_dropouts(data_config)

    modified_data_config_filepath = f"{output_dir}/data_config.yaml"

    with open(modified_data_config_filepath, "w") as file:
        yaml.dump(data_config, file, default_flow_style=False)


    dataset = BacktestStreamedDataset(
        config_filename=modified_data_config_filepath,
        start_time=start_datetime,
        end_time=end_datetime,
    )

    dataloader_kwargs = dict(
        num_workers=num_workers,
        prefetch_factor=2 if num_workers>0 else None,
        multiprocessing_context="spawn" if num_workers>0 else None,
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

    if num_workers>0:
        dataset.presave_pickle(f"{output_dir}/dataset.pkl")

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    forecaster = Forecaster()

    # Loop through the batches
    pbar = tqdm(total=len(dataloader))
    for sample in dataloader:
        # Make predictions for the init-time
        ds_abs_all = forecaster.predict(sample)

        # Save the predictions
        t0 = pd.Timestamp(ds_abs_all.init_time_utc.item())
        filename = f"{output_dir}/{t0}.nc"
        ds_abs_all.to_netcdf(filename)

        pbar.update()

    # Close down
    pbar.close()

    # Clean up
    if num_workers>0:
        os.remove(f"{output_dir}/dataset.pkl")