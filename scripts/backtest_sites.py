"""
A script to run backtest for PVNet for specific sites

Use:

- This script uses hydra to construct the config, just like in `run.py`. So you need to make sure
  that the data config is set up appropriate for the model being run in this script
- The following variables are hard coded near the top of the script and should be changed prior to
  use:
  - number of workers to use;
  - The PVNet model checkpoint (either local or HuggingFace repo details);
  - the time range over which predictions are made;
  - the output directory where the results are stored;

  Outputs netCDF files with the predictions for each t0 in seperate files,
  each file has forecasts for all sites.
  Time resolution of the forecast t0s is the same as the time resolution of the generation data. 

  Warning: this script currently assumes that if you are running the backtest for multiple sites
  (generation data being used has multiple sites).
  that they will all have the same t0s available in generation data,
  if they have non overlapping periods may be best to run this multiple times with
  different generation files for each site

```
python scripts/backtest_sites.py
```

"""
import os

import hydra
import numpy as np
import pandas as pd
import torch
import xarray as xr
from ocf_data_sampler.config import load_yaml_configuration
from ocf_data_sampler.load.load_dataset import get_dataset_dict
from ocf_data_sampler.numpy_sample.common_types import NumpyBatch
from ocf_data_sampler.torch_datasets.datasets.site import SitesDatasetConcurrent
from ocf_data_sampler.torch_datasets.sample.base import batch_to_tensor, copy_batch_to_device
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from pvnet.load_model import get_model_from_checkpoints
from pvnet.models.base_model import BaseModel as PVNetBaseModel

# ------------------------------------------------------------------
# USER CONFIGURED VARIABLES TO RUN THE SCRIPT

num_workers = 2

# Directory path to save results
output_dir: str = "example_repo"

# Local directory to load the PVNet checkpoint from. By default this should pull the best performing
# checkpoint on the val set, set to None if using HF
model_checkpoint_dir: str | None = None


# Location to download exported PVNet model on HF, set to None if using local
hf_model_id: str | None = "openclimatefix/example_repo"
hf_revision: str | None = "95b1658c2b771e567fb3a0379e9bd600e0b1d209"

# Forecasts will be made for all available init times between these
start_datetime = "2024-06-05 00:00"
end_datetime = "2024-06-05 03:00"

# ------------------------------------------------------------------
# DERIVED VARIABLES

# This will run on GPU if it exists
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# GLOBAL VARIABLES

# When sun as elevation below this, the forecast is set to zero
MIN_DAY_ELEVATION = 0

# ------------------------------------------------------------------
# FUNCTIONS

def preds_to_dataarray(preds, model, valid_times, site_ids):
    """Put numpy array of predictions into a dataarray"""

    if model.use_quantile_regression:
        output_labels = [f"forecast_mw_plevel_{int(q*100):02}" for q in model.output_quantiles]
        output_labels[output_labels.index("forecast_mw_plevel_50")] = "forecast_mw"
    else:
        output_labels = ["forecast_mw"]
        preds = preds[..., np.newaxis]
    da = xr.DataArray(
        data=preds,
        dims=["site_id", "target_datetime_utc", "output_label"],
        coords=dict(
            site_id = site_ids,
            target_datetime_utc=valid_times,
            output_label=output_labels,
        ),
    )
    return da

def get_sites_ds(config_path: str) -> xr.Dataset:
    """Load site data from the path in the data config.

    Args:
        config_path: Path to the data configuration file

    Returns:
        xarray.Dataset of PV sites data
    """
    config = load_yaml_configuration(config_path)
    datasets_dict = get_dataset_dict(config.input_data)
    return datasets_dict["site"].to_dataset(name="site")


class ModelPipe:
    """A class to conveniently make and process predictions from batches"""

    def __init__(self, model, ds_site: xr.Dataset, interval_start, interval_end, time_resolution):
        """A class to conveniently make and process predictions from batches

        Args:
            model: PVNet site level model
            ds_site: xarray dataset of pv site true values and capacities
            interval_start: The start timestamp (inclusive) for the prediction interval.
            interval_end: The end timestamp (exclusive) for the prediction interval.
            time_resolution: The time resolution (e.g., in minutes) for the prediction intervals.

        """
        self.model = model
        self.ds_site = ds_site
        self.interval_start = interval_start
        self.interval_end = interval_end
        self.time_resolution = time_resolution

    def predict_batch(self, batch: NumpyBatch) -> xr.Dataset:
        """Run the batch through the model and compile the predictions into an xarray DataArray

        Args:
            batch: A batch containing inputs for a site

        Returns:
            xarray.Dataset of site forecasts for the sample
        """

        tensor_batch = batch_to_tensor(batch)
        # First available timestamp in the sample (this is t0 + interval_start)
        first_time = pd.Timestamp(tensor_batch["site_time_utc"][0][0].item())
        # Compute t0 (true start of forecast)
        t0 = first_time - pd.Timedelta(self.interval_start)

        # Generate valid times for inference (only t0 to t0 + interval_end)
        valid_times = pd.date_range(
            start=t0 + pd.Timedelta(self.time_resolution.astype(int), "min"),
            end=t0 + pd.Timedelta(self.interval_end),
            freq=f"{self.time_resolution.astype(int)}min",
        )
        # Get capacity for this site
        site_capacities = [float(i) for i in self.ds_site["capacity_kwp"].values]
        # Get solar elevation and create sundown mask
        elevation = (tensor_batch['solar_elevation'] - 0.5) * 180
        # We only need elevation mask for forecasted values, not history
        elevation = elevation[:, -valid_times.shape[0]:]
        site_ids = self.ds_site["site_id"].values

        da_sundown_mask = xr.DataArray(
            data=elevation < MIN_DAY_ELEVATION,
            dims=["site_id", "target_datetime_utc"],
            coords=dict(site_id=site_ids,
                        target_datetime_utc=valid_times,
            ),
        )
        with torch.no_grad():
            # Run through model to get 0-1 predictions
            tensor_batch = copy_batch_to_device(tensor_batch, device)
            y_normed = self.model(tensor_batch).detach().cpu().numpy()

        da_normed = preds_to_dataarray(y_normed, self.model, valid_times, site_ids)

        # Multiply normalised forecasts by capacity and clip negatives
        # Define multipliers for each id
        capacity_multipliers = xr.DataArray(
            data=site_capacities,
            dims=["site_id"],
            coords={"site_id": site_ids}
        )
        da_abs = da_normed.clip(0, None) * capacity_multipliers

        # Apply sundown mask
        da_abs = da_abs.where(~da_sundown_mask).fillna(0.0)
        da_abs = da_abs.expand_dims(dim="init_time_utc", axis=0).assign_coords(
            init_time_utc=np.array([t0], dtype="datetime64[ns]")
        )

        return da_abs


@hydra.main(config_path="../configs", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    """Runs the backtest"""

    dataloader_kwargs = dict(
        shuffle=False,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers>0 else None,
        multiprocessing_context="spawn" if num_workers>0 else None,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
        sampler=None,
        batch_sampler=None,
        collate_fn=None,
        timeout=0,
        worker_init_fn=None,
    )

    # Set up output dir
    os.makedirs(output_dir)

    # load yaml file
    unpacked_configuration = load_yaml_configuration(config.datamodule.configuration)

    interval_start = np.timedelta64(
        unpacked_configuration.input_data.site.interval_start_minutes, "m"
    )
    interval_end = np.timedelta64(unpacked_configuration.input_data.site.interval_end_minutes, "m")
    time_resolution = np.timedelta64(
        unpacked_configuration.input_data.site.time_resolution_minutes, "m"
    )

    # Create dataset
    dataset = SitesDatasetConcurrent(
        config.datamodule.configuration, start_time=start_datetime, end_time=end_datetime
    )

    # Load the site data
    ds_sites = get_sites_ds(config.datamodule.configuration)

    # Create a dataloader
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    # Load the PVNet model
    if model_checkpoint_dir:
        model, *_ = get_model_from_checkpoints([model_checkpoint_dir], val_best=True)
        model.eval()
        model.to(device)
    elif hf_model_id:
        model = PVNetBaseModel.from_pretrained(
        model_id=hf_model_id,
        revision=hf_revision,).to(device).eval()
    else:
        raise ValueError("Provide a model checkpoint or a HuggingFace model")

    # Create object to make predictions
    model_pipe = ModelPipe(model, ds_sites, interval_start, interval_end, time_resolution)

    # Loop through the batches
    pbar = tqdm(total=len(dataset))
    for i, batch in enumerate(dataloader):
        try:
            # Make predictions
            ds_abs_all = model_pipe.predict_batch(batch)
            t0 = ds_abs_all.init_time_utc.values[0]
            # Save the predictions
            filename = f"{output_dir}/{t0}.nc"
            ds_abs_all.to_netcdf(filename)

            pbar.update()
        except Exception as e:
            print(f"Exception {e} at batch {i}")
            pass

    pbar.close()
    del dataloader


if __name__ == "__main__":
    main()
