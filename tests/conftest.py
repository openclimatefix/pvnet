import os
import tempfile

import pytest
import pandas as pd
import numpy as np
import xarray as xr
import torch
import hydra

from pvnet.data import SitePresavedDataModule, UKRegionalPresavedDataModule
from pvnet.models import LateFusionModel
from ocf_data_sampler.numpy_sample.common_types import NumpySample, TensorBatch



def time_before_present(dt: pd.Timedelta) -> pd.Timestamp:
    return pd.Timestamp.now(tz=None) - dt


@pytest.fixture
def nwp_data() -> xr.DataArray:
    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/sample_data/nwp_shell.zarr"
    )

    # Last init time was at least 2 hours ago and hour to 3-hour interval
    t0_datetime_utc = time_before_present(pd.Timedelta("2H")).floor("3H")
    ds.init_time.values[:] = pd.date_range(
        t0_datetime_utc - pd.Timedelta(hours=3 * (len(ds.init_time) - 1)),
        t0_datetime_utc,
        freq=pd.Timedelta("3H"),
    )

    # This is important to avoid saving errors
    for v in list(ds.coords.keys()):
        if ds.coords[v].dtype == object:
            ds[v].encoding.clear()

    for v in list(ds.variables.keys()):
        if ds[v].dtype == object:
            ds[v].encoding.clear()

    # Add data to dataset
    ds["UKV"] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.coords]),
        coords=ds.coords,
    )

    # Add stored attributes to DataArray
    ds.UKV.attrs = ds.attrs["_data_attrs"]
    del ds.attrs["_data_attrs"]

    return ds


@pytest.fixture()
def sat_data() -> xr.DataArray:
    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/sample_data/non_hrv_shell.zarr"
    )

    # Change times so they lead up to present. Delayed by at most 1 hour
    t0_datetime_utc = time_before_present(pd.Timedelta(0)).floor("30min")
    t0_datetime_utc = t0_datetime_utc - pd.Timedelta("30min")
    ds.time.values[:] = pd.date_range(
        t0_datetime_utc - pd.Timedelta(minutes=5 * (len(ds.time) - 1)),
        t0_datetime_utc,
        freq=pd.Timedelta("5min"),
    )

    # Add data to dataset
    ds["data"] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.coords]),
        coords=ds.coords,
    )

    # Add stored attributes to DataArray
    ds.data.attrs = ds.attrs["_data_attrs"]
    del ds.attrs["_data_attrs"]

    return ds


def generate_synthetic_sample() -> NumpySample:
    """Generate synthetic sample for testing
    """
    now = pd.Timestamp.now(tz=None)
    sample = {}

    # NWP define
    sample["nwp"] = {
        "ukv": {
            "nwp": torch.rand(11, 11, 24, 24),
            "nwp_init_time_utc": torch.tensor(
                [(now - pd.Timedelta(hours=i)).timestamp() for i in range(11)]
            ),
            "nwp_step": torch.arange(11, dtype=torch.float32),
            "nwp_target_time_utc": torch.tensor(
                [(now + pd.Timedelta(hours=i)).timestamp() for i in range(11)]
            ),
            "nwp_y_osgb": torch.linspace(0, 100, 24),
            "nwp_x_osgb": torch.linspace(0, 100, 24),
        },
        "ecmwf": {
            "nwp": torch.rand(11, 12, 12, 12),
            "nwp_init_time_utc": torch.tensor(
                [(now - pd.Timedelta(hours=i)).timestamp() for i in range(11)]
            ),
            "nwp_step": torch.arange(11, dtype=torch.float32),
            "nwp_target_time_utc": torch.tensor(
                [(now + pd.Timedelta(hours=i)).timestamp() for i in range(11)]
            ),
        },
        "sat_pred": {
            "nwp": torch.rand(12, 11, 24, 24),
            "nwp_init_time_utc": torch.tensor(
                [(now - pd.Timedelta(hours=i)).timestamp() for i in range(12)]
            ),
            "nwp_step": torch.arange(12, dtype=torch.float32),
            "nwp_target_time_utc": torch.tensor(
                [(now + pd.Timedelta(hours=i)).timestamp() for i in range(12)]
            ),
        },
    }

    # Satellite define
    sample["satellite_actual"] = torch.rand(7, 11, 24, 24)
    sample["satellite_time_utc"] = torch.tensor(
        [(now - pd.Timedelta(minutes=5*i)).timestamp() for i in range(7)]
    )
    sample["satellite_x_geostationary"] = torch.linspace(0, 100, 24)
    sample["satellite_y_geostationary"] = torch.linspace(0, 100, 24)

    # GSP define
    sample["gsp"] = torch.rand(21)
    sample["gsp_nominal_capacity_mwp"] = torch.tensor(100.0)
    sample["gsp_effective_capacity_mwp"] = torch.tensor(85.0)
    sample["gsp_time_utc"] = torch.tensor(
        [(now + pd.Timedelta(minutes=30*i)).timestamp() for i in range(21)]
    )
    sample["gsp_t0_idx"] = float(7)
    sample["gsp_id"] = 12
    sample["gsp_x_osgb"] = 123456.0
    sample["gsp_y_osgb"] = 654321.0

    # Solar position define
    sample["solar_azimuth"] = torch.linspace(0, 180, 21)
    sample["solar_elevation"] = torch.linspace(-10, 60, 21)

    return sample


def generate_synthetic_site_sample(
    site_id: int, 
    variation_index: int, 
    add_noise: bool,
) -> xr.Dataset:
    """Generate synthetic site sample that matches site sample structure

    Args:
        site_id: ID for the site
        variation_index: Index to use for coordinate variations
        add_noise: Whether to add random noise to data variables
    """
    now = pd.Timestamp.now(tz=None)

    # Create time and space coordinates
    site_time_coords = pd.date_range(start=now - pd.Timedelta("48H"), periods=197, freq="15min")
    nwp_time_coords = pd.date_range(start=now, periods=50, freq="1h")
    nwp_lat = np.linspace(50.0, 60.0, 24)
    nwp_lon = np.linspace(-10.0, 2.0, 24)
    nwp_channels = np.array(['t2m', 'ssrd', 'ssr', 'sp', 'r', 'tcc', 'u10', 'v10'], dtype='<U5')

    # Generate NWP data
    nwp_init_time = pd.date_range(start=now - pd.Timedelta("12H"), periods=1, freq="12h").repeat(50)
    nwp_steps = pd.timedelta_range(start=pd.Timedelta(0), periods=50, freq="1h")
    nwp_data = np.random.randn(50, 8, 24, 24).astype(np.float32)

    # Generate site data and solar position
    site_data = np.random.rand(197)
    site_lat = 52.5 + variation_index * 0.1
    site_lon = -1.5 - variation_index * 0.05
    site_capacity = 10000.0 * (1.0 + variation_index * 0.01)

    # Calculate time features
    days_since_jan1 = (site_time_coords.dayofyear - 1) / 365.0
    hours_since_midnight = (site_time_coords.hour + site_time_coords.minute / 60.0) / 24.0

    # Calculate trigonometric features
    site_solar_azimuth = np.linspace(0, 360, 197)
    site_solar_elevation = 15 * np.sin(np.linspace(0, 2*np.pi, 197))
    trig_features = {
        "date_sin": np.sin(2 * np.pi * days_since_jan1),
        "date_cos": np.cos(2 * np.pi * days_since_jan1),
        "time_sin": np.sin(2 * np.pi * hours_since_midnight),
        "time_cos": np.cos(2 * np.pi * hours_since_midnight),
    }

    # Create xarray Dataset with all coordinates
    site_data_ds = xr.Dataset(
        data_vars={
            "nwp-ecmwf": (["nwp-ecmwf__target_time_utc", "nwp-ecmwf__channel",
                           "nwp-ecmwf__longitude", "nwp-ecmwf__latitude"], nwp_data),
            "site": (["site__time_utc"], site_data),
        },
        coords={
            # NWP coordinates
            "nwp-ecmwf__latitude": nwp_lat,
            "nwp-ecmwf__longitude": nwp_lon,
            "nwp-ecmwf__channel": nwp_channels,
            "nwp-ecmwf__target_time_utc": nwp_time_coords,
            "nwp-ecmwf__init_time_utc": (["nwp-ecmwf__target_time_utc"], nwp_init_time),
            "nwp-ecmwf__step": (["nwp-ecmwf__target_time_utc"], nwp_steps),

            # Site coordinates
            "site__site_id": np.int32(site_id),
            "site__latitude": site_lat,
            "site__longitude": site_lon,
            "site__capacity_kwp": site_capacity,
            "site__time_utc": site_time_coords,
            "site__solar_azimuth": (["site__time_utc"], site_solar_azimuth),
            "site__solar_elevation": (["site__time_utc"], site_solar_elevation),
            **{f"site__{k}": (["site__time_utc"], v) for k, v in trig_features.items()}
        }
    )

    # Add random noise to data variables if stated
    if add_noise:
        for var in ["site", "nwp-ecmwf"]:
            noise_shape = site_data_ds[var].shape
            noise = np.random.randn(*noise_shape).astype(site_data_ds[var].dtype) * 0.01
            site_data_ds[var] = site_data_ds[var] + noise

    return site_data_ds


@pytest.fixture()
def sample_train_val_datamodule() -> UKRegionalPresavedDataModule:
    """Create a DataModule with synthetic data files for training and validation
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create train and val directories
        os.makedirs(f"{tmpdirname}/train", exist_ok=True)
        os.makedirs(f"{tmpdirname}/val", exist_ok=True)

        # Generate and save synthetic samples
        for i in range(10):
            sample = generate_synthetic_sample()
            torch.save(sample, f"{tmpdirname}/train/{i:08d}.pt")
            torch.save(sample, f"{tmpdirname}/val/{i:08d}.pt")

        # Define DataModule with temporary directory
        dm = UKRegionalPresavedDataModule(
            sample_dir=tmpdirname,
            batch_size=2,
            num_workers=0,
            prefetch_factor=None,
        )

        yield dm


@pytest.fixture()
def sample_site_datamodule() -> SitePresavedDataModule:
    """
    Create a SiteDataModule with synthetic site data in netCDF format
    that matches the structure of the actual site samples
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create train and val directories
        os.makedirs(f"{tmpdirname}/train", exist_ok=True)
        os.makedirs(f"{tmpdirname}/val", exist_ok=True)

        # Generate and save synthetic samples
        for i in range(10):
            site_data = generate_synthetic_site_sample(
                site_id=i % 3 + 1,
                variation_index=i,
                add_noise=True
            )

            # Save as netCDF format for both train and val
            for subset in ["train", "val"]:
                file_path = f"{tmpdirname}/{subset}/{i:08d}.nc"
                site_data.to_netcdf(file_path, mode="w", engine="h5netcdf")

        # Define SiteDataModule with temporary directory
        dm = SitePresavedDataModule(
            sample_dir=tmpdirname,
            batch_size=2,
            num_workers=0,
            prefetch_factor=None,
        )

        yield dm


@pytest.fixture()
def sample_batch(sample_train_val_datamodule) -> TensorBatch:
    return next(iter(sample_train_val_datamodule.train_dataloader()))


@pytest.fixture()
def sample_satellite_batch(sample_batch) -> torch.Tensor:
    return torch.swapaxes(sample_batch["satellite_actual"], 1, 2)


@pytest.fixture()
def sample_site_batch(sample_site_datamodule) -> TensorBatch:
    return next(iter(sample_site_datamodule.train_dataloader()))


@pytest.fixture()
def model_minutes_kwargs() -> dict:
    return dict(
        forecast_minutes=480,
        history_minutes=120,
    )


@pytest.fixture()
def encoder_model_kwargs() -> dict:
    # Used to test encoder model on satellite data
    return dict(
        sequence_length=7,  # 30 minutes of 5 minutely satellite data = 7 time steps
        image_size_pixels=24,
        in_channels=11,
        out_features=128,
    )


@pytest.fixture()
def site_encoder_model_kwargs() -> dict:
    """Used to test site encoder model on PV data with data sampler"""
    return dict(
        sequence_length=60 // 15 + 1,
        num_sites=1,
        out_features=128,
        target_key_to_use="site"
    )


@pytest.fixture()
def raw_late_fusion_model_kwargs(model_minutes_kwargs) -> dict:
    return dict(
        sat_encoder=dict(
            _target_="pvnet.models.late_fusion.encoders.encoders3d.DefaultPVNet",
            _partial_=True,
            in_channels=11,
            out_features=128,
            number_of_conv3d_layers=6,
            conv3d_channels=32,
            image_size_pixels=24,
        ),
        nwp_encoders_dict={
            "ukv": dict(
                _target_="pvnet.models.late_fusion.encoders.encoders3d.DefaultPVNet",
                _partial_=True,
                in_channels=11,
                out_features=128,
                number_of_conv3d_layers=6,
                conv3d_channels=32,
                image_size_pixels=24,
            ),
        },
        add_image_embedding_channel=True,
        # ocf-data-sampler doesn't supprt PV site inputs yet
        pv_encoder=None,
        output_network=dict(
            _target_="pvnet.models.late_fusion.linear_networks.networks.ResFCNet",
            _partial_=True,
            fc_hidden_features=128,
            n_res_blocks=6,
            res_block_layers=2,
            dropout_frac=0.0,
        ),
        location_id_mapping={i:i for i in range(1, 318)},
        embedding_dim=16,
        include_sun=True,
        include_gsp_yield_history=True,
        sat_history_minutes=30,
        nwp_history_minutes={"ukv": 120},
        nwp_forecast_minutes={"ukv": 480},
        min_sat_delay_minutes=0,
        **model_minutes_kwargs,
    )


@pytest.fixture()
def late_fusion_model_kwargs(raw_late_fusion_model_kwargs) -> dict:
    return hydra.utils.instantiate(raw_late_fusion_model_kwargs)


@pytest.fixture()
def late_fusion_model(late_fusion_model_kwargs) -> LateFusionModel:
    return LateFusionModel(**late_fusion_model_kwargs)


@pytest.fixture()
def raw_late_fusion_model_kwargs_site_history(model_minutes_kwargs) -> dict:
    return dict(
        # setting inputs to None/False apart from site history
        sat_encoder=None,
        nwp_encoders_dict=None,
        add_image_embedding_channel=False,
        pv_encoder=None,
        output_network=dict(
            _target_="pvnet.models.late_fusion.linear_networks.networks.ResFCNet",
            _partial_=True,
            fc_hidden_features=128,
            n_res_blocks=6,
            res_block_layers=2,
            dropout_frac=0.0,
        ),
        location_id_mapping=None,
        embedding_dim=None,
        include_sun=False,
        include_gsp_yield_history=False,
        include_site_yield_history=True,
        **model_minutes_kwargs
    )


@pytest.fixture()
def late_fusion_model_kwargs_site_history(raw_late_fusion_model_kwargs_site_history) -> dict:
    return hydra.utils.instantiate(raw_late_fusion_model_kwargs_site_history)


@pytest.fixture()
def late_fusion_model_site_history(late_fusion_model_kwargs_site_history) -> LateFusionModel:
    return LateFusionModel(**late_fusion_model_kwargs_site_history)


@pytest.fixture()
def late_fusion_quantile_model(late_fusion_model_kwargs) -> LateFusionModel:
    return LateFusionModel(output_quantiles=[0.1, 0.5, 0.9], **late_fusion_model_kwargs)
