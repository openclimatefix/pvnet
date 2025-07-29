import os

import dask.array
import pytest
import pandas as pd
import numpy as np
import xarray as xr
import torch
import hydra

from ocf_data_sampler.torch_datasets.sample.site import SiteSample
from pvnet.data.base_datamodule import collate_fn
from ocf_data_sampler.numpy_sample.common_types import NumpySample, TensorBatch
from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration

from pvnet.data import  UKRegionalStreamedDataModule
from pvnet.models import LateFusionModel




_top_test_directory = os.path.dirname(os.path.realpath(__file__))


uk_sat_area_string = """msg_seviri_rss_3km:
    description: MSG SEVIRI Rapid Scanning Service area definition with 3 km resolution
    projection:
        proj: geos
        lon_0: 9.5
        h: 35785831
        x_0: 0
        y_0: 0
        a: 6378169
        rf: 295.488065897014
        no_defs: null
        type: crs
    shape:
        height: 298
        width: 615
    area_extent:
        lower_left_xy: [28503.830075263977, 5090183.970808983]
        upper_right_xy: [-1816744.1169023514, 4196063.827395439]
        units: m
    """


@pytest.fixture(scope="session")
def session_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def sat_zarr_path(session_tmp_path):
    variables = [
        "IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120",
        "IR_134", "VIS006", "VIS008", "WV_062", "WV_073",
    ]
    times = pd.date_range("2023-01-01 00:00", "2023-01-01 23:55", freq="5min")
    y = np.linspace(start=4191563, stop=5304712, num=100)
    x = np.linspace(start=15002, stop=-1824245, num=100)

    coords = (
        ("variable", variables),
        ("time", times),
        ("y_geostationary", y),
        ("x_geostationary", x),
    )

    data = dask.array.zeros(
        shape=tuple(len(coord_values) for _, coord_values in coords),
        chunks=(-1, 10, -1, -1),
        dtype=np.float32,
    )

    attrs = {"area": uk_sat_area_string}

    ds = xr.DataArray(data=data, coords=coords, attrs=attrs).to_dataset(name="data")

    zarr_path = session_tmp_path / "test_sat.zarr"
    ds.to_zarr(zarr_path)

    return zarr_path


@pytest.fixture(scope="session")
def ukv_zarr_path(session_tmp_path):
    init_times = pd.date_range(start="2023-01-01 00:00", freq="180min", periods=24 * 7)
    variables = ["si10", "dswrf", "t", "prate"]
    steps = pd.timedelta_range("0h", "24h", freq="1h")
    x = np.linspace(-239_000, 857_000, 50)
    y = np.linspace(-183_000, 1225_000, 100)
    
    coords = (
        ("init_time", init_times),
        ("variable", variables),
        ("step", steps),
        ("x", x),
        ("y", y),
    )

    data = dask.array.random.uniform(
        low=0,
        high=200,
        size=tuple(len(coord_values) for _, coord_values in coords),
        chunks=(1, -1, -1, 50, 50),
    ).astype(np.float32)

    ds = xr.DataArray(data=data, coords=coords).to_dataset(name="UKV")

    zarr_path = session_tmp_path / "ukv_nwp.zarr"
    ds.to_zarr(zarr_path)
    return zarr_path


@pytest.fixture(scope="session")
def ecmwf_zarr_path(session_tmp_path):
    init_times = pd.date_range(start="2023-01-01 00:00", freq="6h", periods=24 * 7)
    variables = ["t2m", "dswrf", "mcc"]
    steps = pd.timedelta_range("0h", "14h", freq="1h")
    lons = np.arange(-12.0, 3.0, 0.1)
    lats = np.arange(48.0, 65.0, 0.1)

    coords = (
        ("init_time", init_times),
        ("variable", variables),
        ("step", steps),
        ("longitude", lons),
        ("latitude", lats),
    )

    data = dask.array.random.uniform(
        low=0,
        high=200,
        size=tuple(len(coord_values) for _, coord_values in coords),
        chunks=(1, -1, -1, 50, 50),
    ).astype(np.float32)

    ds = xr.DataArray(data=data, coords=coords).to_dataset(name="ECMWF_UK")

    zarr_path = session_tmp_path / "ukv_ecmwf.zarr"
    ds.to_zarr(zarr_path)
    yield zarr_path


@pytest.fixture(scope="session")
def gsp_zarr_path(session_tmp_path):
    times = pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="30min")
    gsp_ids = np.arange(0, 318)
    capacity = np.ones((len(times), len(gsp_ids)))
    generation = np.random.uniform(0, 200, size=(len(times), len(gsp_ids))).astype(np.float32)

    coords = (
        ("datetime_gmt", times),
        ("gsp_id", gsp_ids),
    )

    ds_uk_gsp = xr.Dataset({
        "capacity_mwp": xr.DataArray(capacity, coords=coords),
        "installedcapacity_mwp": xr.DataArray(capacity, coords=coords),
        "generation_mw": xr.DataArray(generation, coords=coords),
    })

    zarr_path = session_tmp_path / "uk_gsp.zarr"
    ds_uk_gsp.to_zarr(zarr_path)
    return zarr_path


@pytest.fixture(scope="session")
def data_config_path(session_tmp_path, sat_zarr_path, ukv_zarr_path, ecmwf_zarr_path, gsp_zarr_path):  
    
    # Populate the config with the generated zarr paths
    config = load_yaml_configuration(f"{_top_test_directory}/test_data/data_config.yaml")
    config.input_data.nwp["ukv"].zarr_path = str(ukv_zarr_path)
    config.input_data.nwp["ecmwf"].zarr_path = str(ecmwf_zarr_path)
    config.input_data.satellite.zarr_path = str(sat_zarr_path)
    config.input_data.gsp.zarr_path = str(gsp_zarr_path)

    filename = f"{session_tmp_path}/data_config.yaml"
    save_yaml_configuration(config, filename)
    return filename


@pytest.fixture(scope="session")
def uk_streamed_datamodule(data_config_path):
    dm = UKRegionalStreamedDataModule(
        configuration=data_config_path,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
    )
    dm.setup(stage="fit")
    return dm


@pytest.fixture()
def sample_batch(uk_streamed_datamodule) -> TensorBatch:
    return next(iter(uk_streamed_datamodule.train_dataloader()))


@pytest.fixture()
def sample_satellite_batch(sample_batch) -> torch.Tensor:
    return torch.swapaxes(sample_batch["satellite_actual"], 1, 2).float()


def generate_synthetic_site_sample(
    site_id: int, 
    variation_index: int, 
    add_noise: bool,
) -> NumpySample:
    """Generate synthetic site sample that matches site sample structure

    Args:
        site_id: ID for the site
        variation_index: Index to use for coordinate variations
        add_noise: Whether to add random noise to data variables
    """
    now = pd.Timestamp.now(tz=None)

    # Create time and space coordinates
    site_time_coords = pd.date_range(start=now - pd.Timedelta("48h"), periods=197, freq="15min")
    nwp_time_coords = pd.date_range(start=now, periods=50, freq="1h")
    nwp_lat = np.linspace(50.0, 60.0, 24)
    nwp_lon = np.linspace(-10.0, 2.0, 24)
    nwp_channels = np.array(['t2m', 'ssrd', 'ssr', 'sp', 'r', 'tcc', 'u10', 'v10'], dtype='<U5')

    # Generate NWP data
    nwp_init_time = pd.date_range(start=now - pd.Timedelta("12h"), periods=1, freq="12h").repeat(50)
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

    return SiteSample(site_data_ds).to_numpy()


@pytest.fixture(scope="session")
def sample_site_batch() -> TensorBatch:
    # Generate and save synthetic samples
    samples  = []
    for i in range(2):
        samples.append(
            generate_synthetic_site_sample(
                site_id=i % 3 + 1,
                variation_index=i,
                add_noise=True
            )
        )
    return collate_fn(samples)


@pytest.fixture()
def model_minutes_kwargs() -> dict:
    return dict(forecast_minutes=480, history_minutes=60)

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
    kwargs = dict(
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
                in_channels=4,
                out_features=128,
                number_of_conv3d_layers=6,
                conv3d_channels=32,
                image_size_pixels=24,
            ),
            "ecmwf": dict(
                _target_="pvnet.models.late_fusion.encoders.encoders3d.DefaultPVNet",
                _partial_=True,
                in_channels=3,
                out_features=128,
                number_of_conv3d_layers=4,
                conv3d_channels=32,
                image_size_pixels=12,
            ),
        },
        
        add_image_embedding_channel=True,
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
        nwp_history_minutes={"ukv": 120, "ecmwf": 120},
        nwp_forecast_minutes={"ukv": 480, "ecmwf": 480},
        min_sat_delay_minutes=0,
        **model_minutes_kwargs,
    )
    return kwargs


@pytest.fixture()
def late_fusion_model_kwargs(raw_late_fusion_model_kwargs) -> dict:
    return hydra.utils.instantiate(raw_late_fusion_model_kwargs)


@pytest.fixture()
def late_fusion_model(late_fusion_model_kwargs) -> LateFusionModel:
    return LateFusionModel(**late_fusion_model_kwargs)


@pytest.fixture()
def raw_late_fusion_model_kwargs_site_history(model_minutes_kwargs) -> dict:
    return dict(
        # Set inputs to None/False apart from site history
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
