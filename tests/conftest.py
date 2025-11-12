import os

import dask.array
import hydra
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.numpy_sample.common_types import TensorBatch
from omegaconf import OmegaConf

from pvnet.datamodule import PVNetDataModule
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
def sat_zarr_path(session_tmp_path) -> str:
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
def ukv_zarr_path(session_tmp_path) -> str:
    init_times = pd.date_range(start="2023-01-01 00:00", freq="180min", periods=24 * 7)
    variables = ["si10", "dswrf", "t", "prate"]
    steps = pd.timedelta_range("0h", "24h", freq="1h")
    x = np.linspace(-239_000, 857_000, 200)
    y = np.linspace(-183_000, 1425_000, 200)
    
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
def ecmwf_zarr_path(session_tmp_path) -> str:
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
def generation_zarr_path(session_tmp_path) -> str:

    times = pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="30min")
    location_ids = np.arange(318)
    # Rough UK bounding box
    lat_min, lat_max = 49.9, 58.7
    lon_min, lon_max = -8.6, 1.8

    # Generate random uniform points
    latitudes = np.random.uniform(lat_min, lat_max, len(location_ids)).astype("float64")
    longitudes = np.random.uniform(lon_min, lon_max, len(location_ids)).astype("float64")

    capacity = np.ones((len(times), len(location_ids)))

    generation = np.random.uniform(0, 200, (len(times), len(location_ids))).astype(np.float32)

    # Build Dataset
    ds_uk = xr.Dataset(
        data_vars={
            "capacity_mwp": (("time_utc", "location_id"), capacity),
            "generation_mw": (("time_utc", "location_id"), generation),
        },
        coords={
            "time_utc": times,
            "location_id": location_ids,
            "latitude": ("location_id", latitudes),
            "longitude": ("location_id", longitudes),
        },
    )

    zarr_path = session_tmp_path / "uk_generation.zarr"
    ds_uk.to_zarr(zarr_path)
    return zarr_path


@pytest.fixture(scope="session")
def data_config_path(
    session_tmp_path, 
    sat_zarr_path, 
    ukv_zarr_path, 
    ecmwf_zarr_path, 
    generation_zarr_path
) -> str:  
    
    # Populate the config with the generated zarr paths
    config = load_yaml_configuration(f"{_top_test_directory}/test_data/data_config.yaml")
    config.input_data.nwp["ukv"].zarr_path = str(ukv_zarr_path)
    config.input_data.nwp["ecmwf"].zarr_path = str(ecmwf_zarr_path)
    config.input_data.satellite.zarr_path = str(sat_zarr_path)
    config.input_data.generation.zarr_path = str(generation_zarr_path)

    filename = f"{session_tmp_path}/data_config.yaml"
    save_yaml_configuration(config, filename)
    return filename


@pytest.fixture(scope="session")
def streamed_datamodule(data_config_path) -> PVNetDataModule:
    dm = PVNetDataModule(
        configuration=data_config_path,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
    )
    dm.setup(stage="fit")
    return dm

@pytest.fixture(scope="session")
def batch(streamed_datamodule) -> TensorBatch:
    return next(iter(streamed_datamodule.train_dataloader()))


@pytest.fixture(scope="session")
def satellite_batch_component(batch) -> torch.Tensor:
    return torch.swapaxes(batch["satellite_actual"], 1, 2).float()


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
        key_to_use="generation",
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
                number_of_conv3d_layers=2,
                stride=[1,2,2],
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
        include_generation_yield_history=True,
        sat_history_minutes=30,
        nwp_history_minutes={"ukv": 120, "ecmwf": 120},
        nwp_forecast_minutes={"ukv": 480, "ecmwf": 480},
        nwp_interval_minutes={"ukv": 60, "ecmwf": 60},
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
def raw_late_fusion_model_kwargs_generation_history(model_minutes_kwargs) -> dict:
    return dict(
        # Set inputs to None/False apart from generation history
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
        include_time=True,
        include_generation_yield_history=True,
        forecast_minutes=480, 
        history_minutes=60,
        interval_minutes=30,
    )


@pytest.fixture()
def late_fusion_model_kwargs_generation_history(raw_late_fusion_model_kwargs_generation_history) -> dict:
    return hydra.utils.instantiate(raw_late_fusion_model_kwargs_generation_history)


@pytest.fixture()
def late_fusion_model_generation_history(late_fusion_model_kwargs_generation_history) -> LateFusionModel:
    return LateFusionModel(**late_fusion_model_kwargs_generation_history)


@pytest.fixture()
def late_fusion_quantile_model(late_fusion_model_kwargs) -> LateFusionModel:
    return LateFusionModel(output_quantiles=[0.1, 0.5, 0.9], **late_fusion_model_kwargs)


@pytest.fixture
def trainer_cfg():
    def _make(trainer_dict):
        return OmegaConf.create({"trainer": trainer_dict})
    return _make
