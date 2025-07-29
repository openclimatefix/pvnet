import os
import tempfile
import pytest
import torch
import numpy as np
import pandas as pd
import xarray as xr
from pvnet.data import SitePresavedDataModule, SiteStreamedDataModule, UKRegionalPresavedDataModule



@pytest.fixture
def temp_pt_sample_dir():
    """Create temporary directory with synthetic PT samples"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create train and val directories
        os.makedirs(f"{tmpdirname}/train", exist_ok=True)
        os.makedirs(f"{tmpdirname}/val", exist_ok=True)

        # Generate and save synthetic samples
        for i in range(5):
            sample = {
                "gsp": torch.rand(21),
                "gsp_time_utc": torch.tensor(list(range(21))),
                "gsp_nominal_capacity_mwp": torch.tensor(100.0),
                "gsp_id": 12
            }
            torch.save(sample, f"{tmpdirname}/train/{i:08d}.pt")
            torch.save(sample, f"{tmpdirname}/val/{i:08d}.pt")

        yield tmpdirname


@pytest.fixture
def temp_nc_sample_dir():
    """Create temporary directory with synthetic NC site samples"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create train and val directories
        os.makedirs(f"{tmpdirname}/train", exist_ok=True)
        os.makedirs(f"{tmpdirname}/val", exist_ok=True)

        # Create config file
        config_path = f"{tmpdirname}/data_configuration.yaml"
        with open(config_path, "w") as f:
            f.write(f"sample_dir: {tmpdirname}\n")

        # Generate and save synthetic site samples
        for i in range(5):
            site_time = pd.date_range("2023-01-01", periods=10, freq="15min")
            ds = xr.Dataset(
                data_vars={
                    "site": (["site__time_utc"], np.random.rand(10)),
                },
                coords={
                    "site__time_utc": site_time,
                    "site__site_id": np.int32(i % 3 + 1),
                    "site__latitude": 52.5,
                    "site__longitude": -1.5,
                    "site__capacity_kwp": 10000.0,
                }
            )

            ds.to_netcdf(f"{tmpdirname}/train/{i:08d}.nc", mode="w", engine="h5netcdf")
            ds.to_netcdf(f"{tmpdirname}/val/{i:08d}.nc", mode="w", engine="h5netcdf")

        yield tmpdirname


def test_init(temp_pt_sample_dir):
    """Test DataModule initialization"""
    dm = UKRegionalPresavedDataModule(
        sample_dir=temp_pt_sample_dir,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
    )


def test_iter(temp_pt_sample_dir):
    """Test iteration through DataModule"""
    dm = UKRegionalPresavedDataModule(
        sample_dir=temp_pt_sample_dir,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
    )

    # Verify existing keys
    batch = next(iter(dm.train_dataloader()))
    assert "gsp" in batch


def test_iter_multiprocessing(temp_pt_sample_dir):
    """Test DataModule with multiple workers"""
    dm = UKRegionalPresavedDataModule(
        sample_dir=temp_pt_sample_dir,
        batch_size=1,
        num_workers=2,
        prefetch_factor=1,
    )

    batches = [batch for _, batch in zip(range(2), dm.train_dataloader())]
    assert len(batches)==2


def test_site_init_sample_dir(temp_nc_sample_dir):
    """Test SiteDataModule initialization with sample dir"""
    _ = SitePresavedDataModule(
        sample_dir=temp_nc_sample_dir,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
    )


def test_site_init_config(temp_nc_sample_dir):
    """Test SiteDataModule initialization with config file"""
    config_path = f"{temp_nc_sample_dir}/data_configuration.yaml"

    _ = SiteStreamedDataModule(
        configuration=config_path,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
    )