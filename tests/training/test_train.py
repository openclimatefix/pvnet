"""Tests and fixtures for CPU-only Trainer and offline W&B logging."""

from pathlib import Path

import pytest
from omegaconf import DictConfig

from pvnet.training.train import train as pvnet_train


@pytest.fixture()
def wandb_offline_env(monkeypatch, session_tmp_path):
    """Put W&B offline, quiet; force CPU."""
    save_dir = str(session_tmp_path / "wandb")
    return save_dir


@pytest.fixture()
def trainer_cfg_cpu():
    """Tiny CPU-only Trainer config."""
    return dict(
        _target_="lightning.pytorch.Trainer",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        accelerator="cpu",
        enable_checkpointing=True,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )


@pytest.fixture()
def logger_cfg(wandb_offline_env):
    """W&B logger config."""
    return {
        "wandb": {
            "_target_": "lightning.pytorch.loggers.wandb.WandbLogger",
            "project": "pvnet-tests",
            "save_dir": wandb_offline_env,
            "offline": True,
            "name": "train-offline-integration",
            "log_model": False,
        }
    }


@pytest.fixture()
def ckpt_cfg(wandb_offline_env):
    """ModelCheckpoint config."""
    return {
        "ckpt": {
            "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
            "dirpath": str(Path(wandb_offline_env).parent / "ckpts"),
            "save_last": True,
            "save_top_k": 1,
            "monitor": "MAE/val",
            "mode": "min",
        }
    }


def build_lit_late_fusion_cfg(
    target_key: str,
    interval_minutes: int,
    include_time: bool,
    forecast_minutes: int = 480,
    history_minutes: int = 60,
):
    """Build config for PVNetLightningModule + minimal LateFusionModel."""
    return {
        "_target_": "pvnet.training.lightning_module.PVNetLightningModule",
        "model": {
            "_target_": "pvnet.models.LateFusionModel",
            "target_key": target_key,
            "sat_encoder": None,
            "nwp_encoders_dict": None,
            "add_image_embedding_channel": False,
            "pv_encoder": None,
            "output_network": {
                "_target_": "pvnet.models.late_fusion.linear_networks.networks.ResFCNet",
                "_partial_": True,
                "fc_hidden_features": 128,
                "n_res_blocks": 6,
                "res_block_layers": 2,
                "dropout_frac": 0.0,
            },
            "location_id_mapping": None,
            "embedding_dim": None,
            "include_sun": False,
            "include_time": include_time,
            "include_site_yield_history": target_key == "site",
            "include_gsp_yield_history": target_key == "gsp",
            "forecast_minutes": forecast_minutes,
            "history_minutes": history_minutes,
            "interval_minutes": interval_minutes,
        },
        "optimizer": {
            "_target_": "pvnet.optimizers.Adam",
            "lr": 1e-3,
        },
        "save_all_validation_results": False,
    }


def test_train_site(
    site_data_config_path,
    trainer_cfg_cpu,
    logger_cfg,
    ckpt_cfg,
):
    """Train site model with W&B offline."""
    cfg = DictConfig({
        "seed": 42,
        "datamodule": {
            "_target_": "pvnet.datamodule.SitesDataModule",
            "configuration": str(site_data_config_path),
            "batch_size": 2,
            "num_workers": 0,
            "prefetch_factor": None,
        },
        "model": build_lit_late_fusion_cfg(
            target_key="site",
            interval_minutes=15,
            include_time=True,
        ),
        "logger": logger_cfg,
        "callbacks": ckpt_cfg,
        "trainer": trainer_cfg_cpu,
    })

    pvnet_train(cfg)


def test_train_pv(
    uk_data_config_path,
    trainer_cfg_cpu,
    logger_cfg,
    ckpt_cfg,
):
    """Train GSP model with W&B offline."""
    cfg = DictConfig({
        "seed": 42,
        "datamodule": {
            "_target_": "pvnet.datamodule.UKRegionalDataModule",
            "configuration": str(uk_data_config_path),
            "batch_size": 2,
            "num_workers": 0,
            "prefetch_factor": None,
        },
        "model": build_lit_late_fusion_cfg(
            target_key="gsp",
            interval_minutes=30,
            include_time=False,
        ),
        "logger": logger_cfg,
        "callbacks": ckpt_cfg,
        "trainer": trainer_cfg_cpu,
    })

    pvnet_train(cfg)
