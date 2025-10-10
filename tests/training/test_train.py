from __future__ import annotations

import pytest
from omegaconf import DictConfig
from pvnet.training.train import train as pvnet_train


def _wandb_offline(monkeypatch: pytest.MonkeyPatch, save_dir: str) -> None:
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_START_METHOD", "thread")
    monkeypatch.setenv("WANDB_DIR", save_dir)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def _minimal_lit_with_late_fusion(
    *,
    target_key: str,
    interval_minutes: int,
    include_time: bool,
    forecast_minutes: int = 480,
    history_minutes: int = 60,
) -> dict:
    """
    Config snippet for PVNetLightningModule + minimal LateFusionModel.
    - Sets the correct interval for each stream (site=15, gsp=30).
    - Flips include_*_yield_history flags based on target.
    - Allows disabling time features when the datamodule doesn't provide them.
    """
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
            "include_gsp_yield_history": target_key in ("gsp", "gsp_yield"),
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


def _common_trainer() -> dict:
    return {
        "_target_": "lightning.pytorch.Trainer",
        "max_epochs": 1,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "accelerator": "cpu",
        "devices": 1,
        "logger": True,
        "enable_checkpointing": True,
        "log_every_n_steps": 1,
    }


def _wandb_logger(save_dir: str) -> dict:
    return {
        "wandb": {
            "_target_": "lightning.pytorch.loggers.wandb.WandbLogger",
            "project": "pvnet-tests",
            "save_dir": save_dir,
            "offline": True,
            "name": "train-offline-integration",
        }
    }


def _checkpoint_callback(save_dir: str) -> dict:
    from pathlib import Path
    ckpt_dir = str(Path(save_dir) / "ckpts")
    return {
        "ckpt": {
            "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
            "dirpath": ckpt_dir,
            "save_last": True,
            "save_top_k": 1,
            "monitor": "MAE/val",
            "mode": "min",
        }
    }


def test_train_site(session_tmp_path, site_data_config_path, monkeypatch: pytest.MonkeyPatch):
    """
    Site-level integration:
    - SitesDataModule (15-min cadence)
    - PVNetLightningModule + minimal LateFusion (target_key='site')
    - W&B offline + checkpoint
    """
    _wandb_offline(monkeypatch, str(session_tmp_path / "wandb"))

    cfg = DictConfig({
        "seed": 42,
        "datamodule": {
            "_target_": "pvnet.datamodule.SitesDataModule",
            "configuration": str(site_data_config_path),
            "batch_size": 2,
            "num_workers": 0,
            "prefetch_factor": None,
        },
        "model": _minimal_lit_with_late_fusion(
            target_key="site",
            interval_minutes=15,
            include_time=True,
        ),
        "logger": _wandb_logger(str(session_tmp_path)),
        "callbacks": _checkpoint_callback(str(session_tmp_path)),
        "trainer": _common_trainer(),
    })

    pvnet_train(cfg)


def test_train_pv(session_tmp_path, uk_data_config_path, monkeypatch: pytest.MonkeyPatch):
    """
    GSP PV integration:
    - UKRegionalDataModule (30-min cadence in test fixtures)
    - PVNetLightningModule + minimal LateFusion (target_key='gsp', 30-min interval)
    - W&B offline + checkpoint
    """
    _wandb_offline(monkeypatch, str(session_tmp_path / "wandb"))

    cfg = DictConfig({
        "seed": 42,
        "datamodule": {
            "_target_": "pvnet.datamodule.UKRegionalDataModule",
            "configuration": str(uk_data_config_path),
            "batch_size": 2,
            "num_workers": 0,
            "prefetch_factor": None,
        },
        "model": _minimal_lit_with_late_fusion(
            target_key="gsp",
            interval_minutes=30,
            include_time=False,
        ),
        "logger": _wandb_logger(str(session_tmp_path)),
        "callbacks": _checkpoint_callback(str(session_tmp_path)),
        "trainer": _common_trainer(),
    })

    pvnet_train(cfg)
