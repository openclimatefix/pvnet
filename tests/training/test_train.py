"""Tests and fixtures for CPU-only Trainer and offline W&B logging."""

from pathlib import Path

import torch
import pytest
import wandb
from omegaconf import DictConfig

from pvnet.training.train import train as pvnet_train


@pytest.fixture()
def wandb_save_dir(session_tmp_path) -> str:
    """Return W&B save dir under session temp path."""
    save_dir = str(session_tmp_path / "wandb")
    return save_dir


@pytest.fixture()
def trainer_cfg_cpu() -> dict:
    """Tiny CPU-only Trainer config."""
    return {
        "_target_": "lightning.pytorch.Trainer",
        "max_epochs": 1,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "accelerator": "cpu",
        "enable_checkpointing": True,
        "log_every_n_steps": 1,
        "enable_progress_bar": False,
    }


@pytest.fixture()
def logger_cfg(wandb_save_dir: str) -> dict:
    """W&B logger config."""
    return {
        "wandb": {
            "_target_": "lightning.pytorch.loggers.wandb.WandbLogger",
            "project": "pvnet-tests",
            "save_dir": wandb_save_dir,
            "offline": True,
            "name": "train-offline-integration",
            "log_model": False,
        }
    }


@pytest.fixture()
def ckpt_cfg(wandb_save_dir: str) -> dict:
    """ModelCheckpoint config."""
    return {
        "ckpt": {
            "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
            "dirpath": str(Path(wandb_save_dir).parent / "ckpts"),
            "save_last": True,
            "save_top_k": 1,
            "monitor": "MAE/val",
            "mode": "min",
        }
    }


def build_lit_late_fusion_cfg(
    interval_minutes: int,
    include_time: bool,
    forecast_minutes: int = 480,
    history_minutes: int = 60,
) -> dict:
    """Build config for PVNetLightningModule + minimal LateFusionModel."""
    return {
        "_target_": "pvnet.training.lightning_module.PVNetLightningModule",
        "model": {
            "_target_": "pvnet.models.LateFusionModel",
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
            "include_generation_history": True,
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

def test_train_pvnet(
    data_config_path,
    trainer_cfg_cpu,
    logger_cfg,
    ckpt_cfg,
    wandb_save_dir
):
    """Train pvnet model with W&B offline."""
    cfg = DictConfig({
        "seed": 42,
        "datamodule": {
            "_target_": "pvnet.datamodule.PVNetDataModule",
            "train_periods": [[None, None]],
            "val_periods": [[None, None]],
            "configuration": str(data_config_path),
            "batch_size": 2,
            "num_workers": 0,
            "prefetch_factor": None,
        },
        "model": build_lit_late_fusion_cfg(
            interval_minutes=30,
            include_time=False,
        ),
        "logger": logger_cfg,
        "callbacks": ckpt_cfg,
        "trainer": trainer_cfg_cpu,
        "model_name": "test_model",
        "ckpt_path": None,
    })

    pvnet_train(cfg)

    # Check that checkpoint exists
    ckpt_paths = list(Path(wandb_save_dir).parent.glob("*/last.ckpt"))
    assert len(ckpt_paths) == 1,  f"expected one checkpoint called last.ckpt at end of epoch, got {ckpt_paths}"

def assert_same_dict(dict0: dict, dict1: dict, desc: str = ""):
    """
    Helper function to compare nested dictionaries containing torch tensors.
    """
    assert dict0.keys() == dict1.keys(), f"Key mismatch: {dict0.keys()}, {dict1.keys()}"
    for key, value in dict0.items():
        if isinstance(value, dict):
            assert_same_dict(value, dict1[key], desc=f"{desc}.{key}")
        else:
            torch.testing.assert_close(dict1[key], value, atol=1e-9, rtol=0, 
                                       msg=f"Checkpoints different for key '{key}' in {desc}: {dict1[key]} vs {value}")

    
def test_checkpoint_load(
    data_config_path,
    trainer_cfg_cpu,
    logger_cfg,
    ckpt_cfg,
    wandb_save_dir
):
    """Test saving and loading from checkpoint from previous test"""
    ckpt_epoch0_path = list(Path(wandb_save_dir).parent.glob("*/epoch=0*.ckpt"))[0]

    # Extend max_epochs by 1 to continue training from checkpoint
    trainer_cfg_cpu['max_epochs'] = 2

    cfg = DictConfig({
            "seed": 42,
            "datamodule": {
                "_target_": "pvnet.datamodule.PVNetDataModule",
                "train_periods": [[None, None]],
                "val_periods": [[None, None]],
                "configuration": str(data_config_path),
                "batch_size": 2,
                "num_workers": 0,
                "prefetch_factor": None,
            },
            "model": build_lit_late_fusion_cfg(
                interval_minutes=30,
                include_time=False,
            ),
            "logger": logger_cfg,
            "callbacks": ckpt_cfg,
            "trainer": trainer_cfg_cpu,
            "model_name": "test_model",
            "ckpt_path": ckpt_epoch0_path,
        })

    # Repeat training twice to check that we get the same results from the same checkpoint
    for _ in range(2):
        # Make sure wandb is reset before running pvnet
        wandb.finish()
        pvnet_train(cfg)

    # Check there are now two files at end of epoch=1
    ckpt_epoch1_path = list(Path(wandb_save_dir).parent.glob("*/epoch=1*.ckpt"))
    assert len(ckpt_epoch1_path) == 2, f"expected 2 checkpoints at end of epoch 2, got {len(ckpt_epoch1_path)}, {ckpt_epoch1_path}"

    # Load both checkpoints and compare 
    ckpt0 = torch.load(ckpt_epoch1_path[0], map_location="cpu", weights_only=False)
    ckpt1 = torch.load(ckpt_epoch1_path[1], map_location="cpu", weights_only=False)

    # Compare state_dict
    assert_same_dict(ckpt0['state_dict'], ckpt1['state_dict'], desc="state_dict")

    # Compare optimizer state which is stored at each step
    assert_same_dict(ckpt0['optimizer_states'][-1], ckpt1['optimizer_states'][-1], desc="optimizer_states")
    
