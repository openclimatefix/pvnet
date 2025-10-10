from __future__ import annotations
import sys
import types
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningModule, LightningDataModule
from omegaconf import DictConfig

from pvnet.training.train import train as pvnet_train



class _ToyDataset(Dataset):
    def __init__(self, n: int = 32, d: int = 4):
        self.x = torch.randn(n, d)
        self.y = self.x.sum(dim=1, keepdim=True) + 0.05 * torch.randn(n, 1)

    def __len__(self): return self.x.size(0)
    def __getitem__(self, i): return self.x[i], self.y[i]


class DummyDataModule(LightningDataModule):
    """Minimal DataModule; exposes `configuration` path because train() copies it."""
    def __init__(self, batch_size: int = 8, configuration: str | None = None):
        super().__init__()
        self.batch_size = batch_size
        self.configuration = configuration

    def setup(self, stage=None):
        self.train_ds = _ToyDataset(32, 4)
        self.val_ds = _ToyDataset(16, 4)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)


class DummyModule(LightningModule):
    """Logs MAE/val so ModelCheckpoint(monitor='MAE/val') has a valid metric."""
    def __init__(self, lr: float = 1e-2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
        self.loss = nn.L1Loss()
        self.lr = lr

    def forward(self, x)
        return self.net(x)

    def training_step(self, batch, _):
        x, y = batch
        yhat = self(x)
        loss = self.loss(yhat, y)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        yhat = self(x)
        val_mae = self.loss(yhat, y)
        self.log("MAE/val", val_mae, on_epoch=True)
        return val_mae

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def test_train_with_wandb_offline(session_tmp_path, site_data_config_path):
    """
    Smoke test proving Wandb offline doesn't block train()
    """
    mod_name = "pvnet._test_components"
    mod = types.ModuleType(mod_name)
    mod.DummyDataModule = DummyDataModule
    mod.DummyModule = DummyModule
    sys.modules[mod_name] = mod

    cfg = DictConfig({
        "seed": 42,
        "datamodule": {
            "_target_": "pvnet._test_components.DummyDataModule",
            "batch_size": 8,
            "configuration": str(site_data_config_path),
        },
        "model": {
            "_target_": "pvnet._test_components.DummyModule",
            "lr": 1e-2,
        },
        "logger": {
            "wandb": {
                "_target_": "lightning.pytorch.loggers.wandb.WandbLogger",
                "project": "pvnet-tests",
                "save_dir": str(session_tmp_path),
                "offline": True,
                "name": "unit",
            }
        },
        "callbacks": {
            "ckpt": {
                "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                "dirpath": str(session_tmp_path / "ckpts"),
                "save_last": True,
                "save_top_k": 1,
                "monitor": "MAE/val",
                "mode": "min",
            }
        },
        "trainer": {
            "_target_": "lightning.pytorch.Trainer",
            "max_epochs": 1,
            "limit_train_batches": 2,
            "limit_val_batches": 2,
            "accelerator": "cpu",
            "devices": 1,
            "logger": True,
            "enable_checkpointing": True,
            "log_every_n_steps": 1,
        },
    })

    pvnet_train(cfg)
