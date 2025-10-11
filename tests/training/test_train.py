from __future__ import annotations

from omegaconf import DictConfig
from pvnet.training.train import train as pvnet_train


def test_train_site(
    site_data_config_path,
    trainer_cfg_cpu,
    logger_cfg,
    ckpt_cfg,
    build_lit_late_fusion_cfg,
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
    build_lit_late_fusion_cfg,
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
