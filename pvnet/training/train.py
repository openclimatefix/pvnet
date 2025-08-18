"""Training"""
import logging
import os
import shutil

import hydra
from lightning.pytorch import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from pvnet.data.base_datamodule import BasePresavedDataModule
from pvnet.utils import (
    DATA_CONFIG_NAME,
    DATAMODULE_CONFIG_NAME,
    FULL_CONFIG_NAME,
    MODEL_CONFIG_NAME,
    remove_model_config_circular_ref,
)

log = logging.getLogger(__name__)


def resolve_monitor_loss(output_quantiles: list | None) -> str:
    """Return the desired metric to monitor based on whether quantile regression is being used.

    The adds the option to use something like:
        monitor: "${resolve_monitor_loss:${model.model.output_quantiles}}"

    in early stopping and model checkpoint callbacks so the callbacks config does not need to be
    modified depending on whether quantile regression is being used or not.
    """
    if output_quantiles is None:
        return "MAE/val"
    else:
        return "quantile_loss/val"


OmegaConf.register_new_resolver("resolve_monitor_loss", resolve_monitor_loss)


def train(config: DictConfig) -> None:
    """Contains training pipeline.

    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning loggers
    loggers: list[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            loggers.append(hydra.utils.instantiate(lg_conf))

    # Init lightning callbacks
    callbacks: list[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(cb_conf))

    # Align the wandb id with the checkpoint path
    # - only works if wandb logger and model checkpoint used
    # - this makes it easy to push the model to huggingface
    use_wandb_logger = False
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            use_wandb_logger = True
            wandb_logger = logger
            break

    # Set the output directory based in the wandb-id of the run
    if use_wandb_logger:
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                # Calling the .experiment property instantiates a wandb run
                wandb_id = wandb_logger.experiment.id

                # Save the run results to the expected parent folder but with the folder name
                # set by the wandb ID
                save_dir = "/".join(callback.dirpath.split("/")[:-1] + [wandb_id])

                callback.dirpath = save_dir

                # Save the model config - exclude model_config circular reference
                os.makedirs(save_dir, exist_ok=True)
                cleaned_model_config = remove_model_config_circular_ref(config.model)
                OmegaConf.save(cleaned_model_config, f"{save_dir}/{MODEL_CONFIG_NAME}")


                # If using pre-saved samples we need to extract the data config from the directory
                # those samples were saved to
                if isinstance(datamodule, BasePresavedDataModule):
                    data_config = f"{config.datamodule.sample_dir}/{DATA_CONFIG_NAME}"

                    # We also save the datamodule config used to create the samples to the output 
                    # directory and to wandb
                    shutil.copyfile(
                        f"{config.datamodule.sample_dir}/{DATAMODULE_CONFIG_NAME}", 
                        f"{save_dir}/{DATAMODULE_CONFIG_NAME}"
                    )
                    wandb_logger.experiment.save(
                        f"{save_dir}/{DATAMODULE_CONFIG_NAME}", 
                        base_path=save_dir,
                    )
                else:
                    # If we are streaming batches the data config is defined and we don't need to
                    # save the datamodule config separately
                    data_config = config.datamodule.configuration

                # Save the data config to the output directory and to wandb
                shutil.copyfile(data_config, f"{save_dir}/{DATA_CONFIG_NAME}")
                wandb_logger.experiment.save(f"{save_dir}/{DATA_CONFIG_NAME}", base_path=save_dir)

                # Save the full hydra config - exclude model_config circular reference
                cleaned_config = remove_model_config_circular_ref(config)
                OmegaConf.save(cleaned_config, f"{save_dir}/{FULL_CONFIG_NAME}")
                wandb_logger.experiment.save(f"{save_dir}/{FULL_CONFIG_NAME}", base_path=save_dir)
                
                break

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=loggers,
        _convert_="partial",
        callbacks=callbacks,
    )

    # Train the model completely
    trainer.fit(model=model, datamodule=datamodule)
