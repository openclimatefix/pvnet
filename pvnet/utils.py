"""Utils"""
import logging
from collections.abc import Sequence
from typing import Optional

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import pandas as pd
import pylab
import rich.syntax
import rich.tree
import torch
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
MODEL_CONFIG_NAME = "model_config.yaml"
DATA_CONFIG_NAME = "data_config.yaml"
DATAMODULE_CONFIG_NAME = "datamodule_config.yaml"
MODEL_CARD_NAME = "README.md"


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities.

    Controlled by main config file:
    - forcing debug friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


def empty(*args, **kwargs):
    """Returns nothing"""
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: list[pl.Callback],
    logger: list[Logger],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def plot_batch_forecasts(
    batch,
    y_hat,
    batch_idx=None,
    quantiles=None,
    key_to_plot: str = "gsp",
    timesteps_to_plot: Optional[list[int]] = None,
):
    """Plot a batch of data and the forecast from that batch"""

    y_key = key_to_plot
    y_id_key = f"{key_to_plot}_id"
    time_utc_key = f"{key_to_plot}_time_utc"
    y = batch[y_key].cpu().numpy()  # Select the one it is trained on
    y_hat = y_hat.cpu().numpy()
    # Select between the timesteps in timesteps to plot
    plotting_name = key_to_plot.upper()

    gsp_ids = batch[y_id_key].cpu().numpy().squeeze()

    times_utc = batch[time_utc_key].cpu().numpy().squeeze().astype("datetime64[ns]")
    times_utc = [pd.to_datetime(t) for t in times_utc]
    if timesteps_to_plot is not None:
        y = y[:, timesteps_to_plot[0] : timesteps_to_plot[1]]
        y_hat = y_hat[:, timesteps_to_plot[0] : timesteps_to_plot[1]]
        times_utc = [t[timesteps_to_plot[0] : timesteps_to_plot[1]] for t in times_utc]

    batch_size = y.shape[0]

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for i, ax in enumerate(axes.ravel()):
        if i >= batch_size:
            ax.axis("off")
            continue
        ax.plot(times_utc[i], y[i], marker=".", color="k", label=r"$y$")

        if quantiles is None:
            ax.plot(
                times_utc[i][-len(y_hat[i]) :], y_hat[i], marker=".", color="r", label=r"$\hat{y}$"
            )
        else:
            cm = pylab.get_cmap("twilight")
            for nq, q in enumerate(quantiles):
                ax.plot(
                    times_utc[i][-len(y_hat[i]) :],
                    y_hat[i, :, nq],
                    color=cm(q),
                    label=r"$\hat{y}$" + f"({q})",
                    alpha=0.7,
                )

        ax.set_title(f"ID: {gsp_ids[i]} | {times_utc[i][0].date()}", fontsize="small")

        xticks = [t for t in times_utc[i] if t.minute == 0][::2]
        ax.set_xticks(ticks=xticks, labels=[f"{t.hour:02}" for t in xticks], rotation=90)
        ax.grid()

    axes[0, 0].legend(loc="best")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (hour of day)")

    if batch_idx is not None:
        title = f"Normed {plotting_name} output : batch_idx={batch_idx}"
    else:
        title = f"Normed {plotting_name} output"
    plt.suptitle(title)
    plt.tight_layout()

    return fig


def _check_shape_and_raise(
    data_key: str,
    tensor: torch.Tensor,
    expected_shape: tuple,
    dim_names: list[str]
) -> None:
    """
    Checks if actual shape matches expected shape - raises a detailed error on mismatch.

    Args:
        data_key: A string identifying the data being checked (e.g., "NWP.ukv").
        tensor: The tensor object whose shape is to be validated.
        expected_shape: The expected shape of the tensor.
        dim_names: A list of names for each dimension for more descriptive errors.

    Raises:
        ValueError: If the shapes do not match.
    """
    actual_shape = tuple(tensor.shape)
    if actual_shape == expected_shape:
        return

    if len(actual_shape) != len(expected_shape):
        raise ValueError(
            f"Shape mismatch for '{data_key}': Incorrect number of dimensions. "
            f"Expected {len(expected_shape)} dims with shape {expected_shape}, "
            f"but got {len(actual_shape)} dims with shape {actual_shape}."
        )

    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if actual != expected:
            dim_name = dim_names[i] if i < len(dim_names) else f"dim_{i}"
            raise ValueError(
                f"Shape mismatch for '{data_key}' in dimension '{dim_name}'. "
                f"Expected size {expected}, but got {actual}. "
                f"Full Expected Shape: {expected_shape}, Full Actual Shape: {actual_shape}."
            )


def validate_batch_against_config(
    batch: dict,
    model_config: DictConfig,
    sat_interval_minutes: int = 5,
    gsp_interval_minutes: int = 30,
) -> None:
    """
    Validates the shapes of tensors in a batch against the model's configuration.

    Context-aware error messages when a batch's tensor shape does not match the 
    shape expected by the model configuration.

    Args:
        batch: A dictionary of tensors from the dataloader.
        model_config: The model's configuration object (e.g., config.model).
        sat_interval_minutes: The time resolution of the satellite data in minutes.
        gsp_interval_minutes: The time resolution of the GSP data in minutes.

    Raises:
        ValueError: If a tensor shape mismatches the expected shape derived from the config.
    """
    log = get_logger(__name__)
    log.info("Performing batch shape validation against model config.")
    dim_names = ["batch", "time", "channels", "height", "width", "sites"]

    if "nwp" in batch and "nwp_encoders_dict" in model_config:
        for source, nwp_data_dict in batch["nwp"].items():
            if source not in model_config.nwp_encoders_dict:
                continue

            nwp_tensor = nwp_data_dict["nwp"]
            
            cfg = model_config.nwp_encoders_dict[source]
            hist_mins = model_config.nwp_history_minutes[source]
            fcst_mins = model_config.nwp_forecast_minutes[source]
            interval = model_config.nwp_interval_minutes[source]
            exp_time = (hist_mins + fcst_mins) // interval + 1
            
            expected_shape = (
                nwp_tensor.shape[0],
                exp_time,
                cfg.keywords['in_channels'],
                cfg.keywords['image_size_pixels'],
                cfg.keywords['image_size_pixels'],
            )
            _check_shape_and_raise(f"NWP.{source}",
            nwp_tensor,
            expected_shape,
            dim_names)

    if "sat" in batch and "sat_encoder" in model_config:
        sat_tensor = batch["sat"]
        cfg = model_config.sat_encoder
        exp_time = model_config.sat_history_minutes // sat_interval_minutes + 1
        expected_shape = (
            sat_tensor.shape[0],
            exp_time,
            cfg.keywords['in_channels'],
            cfg.keywords['image_size_pixels'],
            cfg.keywords['image_size_pixels'],
        )
        _check_shape_and_raise("Satellite", sat_tensor, expected_shape, dim_names)

    if "gsp" in batch:
        gsp_tensor = batch["gsp"]
        
        history_len = model_config.history_minutes // gsp_interval_minutes
        forecast_len = model_config.forecast_minutes // gsp_interval_minutes
        expected_total_len = history_len + forecast_len + 1
        
        expected_shape = (gsp_tensor.shape[0], expected_total_len)
        _check_shape_and_raise("GSP Target", gsp_tensor, expected_shape, dim_names)

    log.info("Batch shape validation success.")
