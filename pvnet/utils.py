"""Utils"""
import logging

import rich.syntax
import rich.tree
import torch
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


PYTORCH_WEIGHTS_NAME = "model_weights.safetensors"
MODEL_CONFIG_NAME = "model_config.yaml"
DATA_CONFIG_NAME = "data_config.yaml"
DATAMODULE_CONFIG_NAME = "datamodule_config.yaml"
FULL_CONFIG_NAME =  "full_experiment_config.yaml"
MODEL_CARD_NAME = "README.md"



def run_config_utilities(config: DictConfig) -> None:
    """A couple of optional utilities.

    Controlled by main config file:
    - forcing debug friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    # Enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # Force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        logger.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # Disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: tuple[str] = (
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
    logger.info("Performing batch shape validation against model config.")
    dim_names = ["batch", "time", "channels", "height", "width"]

    if "nwp" in batch and "nwp_encoders_dict" in model_config:
        if "nwp" not in batch:
            raise ValueError(
                "Model configured with 'nwp_encoders_dict' - 'nwp' data missing from batch."
            )
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
        if "sat" not in batch:
            raise ValueError(
                "Model configured with 'sat_encoder' - 'sat' data missing from batch."
            )
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

    logger.info("Batch shape validation successful.")
