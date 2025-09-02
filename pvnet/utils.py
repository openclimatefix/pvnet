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
        if config.datamodule.get("prefetch_factor"):
            config.datamodule.prefetch_factor = None

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
        if (field == "model" and isinstance(config_section, DictConfig) and 
            "model_config" in config_section):
            config_section = OmegaConf.create(config_section)
            del config_section.model_config

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
    """Checks if tensor shape matches expected - raises detailed error on mismatch."""
    actual_shape = tuple(tensor.shape)
    if actual_shape == expected_shape:
        return
    
    if len(actual_shape) != len(expected_shape):
        raise ValueError(
            f"Shape mismatch for '{data_key}': Expected {len(expected_shape)} dims "
            f"{expected_shape}, got {len(actual_shape)} dims {actual_shape}."
        )
    
    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if actual != expected:
            dim_name = dim_names[i] if i < len(dim_names) else f"dim_{i}"
            raise ValueError(
                f"Shape mismatch for '{data_key}' in '{dim_name}': "
                f"expected {expected}, got {actual}. "
                f"Expected: {expected_shape}, Actual: {actual_shape}."
            )


def validate_batch_against_config(
    batch: dict, 
    model_config, 
    sat_interval_minutes: int = 5, 
    gsp_interval_minutes: int = 30, 
    site_interval_minutes: int = 30
) -> None:
    """Validates tensor shapes in batch against model configuration."""
    logger.info("Performing batch shape validation against model config.")
    
    is_instantiated = (
        hasattr(model_config, 'nwp_encoders_dict') and 
        not isinstance(model_config, (dict, DictConfig))
    )
    if not is_instantiated:
        logger.info("Batch shape validation successful!")
        return
    
    dim_names = ["batch", "time", "channels", "height", "width"]
    
    # NWP validation
    if hasattr(model_config, 'nwp_encoders_dict') and "nwp" in batch:
        for source, nwp_data in batch["nwp"].items():
            if source in model_config.nwp_encoders_dict:
                encoder = model_config.nwp_encoders_dict[source]
                expected_shape = (
                    nwp_data["nwp"].shape[0], 
                    encoder.sequence_length, 
                    encoder.in_channels, 
                    encoder.image_size_pixels, 
                    encoder.image_size_pixels
                )
                _check_shape_and_raise(
                    f"NWP.{source}", nwp_data["nwp"], expected_shape, dim_names
                )
    
    # Satellite validation
    if hasattr(model_config, 'sat_encoder') and "sat" in batch:
        encoder = model_config.sat_encoder
        expected_shape = (
            batch["sat"].shape[0], 
            encoder.sequence_length, 
            encoder.in_channels, 
            encoder.image_size_pixels, 
            encoder.image_size_pixels
        )
        _check_shape_and_raise("Satellite", batch["sat"], expected_shape, dim_names)
    
    # GSP/Site validation
    target_configs = [
        ("gsp", gsp_interval_minutes, "GSP"), 
        ("site", site_interval_minutes, "Site")
    ]
    for key, interval, name in target_configs:
        if key in batch and hasattr(model_config, 'history_minutes') and \
           hasattr(model_config, 'forecast_minutes'):
            history_len = model_config.history_minutes // interval
            forecast_len = model_config.forecast_minutes // interval
            expected_len = history_len + forecast_len + 1
            _check_shape_and_raise(
                f"{name} Target", 
                batch[key], 
                (batch[key].shape[0], expected_len), 
                ["batch", "time"]
            )
    
    logger.info("Batch shape validation successful!")


def extract_raw_config(model_config) -> DictConfig:
    """Extract raw configuration from model_config"""
    if isinstance(model_config, dict):
        return OmegaConf.create(model_config)
    else:
        raise TypeError(f"Expected DictConfig or dict, got {type(model_config)}")


def remove_model_config_circular_ref(config: DictConfig) -> DictConfig:
    """Remove model_config circular reference from config before saving."""
    config_copy = OmegaConf.create(config)
    
    for path in ["model_config", "model.model_config"]:
        if OmegaConf.select(config_copy, path) is not None:
            OmegaConf.set(config_copy, path, OmegaConf.MISSING)
    
    return config_copy
