"""Utils"""
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import rich.syntax
import rich.tree
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from pvnet.models.base_model import BaseModel

logger = logging.getLogger(__name__)


PYTORCH_WEIGHTS_NAME = "model_weights.safetensors"
MODEL_CONFIG_NAME = "model_config.yaml"
DATA_CONFIG_NAME = "data_config.yaml"
DATAMODULE_CONFIG_NAME = "datamodule_config.yaml"
FULL_CONFIG_NAME =  "full_experiment_config.yaml"
MODEL_CARD_NAME = "README.md"

_UNSUPPORTED_STRATEGIES = {"ddp", "ddp_spawn", "ddp_fork", "dp", "deepspeed", "fsdp"}


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

        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


def validate_batch_against_config(
    batch: dict, 
    model: "BaseModel",
) -> None:
    """Validates tensor shapes in batch against model configuration."""
    logger.info("Performing batch shape validation against model config.")
    
    # NWP validation
    if hasattr(model, 'nwp_encoders_dict'):
        if "nwp" not in batch:
            raise ValueError(
                "Model configured with 'nwp_encoders_dict' but 'nwp' data missing from batch."
            )
            
        for source, nwp_data in batch["nwp"].items():
            if source in model.nwp_encoders_dict:

                enc = model.nwp_encoders_dict[source]                
                expected_channels = enc.in_channels
                if model.add_image_embedding_channel:
                    expected_channels -= 1

                expected = (nwp_data["nwp"].shape[0], enc.sequence_length, 
                           expected_channels, enc.image_size_pixels, enc.image_size_pixels)
                if tuple(nwp_data["nwp"].shape) != expected:
                    actual_shape = tuple(nwp_data['nwp'].shape)
                    raise ValueError(
                        f"NWP.{source} shape mismatch: expected {expected}, got {actual_shape}"
                    )

    # Satellite validation
    if hasattr(model, 'sat_encoder'):
        if "satellite_actual" not in batch:
            raise ValueError(
                "Model configured with 'sat_encoder' but 'satellite_actual' missing from batch."
            )

        enc = model.sat_encoder
        expected_channels = enc.in_channels
        if model.add_image_embedding_channel:
            expected_channels -= 1

        expected = (batch["satellite_actual"].shape[0], enc.sequence_length, expected_channels, 
                enc.image_size_pixels, enc.image_size_pixels)
        if tuple(batch["satellite_actual"].shape) != expected:
            actual_shape = tuple(batch['satellite_actual'].shape)
            raise ValueError(f"Satellite shape mismatch: expected {expected}, got {actual_shape}")

    # GSP/Site validation
    key = model._target_key
    if key in batch:
        total_minutes = model.history_minutes + model.forecast_minutes
        interval = model.interval_minutes
        expected_len = total_minutes // interval + 1
        expected = (batch[key].shape[0], expected_len)
        if tuple(batch[key].shape) != expected:
            actual_shape = tuple(batch[key].shape)
            raise ValueError(
                f"{key.upper()} shape mismatch: expected {expected}, got {actual_shape}"
            )

    logger.info("Batch shape validation successful!")


def _is_multi_device_sequence(x) -> bool:
    return isinstance(x, Sequence) and not isinstance(x, (str, bytes)) and len(x) > 1


def validate_gpu_config(config: DictConfig) -> None:
    """Abort if multi-GPU, distributed, or multi-node training requested."""
    tr = config.get("trainer", {})

    # Devices: allow single device and block multiples
    dev = tr.get("devices")
    if (isinstance(dev, int) and dev > 1) or _is_multi_device_sequence(dev):
        raise ValueError("Parallel training not supported. Use `trainer.devices: 1`.")

    # Distributed strategies: not supported
    strat = str(tr.get("strategy", "")).lower()
    if strat in _UNSUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported strategy '{strat}'. Remove or set to null.")

    # Multi-node: not supported
    if tr.get("num_nodes", 1) != 1:
        raise ValueError("Multi-node training not supported (set num_nodes=1).")
