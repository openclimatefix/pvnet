"""Utils"""

import logging
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
FULL_CONFIG_NAME = "full_experiment_config.yaml"
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
    if model.include_nwp:
        if (nwp_dict := batch.get("nwp")) is None:
            raise ValueError("Model uses NWP data but 'nwp' missing from batch.")

        for source, enc in model.nwp_encoders_dict.items():
            if (src_data := nwp_dict.get(source)) is None:
                raise ValueError(f"NWP source '{source}' missing from batch['nwp'].")

            nwp_tensor = src_data["nwp"]
            exp_ch = enc.in_channels - int(model.add_image_embedding_channel)
            _, actual_seq, actual_ch, h, w = nwp_tensor.shape

            if (actual_seq != enc.sequence_length or actual_ch != exp_ch or 
                h != enc.image_size_pixels or w != enc.image_size_pixels):
                msg = (
                    f"NWP.{source} mismatch: Exp {enc.sequence_length}seq, {exp_ch}ch. "
                    f"Got {actual_seq}seq, {actual_ch}ch"
                )
                raise ValueError(msg)

    # Satellite validation
    if model.include_sat:
        if (sat_data := batch.get("satellite_actual")) is None:
            raise ValueError("Model uses sat data but 'satellite_actual' missing from batch.")

        enc = model.sat_encoder
        exp_ch = enc.in_channels - int(model.add_image_embedding_channel)
        _, actual_seq, actual_ch, h, w = sat_data.shape

        if actual_ch != exp_ch or h != enc.image_size_pixels or w != enc.image_size_pixels:
            msg = (
                f"Sat mismatch: Exp {exp_ch}ch, {enc.image_size_pixels}px. "
                f"Got {actual_ch}ch, {h}x{w}px"
            )
            raise ValueError(msg)

        if actual_seq < enc.sequence_length:
            raise ValueError(f"Sat too short: exp {enc.sequence_length}, got {actual_seq}")

    key = "generation"
    if key in batch:
        total_minutes = model.history_minutes + model.forecast_minutes
        expected_len = total_minutes // model.interval_minutes + 1
        expected_shape = (batch[key].shape[0], expected_len)
        actual_shape = tuple(batch[key].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Generation data shape mismatch: expected {expected_shape}, got {actual_shape}"
            )

    logger.info("Batch shape validation successful!")


def validate_gpu_config(config: DictConfig) -> None:
    """Abort if multiple GPUs requested by mistake i.e. `devices: 2` instead of `[2]`."""
    tr = config.get("trainer", {})
    dev = tr.get("devices")

    if isinstance(dev, int) and dev > 1:
        raise ValueError(
            f"Detected `devices: {dev}` — this requests {dev} GPUs. "
            "If you meant a specific GPU (e.g. GPU 2), use `devices: [2]`. "
            "Parallel training not supported."
        )
