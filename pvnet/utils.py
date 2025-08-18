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
    model_config,
    sat_interval_minutes: int = 5,
    gsp_interval_minutes: int = 30,
    site_interval_minutes: int = 30,
) -> None:
    """
    Validates tensor shapes in a batch against model configuration.
    
    Only validates when model_config is an instantiated model object with encoder attributes.
    For DictConfig objects, performs no validation and logs success.
    
    Args:
        batch: Dictionary of tensors from the dataloader.
        model_config: Instantiated model object or config dict.
        sat_interval_minutes: Time resolution of satellite data in minutes.
        gsp_interval_minutes: Time resolution of GSP data in minutes.  
        site_interval_minutes: Time resolution of site data in minutes.
    """
    logger.info("Performing batch shape validation against model config.")
    dim_names = ["batch", "time", "channels", "height", "width"]

    is_instantiated_model = (
        hasattr(model_config, 'nwp_encoders_dict') and 
        not isinstance(model_config, (dict, DictConfig))
    )
    
    if is_instantiated_model:
        model = model_config
        
        # NWP validation
        if hasattr(model, 'nwp_encoders_dict') and "nwp" in batch:
            for source, nwp_data in batch["nwp"].items():
                if source not in model.nwp_encoders_dict:
                    continue

                nwp_tensor = nwp_data["nwp"]
                encoder = model.nwp_encoders_dict[source]
                expected_seq_len = encoder.sequence_length
                
                expected_shape = (
                    nwp_tensor.shape[0],
                    expected_seq_len,
                    encoder.in_channels,
                    encoder.image_size_pixels,
                    encoder.image_size_pixels,
                )
                _check_shape_and_raise(f"NWP.{source}", nwp_tensor, expected_shape, dim_names)

        # Satellite validation
        if hasattr(model, 'sat_encoder') and "sat" in batch:
            sat_tensor = batch["sat"]
            encoder = model.sat_encoder
            expected_seq_len = encoder.sequence_length

            expected_shape = (
                sat_tensor.shape[0],
                expected_seq_len,
                encoder.in_channels,
                encoder.image_size_pixels,
                encoder.image_size_pixels,
            )
            _check_shape_and_raise("Satellite", sat_tensor, expected_shape, dim_names)

        # GSP validation
        if "gsp" in batch:
            gsp_tensor = batch["gsp"]
            
            if hasattr(model, 'history_minutes') and hasattr(model, 'forecast_minutes'):
                history_len = model.history_minutes // gsp_interval_minutes
                forecast_len = model.forecast_minutes // gsp_interval_minutes
                expected_total_len = history_len + forecast_len + 1

                expected_shape = (gsp_tensor.shape[0], expected_total_len)
                gsp_dim_names = ["batch", "time"]
                _check_shape_and_raise("GSP Target", gsp_tensor, expected_shape, gsp_dim_names)

        # Site validation
        if "site" in batch:
            site_tensor = batch["site"]
            
            if hasattr(model, 'history_minutes') and hasattr(model, 'forecast_minutes'):
                history_len = model.history_minutes // site_interval_minutes
                forecast_len = model.forecast_minutes // site_interval_minutes
                expected_total_len = history_len + forecast_len + 1

                expected_shape = (site_tensor.shape[0], expected_total_len)
                site_dim_names = ["batch", "time"]
                _check_shape_and_raise("Site Target", site_tensor, expected_shape, site_dim_names)

    logger.info("Batch shape validation successful!")


def extract_raw_config(model_config) -> DictConfig:
    """Extract raw configuration dictionary from model_config"""
    
    # # Return as-is if already a dict/DictConfig
    # if isinstance(model_config, (dict, DictConfig)):
    #     return model_config
    
    # Extract from instantiated model
    if hasattr(model_config, 'nwp_encoders_dict'):
        config_dict = {'nwp_encoders_dict': {}, 'sat_encoder': {}}
        
        timing_attrs = ['forecast_minutes', 'history_minutes', 'sat_history_minutes', 
                       'nwp_history_minutes', 'nwp_forecast_minutes', 'nwp_interval_minutes']
        
        for attr in timing_attrs:
            if hasattr(model_config, attr):
                config_dict[attr] = getattr(model_config, attr)
        
        if hasattr(model_config, 'nwp_encoders_dict'):
            for source, encoder in model_config.nwp_encoders_dict.items():
                config_dict['nwp_encoders_dict'][source] = {
                    'in_channels': getattr(encoder, 'in_channels', None),
                    'image_size_pixels': getattr(encoder, 'image_size_pixels', None),
                }
        
        if hasattr(model_config, 'sat_encoder'):
            config_dict['sat_encoder'] = {
                'in_channels': getattr(model_config.sat_encoder, 'in_channels', None),
                'image_size_pixels': getattr(model_config.sat_encoder, 'image_size_pixels', None),
            }
        
        return OmegaConf.create(config_dict)
    
    return model_config


def remove_model_config_circular_ref(config: DictConfig) -> DictConfig:
    """Remove model_config circular reference from config before saving."""
    config_copy = OmegaConf.create(config)
    
    # Remove model_config at root level or nested under model
    if "model_config" in config_copy:
        del config_copy.model_config
    elif "model" in config_copy and "model_config" in config_copy.model:
        del config_copy.model.model_config
        
    return config_copy
