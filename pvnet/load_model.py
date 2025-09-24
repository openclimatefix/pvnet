"""Load a model from its checkpoint directory"""

import glob
import os

import hydra
import torch
import yaml

from pvnet.models.ensemble import Ensemble
from pvnet.utils import (
    DATA_CONFIG_NAME,
    DATAMODULE_CONFIG_NAME,
    FULL_CONFIG_NAME,
    MODEL_CONFIG_NAME,
)


def get_model_from_checkpoints(
    checkpoint_dir_paths: list[str],
    val_best: bool = True,
) -> tuple[torch.nn.Module, dict, str, str | None, str | None]:
    """Load a model from its checkpoint directory

    Returns:
        tuple:
            model: nn.Module of pretrained model.
            model_config: path to model config used to train the model.
            data_config: path to data config used to create samples for the model.
            datamodule_config: path to datamodule used to create samples e.g train/test split info.
            experiment_configs: path to the full experimental config.

    """
    is_ensemble = len(checkpoint_dir_paths) > 1

    model_configs = []
    models = []
    data_configs = []
    datamodule_configs = []
    experiment_configs = []

    for path in checkpoint_dir_paths:

        # Load lightning training module
        with open(f"{path}/{MODEL_CONFIG_NAME}") as cfg:
            model_config = yaml.load(cfg, Loader=yaml.FullLoader)

        lightning_module = hydra.utils.instantiate(model_config)

        if val_best:
            # Only one epoch (best) saved per model
            files = glob.glob(f"{path}/epoch*.ckpt")
            if len(files) != 1:
                raise ValueError(
                    f"Found {len(files)} checkpoints @ {path}/epoch*.ckpt. Expected one."
                )
            # TODO: Loading with weights_only=False is not recommended
            checkpoint = torch.load(files[0], map_location="cpu", weights_only=True)
        else:
            checkpoint = torch.load(f"{path}/last.ckpt", map_location="cpu", weights_only=True)

        lightning_module.load_state_dict(state_dict=checkpoint["state_dict"])

        # Extract the model from the lightning module
        models.append(lightning_module.model)
        model_configs.append(model_config["model"])

        # Store the data config used for the model
        data_config = f"{path}/{DATA_CONFIG_NAME}"

        if os.path.isfile(data_config):
            data_configs.append(data_config)
        else:
            raise FileNotFoundError(f"File {data_config} does not exist")

        # TODO: This should be removed in a future release since no new models will be trained on 
        # presaved samples
        # Check for datamodule config
        # This only exists if the model was trained with presaved samples
        datamodule_config = f"{path}/{DATAMODULE_CONFIG_NAME}"
        if os.path.isfile(datamodule_config):
            datamodule_configs.append(datamodule_config)
        else:
            datamodule_configs.append(None)

        # Check for experiment config
        # For backwards compatibility - this might always exist
        experiment_config = f"{path}/{FULL_CONFIG_NAME}"
        if os.path.isfile(datamodule_config):
            experiment_configs.append(experiment_config)
        else:
            experiment_configs.append(None)

    if is_ensemble:
        model_config = {
            "_target_": "pvnet.models.ensemble.Ensemble",
            "model_list": model_configs,
        }
        model = Ensemble(model_list=models)

    else:
        model_config = model_configs[0]
        model = models[0]

    # Assume if using an ensemble that the members were trained on the same input data
    data_config = data_configs[0]
    datamodule_config = datamodule_configs[0]

    # TODO: How should we save the experimental configs if we had an ensemble?
    experiment_config = experiment_configs[0]

    return model, model_config, data_config, datamodule_config, experiment_config
