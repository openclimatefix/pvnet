"""Script to migrate old PVNet models which are hosted on huggingface to current version

This script can be used to update models from version >= v4.1
"""

import datetime
import os
import tempfile
from importlib.metadata import version

import torch
import yaml
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, file_exists
from safetensors.torch import save_file

from pvnet.models.base_model import BaseModel
from pvnet.utils import DATA_CONFIG_NAME, MODEL_CARD_NAME, MODEL_CONFIG_NAME, PYTORCH_WEIGHTS_NAME

# ------------------------------------------
# USER SETTINGS

# The huggingface commit of the model you want to update
repo_id: str = "openclimatefix/pvnet_uk_region"
revision: str = "6feaa986a6bed3cc6c7961c6bf9e92fb15acca6a"

# The local directory which will be downloaded to
# If set to None a temporary directory will be used
local_dir: str | None = None

# Whether to upload the migrated model back to the huggingface - else just saved locally
upload: bool = False

# ------------------------------------------
# SETUP

if local_dir is None:
    temp_dir = tempfile.TemporaryDirectory()
    save_dir = temp_dir.name

else:
    os.makedirs(local_dir, exist_ok=False)
    save_dir = local_dir

# Set up huggingface API
api = HfApi()

# Download the model repo
_ = api.snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=save_dir,
    force_download=True,
)

# ------------------------------------------
# MIGRATION STEPS

# Modify the model config
with open(f"{save_dir}/{MODEL_CONFIG_NAME}") as cfg:
    model_config = yaml.load(cfg, Loader=yaml.FullLoader)

# Get rid of the optimiser - we don't store this anymore
if "optimizer" in model_config:
    del model_config["optimizer"]

# This parameter has been moved out of the model to the pytorch lightning module
if "save_validation_results_csv" in model_config:
    del model_config["save_validation_results_csv"]

# This parameter has been removed
if "adapt_batches" in model_config:
    del model_config["adapt_batches"]

# This parameter has been removed
if "target_key" in model_config:
    if model_config["target_key"] == "site":
        if "include_site_yield_history" in model_config:
            model_config["include_generation_yield_history"] = model_config.pop(
                "include_site_yield_history"
            )

    if model_config["target_key"] == "gsp" or not model_config["target_key"]:
        if "include_site_yield_history" in model_config:
            del model_config["include_site_yield_history"]

    del model_config["target_key"]

if "include_gsp_yield_history" in model_config:
    # Set to false on all current gsp models and now defaults to false
    # Also false for all site models so can be removed
    del model_config["include_gsp_yield_history"]

# Rename the top level model
if model_config["_target_"] == "pvnet.models.multimodal.multimodal.Model":
    model_config["_target_"] = "pvnet.models.LateFusionModel"
elif model_config["_target_"] == "pvnet.models.LateFusionModel":
    pass
else:
    raise Exception("Unknown model: " + model_config["_target_"])

# Re-find the model components in the new package structure
if model_config.get("nwp_encoders_dict", None) is not None:
    for k, v in model_config["nwp_encoders_dict"].items():
        v["_target_"] = (
            v["_target_"]
            .replace("multimodal", "late_fusion")
            .replace("ResConv3DNet2", "ResConv3DNet")
        )


for component in ["sat_encoder", "pv_encoder", "output_network"]:
    if model_config.get(component, None) is not None:
        model_config[component]["_target_"] = (
            model_config[component]["_target_"]
            .replace("multimodal", "late_fusion")
            .replace("ResConv3DNet2", "ResConv3DNet")
            .replace("ResFCNet2", "ResFCNet")
        )

with open(f"{save_dir}/{MODEL_CONFIG_NAME}", "w") as f:
    yaml.dump(model_config, f, sort_keys=False, default_flow_style=False)

# Modify the data config
with open(f"{save_dir}/{DATA_CONFIG_NAME}") as cfg:
    data_config = yaml.load(cfg, Loader=yaml.FullLoader)

# Reformat gsp/site generation input data config to new format
if "gsp" in data_config["input_data"]:
    data_config["input_data"]["generation"] = data_config["input_data"].pop("gsp")

    if "boundaries_version" in data_config["input_data"]["generation"]:
        del data_config["input_data"]["generation"]["boundaries_version"]

if "site" in data_config["input_data"]:
    data_config["input_data"]["generation"] = data_config["input_data"].pop("site")

    data_config["input_data"]["generation"]["zarr_path"] = data_config["input_data"][
        "generation"
    ].pop("file_path")

    if "metadata_file_path" in data_config["input_data"]["generation"]:
        del data_config["input_data"]["generation"]["metadata_file_path"]

with open(f"{save_dir}/{DATA_CONFIG_NAME}", "w") as f:
    yaml.dump(data_config, f, sort_keys=False, default_flow_style=False)

# Resave the model weights as safetensors if in old format
if os.path.exists(f"{save_dir}/pytorch_model.bin"):
    state_dict = torch.load(f"{save_dir}/pytorch_model.bin", map_location="cpu", weights_only=True)
    save_file(state_dict, f"{save_dir}/{PYTORCH_WEIGHTS_NAME}")
    os.remove(f"{save_dir}/pytorch_model.bin")
else:
    assert os.path.exists(f"{save_dir}/{PYTORCH_WEIGHTS_NAME}")

# Add a note to the model card to say the model has been migrated
with open(f"{save_dir}/{MODEL_CARD_NAME}", "a") as f:
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    pvnet_version = version("pvnet")
    f.write(
        f"\n\n---\n**Migration Note**: This model was migrated on {current_date} "
        f"to pvnet version {pvnet_version}\n"
    )

# ------------------------------------------
# CHECKS

# Check the model can be loaded
model = BaseModel.from_pretrained(model_id=save_dir, revision=None)

print("Model checkpoint successfully migrated")

# ------------------------------------------
# UPLOAD TO HUGGINGFACE

if upload:
    print("Uploading migrated model to huggingface")

    operations = []
    for file in [MODEL_CARD_NAME, MODEL_CONFIG_NAME, PYTORCH_WEIGHTS_NAME, DATA_CONFIG_NAME]:
        # Stage modified files for upload
        operations.append(
            CommitOperationAdd(
                path_in_repo=file,  # Name of the file in the repo
                path_or_fileobj=f"{save_dir}/{file}",  # Local path to the file
            ),
        )

    # Remove old pytorch weights file if it exists in the most recent commit
    if file_exists(repo_id, "pytorch_model.bin"):
        operations.append(CommitOperationDelete(path_in_repo="pytorch_model.bin"))

    commit_info = api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message=f"Migrate model (HF commit {revision[:7]}) to pvnet version {pvnet_version}",
    )

    # Print the most recent commit hash
    c = api.list_repo_commits(repo_id=repo_id, repo_type="model")[0]

    print(
        f"\nThe latest commit is now: \n"
        f"    date: {c.created_at} \n"
        f"    commit hash: {c.commit_id}\n"
        f"    by: {c.authors}\n"
        f"    title: {c.title}\n"
    )

if local_dir is None:
    temp_dir.cleanup()
