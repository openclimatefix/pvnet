"""Script to migrate old PVNet models (v4.1) which are hosted on huggingface to current version"""
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
repo_id: str = "openclimatefix/pvnet_uk_region_day_ahead"
revision: str = "e12f7d7c16a00c41b03aecb787a7e992fbd4235b"

# The local directory which will be downloaded to
# If set to None a temporary directory will be used
local_dir: str | None = None 

# Whether to upload the migrated model back to the huggingface - else just saved locally
upload: bool = True

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
del model_config["optimizer"]

# Rename the top level model
if model_config["_target_"]=="pvnet.models.multimodal.multimodal.Model":
    model_config["_target_"] = "pvnet.models.LateFusionModel"
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

# Resave the model weights as safetensors
state_dict = torch.load(f"{save_dir}/pytorch_model.bin", map_location="cpu", weights_only=True)
save_file(state_dict, f"{save_dir}/{PYTORCH_WEIGHTS_NAME}")
os.remove(f"{save_dir}/pytorch_model.bin")

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
                path_in_repo=file, # Name of the file in the repo
                path_or_fileobj=f"{save_dir}/{file}", # Local path to the file
            ),
        )

    # Remove old pytorch weights file if it exists in the most recent commit
    if file_exists(repo_id, "pytorch_model.bin"):
        operations.append(
            CommitOperationDelete(path_in_repo="pytorch_model.bin")
        )

    commit_info = api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message=f"Migrate model to pvnet version {pvnet_version}",
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
