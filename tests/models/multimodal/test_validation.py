import pytest
import torch
from omegaconf import OmegaConf

from pvnet.utils import validate_batch_against_config
from pvnet.models.multimodal.multimodal import Model


def test_validate_batch_against_config(sample_batch: dict, multimodal_model: Model):
    
    if "satellite_actual" in sample_batch:
        sample_batch["sat"] = sample_batch.pop("satellite_actual")
    
    # Assert valid data passes the check without error
    validate_batch_against_config(sample_batch, multimodal_model.hparams)

    # Intentionally corrupt data and assert that error is raised
    invalid_batch = sample_batch.copy()    
    invalid_batch["sat"] = sample_batch["sat"][:, :-1]

    with pytest.raises(ValueError, match="Shape mismatch for 'Satellite' in dimension 'time'"):
        validate_batch_against_config(invalid_batch, multimodal_model.hparams)
