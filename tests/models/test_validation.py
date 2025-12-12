"""Tests for model and trainer configuration validation utilities."""

import pytest
import torch

from pvnet.utils import validate_batch_against_config, validate_gpu_config


def test_validate_batch_against_config(
    batch: dict,
    late_fusion_model,
):
    """Test batch validation utility function."""
    # This should pass as full uk_batch is valid
    validate_batch_against_config(batch=batch, model=late_fusion_model)


def test_validate_batch_against_config_raises_error(late_fusion_model):
    """Test that the validation raises an error for a mismatched batch."""
    # Create batch that is missing required NWP data
    minimal_batch = {"generation": torch.randn(2, 17)}
    with pytest.raises(
        ValueError,
        match="Model configured with 'nwp_encoders_dict' but 'nwp' data missing from batch.",
    ):
        validate_batch_against_config(batch=minimal_batch, model=late_fusion_model)


@pytest.mark.parametrize(
    "trainer",
    [
        {"devices": 1},
        {"devices": [0]},
        {"accelerator": "cpu"},
    ],
    ids=["devices=1", "devices=[0]", "accelerator=cpu"],
)
def test_validate_gpu_config_single_device(trainer_cfg, trainer):
    """Accept single GPU or explicit CPU configurations."""
    validate_gpu_config(trainer_cfg(trainer))


@pytest.mark.parametrize(
    "trainer",
    [
        {"devices": 2},
    ],
    ids=["devices=2"],
)
def test_validate_gpu_config_multiple_devices(trainer_cfg, trainer):
    """Reject accidental multi-GPU setups."""
    with pytest.raises(ValueError, match="Parallel training not supported"):
        validate_gpu_config(trainer_cfg(trainer))
