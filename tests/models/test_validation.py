"""Tests for model and trainer configuration validation utilities."""

import pytest
import torch

from pvnet.utils import validate_batch_against_config, validate_gpu_config


def test_validate_batch_against_config(
    uk_batch: dict,
    late_fusion_model,
):
    """Test batch validation utility function."""
    # This should pass as full uk_batch is valid
    validate_batch_against_config(batch=uk_batch, model=late_fusion_model)


def test_validate_batch_against_config_raises_error(late_fusion_model):
    """Test that the validation raises an error for a mismatched batch."""
    # Create batch that is missing required NWP data
    minimal_batch = {"gsp": torch.randn(2, 17)}
    with pytest.raises(
        ValueError,
        match="Model configured with 'nwp_encoders_dict' but 'nwp' data missing from batch.",
    ):
        validate_batch_against_config(batch=minimal_batch, model=late_fusion_model)


def test_validate_gpu_config_single_device_ok(trainer_cfg):
    """Accept single device."""
    validate_gpu_config(trainer_cfg({"devices": 1}))
    validate_gpu_config(trainer_cfg({"devices": [0]}))
    validate_gpu_config(trainer_cfg({"accelerator": "cpu"}))


@pytest.mark.parametrize("trainer", [
    {"devices": 2},
    {"devices": [0, 1]},
])
def test_validate_gpu_config_multiple_devices_raises(trainer_cfg, trainer):
    """Reject multiple devices."""
    with pytest.raises(ValueError, match="Parallel training"):
        validate_gpu_config(trainer_cfg(trainer))


@pytest.mark.parametrize("strategy", ["ddp", "deepspeed", "fsdp"])
def test_validate_gpu_config_distributed_strategy_raises(trainer_cfg, strategy):
    """Reject distributed strategies not supported by PVNet."""
    with pytest.raises(ValueError, match="Unsupported strategy"):
        validate_gpu_config(trainer_cfg({"strategy": strategy}))
