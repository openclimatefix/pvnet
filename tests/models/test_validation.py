"""Tests for model and trainer configuration validation utilities."""

import pytest
import torch

from pvnet.utils import validate_batch_against_config, validate_gpu_config


def test_validate_batch_against_config(batch, late_fusion_model):
    """Test batch validation utility function."""
    # This should pass as full uk_batch is valid
    validate_batch_against_config(batch=batch, model=late_fusion_model)


def test_validate_batch_against_config_raises_error(late_fusion_model):
    """Test that the validation raises an error for a mismatched batch."""
    # Create batch that is missing required NWP data
    minimal_batch = {"generation": torch.randn(2, 17)}
    with pytest.raises(ValueError, match="Model uses NWP data but 'nwp' missing from batch."):
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


def test_validate_gpu_config_multiple_devices(trainer_cfg):
    """Reject accidental multi-GPU setups."""
    with pytest.raises(ValueError, match="Parallel training not supported"):
        validate_gpu_config(trainer_cfg({"devices": 2}))


def test_validate_batch_longer_sequence(batch, late_fusion_model):
    enc = late_fusion_model.sat_encoder
    actual_ch = enc.in_channels - int(late_fusion_model.add_image_embedding_channel)

    exp_len = late_fusion_model.history_len + late_fusion_model.forecast_len + 1
    bsz = batch["generation"].shape[0]

    longer_batch = {
        "satellite_actual": torch.randn(
            bsz,
            enc.sequence_length + 5,
            actual_ch,
            enc.image_size_pixels,
            enc.image_size_pixels,
        ),
        "nwp": batch["nwp"],
        "generation": batch["generation"],
        "solar_azimuth": torch.randn(bsz, exp_len),
        "solar_elevation": torch.randn(bsz, exp_len),
    }

    validate_batch_against_config(batch=longer_batch, model=late_fusion_model)


def test_validate_batch_against_shorter_sequence(late_fusion_model):
    """Test validation raises error when sequence shorter than required"""

    # Configured for sat only as of the moment
    late_fusion_model.include_nwp = False 
    enc = late_fusion_model.sat_encoder
    actual_ch = enc.in_channels - int(late_fusion_model.add_image_embedding_channel)
    
    short_batch = {
        "satellite_actual": torch.randn(
            1, 
            1,
            actual_ch, 
            enc.image_size_pixels, 
            enc.image_size_pixels
        ),
    }
    
    with pytest.raises(ValueError, match="Sat too short"):
        validate_batch_against_config(batch=short_batch, model=late_fusion_model)
