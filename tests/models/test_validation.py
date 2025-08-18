"""Tests for model validation utility function."""

import pytest
from omegaconf import DictConfig

from pvnet.utils import validate_batch_against_config


def test_validate_batch_against_config(
    uk_batch: dict, 
    raw_late_fusion_model_kwargs: dict,
):
    """Test batch validation utility function."""

    if "satellite_actual" in uk_batch:
        uk_batch["sat"] = uk_batch.pop("satellite_actual")

    model_config = DictConfig(raw_late_fusion_model_kwargs)
    validate_batch_against_config(batch=uk_batch, model_config=model_config)
    
    # Test with different interval parameters
    validate_batch_against_config(
        batch=uk_batch, 
        model_config=model_config,
        sat_interval_minutes=10,
        gsp_interval_minutes=15,
        site_interval_minutes=60
    )
    
    # Test with missing optional batch keys - should not raise errors
    minimal_batch = {"gsp": uk_batch.get("gsp", [])}
    validate_batch_against_config(batch=minimal_batch, model_config=model_config)
    
    # Test with empty batch - should not raise errors
    empty_batch = {}
    validate_batch_against_config(batch=empty_batch, model_config=model_config)
    
    # Test with different config structures
    minimal_config = DictConfig({"some_other_key": "value"})
    validate_batch_against_config(batch=uk_batch, model_config=minimal_config)
    
    # Test function completes properly
    if "nwp" in uk_batch:
        nwp_only_batch = {"nwp": uk_batch["nwp"]}
        validate_batch_against_config(batch=nwp_only_batch, model_config=model_config)
    
    if "sat" in uk_batch:
        sat_only_batch = {"sat": uk_batch["sat"]}
        validate_batch_against_config(batch=sat_only_batch, model_config=model_config)