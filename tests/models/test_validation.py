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

    # Convert dict to DictConfig to match function expectations
    model_config = DictConfig(raw_late_fusion_model_kwargs)

    # Assert valid data passes the check without error
    validate_batch_against_config(batch=uk_batch, model_config=model_config)

    # Test missing satellite data when model expects it
    if "sat_encoder" in model_config:
        batch_missing_sat = uk_batch.copy()
        batch_missing_sat.pop("sat", None)
        
        with pytest.raises(
            ValueError, 
            match="Model configured with 'sat_encoder' but 'sat' data is missing from batch"
        ):
            validate_batch_against_config(batch=batch_missing_sat, model_config=model_config)

    # Test missing NWP data when model expects it
    if "nwp_encoders_dict" in model_config:
        batch_missing_nwp = uk_batch.copy()
        batch_missing_nwp.pop("nwp", None)
        
        with pytest.raises(
            ValueError, 
            match="Model configured with 'nwp_encoders_dict' but 'nwp' data is missing from batch"
        ):
            validate_batch_against_config(batch=batch_missing_nwp, model_config=model_config)

    # Intentionally corrupt satellite data shape - assert that an error is raised
    if "sat" in uk_batch:
        invalid_batch = uk_batch.copy()
        invalid_batch["sat"] = uk_batch["sat"][:, :-1]

        with pytest.raises(ValueError, match="Shape mismatch for 'Satellite' in dimension 'time'"):
            validate_batch_against_config(batch=invalid_batch, model_config=model_config)

    # Test site data validation if present
    if "site" in uk_batch:
        invalid_site_batch = uk_batch.copy()
        invalid_site_batch["site"] = uk_batch["site"][:, :-1]

        with pytest.raises(
            ValueError, 
            match="Shape mismatch for 'Site Target' in dimension 'time'"
        ):
            validate_batch_against_config(batch=invalid_site_batch, model_config=model_config)