"""Tests for model validation utility function."""

import pytest
import torch

from pvnet.utils import validate_batch_against_config


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
    
    # Assert that correct ValueError is raised
    with pytest.raises(
        ValueError, 
        match="Model configured with 'nwp_encoders_dict' but 'nwp' data missing from batch."
    ):
        validate_batch_against_config(batch=minimal_batch, model=late_fusion_model)
