"""Tests for model validation utility function."""

import pytest

from pvnet.models.late_fusion.late_fusion import LateFusionModel
from pvnet.utils import validate_batch_against_config


def test_validate_batch_against_config(
    uk_batch: dict, 
    late_fusion_model: LateFusionModel,
    late_fusion_model_kwargs: dict,
):
    """Test batch validation utility function."""

    if "satellite_actual" in uk_batch:
        uk_batch["sat"] = uk_batch.pop("satellite_actual")

    # The model config is now passed in as a separate fixture
    model_config = late_fusion_model_kwargs

    # Assert valid data passes check without error
    validate_batch_against_config(uk_batch, model_config)

    # Intentionally corrupt data - assert that error is raised
    invalid_batch = uk_batch.copy()
    invalid_batch["sat"] = uk_batch["sat"][:, :-1]

    with pytest.raises(ValueError, match="Shape mismatch for 'Satellite' in dimension 'time'"):
        validate_batch_against_config(invalid_batch, model_config)