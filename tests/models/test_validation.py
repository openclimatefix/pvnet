"""Tests for model validation utility function."""


from pvnet.utils import validate_batch_against_config


def test_validate_batch_against_config(
   uk_batch: dict, 
   late_fusion_model,
):
   """Test batch validation utility function."""

   if "satellite_actual" in uk_batch:
       uk_batch["sat"] = uk_batch.pop("satellite_actual")

   validate_batch_against_config(batch=uk_batch, model_config=late_fusion_model)

   # Test with different interval parameters
   validate_batch_against_config(
       batch=uk_batch, 
       model_config=late_fusion_model,
       sat_interval_minutes=10,
       gsp_interval_minutes=30,
       site_interval_minutes=60
   )

   # Test with missing optional batch keys - should not raise errors
   minimal_batch = {"gsp": uk_batch.get("gsp", [])}
   validate_batch_against_config(batch=minimal_batch, model_config=late_fusion_model)

   # Test with empty batch - should not raise errors
   empty_batch = {}
   validate_batch_against_config(batch=empty_batch, model_config=late_fusion_model)

   # Test function completes properly
   if "nwp" in uk_batch:
       nwp_only_batch = {"nwp": uk_batch["nwp"]}
       validate_batch_against_config(batch=nwp_only_batch, model_config=late_fusion_model)

   if "sat" in uk_batch:
       sat_only_batch = {"sat": uk_batch["sat"]}
       validate_batch_against_config(batch=sat_only_batch, model_config=late_fusion_model)
