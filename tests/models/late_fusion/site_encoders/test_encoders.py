from pvnet.models.late_fusion.site_encoders.encoders import (
    SimpleLearnedAggregator,
    SingleAttentionNetwork,
)


def _test_model_forward(batch, model_class, kwargs, batch_size):
    model = model_class(**kwargs)
    y = model(batch)
    assert tuple(y.shape) == (batch_size, kwargs["out_features"]), y.shape


def _test_model_backward(batch, model_class, kwargs):
    model = model_class(**kwargs)
    y = model(batch)
    # Backwards on sum drives sum to zero
    y.sum().backward()


# Test model forward on all models
def test_simplelearnedaggregator_forward(sample_pv_batch, site_encoder_model_kwargs):
    _test_model_forward(
        sample_pv_batch,
        SimpleLearnedAggregator,
        site_encoder_model_kwargs,
        batch_size=8,
    )


def test_singleattentionnetwork_forward(sample_site_batch, site_encoder_model_kwargs_dsampler):
    _test_model_forward(
        sample_site_batch,
        SingleAttentionNetwork,
        site_encoder_model_kwargs_dsampler,
        batch_size=2,
    )


# Test model backward on all models
def test_simplelearnedaggregator_backward(sample_pv_batch, site_encoder_model_kwargs):
    _test_model_backward(sample_pv_batch, SimpleLearnedAggregator, site_encoder_model_kwargs)


def test_singleattentionnetwork_backward(sample_site_batch, site_encoder_model_kwargs_dsampler):
    _test_model_backward(
        sample_site_batch, SingleAttentionNetwork, site_encoder_model_kwargs_dsampler
    )
