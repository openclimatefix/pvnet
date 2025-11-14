from pvnet.models.late_fusion.site_encoders.encoders import SingleAttentionNetwork


def _test_model_forward(batch, model_class, kwargs, batch_size):
    model = model_class(**kwargs)
    y = model(batch)
    assert tuple(y.shape) == (batch_size, kwargs["out_features"]), y.shape


def _test_model_backward(batch, model_class, kwargs):
    model = model_class(**kwargs)
    y = model(batch)
    # Backwards on sum drives sum to zero
    y.sum().backward()


def test_singleattentionnetwork_forward(batch, site_encoder_model_kwargs):
    _test_model_forward(
        batch,
        SingleAttentionNetwork,
        site_encoder_model_kwargs,
        batch_size=2,
    )


def test_singleattentionnetwork_backward(batch, site_encoder_model_kwargs):
    _test_model_backward(batch, SingleAttentionNetwork, site_encoder_model_kwargs)
