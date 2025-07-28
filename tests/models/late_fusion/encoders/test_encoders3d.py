from pvnet.models.late_fusion.encoders.encoders3d import (
    DefaultPVNet,
    ResConv3DNet,
)


def _test_model_forward(batch, model_class, model_kwargs):
    model = model_class(**model_kwargs)
    y = model(batch)
    assert tuple(y.shape) == (2, model_kwargs["out_features"]), y.shape


def _test_model_backward(batch, model_class, model_kwargs):
    model = model_class(**model_kwargs)
    y = model(batch)
    # Backwards on sum drives sum to zero
    y.sum().backward()


# Test model forward on all models
def test_defaultpvnet_forward(sample_satellite_batch, encoder_model_kwargs):
    _test_model_forward(sample_satellite_batch, DefaultPVNet, encoder_model_kwargs)


def test_resconv3dnet_forward(sample_satellite_batch, encoder_model_kwargs):
    _test_model_forward(sample_satellite_batch, ResConv3DNet, encoder_model_kwargs)


# Test model backward on all models
def test_defaultpvnet_backward(sample_satellite_batch, encoder_model_kwargs):
    _test_model_backward(sample_satellite_batch, DefaultPVNet, encoder_model_kwargs)


def test_resconv3dnet_backward(sample_satellite_batch, encoder_model_kwargs):
    _test_model_backward(sample_satellite_batch, ResConv3DNet, encoder_model_kwargs)
