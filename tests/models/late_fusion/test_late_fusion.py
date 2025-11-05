def test_model_forward(late_fusion_model, batch):
    y = late_fusion_model(batch)

    # Check output is the correct shape: [batch size=2, forecast_len=16]
    assert tuple(y.shape) == (2, 16), y.shape

def test_model_forward_site_history(late_fusion_model_generation_history, batch):

    y = late_fusion_model_generation_history(batch)

    # Check output is the correct shape: [batch size=2, forecast_len=32]
    assert tuple(y.shape) == (2, 16), y.shape


def test_model_backward(late_fusion_model, batch):
    y = late_fusion_model(batch)

    # Backwards on sum drives sum to zero
    y.sum().backward()


def test_quantile_model_forward(late_fusion_quantile_model, batch):
    y_quantiles = late_fusion_quantile_model(batch)

    # Check output is the correct shape: [batch size=2, forecast_len=16,  num_quantiles=3]
    assert tuple(y_quantiles.shape) == (2, 16, 3), y_quantiles.shape


def test_quantile_model_backward(late_fusion_quantile_model, batch):

    y_quantiles = late_fusion_quantile_model(batch)

    # Backwards on sum drives sum to zero
    y_quantiles.sum().backward()
