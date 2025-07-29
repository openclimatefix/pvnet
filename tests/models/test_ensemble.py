from pvnet.models.ensemble import Ensemble


def test_model_init(late_fusion_model):
    # Without weighting
    ensemble_model = Ensemble(model_list=[late_fusion_model] * 3, weights=None)

    # With weighting
    ensemble_model = Ensemble(model_list=[late_fusion_model] * 3, weights=[1, 2, 3])


def test_model_forward(late_fusion_model, sample_batch):
    ensemble_model = Ensemble(model_list=[late_fusion_model] * 3)

    y = ensemble_model(sample_batch)

    # Check output is the correct shape: [batch size=2, forecast_len=16]
    assert tuple(y.shape) == (2, 16), y.shape


def test_quantile_model_forward(late_fusion_quantile_model, sample_batch):
    ensemble_model = Ensemble(model_list=[late_fusion_quantile_model] * 3)

    y_quantiles = ensemble_model(sample_batch)

    # Check output is the correct shape: [batch size=2, forecast_len=16, num_quantiles=3]
    assert tuple(y_quantiles.shape) == (2, 16, 3), y_quantiles.shape
