import pytest
import torch
from pvnet.models.base_model import BaseModel


def test_parse_gmm_params_shapes_and_constraints(
    gmm_model_factory, build_y_gmm_from_params
):
    torch.manual_seed(0)

    B, H, K = 4, 5, 3
    model = gmm_model_factory(forecast_len=H, num_components=K)

    mus = torch.randn(B, H, K)
    sigma_raws = torch.randn(B, H, K)  # will go through softplus + 1e-3
    logits = torch.randn(B, H, K)  # will go through softmax

    y_gmm = build_y_gmm_from_params(mus, sigma_raws, logits)
    out_mus, out_sigmas, out_pis = model._parse_gmm_params(y_gmm)

    assert out_mus.shape == (B, H, K)
    assert out_sigmas.shape == (B, H, K)
    assert out_pis.shape == (B, H, K)

    assert torch.all(out_sigmas > 0)
    assert torch.all(out_sigmas >= 1e-3)

    sums = out_pis.sum(dim=-1)  # [B, H]
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    assert torch.all(out_pis >= 0)


def test_parse_gmm_params_softplus_offset(gmm_model_factory, build_y_gmm_from_params):
    """Check the exact softplus + 1e-3 transform for sigma."""
    B, H, K = 1, 2, 2
    model = gmm_model_factory(forecast_len=H, num_components=K)

    mus = torch.zeros(B, H, K)
    sigma_raws = torch.tensor(
        [[[-10.0, 0.0], [-5.0, 1.5]]], dtype=torch.float
    )  # [1,2,2]
    logits = torch.zeros(B, H, K)

    y_gmm = build_y_gmm_from_params(mus, sigma_raws, logits)
    _, sigmas, _ = model._parse_gmm_params(y_gmm)

    expected = torch.nn.functional.softplus(sigma_raws) + 1e-3
    assert torch.allclose(sigmas, expected, atol=1e-7)


def test_gmm_to_prediction_matches_manual_expectation(
    gmm_model_factory, build_y_gmm_from_params
):
    """
    Build a tiny case we can compute by hand.
    Use equal logits -> equal weights, and simple means.
    """
    B, H, K = 1, 2, 2
    model = gmm_model_factory(forecast_len=H, num_components=K)

    mus = torch.tensor([[[2.0, 6.0], [10.0, 14.0]]])  # [1,2,2]
    sigma_raws = torch.zeros(B, H, K)
    logits = torch.zeros(B, H, K)  # equal weights

    y_gmm = build_y_gmm_from_params(mus, sigma_raws, logits)
    pred = model._gmm_to_prediction(y_gmm)  # [B, H]

    expected = torch.tensor([[4.0, 12.0]])
    assert torch.allclose(pred, expected, atol=1e-6)


def test_sample_from_gmm_empirical_mean_near_expectation(gmm_model_factory):
    """
    Sampling should yield an empirical mean near the analytic mixture mean.
    Use small variances and many samples to tighten the estimate.
    """
    torch.manual_seed(123)

    B, H, K = 2, 3, 3
    model = gmm_model_factory(forecast_len=H, num_components=K)

    mus = torch.tensor(
        [
            [[0.0, 5.0, 10.0], [1.0, 6.0, 11.0], [2.0, 7.0, 12.0]],
            [[3.0, 8.0, 13.0], [4.0, 9.0, 14.0], [5.0, 10.0, 15.0]],
        ]
    )
    sigmas = torch.full_like(mus, 0.05)

    logits = torch.tensor([0.0, 1.0, 2.0]).repeat(B, H, 1)
    pis = torch.softmax(logits, dim=-1)

    n_samples = 5000
    samples = model._sample_from_gmm(mus, sigmas, pis, n_samples=n_samples)  # [S,B,H]

    empirical_mean = samples.mean(dim=0)  # [B,H]
    analytic_mean = (pis * mus).sum(dim=-1)

    assert torch.allclose(empirical_mean, analytic_mean, atol=0.15)


def test_quantiles_and_gmm_mutually_exclusive():
    """Constructor should forbid using both quantiles and GMM simultaneously."""
    with pytest.raises(ValueError):
        _ = BaseModel(
            history_minutes=0,
            forecast_minutes=60,
            output_quantiles=[0.1, 0.5, 0.9],
            num_gmm_components=2,
            interval_minutes=30,
        )
