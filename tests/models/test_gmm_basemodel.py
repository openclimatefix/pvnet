import math
import pytest
import torch


from pvnet.models.base_model import BaseModel


class _MinimalGMMModel(BaseModel):
    """Tiny subclass to use BaseModel's GMM helpers without extra encoders."""

    def __init__(self, forecast_len=3, num_components=2):
        # 30-min intervals so forecast_minutes = forecast_len * 30
        super().__init__(
            history_minutes=0,
            forecast_minutes=forecast_len * 30,
            output_quantiles=None,
            num_gmm_components=num_components,
            interval_minutes=30,
        )
        # Flags not used in GMM helpers but defined to be safe
        self.include_sat = False
        self.include_nwp = False
        self.include_sun = False


def _build_y_gmm_from_params(mus, sigma_raws, logits):
    """
    Helper: stack params into the flat y_gmm expected by BaseModel._parse_gmm_params.

    mus, sigma_raws, logits shapes: [B, H, K]
    Returns y_gmm with shape [B, H*K*3]
    """
    B, H, K = mus.shape
    params = torch.stack([mus, sigma_raws, logits], dim=-1)  # [B, H, K, 3]
    return params.reshape(B, H * K * 3)


def test_parse_gmm_params_shapes_and_constraints():
    torch.manual_seed(0)

    B, H, K = 4, 5, 3
    model = _MinimalGMMModel(forecast_len=H, num_components=K)

    # Random pre-activations; sigma_raw can be anything, logits too.
    mus = torch.randn(B, H, K)
    sigma_raws = torch.randn(B, H, K)  # will go through softplus + 1e-3
    logits = torch.randn(B, H, K)  # will go through softmax

    y_gmm = _build_y_gmm_from_params(mus, sigma_raws, logits)

    out_mus, out_sigmas, out_pis = model._parse_gmm_params(y_gmm)

    assert out_mus.shape == (B, H, K)
    assert out_sigmas.shape == (B, H, K)
    assert out_pis.shape == (B, H, K)

    # Sigmas must be strictly positive and >= 1e-3 (because of +1e-3).
    assert torch.all(out_sigmas > 0)
    assert torch.all(out_sigmas >= 1e-3)

    # Pis are a softmax over components: each row sums to ~1, non-negative.
    sums = out_pis.sum(dim=-1)  # [B, H]
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    assert torch.all(out_pis >= 0)


def test_parse_gmm_params_softplus_offset():
    """Check the exact softplus + 1e-3 transform for sigma."""
    B, H, K = 1, 2, 2
    model = _MinimalGMMModel(forecast_len=H, num_components=K)

    mus = torch.zeros(B, H, K)
    # Choose a very negative raw to make softplus small but positive:
    sigma_raws = torch.tensor([[[-10.0, 0.0], [-5.0, 1.5]]])  # [1,2,2]
    logits = torch.zeros(B, H, K)  # irrelevant here

    y_gmm = _build_y_gmm_from_params(mus, sigma_raws, logits)
    _, sigmas, _ = model._parse_gmm_params(y_gmm)

    expected = torch.nn.functional.softplus(sigma_raws) + 1e-3
    assert torch.allclose(sigmas, expected, atol=1e-7)


def test_gmm_to_prediction_matches_manual_expectation():
    """
    Build a tiny case we can compute by hand.
    Use equal logits -> equal weights, and simple means.
    """
    B, H, K = 1, 2, 2
    model = _MinimalGMMModel(forecast_len=H, num_components=K)

    # Means we can average easily:
    # t0: mu=[2.0, 6.0], t1: mu=[10.0, 14.0]
    mus = torch.tensor([[[2.0, 6.0], [10.0, 14.0]]])  # [1,2,2]

    # Raw sigmas don't matter for the expectation; keep simple.
    sigma_raws = torch.zeros(B, H, K)

    # Equal logits -> softmax = [0.5, 0.5]
    logits = torch.zeros(B, H, K)

    y_gmm = _build_y_gmm_from_params(mus, sigma_raws, logits)

    # Model under test
    pred = model._gmm_to_prediction(y_gmm)  # [B, H]

    # Manual E[Y] = sum(pi * mu) = 0.5*(2+6)=4, 0.5*(10+14)=12
    expected = torch.tensor([[4.0, 12.0]])
    assert torch.allclose(pred, expected, atol=1e-6)


def test_sample_from_gmm_empirical_mean_near_expectation():
    """
    Sampling should yield an empirical mean near the analytic mixture mean.
    Use small variances and many samples to tighten the estimate.
    """
    torch.manual_seed(123)

    B, H, K = 2, 3, 3
    model = _MinimalGMMModel(forecast_len=H, num_components=K)

    # Set mus to a simple grid
    mus = torch.tensor(
        [
            [[0.0, 5.0, 10.0], [1.0, 6.0, 11.0], [2.0, 7.0, 12.0]],
            [[3.0, 8.0, 13.0], [4.0, 9.0, 14.0], [5.0, 10.0, 15.0]],
        ]
    )  # [B=2,H=3,K=3]

    # Very small stds to reduce sampling variance
    sigmas = torch.full_like(mus, 0.05)

    # Choose non-uniform, normalized weights per time; e.g., logits = [0, 1, 2]
    logits = torch.tensor([0.0, 1.0, 2.0]).repeat(B, H, 1)
    pis = torch.softmax(logits, dim=-1)

    # Sample a lot to stabilize mean
    n_samples = 5000
    samples = model._sample_from_gmm(mus, sigmas, pis, n_samples=n_samples)  # [S,B,H]

    # Empirical means over samples
    empirical_mean = samples.mean(dim=0)  # [B,H]

    # Analytic mixture mean
    analytic_mean = (pis * mus).sum(dim=-1)  # [B,H]

    # With small variance and many samples, these should be close
    assert torch.allclose(empirical_mean, analytic_mean, atol=0.15)


def test_quantiles_and_gmm_mutually_exclusive():
    """Constructor should forbid using both quantiles and GMM simultaneously."""
    with pytest.raises(ValueError):
        _ = _MinimalGMMModel(forecast_len=2, num_components=2)
        _ = BaseModel(
            history_minutes=0,
            forecast_minutes=60,
            output_quantiles=[0.1, 0.5, 0.9],
            num_gmm_components=2,
            interval_minutes=30,
        )
