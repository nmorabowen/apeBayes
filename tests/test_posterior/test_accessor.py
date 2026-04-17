"""Tests for PosteriorAccessor sigma_src / sigma_pred helpers."""
from __future__ import annotations

import numpy as np


def test_sigma_src_matches_sigma_run(stub_posterior_v8):
    """sigma_src() is the Python-facing alias for sigma_run()."""
    a = stub_posterior_v8.sigma_src()
    b = stub_posterior_v8.sigma_run()
    assert a.shape == b.shape
    np.testing.assert_array_equal(a, b)


def test_sigma_pred_with_array_denominator(stub_posterior_v8):
    """sigma_pred accepts a precomputed sigma_GM array."""
    p = stub_posterior_v8
    gm = np.sqrt(p.sigma_src() ** 2 + p.sigma_inter() ** 2)
    # Heteroskedastic residual (S, K) -> pick ref column explicitly.
    pred = p.sigma_pred(gm, ref_idx=0)
    eps_eff = p.sigma_eps_effective()[:, 0]
    expected = np.sqrt(gm ** 2 + eps_eff ** 2)
    np.testing.assert_allclose(pred, expected)


def test_sigma_pred_with_callable_denominator(stub_posterior_v8):
    """sigma_pred accepts a callable that assembles sigma_GM from the accessor."""
    p = stub_posterior_v8

    def sigma_gm(post):
        return np.sqrt(post.sigma_src() ** 2 + post.sigma_inter() ** 2)

    pred = p.sigma_pred(sigma_gm, ref_idx=0)
    assert pred.shape == (p.n_samples,)
    # Strictly larger than σ_GM since residual > 0
    gm = sigma_gm(p)
    assert np.all(pred >= gm)


def test_sigma_pred_broadcasts_when_ref_idx_omitted(stub_posterior_v8):
    """Without ref_idx, heteroskedastic residual broadcasts across configs."""
    p = stub_posterior_v8
    gm = np.sqrt(p.sigma_src() ** 2 + p.sigma_inter() ** 2)
    pred = p.sigma_pred(gm)  # (S, K)
    assert pred.ndim == 2
    assert pred.shape == (p.n_samples, 16)


def test_sigma_pred_homoskedastic(stub_dataset):
    """Homoskedastic path: sigma_eps (S,) -> sigma_pred (S,)."""
    from apeBayes.posterior import PosteriorAccessor
    from tests.conftest import _make_idata  # type: ignore[import-not-found]

    rng = np.random.default_rng(3)
    S = 40
    sigma_src = np.abs(rng.normal(0.55, 0.05, size=S))
    sigma_inter = np.abs(rng.normal(0.30, 0.04, size=S))
    sigma_eps_scalar = np.abs(rng.normal(0.10, 0.02, size=S))
    nu = np.abs(rng.normal(25.0, 3.0, size=S)) + 3.0
    idata = _make_idata(
        sigma_src=sigma_src, sigma_inter=sigma_inter,
        sigma_eps=sigma_eps_scalar, nu=nu,
    )
    p = PosteriorAccessor(idata, stub_dataset)

    gm = np.sqrt(p.sigma_src() ** 2 + p.sigma_inter() ** 2)
    pred = p.sigma_pred(gm)
    assert pred.shape == (S,)
    # Student-t correction: σ_eps_eff = σ_eps * sqrt(ν / (ν-2))
    factor = np.sqrt(nu / (nu - 2.0))
    expected = np.sqrt(gm ** 2 + (sigma_eps_scalar * factor) ** 2)
    np.testing.assert_allclose(pred, expected)
