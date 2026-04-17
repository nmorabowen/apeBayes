"""Tests for BayesEpistemicModel.decision_report()."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apeBayes import BayesEpistemicModel
from apeBayes.model import RandomSlopesInteractionModel


@pytest.fixture
def model_v8(synthetic_long_df, default_config, stub_posterior_v8):
    """BayesEpistemicModel with a v8-stub posterior attached (no MCMC)."""
    m = BayesEpistemicModel(synthetic_long_df, cfg=default_config)
    # Shortcut: inject the stub posterior and the matching builder.
    m._posterior = stub_posterior_v8  # type: ignore[attr-defined]
    m._builder = RandomSlopesInteractionModel()  # type: ignore[attr-defined]
    return m


def test_decision_report_schema(model_v8):
    df = model_v8.decision_report()
    assert isinstance(df, pd.DataFrame)
    assert "Config" in df.columns
    assert len(df) == 16
    # Canonical \u03b2_GM always present
    for col in ("beta_gm_med", "beta_gm_lo", "beta_gm_hi"):
        assert col in df.columns
    # Full triple by default
    for col in ("beta_src_med", "beta_pred_med"):
        assert col in df.columns
    # P_eq ladder under gm
    for a in (0.4, 0.7, 1.1):
        assert f"P_eq_{a:g}_gm" in df.columns
    # Sensitivity columns at \u03b1_eq
    assert "P_eq_0.4_src" in df.columns
    assert "P_eq_0.4_pred" in df.columns
    # Decision + robustness
    assert "decision" in df.columns
    assert "denominator_robust" in df.columns


def test_decision_labels_are_valid(model_v8):
    df = model_v8.decision_report()
    assert set(df["decision"]).issubset({"equivalent", "inequivalent", "undecided"})


def test_p_eq_in_unit_interval(model_v8):
    df = model_v8.decision_report()
    for col in df.columns:
        if col.startswith("P_eq_"):
            assert df[col].between(0, 1).all(), col


def test_denominator_subset_skips_unrequested_cols(model_v8):
    """Requesting a subset of denominators omits the others' columns."""
    df = model_v8.decision_report(denominators=("src", "gm"))
    assert "beta_src_med" in df.columns
    assert "beta_gm_med" in df.columns
    assert "beta_pred_med" not in df.columns
    # P_eq sensitivity at \u03b1_eq only for the non-gm denoms requested
    assert "P_eq_0.4_src" in df.columns
    assert "P_eq_0.4_pred" not in df.columns


def test_gm_required_for_decision(model_v8):
    with pytest.raises(ValueError, match="gm"):
        model_v8.decision_report(denominators=("src", "pred"))


def test_ref_row_is_equivalent(model_v8):
    df = model_v8.decision_report()
    ref = model_v8.data.ref_label
    row = df[df["Config"] == ref].iloc[0]
    # Reference contrast is identically zero \u2192 P_eq = 1 for every \u03b1.
    assert row["beta_gm_med"] == 0.0
    assert row["P_eq_0.4_gm"] == 1.0
    assert row["decision"] == "equivalent"


def test_engineered_equivalent_and_inequivalent(
    synthetic_long_df, default_config, stub_dataset,
):
    """Construct a posterior where rows are pre-known equivalent / inequivalent."""
    from apeBayes.posterior import PosteriorAccessor
    from tests.conftest import _make_idata  # type: ignore[import-not-found]

    rng = np.random.default_rng(11)
    S = 400
    sigma_src = np.abs(rng.normal(0.6, 0.02, size=S))
    sigma_inter = np.abs(rng.normal(0.2, 0.02, size=S))
    # \u03c3_GM median ~ \u221a(0.36 + 0.04) = 0.632
    mu0 = np.zeros(S)
    mu_config = np.zeros((S, 16))
    # Config 0 is the reference (\u0394\u03bc = 0 \u2192 equivalent)
    # Config 1 gets a small bias well under \u03b1_eq \u00b7 \u03c3_GM \u2248 0.25 \u2192 equivalent
    mu_config[:, 1] = rng.normal(0.05, 0.01, size=S)
    # Config 2 gets a bias far exceeding the band \u2192 inequivalent
    mu_config[:, 2] = rng.normal(1.0, 0.02, size=S)
    # sum-to-zero for the rest
    sigma_eps = np.abs(rng.normal(0.10, 0.01, size=(S, 16)))
    nu = np.abs(rng.normal(25.0, 1.0, size=S)) + 3.0
    idata = _make_idata(
        sigma_src=sigma_src, sigma_inter=sigma_inter,
        mu0=mu0, mu_config=mu_config, sigma_eps=sigma_eps, nu=nu,
    )
    post = PosteriorAccessor(idata, stub_dataset)

    m = BayesEpistemicModel(synthetic_long_df, cfg=default_config)
    m._posterior = post
    m._builder = RandomSlopesInteractionModel()
    df = m.decision_report(ref=post.data.config_labels[0])
    row1 = df[df["Config"] == post.data.config_labels[1]].iloc[0]
    row2 = df[df["Config"] == post.data.config_labels[2]].iloc[0]
    assert row1["decision"] == "equivalent"
    assert row2["decision"] == "inequivalent"


def test_denominator_robust_flag_true_when_all_agree(model_v8):
    """When \u03b2_src, \u03b2_GM, \u03b2_pred all clear the band, denominator_robust is True."""
    df = model_v8.decision_report()
    # The reference row definitely has all three at P_eq = 1
    ref = model_v8.data.ref_label
    row = df[df["Config"] == ref].iloc[0]
    assert row["denominator_robust"]


def test_custom_alpha_eq_must_be_in_ladder(model_v8):
    with pytest.raises(ValueError, match="alpha_ladder"):
        model_v8.decision_report(alpha_eq=0.5, alpha_ladder=(0.4, 0.7, 1.1))
