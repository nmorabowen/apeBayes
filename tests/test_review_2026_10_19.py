"""Regression tests for the fixes from the 2026-10-19 code / statistics /
model reviews. No MCMC: the posterior is a point mass at known parameters
(see ``review_helpers``)."""

from __future__ import annotations

import numpy as np
import pytest

from review_helpers import point_mass_model, truth


# ── A2: linear predictor includes λ_case, γ_sr (and v10 u, v) ───────────

@pytest.mark.parametrize("case_terms", [False, True])
def test_linear_predictor_reproduces_noise_free_data(case_terms):
    t = truth(case_terms=case_terms)
    m = point_mass_model(t)
    mu = m.posterior.linear_predictor()
    np.testing.assert_allclose(mu[0], m.data.y, atol=1e-12)


@pytest.mark.parametrize("case_terms", [False, True])
def test_r2_conditional_is_one_on_noise_free_data(case_terms):
    m = point_mass_model(truth(case_terms=case_terms))
    r2 = m.r2_table()
    assert r2["R2_cond_med"].iloc[0] == pytest.approx(1.0, abs=1e-12)
    fv = m.fitted_values()
    assert np.abs(fv["resid"]).max() < 1e-12


def test_record_effect_at_matches_predictor():
    t = truth(case_terms=True)
    m = point_mass_model(t)
    p = m.posterior
    d = m.data
    rec = p.record_effect_at(run_idx=2, station_idx=1)  # (S, K)
    sel = (d.run_idx == 2) & (d.station_idx == 1)
    want = p.record_term()[0, sel]
    np.testing.assert_allclose(rec[0, d.config_idx[sel]], want, atol=1e-12)


# ── A1: variance budget is λ-aware, per draw, one footing ───────────────

def test_variance_budget_run_row_is_lambda_weighted():
    t = truth()
    t["sigma_inter"] = 0.2
    m = point_mass_model(t)
    vb = m.variance_budget_table().set_index("component")
    lam2 = np.mean(t["lambda_case"] ** 2)  # balanced: each Case is 1/4 of obs
    run = vb.loc[[i for i in vb.index if i.startswith("Run")][0]]
    assert run["var_med"] == pytest.approx(lam2 * 0.5 ** 2)
    inter = vb.loc[[i for i in vb.index if i.startswith("Interaction")][0]]
    assert inter["var_med"] == pytest.approx(0.2 ** 2)
    assert vb["pct_mean"].sum() == pytest.approx(100.0)
    # per-draw shares: with a point-mass posterior they equal var / total
    total = vb["var_med"].sum()
    np.testing.assert_allclose(vb["pct_med"], 100 * vb["var_med"] / total)


def test_variance_budget_finite_residual_zero_on_noise_free_data():
    m = point_mass_model(truth(case_terms=True))
    vb = m.variance_budget_table(kind="finite").set_index("component")
    assert vb.loc["Residual disp. Var(y−μ)", "var_med"] == pytest.approx(0.0, abs=1e-20)
    assert {"Case×rupture  Var(u)", "Case×sta×rup  Var(v)"} <= set(vb.index)


def test_variance_budget_case_rows_present_for_v10():
    m = point_mass_model(truth(case_terms=True))
    names = m.variance_budget_table()["component"].tolist()
    assert "Case×rupture  σ²_u" in names and "Case×sta×rup  σ²_v" in names


# ── A3: variance_component_table(configs=...) labels the right rows ─────

def test_variance_component_table_subset_labels():
    t = truth()
    m = point_mass_model(t)
    vc = m.variance_component_table(configs=["4D", "1A"]).set_index("component")
    assert vc.loc["sigma_eps[4D]", "med"] == pytest.approx(t["sigma_eps"][15])
    assert vc.loc["sigma_eps[1A]", "med"] == pytest.approx(t["sigma_eps"][0])
