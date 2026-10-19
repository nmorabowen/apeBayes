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
