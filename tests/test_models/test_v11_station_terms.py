"""v11 ("ctS"): CaseRecordModel with Case×station and Tier×station terms.

No MCMC: graphs are checked at build time and the post-processing on a
point-mass posterior at known parameters (``review_helpers``).
"""

from __future__ import annotations

import numpy as np
import pytest
from review_helpers import TIERS, crossed_config, crossed_df, point_mass_model, truth

from apeBayes.data import encode_dataset
from apeBayes.model import CaseRecordModel
from apeBayes.model.registry import builder_from_spec, builder_spec, model_version

# Free variables and log-density of the 0.4.0 v10 graph on the review design
# (computed from the 0.4.0 source of case_record.py, commit 2548760).
_V10_VARS = ["mu0", "mu_config_zerosum__", "delta_st_zerosum__", "sigma_run_log__",
             "z_run_zerosum__", "sigma_inter_log__", "z_inter_zerosum__", "log_lambda_free",
             "sigma_u_case_log__", "z_u_zerosum__", "sigma_v_case_log__", "z_v_zerosum__",
             "sigma_eps_case_log__", "nu_minus2_log__"]
_V10_LOGP = -1070.472839872557


def _build(builder):
    d = encode_dataset(crossed_df(truth(case_terms=True), noise=True), crossed_config())
    return builder.build(d, crossed_config())


def _logp(m):
    rng = np.random.default_rng(0)
    pt = {k: rng.normal(size=np.shape(v)) * 0.3 for k, v in m.initial_point().items()}
    return float(m.compile_logp()(pt))


def _v11_model():
    m = point_mass_model(truth(case_terms=True, station_terms=True))
    m._builder = CaseRecordModel(case_station=True, tier_station=True)
    return m


# ── graph ───────────────────────────────────────────────────────────────

def test_station_terms_off_is_the_040_v10_graph():
    m = _build(CaseRecordModel())
    assert [v.name for v in m.value_vars] == _V10_VARS
    assert _logp(m) == pytest.approx(_V10_LOGP, rel=1e-10)


def test_v11_graph_adds_both_terms():
    m = _build(CaseRecordModel(case_station=True, tier_station=True))
    names = [v.name for v in m.value_vars]
    assert names[:len(_V10_VARS) - 2] == _V10_VARS[:-2]
    for v in ("sigma_case_station_log__", "z_cs_zerosum__",
              "sigma_tier_station_log__", "z_ts_zerosum__"):
        assert v in names
    assert m.named_vars_to_dims["case_station"] == ("Case", "Station")
    assert m.named_vars_to_dims["tier_station"] == ("Tier", "Station")
    assert list(m.coords["Tier"]) == TIERS


@pytest.mark.parametrize("cs, ts", [(True, False), (False, True)])
def test_each_station_term_alone(cs, ts):
    m = _build(CaseRecordModel(case_station=cs, tier_station=ts))
    assert ("case_station" in m.named_vars) is cs
    assert ("tier_station" in m.named_vars) is ts


def test_station_term_is_zero_sum_over_stations():
    m = _build(CaseRecordModel(case_station=True, tier_station=True))
    draw = __import__("pymc").draw(m["case_station"], random_seed=1)
    np.testing.assert_allclose(draw.sum(axis=1), 0.0, atol=1e-12)
    draw = __import__("pymc").draw(m["tier_station"], random_seed=1)
    np.testing.assert_allclose(draw.sum(axis=1), 0.0, atol=1e-12)


def test_negative_station_scale_rejected():
    with pytest.raises(ValueError, match="sigma_cs"):
        CaseRecordModel(case_station=True, sigma_cs=0.0)


# ── spec / version ──────────────────────────────────────────────────────

def test_v10_spec_has_no_station_options():
    spec = builder_spec(CaseRecordModel())
    assert spec["version"] == "v10"
    assert "case_station" not in spec["options"]


@pytest.mark.parametrize("cs, ts, version", [
    (True, True, "v11"), (True, False, "v10+cS"), (False, True, "v10+tS"),
])
def test_spec_roundtrip_and_version(cs, ts, version):
    b = CaseRecordModel(case_station=cs, tier_station=ts, sigma_cs=0.2)
    spec = builder_spec(b)
    assert spec["version"] == version == model_version(b)
    b2 = builder_from_spec(spec)
    assert b2.options() == b.options()
    assert version in b2.description


def test_detect_v11_from_variables():
    from apeBayes.facade import _detect_builder

    b, tag = _detect_builder(_v11_model().posterior)
    assert tag == "v11"
    assert b.options()["case_station"] and b.options()["tier_station"]


def test_tier_mapping():
    d = encode_dataset(crossed_df(truth()), crossed_config())
    levels, cfg_to_tier, obs = d.tier_mapping()
    assert levels == TIERS
    for lbl, k in zip(d.config_labels, range(d.n_configs), strict=True):
        assert levels[cfg_to_tier[k]] == str(lbl)[0]
    np.testing.assert_array_equal(obs, cfg_to_tier[d.config_idx])


# ── post-processing ─────────────────────────────────────────────────────

def test_linear_predictor_includes_station_terms():
    m = _v11_model()
    np.testing.assert_allclose(m.posterior.linear_predictor()[0], m.data.y, atol=1e-12)
    assert m.r2_table()["R2_cond_med"].iloc[0] == pytest.approx(1.0, abs=1e-12)


def test_station_terms_values():
    t = truth(case_terms=True, station_terms=True)
    p = _v11_model().posterior
    d = p.data
    want = (t["case_station"][d.case_mapping()[2], d.station_idx]
            + t["tier_station"][d.tier_mapping()[2], d.station_idx])
    np.testing.assert_allclose(p.station_terms()[0], want)


@pytest.mark.parametrize("kind", ["superpopulation", "finite"])
def test_variance_budget_has_station_rows(kind):
    t = truth(case_terms=True, station_terms=True)
    m = _v11_model()
    vb = m.variance_budget_table(kind=kind).set_index("component")
    d = m.data
    want_w = np.var(t["case_station"][d.case_mapping()[2], d.station_idx])
    want_k = np.var(t["tier_station"][d.tier_mapping()[2], d.station_idx])
    assert vb.loc["Case×station  Var(w)", "var_med"] == pytest.approx(want_w)
    assert vb.loc["Tier×station  Var(κ)", "var_med"] == pytest.approx(want_k)
    assert list(vb.index)[-1].startswith("Residual")


def test_sigma_gm_default_is_v10_formula_and_case_argument():
    t = truth(case_terms=True, station_terms=True)
    p = _v11_model().posterior
    b = CaseRecordModel(case_station=True, tier_station=True)
    ref = 3  # Case D: reference 4D in crossed_config
    v10 = np.sqrt(t["sigma_run"] ** 2 + t["sigma_inter"] ** 2
                  + t["sigma_u_case"][ref] ** 2 + t["sigma_v_case"][ref] ** 2)
    assert b.sigma_GM(p)[0] == pytest.approx(v10)
    assert b.sigma_GM(p, case="D")[0] == pytest.approx(v10)
    want_a = np.sqrt(t["lambda_case"][0] ** 2 * t["sigma_run"] ** 2 + t["sigma_inter"] ** 2
                     + t["sigma_u_case"][0] ** 2 + t["sigma_v_case"][0] ** 2)
    assert b.sigma_GM(p, case="A")[0] == pytest.approx(want_a)


def test_offsets_do_not_contain_station_terms():
    """Δμ comes from mu_config alone: adding station terms leaves the offset table."""
    t0 = truth(case_terms=True)
    m0 = point_mass_model(t0)
    m1 = _v11_model()
    m0._builder = m1._builder = CaseRecordModel()
    p0 = m0.posterior.mu_config() - m0.posterior.mu_config()[:, [m0.data.ref_idx]]
    p1 = m1.posterior.mu_config() - m1.posterior.mu_config()[:, [m1.data.ref_idx]]
    np.testing.assert_allclose(p0, p1)


def test_loro_mean_uses_station_terms():
    """New rupture at a known station: the predictive mean adds w + κ."""
    from apeBayes.model.cv import _fold_lpd

    t = truth(case_terms=True, station_terms=True)
    m11 = _v11_model()
    t10 = {k: v for k, v in t.items() if k not in
           ("case_station", "tier_station", "sigma_case_station", "sigma_tier_station")}
    m10 = point_mass_model(t10)
    test = crossed_df(t).query("rupture == 'rup_1'")
    cfg = crossed_config()
    lp11 = _fold_lpd(m11.posterior, test, cfg, "run", None, 0)
    lp10 = _fold_lpd(m10.posterior, test, cfg, "run", None, 0)
    assert lp11 > lp10


def test_loso_covariance_adds_station_scales():
    """New station: σ_cs² and σ_ts² enter the predictive covariance."""
    from apeBayes.model.cv import _fold_lpd

    t = truth(case_terms=True, station_terms=True)
    m11 = _v11_model()
    t10 = {k: v for k, v in t.items() if k not in
           ("case_station", "tier_station", "sigma_case_station", "sigma_tier_station")}
    m10 = point_mass_model(t10)
    test = crossed_df(t).query("sta == 'sta_2'")
    cfg = crossed_config()
    assert _fold_lpd(m11.posterior, test, cfg, "station", None, 0) != pytest.approx(
        _fold_lpd(m10.posterior, test, cfg, "station", None, 0))
