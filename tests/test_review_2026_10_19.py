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


# ── Reproducibility pin: default graphs are those of main @ 53abef8 ─────

_MAIN_GRAPHS = {
    "v8": (["mu0", "tau_config_log__", "z_config_free", "delta_st_free",
            "sigma_run_log__", "z_run_free", "sigma_inter_log__", "z_inter_free",
            "log_lambda_free", "sigma_eps_config_log__", "nu_minus2_log__"],
           -915.7262214799372),
    "v81": (["mu0", "tau_config_log__", "z_config_free", "delta_st_free",
             "sigma_run_log__", "z_run_free", "sigma_inter_log__", "z_inter_free",
             "log_lambda_free", "log_sigma_eps_bar", "tau_sigma_eps_log__",
             "z_sigma_eps", "nu_minus2_log__"], -825.4180093769496),
    "v4": (["mu0", "tau_config_log__", "z_config_free", "delta_st_free",
            "sigma_run_log__", "z_run_free", "log_lambda_free",
            "sigma_eps_config_log__", "nu_minus2_log__"], -853.0790364283464),
    "v4c": (["mu0", "tau_config_log__", "z_config_free", "delta_st_free",
             "sigma_run_log__", "b_run_free", "log_lambda_free",
             "sigma_eps_config_log__", "nu_minus2_log__"], -853.1566524612343),
}


def _default_builders():
    from apeBayes.model import RandomSlopesInteractionModel, RandomSlopesModel
    return {
        "v8": RandomSlopesInteractionModel(),
        "v81": RandomSlopesInteractionModel(residual_pooling="partial"),
        "v4": RandomSlopesModel(),
        "v4c": RandomSlopesModel(centered_runs=True),
    }


@pytest.mark.parametrize("name", list(_MAIN_GRAPHS))
def test_default_builder_graph_unchanged(name):
    """Same free variables, same order, same log-density as the published code."""
    from apeBayes.data import encode_dataset
    from review_helpers import crossed_config, crossed_df

    d = encode_dataset(crossed_df(truth(), noise=True), crossed_config())
    m = _default_builders()[name].build(d, crossed_config())
    rng = np.random.default_rng(0)
    pt = {k: rng.normal(size=np.shape(v)) * 0.3 for k, v in m.initial_point().items()}
    names, logp = _MAIN_GRAPHS[name]
    assert [v.name for v in m.value_vars] == names
    assert float(m.compile_logp()(pt)) == pytest.approx(logp, rel=1e-10)


# ── B: zero_sum="zsn" ───────────────────────────────────────────────────

def test_zsn_prior_is_symmetric_and_double_zero_sum():
    import pymc as pm

    from apeBayes.data import encode_dataset
    from apeBayes.model import RandomSlopesInteractionModel
    from review_helpers import crossed_config, crossed_df

    d = encode_dataset(crossed_df(truth(), noise=True), crossed_config())
    m = RandomSlopesInteractionModel(zero_sum="zsn").build(d, crossed_config())
    with m:
        pr = pm.sample_prior_predictive(
            draws=4000, random_seed=3,
            var_names=["b_run", "sigma_run", "gamma_sr", "sigma_inter", "mu_config",
                       "tau_config"],
        ).prior
    b = pr["b_run"].values.reshape(-1, 6) / pr["sigma_run"].values.reshape(-1, 1)
    sd = b.std(0)
    assert sd.max() / sd.min() < 1.1  # last level no longer (n-1)x wider
    z = pr["mu_config"].values.reshape(-1, 16) / pr["tau_config"].values.reshape(-1, 1)
    assert z.std(0).max() / z.std(0).min() < 1.1
    g = pr["gamma_sr"].values.reshape(-1, 3, 6)
    np.testing.assert_allclose(g.sum(axis=1), 0, atol=1e-8)  # within each rupture
    np.testing.assert_allclose(g.sum(axis=2), 0, atol=1e-8)  # within each station


def test_last_scheme_prior_is_asymmetric():
    """Documents the published scheme: last rupture has a sqrt(5)x wider prior."""
    import pymc as pm

    from apeBayes.data import encode_dataset
    from apeBayes.model import RandomSlopesInteractionModel
    from review_helpers import crossed_config, crossed_df

    d = encode_dataset(crossed_df(truth(), noise=True), crossed_config())
    with RandomSlopesInteractionModel().build(d, crossed_config()):
        pr = pm.sample_prior_predictive(draws=4000, random_seed=3,
                                        var_names=["b_run", "sigma_run"]).prior
    b = pr["b_run"].values.reshape(-1, 6) / pr["sigma_run"].values.reshape(-1, 1)
    assert b.std(0)[-1] / b.std(0)[0] == pytest.approx(np.sqrt(5), rel=0.1)


# ── A8: builder options serialised; v8.1 loads as v8.1 ──────────────────

def test_builder_spec_roundtrip():
    from apeBayes.model import (
        CaseRecordModel,
        RandomSlopesInteractionModel,
        builder_from_spec,
        builder_spec,
        model_version,
    )

    b = RandomSlopesInteractionModel(
        residual_pooling="partial", residual_tau=1.7, residual_tau_dist="halfcauchy",
        sigma_inter=0.4, residual_log_mean=-1.0, residual_log_sd=2.0, zero_sum="zsn",
    )
    spec = builder_spec(b)
    assert spec["version"] == "v8.1"
    b2 = builder_from_spec(spec)
    assert b2.options() == b.options()
    assert model_version(CaseRecordModel()) == "v10"


def test_bundle_roundtrip_keeps_builder(tmp_path):
    from apeBayes import BayesEpistemicModel
    from apeBayes.model import RandomSlopesInteractionModel

    m = point_mass_model(truth())
    m._builder = RandomSlopesInteractionModel(residual_pooling="partial", residual_tau=1.7)
    path = m.save(tmp_path / "fit")
    m2 = BayesEpistemicModel.load(path)
    assert m2.builder.options() == m._builder.options()


def test_detect_v81_from_variables():
    from apeBayes.facade import _detect_builder

    t = truth()
    t["tau_sigma_eps"] = 0.5
    m = point_mass_model(t)
    _, tag = _detect_builder(m.posterior)
    assert tag == "v8.1"


def test_load_rejects_mismatched_coordinates(tmp_path):
    from apeBayes import BayesEpistemicModel

    m = point_mass_model(truth())
    path = m.save(tmp_path / "fit")
    df = m._df.copy()
    df["sta"] = df["sta"].map({"sta_2": "sta_9", "sta_5": "sta_5", "sta_8": "sta_8"})
    with pytest.raises(ValueError, match="Station"):
        BayesEpistemicModel.load(path, df=df)


# ── A6: convergence gate ────────────────────────────────────────────────

def _two_chain_idata(shift: float, n: int = 500, seed: int = 0):
    import arviz as az
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(2, n))
    x[1] += shift
    return az.from_dict(posterior={"theta": x})


def test_convergence_report_flags_bad_chains():
    from apeBayes.diagnostics import convergence_report

    good = convergence_report(_two_chain_idata(0.0, n=2000))
    assert good["converged"]
    bad = convergence_report(_two_chain_idata(3.0))
    assert not bad["converged"] and bad["max_rhat"] > 1.01
    assert bad["worst_rhat_var"] == "theta"


def test_tables_carry_convergence_flag():
    m = point_mass_model(truth())
    m.idata.attrs.update({
        "convergence_converged": 0, "convergence_max_rhat": 1.2,
        "convergence_min_ess_bulk": 10.0, "convergence_min_ess_tail": 10.0,
        "convergence_divergences": 3,
    })
    rep = m.decision_report()
    assert "converged" in rep.columns and not rep["converged"].any()
    assert m.standardized_bias_table().attrs["converged"] is False
    assert m.variance_budget_table().attrs["converged"] is False


def test_fit_without_builder_warns(monkeypatch):
    import apeBayes.facade as fac

    class _Stop(Exception):
        pass

    def _boom(*a, **k):
        raise _Stop

    monkeypatch.setattr(fac, "sample_model", _boom)
    m = point_mass_model(truth())
    with pytest.warns(FutureWarning, match="FlatConfigModel"), pytest.raises(_Stop):
        m.fit()


# ── A7: separability_check reports λ-relevant SDs and correlation CIs ───

def test_separability_check_sd_ratio_sees_lambda_correlation_does_not():
    from apeBayes import separability_check
    from review_helpers import crossed_config, crossed_df

    t = truth()
    t["gamma_sr"] = np.zeros_like(t["gamma_sr"])
    corr = separability_check(crossed_df(t), crossed_config())
    # pure rank-1 patterns: every correlation is 1 whatever λ is
    assert corr.attrs["min_pairwise_corr"] == pytest.approx(1.0)
    np.testing.assert_allclose(corr.attrs["sd_ratio_to_ref"].to_numpy(),
                               t["lambda_case"], rtol=1e-10)
    ci = corr.attrs["corr_ci"]
    assert len(ci) == 6 and {"lo", "hi"} <= set(ci.columns)


def test_separability_corr_interval_is_wide_with_six_runs():
    from scipy.stats import norm

    r, n = 0.47, 6
    half = norm.ppf(0.975) / np.sqrt(n - 3)
    lo, hi = np.tanh(np.arctanh(r) - half), np.tanh(np.arctanh(r) + half)
    assert lo < -0.5 and hi > 0.9  # the critique's (-0.55, 0.93)


# ── A8 smaller fixes ────────────────────────────────────────────────────

def test_plot_variance_ratio_homoskedastic_divides_by_sigma_src():
    import matplotlib
    matplotlib.use("Agg")
    from apeBayes.plots.variance import plot_variance_ratio

    fig, ax = plot_variance_ratio(np.full(200, 2.0), np.full(200, 0.5), ["1A", "1B"])
    xs = ax.collections[0].get_offsets()[:, 0]
    np.testing.assert_allclose(xs, 0.25)


def test_plot_variance_ratio_keeps_label_column_pairing():
    import matplotlib
    matplotlib.use("Agg")
    from apeBayes.plots.variance import plot_variance_ratio

    S = 100
    sig = np.column_stack([np.full(S, 0.4), np.full(S, 0.1)])  # labels 2A, 1A
    fig, ax = plot_variance_ratio(np.ones(S), sig, ["2A", "1A"])
    ticks = [t.get_text() for t in ax.get_yticklabels()]
    xs = dict(zip(ticks, ax.collections[0].get_offsets()[:, 0]))
    assert xs["1A"] == pytest.approx(0.1) and xs["2A"] == pytest.approx(0.4)


def test_cluster_threshold_does_not_reuse_alpha():
    from apeBayes.analysis.equivalence import epistemic_clusters

    rng = np.random.default_rng(0)
    S = 2000
    mu = np.column_stack([rng.normal(0, 0.01, S), rng.normal(0.5, 0.01, S)])
    sd = np.ones(S)
    # P_eq(|Δμ| < α) is 0 at α=0.4 and 1 at α=0.7; the clustering should
    # follow P_eq, not move its cut with α.
    a = epistemic_clusters(mu, sd, alpha=0.4, labels=["X", "Y"])
    b = epistemic_clusters(mu, sd, alpha=0.7, labels=["X", "Y"])
    assert a["cluster"].nunique() == 2 and b["cluster"].nunique() == 1


def test_empty_config_warns_and_empty_reference_raises():
    from apeBayes.data import encode_dataset
    from review_helpers import crossed_config, crossed_df

    df = crossed_df(truth())
    with pytest.warns(UserWarning, match="no observations"):
        encode_dataset(df[df.TierCase != "2C"], crossed_config())
    with pytest.raises(ValueError, match="Reference config"):
        encode_dataset(df[df.TierCase != "4D"], crossed_config())


# ── A5: leave-one-rupture-out marginal is the generative one ────────────

@pytest.mark.parametrize("case_terms", [False, True])
def test_logo_run_marginal_matches_monte_carlo(case_terms):
    """The exact held-out-rupture density equals an MVN whose covariance is
    estimated from freshly simulated rupture effects (f = 1 + 1/R)."""
    from scipy.stats import multivariate_normal

    from apeBayes.model.cv import _fold_lpd

    t = truth(case_terms=case_terms)
    t["sigma_eps"] = np.full(16, 0.3)
    m = point_mass_model(t, n_draws=2)
    rng = np.random.default_rng(5)
    test = m._df[m._df.rupture == "rup_3"].copy()
    test["edp"] = test["edp"] + rng.normal(0, 0.4, len(test))
    lpd = _fold_lpd(m.posterior, test, m.cfg, "run", None, 0)

    d = m.data
    k = np.array([d.config_label_to_idx(c) for c in test.TierCase])
    c = d.case_mapping()[1][k]
    s = np.array([list(d.station_labels).index(x) for x in test.sta])
    f = 1 + 1 / 6
    M = 200_000
    mean = t["mu0"] + t["mu_config"][k] + t["delta_st"][s]
    b = rng.normal(0, np.sqrt(f) * t["sigma_run"], M)
    g = rng.normal(0, np.sqrt(f) * t["sigma_inter"], (M, 3))
    rec = t["lambda_case"][c][None] * b[:, None] + g[:, s]
    if case_terms:
        u = rng.normal(0, 1, (M, 4)) * np.sqrt(f) * t["sigma_u_case"]
        v = rng.normal(0, 1, (M, 4, 3)) * np.sqrt(f) * t["sigma_v_case"][:, None]
        rec = rec + u[:, c] + v[:, c, s]
    # independent construction: covariance of the simulated record terms
    C = np.cov(rec, rowvar=False) + np.diag(np.full(len(k), 0.3 ** 2))
    ref = multivariate_normal(mean, C).logpdf(test.edp.to_numpy())
    assert lpd == pytest.approx(ref, abs=0.1)
