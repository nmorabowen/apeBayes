r"""
Leave-one-group-out cross-validation (rupture or station).

PSIS-LOO leaves out single observations, which share their rupture and
station effects with the observations kept in, so it cannot tell whether a
model's *record* structure generalises. Leaving out a whole rupture (or a
whole station) and scoring the held-out block jointly can: this is the
check that rejected the rank-1 v8 record structure on the Paper 1 data
(2026-10-19 critique, by 56–75 nats).

For each fold the model is refitted without the group, with a Gaussian
likelihood, and the held-out block is scored by its exact joint marginal
density over the unseen group's random effects, averaged over draws:

.. math::
    \mathrm{lpd}_g = \log \frac{1}{S} \sum_s
        \mathcal{N}\!\left(y_g \mid m_g^{(s)}, C_g^{(s)}\right).

**New rupture** (``group="run"``): the mean is μ₀ + μ_k + δ_s, and

.. math::
    C_{ij} = f_R\left[\sigma_{run}^2 \lambda_{c_i}\lambda_{c_j}
        + \sigma_{inter}^2 \xi_{c_i}\xi_{c_j} [s_i = s_j]
        + \sigma_{u,c_i}^2 [c_i = c_j]
        + \sigma_{v,c_i}^2 [c_i = c_j, s_i = s_j]\right]
        + \sigma_{\varepsilon,k_i}^2 [i = j],

with :math:`f_R = 1 + 1/R` for the R training ruptures (a new level of a
zero-sum effect is offset by the training mean).

**New station** (``group="station"``): the mean adds the known rupture
terms (λ_c b_r, u_cr); the new station's δ is drawn from its prior
N(0, σ_station²) (stations are fixed effects, so this mostly tests that
prior; with 3 stations the result is weak evidence either way), plus the
station-specific interaction terms with :math:`f_S = 1 + 1/S`.

The Gaussian refit is needed for the exact marginal; it is a check of the
mean and record-covariance structure, not of the Student-t tails.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from ..data import encode_dataset
from ..diagnostics.convergence import convergence_report
from ..posterior import PosteriorAccessor
from .sampling import sample_model

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ..config import ModelConfig
    from .base import ModelBuilder


def _gaussian_builder(builder: ModelBuilder) -> ModelBuilder:
    """Copy of ``builder`` with a Gaussian likelihood override."""
    from .registry import builder_from_spec, builder_spec

    spec = builder_spec(builder)
    if "likelihood" in spec["options"] or hasattr(builder, "_likelihood_override"):
        spec["options"]["likelihood"] = "gaussian"
    return builder_from_spec(spec)


def _mvn_logpdf_chol(resid: np.ndarray, C: np.ndarray) -> float:
    L = np.linalg.cholesky(C)
    z = np.linalg.solve(L, resid)
    return float(-0.5 * z @ z - np.log(np.diag(L)).sum() - 0.5 * len(resid) * np.log(2 * np.pi))


def _fold_lpd(
    post: PosteriorAccessor,
    test: pd.DataFrame,
    cfg: ModelConfig,
    group: str,
    max_draws: int | None,
    seed: int,
) -> float:
    d = post.data
    cfg_lbl = {lbl: i for i, lbl in enumerate(d.config_labels)}
    k = np.array([cfg_lbl[str(v)] for v in _config_labels_of(test, cfg)])
    case_levels, config_to_case, _, _ = (
        d.case_mapping() if len(cfg.factors) == 2 else ([""], np.zeros(d.n_configs, int), None, 0)
    )
    c = config_to_case[k]
    S = post.n_samples
    idx = np.arange(S)
    if max_draws is not None and S > max_draws:
        idx = np.sort(np.random.default_rng(seed).choice(S, max_draws, replace=False))

    lam = post.lambda_case()
    xi = post.xi_case()
    s_inter = post.sigma_inter()
    su, sv = post.sigma_u_case(), post.sigma_v_case()
    sig_eps = post.sigma_eps()
    if sig_eps.ndim == 1:
        sig_eps = np.repeat(sig_eps[:, None], d.n_configs, axis=1)
    same_c = (c[:, None] == c[None, :]).astype(float)

    if group == "run":
        st_lbl = {lbl: i for i, lbl in enumerate(d.station_labels)}
        s = np.array([st_lbl[str(v)] for v in test[cfg.station_col]])
        same_s = (s[:, None] == s[None, :]).astype(float)
        f = 1.0 + 1.0 / d.n_runs
        mean = post.mu0()[:, None] + post.mu_config()[:, k] + post.delta_st()[:, s]
    else:  # station
        run_lbl = {lbl: i for i, lbl in enumerate(d.run_labels)}
        r = np.array([run_lbl[str(v)] for v in test[cfg.run_col]])
        same_r = (r[:, None] == r[None, :]).astype(float)
        f = 1.0 + 1.0 / d.n_stations
        b = post.b_run()[:, r]
        mean = post.mu0()[:, None] + post.mu_config()[:, k] + (
            b if lam is None else lam[:, c] * b
        )
        if post.has_case_record_terms:
            mean = mean + post.draws("u_case_run")[:, c, r]

    y = test[cfg.edp_col].to_numpy(dtype=float)
    sig_st2 = cfg.priors.sigma_station ** 2
    lp = np.empty(len(idx))
    for j, sidx in enumerate(idx):
        if group == "run":
            l = np.ones(len(k)) if lam is None else lam[sidx, c]
            C = post.sigma_run()[sidx] ** 2 * np.outer(l, l)
            if s_inter is not None:
                x = np.ones(len(k)) if xi is None else xi[sidx, c]
                C = C + s_inter[sidx] ** 2 * np.outer(x, x) * same_s
            if su is not None:
                C = C + np.outer(su[sidx, c], su[sidx, c]) * same_c
                C = C + np.outer(sv[sidx, c], sv[sidx, c]) * same_c * same_s
            C = f * C
        else:
            C = np.zeros((len(k), len(k)))
            if s_inter is not None:
                x = np.ones(len(k)) if xi is None else xi[sidx, c]
                C = C + s_inter[sidx] ** 2 * np.outer(x, x) * same_r
            if sv is not None:
                C = C + np.outer(sv[sidx, c], sv[sidx, c]) * same_c * same_r
            C = f * C + sig_st2
        C = C + np.diag(sig_eps[sidx, k] ** 2)
        lp[j] = _mvn_logpdf_chol(y - mean[sidx], C)
    return float(logsumexp(lp) - np.log(len(lp)))


def _config_labels_of(df: pd.DataFrame, cfg: ModelConfig) -> np.ndarray:
    if cfg.config_col in df.columns:
        return df[cfg.config_col].astype(str).to_numpy()
    cols = [df[f.column].astype(str) for f in cfg.factors]
    return np.array([cfg.config_sep.join(p) for p in zip(*cols, strict=True)])


def leave_one_group_out(
    df: pd.DataFrame,
    cfg: ModelConfig,
    builder: ModelBuilder,
    *,
    group: Literal["run", "station"] = "run",
    folds: Sequence[str] | None = None,
    max_draws: int | None = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Leave-one-rupture-out (or station-out) predictive density.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data (all groups).
    cfg : ModelConfig
        Configuration; ``cfg.sampling`` controls each refit. The refits use
        a Gaussian likelihood whatever ``cfg.likelihood`` says.
    builder : ModelBuilder
        Model to evaluate (its likelihood override is replaced by Gaussian).
    group : {"run", "station"}
        Which grouping to leave out.
    folds : sequence of str, optional
        Group labels to leave out (default: every level).
    max_draws : int or None
        Posterior draws used for the marginal density (random subset).
    seed : int
        Seed for the draw subset.

    Returns
    -------
    pd.DataFrame
        One row per fold: ``fold``, ``lpd``, ``n_obs``, ``divergences``,
        ``max_rhat``, ``converged``. Sum ``lpd`` for the elpd; compare
        models with :func:`compare_logo`.
    """
    col = cfg.run_col if group == "run" else cfg.station_col
    labels = sorted(df[col].astype(str).unique()) if folds is None else list(folds)
    gcfg = dataclasses.replace(cfg, likelihood="gaussian")
    gbuilder = _gaussian_builder(builder)
    rows = []
    for g in labels:
        mask = df[col].astype(str) == g
        train, test = df[~mask], df[mask]
        data = encode_dataset(train, gcfg)
        idata = sample_model(gbuilder.build(data, gcfg), gcfg.sampling,
                             prior_predictive=False, posterior_predictive=False)
        post = PosteriorAccessor(idata, data)
        rep = convergence_report(idata)
        rows.append({
            "fold": g,
            "lpd": _fold_lpd(post, test, gcfg, group, max_draws, seed),
            "n_obs": int(mask.sum()),
            "divergences": rep["divergences"],
            "max_rhat": rep["max_rhat"],
            "converged": rep["converged"],
        })
    out = pd.DataFrame(rows)
    out.attrs["group"] = group
    return out


def compare_logo(results: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Rank models by summed leave-one-group-out lpd.

    The SE of an elpd difference uses the paired fold differences,
    ``sqrt(F · var(diff))``, as in the 2026-10-19 critique.

    Returns
    -------
    pd.DataFrame
        Index = model name, sorted best first; columns ``elpd``,
        ``elpd_diff`` (vs best), ``se_diff``, ``n_folds``, ``all_converged``.
    """
    names = list(results)
    folds = results[names[0]]["fold"].tolist()
    lp = {n: results[n].set_index("fold").loc[folds, "lpd"].to_numpy() for n in names}
    elpd = {n: float(v.sum()) for n, v in lp.items()}
    best = max(elpd, key=elpd.get)
    rows = []
    for n in names:
        diff = lp[n] - lp[best]
        rows.append({
            "model": n,
            "elpd": elpd[n],
            "elpd_diff": float(diff.sum()),
            "se_diff": float(np.sqrt(len(diff) * np.var(diff, ddof=1))) if n != best else 0.0,
            "n_folds": len(folds),
            "all_converged": bool(results[n]["converged"].all()),
        })
    return pd.DataFrame(rows).set_index("model").sort_values("elpd", ascending=False)
