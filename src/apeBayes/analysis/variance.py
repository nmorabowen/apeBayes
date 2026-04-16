"""
Variance budget and component analysis.

Decomposes total posterior variance into:
  1. Spread across configuration means  Var(μ_config)
  2. Spread across station means        Var(δ_s)
  3. Run-level dispersion               σ_run²
  4. Residual dispersion                E[σ_eps²] (with Student-t adjustment)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import EPS, student_t_var_factor


def variance_budget(
    mu_config: np.ndarray,
    delta_st: np.ndarray,
    sigma_run: np.ndarray,
    sigma_eps: np.ndarray,
    config_idx: np.ndarray,
    ci: tuple[float, float] = (0.05, 0.95),
    *,
    nu: np.ndarray | None = None,
    gamma_sr: np.ndarray | None = None,
) -> pd.DataFrame:
    """Posterior variance budget (four or five components).

    Parameters
    ----------
    mu_config : (S, K) — configuration effects per draw.
    delta_st : (S, J) — station effects per draw.
    sigma_run : (S,) — run-level SD per draw.
    sigma_eps : (S,) or (S, K) — residual SD (homo or hetero).
    config_idx : (N,) — per-observation config index.
    nu : (S,) or None — Student-t DoF for variance adjustment.
    ci : tuple — credible interval quantiles.
    gamma_sr : (S, n_inter) or None
        Station×rupture interaction draws (v8+).  When provided, a fifth
        row "Interaction Var(γ_sr)" is added to the budget between the
        run and residual rows.  When None, the legacy 4-row budget is
        returned (backward compatible with v1–v7).

    Returns
    -------
    pd.DataFrame with columns: component, var_med, var_lo, var_hi, pct_med.
    """
    # 1. Spread across config means
    var_config = np.var(mu_config, axis=1, ddof=0)    # (S,)

    # 2. Spread across station means
    var_station = np.var(delta_st, axis=1, ddof=0)    # (S,)

    # 3. Run dispersion
    var_run = sigma_run ** 2                           # (S,)

    # 4. Residual dispersion (with Student-t adjustment)
    t_factor = np.ones_like(sigma_run)
    if nu is not None:
        t_factor = student_t_var_factor(nu)

    if sigma_eps.ndim == 2:
        # hetero: weighted average over observation mix
        sig_obs = sigma_eps[:, config_idx]             # (S, N)
        var_eps = np.mean(sig_obs ** 2 * t_factor[:, None], axis=1)
    else:
        var_eps = sigma_eps ** 2 * t_factor

    ci_lo, ci_hi = ci

    def _row(name: str, v: np.ndarray) -> dict:
        return {
            "component": name,
            "var_med": float(np.nanmedian(v)),
            "var_lo": float(np.nanquantile(v, ci_lo)),
            "var_hi": float(np.nanquantile(v, ci_hi)),
        }

    rows = [
        _row("Config spread  Var(μ_config)", var_config),
        _row("Station spread Var(δ_s)", var_station),
        _row("Run dispersion σ²_run", var_run),
    ]

    if gamma_sr is not None:
        # Empirical spread across (station, rupture) cells per draw.
        # This is comparable to Var(μ_config) / Var(δ_s): the variance
        # of the effect across its levels, not the hierarchical scale.
        var_inter = np.var(gamma_sr, axis=1, ddof=0)   # (S,)
        rows.append(_row("Interaction   Var(γ_sr)", var_inter))

    rows.append(_row("Residual disp. σ²_ε", var_eps))

    tab = pd.DataFrame(rows)
    total = tab["var_med"].sum()
    tab["pct_med"] = 100.0 * tab["var_med"] / max(total, EPS)

    return tab


def variance_components(
    sigma_run: np.ndarray,
    sigma_eps: np.ndarray,
    labels: list[str],
    ci: tuple[float, float] = (0.05, 0.95),
    *,
    nu: np.ndarray | None = None,
) -> pd.DataFrame:
    """Posterior summaries of scale parameters.

    Returns one row per component: sigma_run, sigma_eps (or sigma_eps_config[k]),
    and optionally nu.
    """
    ci_lo, ci_hi = ci

    def _row(name: str, v: np.ndarray) -> dict:
        return {
            "component": name,
            "med": float(np.median(v)),
            "lo": float(np.quantile(v, ci_lo)),
            "hi": float(np.quantile(v, ci_hi)),
        }

    rows = [_row("sigma_run", sigma_run)]

    if sigma_eps.ndim == 1:
        rows.append(_row("sigma_eps", sigma_eps))
    else:
        for k, lbl in enumerate(labels):
            rows.append(_row(f"sigma_eps[{lbl}]", sigma_eps[:, k]))

    if nu is not None:
        rows.append(_row("nu", nu))

    return pd.DataFrame(rows)
