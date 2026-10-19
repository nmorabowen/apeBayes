"""Fitted values and R² computation.

The predictor-based functions (:func:`fitted_from_predictor`,
:func:`r2_from_predictor`) are the general ones: they take the ``(S, N)``
linear predictor built by ``PosteriorAccessor.linear_predictor`` and are
correct for every variant (λ_case, γ_sr, ξ_case and the v10 Case×record
terms included). :func:`posterior_mean_fitted` / :func:`posterior_r2` build
the predictor as ``mu0 + mu_config + delta + b_run`` and are therefore
only correct for the v1–v3 builders, which have no loadings or
interaction; they are kept for backward compatibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def posterior_mean_fitted(
    y: np.ndarray,
    mu0: np.ndarray,
    mu_config: np.ndarray,
    delta_st: np.ndarray,
    b_run: np.ndarray,
    config_idx: np.ndarray,
    station_idx: np.ndarray,
    run_idx: np.ndarray,
) -> pd.DataFrame:
    """Compute posterior-mean fitted values and residuals.

    Returns a DataFrame with:
      - y            : observed log-EDP
      - yhat_no_run  : E[mu0 + mu_config + delta_st]
      - yhat_with_run: E[mu0 + mu_config + delta_st + b_run]
      - resid        : y − yhat_with_run
    """
    mu0_m = float(mu0.mean())
    mu_cfg_m = mu_config.mean(axis=0)
    delta_m = delta_st.mean(axis=0)
    b_run_m = b_run.mean(axis=0)

    yhat_no_run = mu0_m + mu_cfg_m[config_idx] + delta_m[station_idx]
    yhat_with_run = yhat_no_run + b_run_m[run_idx]
    resid = y - yhat_with_run

    return pd.DataFrame({
        "y": y,
        "yhat_no_run": yhat_no_run,
        "yhat_with_run": yhat_with_run,
        "resid": resid,
    })


def posterior_r2(
    y: np.ndarray,
    mu0: np.ndarray,
    mu_config: np.ndarray,
    delta_st: np.ndarray,
    b_run: np.ndarray,
    config_idx: np.ndarray,
    station_idx: np.ndarray,
    run_idx: np.ndarray,
    ci: tuple[float, float] = (0.05, 0.95),
) -> pd.DataFrame:
    """Posterior distribution of R² (marginal and conditional).

    Vectorized over posterior draws — no Python loops.

    R²_marginal    : variance explained by fixed effects (config + station)
    R²_conditional : variance explained by fixed + random effects
    """
    mu0.shape[0]
    var_y = np.var(y, ddof=1)

    # Broadcast: (S, N)
    mu_no_run = (
        mu0[:, None]
        + mu_config[:, config_idx]
        + delta_st[:, station_idx]
    )  # (S, N)

    mu_with_run = mu_no_run + b_run[:, run_idx]  # (S, N)

    resid_marg = y[None, :] - mu_no_run           # (S, N)
    resid_cond = y[None, :] - mu_with_run          # (S, N)

    r2_marg = 1.0 - np.var(resid_marg, axis=1, ddof=1) / var_y   # (S,)
    r2_cond = 1.0 - np.var(resid_cond, axis=1, ddof=1) / var_y   # (S,)

    ci_lo, ci_hi = ci
    return pd.DataFrame({
        "R2_marginal_med": [float(np.median(r2_marg))],
        "R2_marginal_lo": [float(np.quantile(r2_marg, ci_lo))],
        "R2_marginal_hi": [float(np.quantile(r2_marg, ci_hi))],
        "R2_cond_med": [float(np.median(r2_cond))],
        "R2_cond_lo": [float(np.quantile(r2_cond, ci_lo))],
        "R2_cond_hi": [float(np.quantile(r2_cond, ci_hi))],
    })


def fitted_from_predictor(
    y: np.ndarray,
    mu_fixed: np.ndarray,
    mu_full: np.ndarray,
) -> pd.DataFrame:
    """Posterior-mean fitted values and residuals from ``(S, N)`` predictors.

    Parameters
    ----------
    y : (N,) observed log-EDP.
    mu_fixed : (S, N) fixed part μ₀ + μ_config + δ_s.
    mu_full : (S, N) full linear predictor (fixed + every record term).

    Returns
    -------
    pd.DataFrame
        Columns ``y``, ``yhat_no_run`` (E[fixed part]), ``yhat_with_run``
        (E[full predictor], i.e. including λ·b, γ, ξ, u, v as the model
        has them) and ``resid = y − yhat_with_run``.
    """
    yhat_no_run = mu_fixed.mean(axis=0)
    yhat_with_run = mu_full.mean(axis=0)
    return pd.DataFrame({
        "y": y,
        "yhat_no_run": yhat_no_run,
        "yhat_with_run": yhat_with_run,
        "resid": y - yhat_with_run,
    })


def r2_from_predictor(
    y: np.ndarray,
    mu_fixed: np.ndarray,
    mu_full: np.ndarray,
    ci: tuple[float, float] = (0.05, 0.95),
) -> pd.DataFrame:
    """Posterior R² (marginal and conditional) from ``(S, N)`` predictors.

    R²_marginal uses the fixed part only; R²_conditional uses the full
    predictor. Both are ``1 − Var(y − μ^(s)) / Var(y)`` per draw.
    """
    var_y = np.var(y, ddof=1)
    r2_marg = 1.0 - np.var(y[None, :] - mu_fixed, axis=1, ddof=1) / var_y
    r2_cond = 1.0 - np.var(y[None, :] - mu_full, axis=1, ddof=1) / var_y
    ci_lo, ci_hi = ci
    return pd.DataFrame({
        "R2_marginal_med": [float(np.median(r2_marg))],
        "R2_marginal_lo": [float(np.quantile(r2_marg, ci_lo))],
        "R2_marginal_hi": [float(np.quantile(r2_marg, ci_hi))],
        "R2_cond_med": [float(np.median(r2_cond))],
        "R2_cond_lo": [float(np.quantile(r2_cond, ci_lo))],
        "R2_cond_hi": [float(np.quantile(r2_cond, ci_hi))],
    })


def mu_hat_table(
    mu0: np.ndarray,
    mu_config: np.ndarray,
    labels: list[str],
    ci: tuple[float, float] = (0.05, 0.95),
    *,
    subset_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute configuration intercepts with posterior summaries."""
    if subset_idx is not None:
        mu_sub = mu0[:, None] + mu_config[:, subset_idx]
    else:
        mu_sub = mu0[:, None] + mu_config

    ci_lo, ci_hi = ci
    return pd.DataFrame({
        "Config": labels,
        "mu_med": np.median(mu_sub, axis=0),
        "mu_lo": np.quantile(mu_sub, ci_lo, axis=0),
        "mu_hi": np.quantile(mu_sub, ci_hi, axis=0),
    })


def delta_hat_table(
    delta_st: np.ndarray,
    labels: list[str],
    ci: tuple[float, float] = (0.05, 0.95),
) -> pd.DataFrame:
    """Station effects with posterior summaries."""
    ci_lo, ci_hi = ci
    return pd.DataFrame({
        "Station": labels,
        "delta_med": np.median(delta_st, axis=0),
        "delta_lo": np.quantile(delta_st, ci_lo, axis=0),
        "delta_hi": np.quantile(delta_st, ci_hi, axis=0),
    })
