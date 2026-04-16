"""
Fitted values and R² computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import EPS


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
    S = mu0.shape[0]
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


def mu_hat_table(
    mu0: np.ndarray,
    mu_config: np.ndarray,
    labels: list[str],
    ci: tuple[float, float] = (0.05, 0.95),
    *,
    subset_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    """Configuration intercepts (mu0 + mu_config) with posterior summaries."""
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
