"""
Posterior predictive checks and model validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import flatten_posterior


def posterior_predictive_check(
    y_obs: np.ndarray,
    y_rep: np.ndarray,
    *,
    config_idx: np.ndarray | None = None,
    config_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Basic posterior predictive check: compare observed vs replicated statistics.

    Parameters
    ----------
    y_obs : (N,) observed data.
    y_rep : (S, N) posterior predictive draws.
    config_idx : (N,) optional config indices for group-wise checks.
    config_labels : optional labels for config groups.

    Returns
    -------
    pd.DataFrame with rows for global and per-group checks, columns:
        group, obs_mean, obs_sd, rep_mean_med, rep_sd_med, p_value_mean, p_value_sd
    """
    rows = []

    # Global
    obs_mean = float(np.mean(y_obs))
    obs_sd = float(np.std(y_obs, ddof=1))
    rep_means = np.mean(y_rep, axis=1)
    rep_sds = np.std(y_rep, axis=1, ddof=1)

    rows.append({
        "group": "Global",
        "obs_mean": obs_mean,
        "obs_sd": obs_sd,
        "rep_mean_med": float(np.median(rep_means)),
        "rep_sd_med": float(np.median(rep_sds)),
        "p_value_mean": float(np.mean(rep_means >= obs_mean)),
        "p_value_sd": float(np.mean(rep_sds >= obs_sd)),
    })

    # Per-group
    if config_idx is not None and config_labels is not None:
        for k, lbl in enumerate(config_labels):
            mask = config_idx == k
            if mask.sum() == 0:
                continue
            y_g = y_obs[mask]
            yr_g = y_rep[:, mask]
            g_obs_mean = float(np.mean(y_g))
            g_obs_sd = float(np.std(y_g, ddof=1)) if y_g.size > 1 else 0.0
            g_rep_means = np.mean(yr_g, axis=1)
            g_rep_sds = np.std(yr_g, axis=1, ddof=1) if y_g.size > 1 else np.zeros(yr_g.shape[0])

            rows.append({
                "group": lbl,
                "obs_mean": g_obs_mean,
                "obs_sd": g_obs_sd,
                "rep_mean_med": float(np.median(g_rep_means)),
                "rep_sd_med": float(np.median(g_rep_sds)),
                "p_value_mean": float(np.mean(g_rep_means >= g_obs_mean)),
                "p_value_sd": float(np.mean(g_rep_sds >= g_obs_sd)),
            })

    return pd.DataFrame(rows)
