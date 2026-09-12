"""Posterior predictive checks and model validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..config import FactorSpec, ModelConfig


def posterior_predictive_check(
    y_obs: np.ndarray,
    y_rep: np.ndarray,
    *,
    config_idx: np.ndarray | None = None,
    config_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Compare observed vs replicated statistics for posterior predictive check.

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


def _resolve_case_factor(cfg: ModelConfig, case_factor: str | None) -> FactorSpec:
    """Return the ``FactorSpec`` to treat as "Case" for ``separability_check``.

    Matches ``case_factor`` against each factor's ``name`` first, then its
    ``column``, so callers can pass either the human-readable name (e.g.
    ``"Nonlinearity"``) or the dataframe column (e.g. ``"Case"``). Defaults
    to the last entry in ``cfg.factors`` when ``case_factor`` is ``None``.
    """
    if case_factor is None:
        return cfg.factors[-1]
    for spec in cfg.factors:
        if spec.name == case_factor or spec.column == case_factor:
            return spec
    raise ValueError(
        f"case_factor {case_factor!r} not found among cfg.factors "
        f"(names={cfg.factor_names}, columns={cfg.factor_columns})."
    )


def separability_check(
    df: pd.DataFrame,
    cfg: ModelConfig,
    *,
    case_factor: str | None = None,
) -> pd.DataFrame:
    """Pre-fit separability diagnostic for the random-slopes models.

    ``RandomSlopesModel`` and ``RandomSlopesInteractionModel`` (v4-v9) share
    a single per-run rupture effect ``b_run`` and let each Case level rescale
    it through ``lambda_case``. That parameterisation only identifies
    ``lambda_case`` if the *pattern* of run-to-run variation is genuinely
    shared across Case levels — i.e. a rupture that pushes Case A's EDP up
    should also push Case B's, C's, and D's EDP up (by some Case-specific
    factor). This function checks that assumption directly on the raw data,
    before any model is fit:

    1. Subtract the per-(config, station) mean from ``cfg.edp_col`` so that
       station- and configuration-level offsets do not contaminate the
       run pattern.
    2. Average that residual within each (Case level, run) cell, collapsing
       out station and (within a Case) configuration-level noise.
    3. Correlate the resulting run patterns across Case levels.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format observations with columns ``cfg.config_col``,
        ``cfg.station_col``, ``cfg.run_col``, ``cfg.edp_col``, and the
        factor columns in ``cfg.factor_columns``.
    cfg : ModelConfig
        Model configuration; supplies the column names above and the
        factor list used to locate the "Case" factor.
    case_factor : str | None, optional
        Name or column of the factor to treat as "Case". Defaults to the
        last entry in ``cfg.factors`` (matching how ``RandomSlopesModel``
        picks its Case factor).

    Returns
    -------
    pd.DataFrame
        Square Case-by-Case Pearson correlation matrix of the run patterns,
        indexed and columned by the Case levels. Two extra diagnostics are
        attached as DataFrame attributes:

        - ``result.attrs["min_pairwise_corr"]`` — the minimum pairwise
          correlation over the upper triangle (excluding the diagonal).
          Below roughly **0.4**, the common-rupture-effect assumption
          behind ``RandomSlopesModel`` / ``RandomSlopesInteractionModel``
          is violated: the Case levels do not respond to the same ruptures
          in the same way, and ``lambda_case`` is not identified.
        - ``result.attrs["all_near_one"]`` — ``True`` when every pairwise
          correlation is at least **0.98**. This means the Case levels
          differ by an (approximately) constant offset only, so the
          per-configuration residual pattern has nothing case-specific
          left to carry — ``lambda_case`` would have no signal to fit
          beyond a shared scale.
    """
    case_spec = _resolve_case_factor(cfg, case_factor)
    case_col = case_spec.column

    required = [cfg.config_col, cfg.station_col, cfg.run_col, cfg.edp_col, case_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"separability_check: df is missing columns {missing}.")

    work = df[required].copy()
    work[case_col] = work[case_col].astype(str)

    # (a) subtract the (config, station) mean from the EDP.
    group_mean = work.groupby(
        [cfg.config_col, cfg.station_col],
    )[cfg.edp_col].transform("mean")
    residual = work[cfg.edp_col] - group_mean

    # (b) average the residual per (Case level, run).
    case_run_mean = (
        pd.DataFrame({
            case_col: work[case_col],
            cfg.run_col: work[cfg.run_col],
            "_residual": residual,
        })
        .groupby([case_col, cfg.run_col])["_residual"]
        .mean()
        .reset_index()
    )

    case_levels = case_spec.levels
    if case_levels is None:
        case_levels = sorted(case_run_mean[case_col].unique())
    else:
        case_levels = [str(c) for c in case_levels]
    case_levels = [c for c in case_levels if c in set(case_run_mean[case_col])]

    wide = case_run_mean.pivot(index=cfg.run_col, columns=case_col, values="_residual")
    wide = wide.reindex(columns=case_levels)

    # (c) Case-by-Case Pearson correlation of the run patterns.
    corr: pd.DataFrame = wide.corr()
    corr.index.name = None
    corr.columns.name = None

    n = len(case_levels)
    if n > 1:
        upper = corr.to_numpy()[np.triu_indices(n, k=1)]
        min_pairwise_corr = float(np.nanmin(upper)) if np.any(~np.isnan(upper)) else float("nan")
        all_near_one = bool(np.all(upper >= 0.98))
    else:
        min_pairwise_corr = float("nan")
        all_near_one = True

    corr.attrs["min_pairwise_corr"] = min_pairwise_corr
    corr.attrs["all_near_one"] = all_near_one
    return corr
