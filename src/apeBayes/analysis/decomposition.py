"""
N-way factorial decomposition of epistemic configuration effects.

For K factors the decomposition is:

    μ_{i₁i₂…iₖ} = μ̄ + Σ_k τ^(k)_{iₖ}
                      + Σ_{k<l} γ^(kl)_{iₖ iₗ}
                      + …
                      + residual interaction

Currently implements the 2-factor case in full (Tier × Case).
The N-way generalization slot is reserved.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from ..utils import EPS

# ── 2-factor decomposition ──────────────────────────────────────────────────

def axiswise_decomposition_draws(
    mu_config: np.ndarray,
    factor0_levels: list[str],
    factor1_levels: list[str],
    grid: dict[tuple[str, str], int],
    sigma_run: np.ndarray,
) -> dict[str, Any]:
    """Per-draw axis-wise decomposition for a 2-factor design.

    Decomposes μ_ij = μ̄ + τ_i + κ_j + γ_ij per posterior draw.

    Parameters
    ----------
    mu_config : (S, K)
        Configuration effects.
    factor0_levels : list[str]
        Levels of the first factor (rows = "Tier" / SSI).
    factor1_levels : list[str]
        Levels of the second factor (columns = "Case" / Nonlinearity).
    grid : dict[(str, str), int]
        Maps (level0, level1) → flat config index.
    sigma_run : (S,)
        Run-level SD per draw.

    Returns
    -------
    dict with keys:
        mu_ij   : (S, I, J) reshaped config means
        mu_bar  : (S,) grand mean per draw
        tau     : (S, I) row (factor-0) main effects
        kappa   : (S, J) column (factor-1) main effects
        gamma   : (S, I, J) interaction
        sigma_run, factor0_levels, factor1_levels, grid
    """
    I = len(factor0_levels)
    J = len(factor1_levels)

    # Check completeness
    missing = [
        (a, b)
        for a in factor0_levels
        for b in factor1_levels
        if (a, b) not in grid
    ]
    if missing:
        miss_str = ", ".join(f"{a}{b}" for a, b in missing[:8])
        raise ValueError(
            "Axis-wise decomposition requires a complete factor grid. "
            f"Missing: {miss_str}"
        )

    S = mu_config.shape[0]
    mu_ij = np.empty((S, I, J), dtype=float)
    for i, a in enumerate(factor0_levels):
        for j, b in enumerate(factor1_levels):
            col = grid[(a, b)]
            mu_ij[:, i, j] = mu_config[:, col]

    mu_bar = mu_ij.mean(axis=(1, 2))                           # (S,)
    tau = mu_ij.mean(axis=2) - mu_bar[:, None]                 # (S, I)
    kappa = mu_ij.mean(axis=1) - mu_bar[:, None]               # (S, J)
    gamma = (
        mu_ij
        - mu_bar[:, None, None]
        - tau[:, :, None]
        - kappa[:, None, :]
    )                                                           # (S, I, J)

    return {
        "mu_ij": mu_ij,
        "mu_bar": mu_bar,
        "tau": tau,
        "kappa": kappa,
        "gamma": gamma,
        "sigma_run": sigma_run,
        "factor0_levels": factor0_levels,
        "factor1_levels": factor1_levels,
        "grid": grid,
    }


def axiswise_table(
    decomp: dict[str, Any],
    ci: tuple[float, float] = (0.05, 0.95),
    *,
    ratio_mode: Literal["var_over_sigma", "var_over_sigma2"] = "var_over_sigma",
    factor_names: tuple[str, str] = ("Factor0", "Factor1"),
) -> pd.DataFrame:
    """Summarize axis-wise decomposition into a table.

    Returns variance, ratio-to-σ_run, and percentage for each component
    (factor-0 main, factor-1 main, interaction).
    """
    tau = decomp["tau"]           # (S, I)
    kappa = decomp["kappa"]       # (S, J)
    gamma = decomp["gamma"]       # (S, I, J)
    sigma_run = decomp["sigma_run"]
    S = tau.shape[0]

    var_tau = np.var(tau, axis=1, ddof=0)
    var_kappa = np.var(kappa, axis=1, ddof=0)
    var_gamma = np.var(gamma.reshape(S, -1), axis=1, ddof=0)

    var_total = np.maximum(var_tau + var_kappa + var_gamma, EPS)
    pct_tau = 100.0 * var_tau / var_total
    pct_kappa = 100.0 * var_kappa / var_total
    pct_gamma = 100.0 * var_gamma / var_total

    if ratio_mode == "var_over_sigma":
        denom = np.maximum(sigma_run, EPS)
    else:
        denom = np.maximum(sigma_run ** 2, EPS)

    ci_lo, ci_hi = ci

    def _row(name: str, var_v: np.ndarray, ratio_v: np.ndarray, pct_v: np.ndarray) -> dict:
        return {
            "component": name,
            "var_med": float(np.median(var_v)),
            "var_lo": float(np.quantile(var_v, ci_lo)),
            "var_hi": float(np.quantile(var_v, ci_hi)),
            "ratio_med": float(np.median(ratio_v)),
            "ratio_lo": float(np.quantile(ratio_v, ci_lo)),
            "ratio_hi": float(np.quantile(ratio_v, ci_hi)),
            "pct_med": float(np.median(pct_v)),
            "pct_lo": float(np.quantile(pct_v, ci_lo)),
            "pct_hi": float(np.quantile(pct_v, ci_hi)),
        }

    rows = [
        _row(f"{factor_names[0]} (main)", var_tau, var_tau / denom, pct_tau),
        _row(f"{factor_names[1]} (main)", var_kappa, var_kappa / denom, pct_kappa),
        _row(
            f"{factor_names[0]}×{factor_names[1]} interaction",
            var_gamma, var_gamma / denom, pct_gamma,
        ),
    ]

    return pd.DataFrame(rows)


def level_ranking_tables(
    decomp: dict[str, Any],
    ci: tuple[float, float] = (0.05, 0.95),
    *,
    factor_names: tuple[str, str] = ("Factor0", "Factor1"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-level energy shares and spreads.

    Returns
    -------
    factor0_table : pd.DataFrame
        Per-level tau, energy share for factor 0.
    factor1_table : pd.DataFrame
        Per-level kappa, energy share for factor 1.
    spread_table : pd.DataFrame
        Overall spread (max − min) per axis, normalized by σ_run.
    """
    tau = decomp["tau"]
    kappa = decomp["kappa"]
    sigma_run = decomp["sigma_run"]
    f0_levels = decomp["factor0_levels"]
    f1_levels = decomp["factor1_levels"]

    ci_lo, ci_hi = ci

    def _summ(v: np.ndarray) -> tuple[float, float, float]:
        return (
            float(np.median(v)),
            float(np.quantile(v, ci_lo)),
            float(np.quantile(v, ci_hi)),
        )

    # Energy shares
    tau_energy = tau ** 2
    kappa_energy = kappa ** 2
    w_tau = tau_energy / np.maximum(tau_energy.sum(axis=1, keepdims=True), EPS)
    w_kappa = kappa_energy / np.maximum(kappa_energy.sum(axis=1, keepdims=True), EPS)

    f0_rows = []
    for i, lbl in enumerate(f0_levels):
        t_med, t_lo, t_hi = _summ(tau[:, i])
        w_med, w_lo, w_hi = _summ(w_tau[:, i])
        f0_rows.append({
            factor_names[0]: lbl,
            "effect_med": t_med, "effect_lo": t_lo, "effect_hi": t_hi,
            "share_pct_med": 100.0 * w_med,
            "share_pct_lo": 100.0 * w_lo,
            "share_pct_hi": 100.0 * w_hi,
        })

    f1_rows = []
    for j, lbl in enumerate(f1_levels):
        k_med, k_lo, k_hi = _summ(kappa[:, j])
        w_med, w_lo, w_hi = _summ(w_kappa[:, j])
        f1_rows.append({
            factor_names[1]: lbl,
            "effect_med": k_med, "effect_lo": k_lo, "effect_hi": k_hi,
            "share_pct_med": 100.0 * w_med,
            "share_pct_lo": 100.0 * w_lo,
            "share_pct_hi": 100.0 * w_hi,
        })

    # Spreads
    tau_spread = tau.max(axis=1) - tau.min(axis=1)
    kappa_spread = kappa.max(axis=1) - kappa.min(axis=1)
    tau_over_sig = tau_spread / np.maximum(sigma_run, EPS)
    kappa_over_sig = kappa_spread / np.maximum(sigma_run, EPS)

    spread_rows = []
    for axis_name, spread, ratio in [
        (factor_names[0], tau_spread, tau_over_sig),
        (factor_names[1], kappa_spread, kappa_over_sig),
    ]:
        s_med, s_lo, s_hi = _summ(spread)
        r_med, r_lo, r_hi = _summ(ratio)
        spread_rows.append({
            "axis": axis_name,
            "spread_med": s_med, "spread_lo": s_lo, "spread_hi": s_hi,
            "spread_over_sigma_run_med": r_med,
            "spread_over_sigma_run_lo": r_lo,
            "spread_over_sigma_run_hi": r_hi,
        })

    return (
        pd.DataFrame(f0_rows).sort_values("share_pct_med", ascending=False).reset_index(drop=True),
        pd.DataFrame(f1_rows).sort_values("share_pct_med", ascending=False).reset_index(drop=True),
        pd.DataFrame(spread_rows),
    )
