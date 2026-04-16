"""
Standardized bias analysis.

All functions are stateless: they take posterior draws (numpy arrays)
and metadata, and return pandas DataFrames.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ..utils import student_t_sd_factor


def standardized_bias(
    mu_config: np.ndarray,
    sigma_run: np.ndarray,
    ref_idx: int,
    labels: list[str],
    ci: tuple[float, float] = (0.05, 0.95),
    *,
    subset_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    """Standardized epistemic bias relative to a reference configuration.

    Computes per posterior draw:
        Δμ_k  = mu_config[:, k] − mu_config[:, ref]
        β_k   = Δμ_k / σ_run
        ratio = exp(Δμ_k)                              (multiplicative factor)

    Parameters
    ----------
    mu_config : (S, K) array
        Posterior draws of configuration effects.
    sigma_run : (S,) array
        Posterior draws of run-level SD.
    ref_idx : int
        Index of the reference configuration.
    labels : list[str]
        Configuration labels (length K or len(subset_idx)).
    ci : tuple
        Credible interval quantiles.
    subset_idx : (M,) int array, optional
        If given, only compute for these configuration indices.

    Returns
    -------
    pd.DataFrame
        Columns: Config, std_bias_{med,lo,hi}, dmu_{med,lo,hi}, mult_{med,lo,hi}.
    """
    mu_sub = mu_config[:, subset_idx] if subset_idx is not None else mu_config

    dmu = mu_sub - mu_config[:, [ref_idx]]         # (S, M)
    beta = dmu / sigma_run[:, None]                 # (S, M)
    mult = np.exp(dmu)                              # (S, M)

    ci_lo, ci_hi = ci
    return pd.DataFrame({
        "Config": labels,
        "std_bias_med": np.median(beta, axis=0),
        "std_bias_lo": np.quantile(beta, ci_lo, axis=0),
        "std_bias_hi": np.quantile(beta, ci_hi, axis=0),
        "dmu_med": np.median(dmu, axis=0),
        "dmu_lo": np.quantile(dmu, ci_lo, axis=0),
        "dmu_hi": np.quantile(dmu, ci_hi, axis=0),
        "mult_med": np.median(mult, axis=0),
        "mult_lo": np.quantile(mult, ci_lo, axis=0),
        "mult_hi": np.quantile(mult, ci_hi, axis=0),
    })


def standardized_bias_total(
    mu_config: np.ndarray,
    sigma_run: np.ndarray,
    sigma_eps: np.ndarray,
    ref_idx: int,
    labels: list[str],
    ci: tuple[float, float] = (0.05, 0.95),
    *,
    nu: np.ndarray | None = None,
    subset_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    """Bias standardized by *total* predictive SD (run + residual).

    σ_tot_k = √(σ_run² + σ_eps_eff_k²)
    β_tot_k = Δμ_k / σ_tot_k

    Parameters
    ----------
    sigma_eps : (S,) or (S, K) array
        Residual SD — scalar (homo) or per-config (hetero).
    nu : (S,) array, optional
        Student-t degrees of freedom for variance adjustment.
    """
    if subset_idx is not None:
        mu_sub = mu_config[:, subset_idx]
        if sigma_eps.ndim == 2:
            sig_eps_sub = sigma_eps[:, subset_idx]
        else:
            sig_eps_sub = sigma_eps[:, None] * np.ones((mu_sub.shape[0], mu_sub.shape[1]))
    else:
        mu_sub = mu_config
        if sigma_eps.ndim == 1:
            sig_eps_sub = sigma_eps[:, None] * np.ones((mu_sub.shape[0], mu_sub.shape[1]))
        else:
            sig_eps_sub = sigma_eps

    # Student-t SD adjustment
    if nu is not None:
        sig_eps_sub = sig_eps_sub * student_t_sd_factor(nu)[:, None]

    dmu = mu_sub - mu_config[:, [ref_idx]]
    sigma_tot = np.sqrt(sigma_run[:, None] ** 2 + sig_eps_sub ** 2)
    beta_tot = dmu / sigma_tot

    ci_lo, ci_hi = ci
    return pd.DataFrame({
        "Config": labels,
        "beta_tot_med": np.median(beta_tot, axis=0),
        "beta_tot_lo": np.quantile(beta_tot, ci_lo, axis=0),
        "beta_tot_hi": np.quantile(beta_tot, ci_hi, axis=0),
        "sigma_tot_med": np.median(sigma_tot, axis=0),
        "sigma_tot_lo": np.quantile(sigma_tot, ci_lo, axis=0),
        "sigma_tot_hi": np.quantile(sigma_tot, ci_hi, axis=0),
        "dmu_med": np.median(dmu, axis=0),
        "dmu_lo": np.quantile(dmu, ci_lo, axis=0),
        "dmu_hi": np.quantile(dmu, ci_hi, axis=0),
    })


def bias_probability(
    mu_config: np.ndarray,
    sigma_run: np.ndarray,
    ref_idx: int,
    labels: list[str],
    *,
    mode: Literal["exceed_band", "within_equiv", "positive"] = "exceed_band",
    band: float = 1.0,
    alpha_equiv: float = 0.5,
    subset_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    """Probability summaries of β = Δμ / σ_run.

    Modes
    -----
    - "exceed_band"  : P(|β| > band)
    - "within_equiv" : P(|β| < alpha_equiv)
    - "positive"     : P(β > 0)
    """
    mu_sub = mu_config[:, subset_idx] if subset_idx is not None else mu_config

    dmu = mu_sub - mu_config[:, [ref_idx]]
    beta = dmu / sigma_run[:, None]

    if mode == "exceed_band":
        p = np.mean(np.abs(beta) > band, axis=0)
        label = f"P(|β|>{band:g})"
    elif mode == "within_equiv":
        p = np.mean(np.abs(beta) < alpha_equiv, axis=0)
        label = f"P(|β|<{alpha_equiv:g})"
    elif mode == "positive":
        p = np.mean(beta > 0, axis=0)
        label = "P(β>0)"
    else:
        raise ValueError(f"Unknown mode={mode!r}")

    return pd.DataFrame({
        "Config": labels,
        "beta_med": np.median(beta, axis=0),
        "prob": p,
        "prob_label": label,
    })
