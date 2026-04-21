"""
Standardized bias analysis.

All functions are stateless: they take posterior draws (numpy arrays)
and metadata, and return pandas DataFrames.

The denominator is provided explicitly by the caller (``sigma_denom``),
so the same routine computes β_src, β_GM (canonical), or β_pred depending
on which aleatory SD is passed in. See ``uncertanty_measures.md`` §4 for
the three denominators and §7 for per-variant σ_GM.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def standardized_bias(
    mu_config: np.ndarray,
    sigma_denom: np.ndarray,
    ref_idx: int,
    labels: list[str],
    ci: tuple[float, float] = (0.05, 0.95),
    *,
    subset_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    r"""Standardised epistemic bias ``β = Δμ / σ_denom`` per posterior draw.

    Computes, for each draw :math:`s`:

    .. math::
        \Delta\mu_k^{(s)} = \mu_k^{(s)} - \mu_{\text{ref}}^{(s)},
        \qquad
        \beta_k^{(s)} = \Delta\mu_k^{(s)} / \sigma_{\text{denom}}^{(s)}.

    The caller chooses ``sigma_denom`` to produce β_src, β_GM (canonical),
    or β_pred. See ``uncertanty_measures.md`` §4.

    Parameters
    ----------
    mu_config : (S, K) array
        Posterior draws of configuration effects.
    sigma_denom : (S,) or (S, K) array
        Posterior draws of the denominator SD. ``(S,)`` for per-draw scalar
        denominators (σ_src, σ_GM); ``(S, K)`` for per-configuration
        denominators (σ_pred when the residual is heteroskedastic).
    ref_idx : int
        Index of the reference configuration.
    labels : list[str]
        Configuration labels (length K or ``len(subset_idx)``).
    ci : tuple
        Credible-interval quantiles.
    subset_idx : (M,) int array, optional
        If given, restrict output to these configuration indices.

    Returns
    -------
    pd.DataFrame
        Columns: ``Config, std_bias_{med,lo,hi}, dmu_{med,lo,hi},
        mult_{med,lo,hi}``. Rows ordered by ``labels``.
    """
    mu_sub = mu_config[:, subset_idx] if subset_idx is not None else mu_config
    dmu = mu_sub - mu_config[:, [ref_idx]]  # (S, M)

    if sigma_denom.ndim == 1:
        beta = dmu / sigma_denom[:, None]
    elif sigma_denom.ndim == 2:
        denom = sigma_denom[:, subset_idx] if subset_idx is not None else sigma_denom
        if denom.shape != dmu.shape:
            raise ValueError(
                f"sigma_denom shape {denom.shape} incompatible with mu_config "
                f"subset shape {dmu.shape}"
            )
        beta = dmu / denom
    else:
        raise ValueError(
            f"sigma_denom must be 1-D (S,) or 2-D (S, K), got shape {sigma_denom.shape}"
        )

    mult = np.exp(dmu)

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


def bias_probability(
    mu_config: np.ndarray,
    sigma_denom: np.ndarray,
    ref_idx: int,
    labels: list[str],
    *,
    mode: Literal["exceed_band", "within_equiv", "positive"] = "within_equiv",
    band: float = 1.1,
    alpha_equiv: float = 0.4,
    subset_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    """Probability summaries of β = Δμ / σ_denom.

    Defaults match the spec (``uncertanty_measures.md`` §5–§6):
    ``alpha_equiv=0.4`` is the paper-level α_eq; ``band=1.1`` is the top
    rung of the α ladder (severely biased).

    Modes
    -----
    - ``"exceed_band"``  : P(|β| > band)
    - ``"within_equiv"`` : P(|β| < alpha_equiv) — the spec's P_eq
    - ``"positive"``     : P(β > 0)
    """
    mu_sub = mu_config[:, subset_idx] if subset_idx is not None else mu_config
    dmu = mu_sub - mu_config[:, [ref_idx]]

    if sigma_denom.ndim == 1:
        beta = dmu / sigma_denom[:, None]
    elif sigma_denom.ndim == 2:
        denom = sigma_denom[:, subset_idx] if subset_idx is not None else sigma_denom
        if denom.shape != dmu.shape:
            raise ValueError(
                f"sigma_denom shape {denom.shape} incompatible with mu_config "
                f"subset shape {dmu.shape}"
            )
        beta = dmu / denom
    else:
        raise ValueError(
            f"sigma_denom must be 1-D (S,) or 2-D (S, K), got shape {sigma_denom.shape}"
        )

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


def epistemic_median_ratio(
    mu_config: np.ndarray,
    ref_idx: int,
    labels: list[str],
    ci: tuple[float, float] = (0.05, 0.95),
    *,
    r_band: float | None = 1.25,
    subset_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    r"""Epistemic Median Ratio ``ρ_epi = exp(Δμ)`` — ``uncertanty_measures.md`` §6.

    Physical-unit companion to β (§6.1). Computed directly from the mean
    structure of the posterior, without any σ_GM dependency — this is why
    ρ_epi keeps a bounded CI in the regime where β is lever-arm inflated
    (§6.3). ρ_epi is reported *alongside* β, not as a replacement: the
    equivalence decision gate remains P_eq on β (§5, §8). ``r_band`` yields
    the §6.5 post-hoc reading P(ρ_epi ∈ [1/r, r]) — available as an
    engineering-unit equivalence summary but **not** a decision gate.

    Parameters
    ----------
    mu_config : (S, K) array
        Posterior draws of configuration effects on the log-EDP scale.
    ref_idx : int
        Index of the reference configuration.
    labels : list[str]
        Configuration labels of length K or ``len(subset_idx)``.
    ci : tuple, default (0.05, 0.95)
        Quantile CI for the ρ_epi and Δμ columns. Quantiles commute with
        monotone transforms, so ``rho_*`` and ``exp(dmu_*)`` are identical.
    r_band : float or None, default 1.25
        Engineering ratio for the post-hoc P_rho column (§6.5). If ``None``,
        the P_rho column is omitted.
    subset_idx : (M,) int array, optional
        Restrict output to these configuration indices.

    Returns
    -------
    pd.DataFrame
        Columns: ``Config``, ``rho_{med,lo,hi}``, ``dmu_{med,lo,hi}``, and
        ``P_rho`` if ``r_band`` is given.
    """
    mu_sub = mu_config[:, subset_idx] if subset_idx is not None else mu_config
    dmu = mu_sub - mu_config[:, [ref_idx]]  # (S, M)
    rho = np.exp(dmu)

    ci_lo, ci_hi = ci
    out = {
        "Config": labels,
        "rho_med": np.median(rho, axis=0),
        "rho_lo": np.quantile(rho, ci_lo, axis=0),
        "rho_hi": np.quantile(rho, ci_hi, axis=0),
        "dmu_med": np.median(dmu, axis=0),
        "dmu_lo": np.quantile(dmu, ci_lo, axis=0),
        "dmu_hi": np.quantile(dmu, ci_hi, axis=0),
    }
    if r_band is not None:
        if r_band <= 1.0:
            raise ValueError(f"r_band must be > 1, got {r_band}")
        inside = (rho >= 1.0 / r_band) & (rho <= r_band)
        out["P_rho"] = np.mean(inside, axis=0)

    return pd.DataFrame(out)
