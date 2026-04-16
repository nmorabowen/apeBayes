"""
Shared utilities for apeBayes.

Numerical helpers, posterior-draw manipulation, ordering functions,
and named constants live here so every other module can import them
without circular dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# ── Named constants ─────────────────────────────────────────────────────────

EPS = 1e-12          # safe-division / log-clipping epsilon
RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400


# ── Posterior draw helpers ──────────────────────────────────────────────────

def flatten_posterior(x: Any) -> np.ndarray:
    """Collapse (chain, draw, …) xarray/DataArray → (S, …) numpy array.

    Parameters
    ----------
    x : xarray.DataArray or array-like
        Posterior variable with leading (chain, draw) dimensions.

    Returns
    -------
    np.ndarray
        Array with shape (S, …) where S = n_chains × n_draws.
    """
    arr = x.values if hasattr(x, "values") else np.asarray(x)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    # (chain, draw, ...) → (S, ...)
    return arr.reshape(-1, *arr.shape[2:])


# ── Quantile / summary helpers ──────────────────────────────────────────────

@dataclass(frozen=True)
class DrawSummary:
    """Median + credible-interval summary of a 1-D draw vector."""

    median: float
    lo: float
    hi: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        """Return summary as a dictionary with optional key prefix."""
        return {
            f"{prefix}med": self.median,
            f"{prefix}lo": self.lo,
            f"{prefix}hi": self.hi,
        }


def summarize_draws(
    draws: np.ndarray,
    ci_lo: float = 0.05,
    ci_hi: float = 0.95,
    axis: int = 0,
) -> DrawSummary | np.ndarray:
    """Posterior median + credible interval.

    Parameters
    ----------
    draws : np.ndarray
        1-D (S,) or 2-D (S, K) array of posterior samples.
    ci_lo, ci_hi : float
        Lower/upper quantile bounds.
    axis : int
        Axis along which to compute (default 0 = sample axis).

    Returns
    -------
    DrawSummary
        If input is 1-D.
    np.ndarray of DrawSummary
        If input is 2-D (one per column).
    """
    med = np.median(draws, axis=axis)
    lo = np.quantile(draws, ci_lo, axis=axis)
    hi = np.quantile(draws, ci_hi, axis=axis)

    if np.ndim(med) == 0:
        return DrawSummary(float(med), float(lo), float(hi))

    return np.array([
        DrawSummary(float(m), float(l), float(h))
        for m, l, h in zip(np.atleast_1d(med), np.atleast_1d(lo), np.atleast_1d(hi), strict=False)
    ])


def summarize_to_dict(
    draws: np.ndarray,
    ci_lo: float = 0.05,
    ci_hi: float = 0.95,
    prefix: str = "",
) -> dict[str, float]:
    """Quick {prefix}med / {prefix}lo / {prefix}hi from 1-D draws."""
    return DrawSummary(
        float(np.median(draws)),
        float(np.quantile(draws, ci_lo)),
        float(np.quantile(draws, ci_hi)),
    ).as_dict(prefix)


def summarize_columns(
    draws_2d: np.ndarray,
    labels: list[str],
    ci_lo: float = 0.05,
    ci_hi: float = 0.95,
    prefix: str = "",
    label_col: str = "label",
) -> pd.DataFrame:
    """Summarize (S, K) draws into a tidy DataFrame with one row per column."""
    med = np.median(draws_2d, axis=0)
    lo = np.quantile(draws_2d, ci_lo, axis=0)
    hi = np.quantile(draws_2d, ci_hi, axis=0)
    return pd.DataFrame({
        label_col: labels,
        f"{prefix}med": med,
        f"{prefix}lo": lo,
        f"{prefix}hi": hi,
    })


# ── Student-t variance adjustment ──────────────────────────────────────────

def student_t_var_factor(nu: np.ndarray) -> np.ndarray:
    """Variance inflation factor for Student-t: ν / (ν − 2).

    Clipped to avoid divergence when ν ≤ 2.
    """
    return nu / np.maximum(nu - 2.0, EPS)


def student_t_sd_factor(nu: np.ndarray) -> np.ndarray:
    """SD inflation factor: √(ν / (ν − 2))."""
    return np.sqrt(student_t_var_factor(nu))


# ── Safe numerical ops ──────────────────────────────────────────────────────

def safe_log(x: np.ndarray) -> np.ndarray:
    """Log with clipping of non-positive/non-finite values to EPS."""
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x) & (x > 0.0), x, EPS)
    return np.log(x)


def safe_divide(
    num: np.ndarray,
    den: np.ndarray,
    fill: float = 0.0,
) -> np.ndarray:
    """Element-wise division with EPS floor on denominator."""
    return num / np.maximum(np.abs(den), EPS) * np.sign(den + EPS)


# ── Configuration label parsing ─────────────────────────────────────────────

_TC_PATTERN = re.compile(r"^(\d+)([A-Za-z].*)$")


def parse_config_label(label: str, sep: str = "") -> tuple[str, ...]:
    """Parse a flat config label into factor-level strings.

    If *sep* is non-empty, splits on the separator: "4_D_PM4Sand" → ("4", "D", "PM4Sand").
    If *sep* is empty (legacy mode), tries digit+letter pattern: "4D" → ("4", "D").
    Falls back to returning (label,) if nothing matches.
    """
    label = str(label)
    if sep:
        parts = tuple(label.split(sep))
        return parts if len(parts) > 1 else (label,)
    m = _TC_PATTERN.match(label)
    if m is None:
        return (label,)
    return (m.group(1), m.group(2).upper())


def config_label_sort_key(
    label: str,
    major: str = "tier",
    sep: str = "",
) -> tuple:
    """Sort key for configuration labels.

    Parameters
    ----------
    major : {"tier", "case"}
        Which factor is the primary sort axis.
    sep : str
        Separator used in label generation (passed to parse_config_label).
    """
    parts = parse_config_label(label, sep=sep)
    if len(parts) == 1:
        return (1, label)

    tier_num = int(parts[0]) if parts[0].isdigit() else 0
    case_str = parts[1] if len(parts) > 1 else ""

    if major == "case":
        return (0, case_str, tier_num, label)
    return (0, tier_num, case_str, label)


def order_config_labels(
    labels: list[str],
    major: str = "tier",
    sep: str = "",
) -> list[str]:
    """Sort configuration labels by tier-major or case-major order."""
    return sorted(labels, key=lambda lbl: config_label_sort_key(lbl, major=major, sep=sep))
