"""
Plotting helper utilities — file I/O, label parsing, layout tools.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── File helpers ────────────────────────────────────────────────────────

def ensure_dir(p: str | Path | None) -> Path | None:
    """Create directory (and parents) if needed; return Path or None."""
    if p is None:
        return None
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(
    fig: plt.Figure,
    out_dir: Path | str | None,
    filename: str,
    *,
    prefix: str = "",
    dpi: int = 300,
    close: bool = False,
) -> None:
    """Save figure to *out_dir/prefix+filename* if out_dir is not None."""
    if out_dir is None:
        return
    out_dir = Path(out_dir)
    fig.savefig(out_dir / f"{prefix}{filename}", dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)


# ── Label parsing (Tier × Case convention) ──────────────────────────────

_TC_RE = re.compile(r"^(\d+)([A-Za-z].*)$")


def parse_tier_case(label: str) -> tuple[int, str] | None:
    """Parse '4D' → (4, 'D'); return None if no match."""
    m = _TC_RE.match(str(label))
    if m is None:
        return None
    return int(m.group(1)), m.group(2)


def case_color(label: str, fallback: str = "#3C3C3C") -> str:
    """Return the CASE_COLORS colour for a config label like '4D' → 'D' → SAGE."""
    from .style import CASE_COLORS, CHARCOAL
    tc = parse_tier_case(str(label))
    if tc is None:
        return fallback
    return CASE_COLORS.get(tc[1], fallback)


def case_colors_for_labels(labels: list[str]) -> list[str]:
    """Return a list of Case-based colours aligned with *labels*."""
    return [case_color(lbl) for lbl in labels]


def add_tier_case_columns(df: pd.DataFrame, col: str = "Config") -> pd.DataFrame:
    """Add 'Tier' and 'Case' integer/string columns from a TierCase column."""
    tc = df[col].astype(str)
    tier = tc.str.extract(r"^(\d+)")[0].astype("Int64")
    case = tc.str.extract(r"^\d+([A-Za-z].*)$")[0].astype(str)
    out = df.copy()
    out["Tier"] = tier
    out["Case"] = case
    return out


def order_config_labels(
    labels: list[str],
    by: Literal["tier", "case", "input"] = "tier",
) -> list[str]:
    """Sort configuration labels in tier-major or case-major order.

    Parameters
    ----------
    by : str
        ``"tier"``  → 1A, 1B, 1C, …, 2A, 2B, …
        ``"case"``  → 1A, 2A, 3A, …, 1B, 2B, …
        ``"input"`` → unchanged.
    """
    if by == "input":
        return list(labels)

    def _key(lbl: str):
        parsed = parse_tier_case(lbl)
        if parsed is None:
            return (1, 999, "~", lbl)
        tier, case = parsed
        if by == "tier":
            return (0, tier, case.upper(), lbl)
        else:  # case
            return (0, case.upper(), tier, lbl)

    return sorted(labels, key=_key)


# ── Pivot helpers (for heatmap-style plots) ─────────────────────────────

def pivot_tier_case(
    df: pd.DataFrame,
    value_col: str,
    config_col: str = "Config",
) -> pd.DataFrame:
    """Pivot a table with a Config column into a Tier × Case matrix."""
    tmp = add_tier_case_columns(df, col=config_col)
    return tmp.pivot(index="Tier", columns="Case", values=value_col).sort_index()


# ── Symmetric colour bounds ─────────────────────────────────────────────

def symmetric_bounds(arr: np.ndarray) -> tuple[float, float]:
    """Return (-max|x|, +max|x|) for diverging-colour scales."""
    mx = float(np.nanmax(np.abs(arr)))
    if not np.isfinite(mx) or mx == 0:
        mx = 1e-6
    return -mx, mx


# ── Safe log for positive quantities ────────────────────────────────────

def safe_log_pos(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Log that clips non-positive / non-finite values to eps first."""
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x) & (x > 0.0), x, eps)
    return np.log(x)


# ── EDP scale utilities ────────────────────────────────────────────────

def to_edp_scale(x: Any, original: bool) -> np.ndarray:
    """Optionally exponentiate log-EDP back to natural EDP scale."""
    arr = np.asarray(x, dtype=float)
    return np.exp(arr) if original else arr


def edp_axis_label(original: bool) -> str:
    return "EDP" if original else "log EDP"
