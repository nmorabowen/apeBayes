"""
Publication-quality plot style for apeBayes.

Palette calibrated 2026-04-15 for protanopia.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# -- Named colours -------------------------------------------------------------

NAVY:       str = "#002147"
TEAL:       str = "#1B6B8A"
STEEL:      str = "#4A90B8"
ICE:        str = "#C6DCF0"
AMBER:      str = "#C49A20"
SAGE:       str = "#5D8A78"
TAUPE:      str = "#B0A08E"
CHARCOAL:   str = "#3C3C3C"

# -- Primary categorical palette (8 colours, protanopia-safe) ------------------

PALETTE: list[str] = [
    NAVY,       # 0  deepest blue
    TEAL,       # 1  blue-green
    STEEL,      # 2  mid-blue
    AMBER,      # 3  warm pop
    SAGE,       # 4  earth green
    TAUPE,      # 5  warm neutral
    ICE,        # 6  lightest blue (fills / subtle)
    CHARCOAL,   # 7  near-black
]

# -- Semantic colour maps (paper-specific) -------------------------------------

CASE_COLORS: dict[str, str] = {
    "A": AMBER,
    "B": NAVY,
    "C": TEAL,
    "D": SAGE,
}

TIER_COLORS: dict[str, str] = {
    "1": ICE,
    "2": STEEL,
    "3": TEAL,
    "4": NAVY,
}

STATION_COLORS: dict[str, str] = {
    "sta_2": NAVY,
    "sta_5": AMBER,
    "sta_8": TEAL,
}

RUPTURE_COLORS: dict[str, str] = {
    "rup_bl_1": NAVY,
    "rup_bl_2": TEAL,
    "rup_bl_3": STEEL,
    "rup_bl_4": AMBER,
    "rup_bl_5": SAGE,
    "rup_bl_6": TAUPE,
}

VARIANCE_COLORS: dict[str, str] = {
    "between_case":    NAVY,
    "between_station": TEAL,
    "between_rupture": STEEL,
    "interaction":     AMBER,
    "residual":        TAUPE,
}

# -- Colormaps -----------------------------------------------------------------

_DIV_NODES = [NAVY, "#6B9CC4", "#FFFFFF", "#C47070", "#8B1A1A"]
CMAP_DIV_OBJ = mcolors.LinearSegmentedColormap.from_list(
    "NavyCrimson", [mcolors.to_rgb(c) for c in _DIV_NODES], N=256,
)
mpl.colormaps.register(CMAP_DIV_OBJ, name="NavyCrimson")
CMAP_DIV: str = "NavyCrimson"

_SEQ_NODES = [ICE, STEEL, TEAL, NAVY]
CMAP_SEQ_OBJ = mcolors.LinearSegmentedColormap.from_list(
    "IceNavy", [mcolors.to_rgb(c) for c in _SEQ_NODES], N=256,
)
mpl.colormaps.register(CMAP_SEQ_OBJ, name="IceNavy")
CMAP_SEQ: str = "IceNavy"

REF_COLOR: str = CHARCOAL

# -- Figure dimensions (WileyNJDv5 Times1COL) ---------------------------------

FULL_WIDTH: float = 7.0
HALF_WIDTH: float = 3.4

# -- rcParams defaults ---------------------------------------------------------

_BASE_RC: dict[str, object] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.6,
    "axes.titleweight": "bold",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": mpl.cycler(color=PALETTE),
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "legend.frameon": False,
    "legend.handlelength": 1.2,
    "figure.dpi": 150,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def apply_style(
    fontsize: float = 8.0,
    context: str = "paper",
) -> None:
    """Apply apeBayes publication style globally."""
    scale = {"paper": 1.0, "notebook": 1.25, "talk": 1.6}.get(context, 1.0)
    fs = fontsize * scale

    rc = dict(_BASE_RC)
    rc.update({
        "font.size": fs,
        "axes.titlesize": fs + 1,
        "axes.labelsize": fs,
        "xtick.labelsize": max(fs - 0.5, 5),
        "ytick.labelsize": max(fs - 0.5, 5),
        "legend.fontsize": max(fs - 0.5, 5),
        "figure.titlesize": fs + 2,
        "figure.titleweight": "bold",
    })

    plt.rcParams.update(rc)


def fs(delta: float = 0.0, floor: float = 5.0) -> float:
    """Return ``rcParams['font.size'] + delta``, clamped at ``floor``.

    Use inside plot functions instead of hardcoded ``fontsize=N`` literals
    so that per-element overrides scale with ``apply_style(context=...)``.
    The standards in ``reference_paper_figure_standards.md`` assume an 8 pt
    paper base; calls like ``fs(-1)`` (tick labels, 7 pt at paper) then
    render as 9 pt at notebook context without further changes.

    Canonical deltas for the composite-subplot table in that spec:

    ====================  =======  =================
    Role                  Delta    Paper-ctx result
    ====================  =======  =================
    Panel title / body    ``  0``  8 pt
    Tick labels / small   ``-1``   7 pt
    Small annotation      ``-2``   6 pt
    Dense legend          ``-3``   5 pt (= floor)
    Panel subtitle        ``+1``   9 pt
    Figure suptitle       ``+2``   10 pt
    ====================  =======  =================

    Parameters
    ----------
    delta : float, default 0.0
        Offset in points from the current base font size.
    floor : float, default 5.0
        Minimum returned size (legibility guardrail; 5 pt is the
        Wiley-guideline smallest acceptable figure text).

    Returns
    -------
    float
        Font size in points, suitable for ``fontsize=...`` kwargs.
    """
    return max(plt.rcParams["font.size"] + delta, floor)
