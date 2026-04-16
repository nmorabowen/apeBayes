"""Plotting utilities for apeBayes.

Submodules
----------
style    : rcParams, colour palettes, publication defaults.
helpers  : save/annotate/label/layout utilities.
bias     : bias heatmaps, triptychs, forest plots.
variance : variance-budget bars, pie, tornado.
equivalence : equivalence matrices, dendrograms, sweep curves.
posterior : density / CDF overlays, PPC, trace & rank.
"""

from .style import (
    apply_style, PALETTE, CMAP_DIV, CMAP_SEQ,
    CASE_COLORS, TIER_COLORS, STATION_COLORS, RUPTURE_COLORS,
    VARIANCE_COLORS, FULL_WIDTH, HALF_WIDTH,
    NAVY, TEAL, STEEL, ICE, AMBER, SAGE, TAUPE, CHARCOAL,
)
from .helpers import savefig, ensure_dir, add_tier_case_columns

__all__ = [
    "apply_style",
    "PALETTE", "CMAP_DIV", "CMAP_SEQ",
    "CASE_COLORS", "TIER_COLORS", "STATION_COLORS",
    "RUPTURE_COLORS", "VARIANCE_COLORS",
    "FULL_WIDTH", "HALF_WIDTH",
    "NAVY", "TEAL", "STEEL", "ICE", "AMBER", "SAGE", "TAUPE", "CHARCOAL",
    "savefig", "ensure_dir", "add_tier_case_columns",
]
