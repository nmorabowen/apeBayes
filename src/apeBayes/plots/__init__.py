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

from .helpers import add_tier_case_columns, ensure_dir, savefig
from .style import (
    AMBER,
    CASE_COLORS,
    CHARCOAL,
    CMAP_DIV,
    CMAP_SEQ,
    FULL_WIDTH,
    HALF_WIDTH,
    ICE,
    NAVY,
    PALETTE,
    RUPTURE_COLORS,
    SAGE,
    STATION_COLORS,
    STEEL,
    TAUPE,
    TEAL,
    TIER_COLORS,
    VARIANCE_COLORS,
    apply_style,
)

__all__ = [
    "AMBER",
    "CASE_COLORS",
    "CHARCOAL",
    "CMAP_DIV",
    "CMAP_SEQ",
    "FULL_WIDTH",
    "HALF_WIDTH",
    "ICE",
    "NAVY",
    "PALETTE",
    "RUPTURE_COLORS",
    "SAGE",
    "STATION_COLORS",
    "STEEL",
    "TAUPE",
    "TEAL",
    "TIER_COLORS",
    "VARIANCE_COLORS",
    "add_tier_case_columns",
    "apply_style",
    "ensure_dir",
    "savefig",
]
