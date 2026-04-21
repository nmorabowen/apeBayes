"""Validation-parameter visualisations — §7 of uncertanty_measures.md.

One plot function dispatches on the §7.4 rule attached to the validation
table (``"threshold"`` / ``"fractional"`` / ``"ci_overlap"``). Each rule
has a natural reference element:

- ``"threshold"``  : dashed vertical line at θ, fail side shaded light red.
- ``"fractional"`` : dashed vertical at target, ±τ_val band shaded teal.
- ``"ci_overlap"`` : horizontal shaded band at the benchmark HDI.

Error bars are the posterior CI on ``ÊDP^med``; colour indicates pass/fail
and the posterior mass P(rule holds) is annotated per row.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .helpers import ensure_dir, order_config_labels, savefig
from .style import CHARCOAL, HALF_WIDTH, NAVY, PALETTE, TEAL, fs

if TYPE_CHECKING:
    from pathlib import Path


# Accent colour for "fail" rows; complements NAVY (pass) without being
# green/red contrast that fails the protanopia-safe palette rule.
_FAIL_COLOR: str = "#B84A39"


def plot_validation_decision(
    val_df: pd.DataFrame,
    *,
    rule: Literal["threshold", "fractional", "ci_overlap"] | None = None,
    theta: float | None = None,
    direction: Literal["below", "above"] = "below",
    target: float | None = None,
    tau_val: float | None = None,
    target_hdi: tuple[float, float] | None = None,
    p_star_val: float = 0.95,
    order: Literal["tier", "case", "input"] = "tier",
    ref_label: str | None = None,
    label_col: str = "Config",
    figsize: tuple[float, float] = (HALF_WIDTH, 4.5),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "validation_decision.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    r"""Forest-style decision plot for the validation parameter — §7.

    Renders one horizontal error-bar per configuration at the posterior CI
    on ``ÊDP^med = exp(u₀ + Δμ)``. Rows are colour-coded by pass / fail
    under the chosen §7.4 rule, and the per-row posterior mass
    ``P(rule holds)`` is annotated on the right. A rule-specific reference
    element (threshold line, target band, or benchmark HDI band) is
    overlaid so the reader can read off pass/fail by eye.

    Parameters
    ----------
    val_df : pd.DataFrame
        Output of :func:`apeBayes.analysis.validation.validation_decision`
        or :meth:`BayesEpistemicModel.validation_decision_table`. Must
        carry ``Config``, ``edp_{med,lo,hi}``, ``posterior_mass``,
        ``decision``, and ``rule`` columns.
    rule : {"threshold", "fractional", "ci_overlap"}, optional
        Which §7.4 rule to render. If omitted, inferred from the
        DataFrame's ``rule`` column.
    theta : float, optional
        Threshold on ``ÊDP^med`` (required for ``rule='threshold'``).
    direction : {"below", "above"}, default "below"
        Passing side of ``theta``.
    target : float, optional
        External benchmark EDP in physical units (required for
        ``rule='fractional'``).
    tau_val : float, optional
        Fractional tolerance (required for ``rule='fractional'``).
    target_hdi : tuple[float, float], optional
        Physical-unit interval on the benchmark (required for
        ``rule='ci_overlap'``).
    p_star_val : float, default 0.95
        Gate for pass/fail. Shown in the title; ``val_df`` already carries
        the decision column.
    order : {"tier", "case", "input"}, default "tier"
        Row ordering on the y-axis.
    ref_label : str, optional
        Reference configuration; if present it is dropped from the plot.
    label_col : str, default "Config"
        Config-label column name.
    figsize, out_dir, prefix, filename
        Standard figure-save quartet.

    Returns
    -------
    fig, ax
    """
    out_dir = ensure_dir(out_dir)
    df = val_df.copy()
    if ref_label is not None:
        df = df[df[label_col] != ref_label]

    if rule is None:
        if "rule" not in df.columns or df["rule"].nunique() != 1:
            raise ValueError(
                "rule= not provided and val_df lacks a single 'rule' value"
            )
        rule = str(df["rule"].iloc[0])  # type: ignore[assignment]

    labels = order_config_labels(df[label_col].astype(str).tolist(), by=order)
    df = df.set_index(label_col).loc[labels].reset_index()

    med = df["edp_med"].to_numpy(dtype=float)
    lo = df["edp_lo"].to_numpy(dtype=float)
    hi = df["edp_hi"].to_numpy(dtype=float)
    mass = df["posterior_mass"].to_numpy(dtype=float)
    decisions = df["decision"].astype(str).to_numpy()

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    y = np.arange(len(labels))[::-1]

    # Per-row colouring: NAVY for pass, crimson for fail.
    row_colors = [NAVY if d == "pass" else _FAIL_COLOR for d in decisions]

    # Error bars, one at a time so each picks up its own colour.
    for med_i, lo_i, hi_i, y_i, c in zip(med, lo, hi, y, row_colors):
        ax.errorbar(
            [med_i], [y_i],
            xerr=[[med_i - lo_i], [hi_i - med_i]],
            fmt="o", color=c, ecolor=c,
            capsize=3, lw=1.3, ms=4,
        )

    # Rule-specific reference overlay.
    if rule == "threshold":
        if theta is None:
            raise ValueError("theta= required for rule='threshold'")
        ax.axvline(
            theta, color=CHARCOAL, lw=1.0, ls="--",
            label=rf"$\theta = {theta:g}$",
        )
        xlim = ax.get_xlim()
        if direction == "below":
            ax.axvspan(theta, xlim[1], alpha=0.07, color=_FAIL_COLOR, zorder=0)
        else:
            ax.axvspan(xlim[0], theta, alpha=0.07, color=_FAIL_COLOR, zorder=0)
        ax.set_xlim(xlim)

    elif rule == "fractional":
        if target is None or tau_val is None:
            raise ValueError(
                "target= and tau_val= required for rule='fractional'"
            )
        ax.axvline(
            target, color=CHARCOAL, lw=1.0, ls="--",
            label=rf"target $={target:g}$",
        )
        lo_band = target * (1.0 - tau_val)
        hi_band = target * (1.0 + tau_val)
        ax.axvspan(
            lo_band, hi_band, alpha=0.15, color=TEAL,
            label=rf"$\pm\tau_{{\mathrm{{val}}}} = \pm{tau_val:.0%}$ band",
            zorder=0,
        )

    elif rule == "ci_overlap":
        if target_hdi is None:
            raise ValueError("target_hdi= required for rule='ci_overlap'")
        lo_t, hi_t = target_hdi
        ax.axvspan(
            lo_t, hi_t, alpha=0.15, color=PALETTE[3],
            label=rf"target HDI $[{lo_t:g}, {hi_t:g}]$", zorder=0,
        )

    else:
        raise ValueError(f"unknown rule {rule!r}")

    # Posterior-mass annotations on the right edge.
    xmax = ax.get_xlim()[1]
    xspan = xmax - ax.get_xlim()[0]
    for y_i, m_i, d_i in zip(y, mass, decisions):
        ax.text(
            xmax + 0.02 * xspan, y_i, f"P={m_i:.2f}",
            va="center", ha="left", fontsize=fs(-2),
            color="0.25" if d_i == "pass" else _FAIL_COLOR,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fs(-1))
    ax.set_xlabel(
        r"$\widehat{\mathrm{EDP}}^{\mathrm{med}}$ (physical units)",
        fontsize=fs(-1),
    )
    ax.set_title(
        rf"Validation — rule={rule}, $P^{{\star}}_{{\mathrm{{val}}}}="
        rf"{p_star_val:g}$",
        fontsize=fs(),
    )
    ax.legend(loc="best", fontsize=fs(-2), framealpha=0.9)
    ax.grid(True, axis="x", alpha=0.25, lw=0.5, zorder=0)

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax
