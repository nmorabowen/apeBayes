"""
Variance-budget visualizations: waterfall, lollipop components,
decomposition bars, sigma stability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .helpers import savefig, ensure_dir, order_config_labels
from .style import PALETTE, CMAP_DIV, FULL_WIDTH, HALF_WIDTH, VARIANCE_COLORS


# ── Color mapping for variance components ─────────────────────────────────

_BUDGET_COLOR_ORDER: list[str] = [
    VARIANCE_COLORS["between_case"],     # Config spread
    VARIANCE_COLORS["between_station"],  # Station spread
    VARIANCE_COLORS["between_rupture"],  # Run dispersion
    VARIANCE_COLORS["interaction"],      # Interaction
    VARIANCE_COLORS["residual"],         # Residual
]


def _budget_colors(names: list[str]) -> list[str]:
    """Map variance-budget component names to semantic colours."""
    out = []
    for i, n in enumerate(names):
        if "Config" in n:
            out.append(VARIANCE_COLORS["between_case"])
        elif "Station" in n:
            out.append(VARIANCE_COLORS["between_station"])
        elif "Run" in n:
            out.append(VARIANCE_COLORS["between_rupture"])
        elif "Interaction" in n or "γ" in n:
            out.append(VARIANCE_COLORS["interaction"])
        elif "Residual" in n or "ε" in n:
            out.append(VARIANCE_COLORS["residual"])
        else:
            out.append(PALETTE[i % len(PALETTE)])
    return out


# ── Variance budget: waterfall chart ─────────────────────────────────────

def plot_variance_budget_waterfall(
    budget_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "variance_budget_waterfall.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    """Waterfall chart of variance components.

    Each bar starts where the previous one ended, showing how the total
    variance accumulates.  Much more readable than a stacked bar when one
    component dominates.

    Expects ``budget_df`` from ``variance_budget_table()`` with columns:
    component, var_med, var_lo, var_hi, pct_med.
    """
    out_dir = ensure_dir(out_dir)
    df = budget_df.copy()

    names = df["component"].tolist()
    vals = df["var_med"].to_numpy(dtype=float)
    pcts = df["pct_med"].to_numpy(dtype=float)
    colors = _budget_colors(names)

    short_names = []
    for n in names:
        # Shorten long component names for the axis
        if "Interaction" in n:
            short_names.append("Interaction")
            continue
        for prefix_str in ("Config spread", "Station spread", "Run dispersion", "Residual disp."):
            if prefix_str in n:
                short_names.append(prefix_str.replace(" spread", "").replace(" dispersion", "").replace(" disp.", ""))
                break
        else:
            short_names.append(n[:12])

    fig, ax = plt.subplots(figsize=figsize)

    bottom = 0.0
    x = np.arange(len(names))
    for i, (name, val, pct) in enumerate(zip(short_names, vals, pcts)):
        bar = ax.bar(i, val, bottom=bottom, color=colors[i], edgecolor="white",
                     lw=0.5, width=0.65)
        # Label: percentage inside bar, absolute value above
        mid_y = bottom + val / 2
        if pct > 3:
            ax.text(i, mid_y, f"{pct:.0f}%", ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")
        ax.text(i, bottom + val + 0.001, f"{val:.4f}", ha="center", va="bottom",
                fontsize=6, color="0.3")
        bottom += val

    # Total line
    ax.axhline(bottom, color="0.3", lw=0.8, ls=":", zorder=0)
    ax.text(len(names) - 0.3, bottom, f"Total = {bottom:.4f}",
            ha="right", va="bottom", fontsize=7, color="0.3")

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylabel("Variance (log EDP)²")
    ax.set_title("Variance budget (waterfall)")
    ax.set_xlim(-0.5, len(names) - 0.5)

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax


# ── Variance budget: original stacked bar (kept for backward compat) ─────

def plot_variance_budget_bars(
    budget_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "variance_budget.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    """Stacked horizontal bar of variance shares (original v7 style)."""
    out_dir = ensure_dir(out_dir)
    df = budget_df.copy()

    components = df["component"].tolist()
    pcts = df["pct_med"].to_numpy(dtype=float)
    colors = _budget_colors(components)

    fig, ax = plt.subplots(figsize=figsize)
    left = 0.0
    for i, (comp, pct) in enumerate(zip(components, pcts)):
        ax.barh(0, pct, left=left, color=colors[i], edgecolor="white",
                lw=0.4, label=comp, height=0.5)
        if pct > 5:
            ax.text(left + pct / 2, 0, f"{pct:.0f}%",
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold")
        left += pct

    ax.set_xlim(0, 100)
    ax.set_xlabel("Variance share (%)")
    ax.set_yticks([])
    ax.legend(loc="upper right", frameon=True, fontsize=7)
    ax.set_title("Variance budget")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax


# ── Variance components: lollipop chart ──────────────────────────────────

def plot_variance_components_lollipop(
    comp_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "variance_components_lollipop.pdf",
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    """Lollipop chart of all scale parameters with CI whiskers.

    Expects ``comp_df`` from ``variance_component_table()`` with columns:
    component, med, lo, hi.  Works for any mix of sigma_run, sigma_eps[k], nu.

    When nu is present, it gets its own right-hand panel so its wide CI
    doesn't compress the sigma scales.
    """
    out_dir = ensure_dir(out_dir)
    df = comp_df.copy()

    has_nu = df["component"].str.contains("nu").any()
    df_sigma = df[~df["component"].str.contains("nu")].reset_index(drop=True)
    df_nu = df[df["component"].str.contains("nu")].reset_index(drop=True)

    if has_nu:
        if figsize is None:
            figsize = (FULL_WIDTH, max(0.28 * len(df_sigma) + 0.8, 3.0))
        fig, axes = plt.subplots(
            1, 2, figsize=figsize, constrained_layout=True,
            gridspec_kw={"width_ratios": [3, 1]},
        )
        ax_sigma, ax_nu = axes
    else:
        if figsize is None:
            figsize = (HALF_WIDTH, max(0.28 * len(df_sigma) + 0.8, 3.0))
        fig, ax_sigma = plt.subplots(figsize=figsize)
        axes = np.array([ax_sigma])

    # -- Sigma panel --
    labels_s = df_sigma["component"].tolist()
    med_s = df_sigma["med"].to_numpy(dtype=float)
    lo_s = df_sigma["lo"].to_numpy(dtype=float)
    hi_s = df_sigma["hi"].to_numpy(dtype=float)
    y_s = np.arange(len(labels_s))

    colors_s = []
    for lbl in labels_s:
        if "run" in lbl:
            colors_s.append(VARIANCE_COLORS["between_rupture"])
        elif "eps" in lbl:
            colors_s.append(VARIANCE_COLORS["residual"])
        else:
            colors_s.append(PALETTE[6])

    for i in range(len(labels_s)):
        ax_sigma.plot([lo_s[i], hi_s[i]], [y_s[i], y_s[i]],
                      color=colors_s[i], lw=1.5, solid_capstyle="round")
    ax_sigma.scatter(med_s, y_s, s=40, c=colors_s, zorder=5,
                     edgecolors="white", linewidth=0.5)
    for i in range(len(labels_s)):
        ax_sigma.text(hi_s[i] + (hi_s.max() - lo_s.min()) * 0.02, y_s[i],
                      f"{med_s[i]:.3f}", va="center", fontsize=7, color="0.3")

    ax_sigma.set_yticks(y_s)
    ax_sigma.set_yticklabels(labels_s, fontsize=7)
    ax_sigma.invert_yaxis()
    ax_sigma.set_xlabel("Posterior value")
    ax_sigma.set_title("Scale parameters (median + 90% CI)")

    patches = []
    if any("run" in l for l in labels_s):
        patches.append(mpatches.Patch(color=VARIANCE_COLORS["between_rupture"],
                                      label=r"$\sigma_{\mathrm{run}}$"))
    if any("eps" in l for l in labels_s):
        patches.append(mpatches.Patch(color=VARIANCE_COLORS["residual"],
                                      label=r"$\sigma_{\varepsilon}$"))
    if patches:
        ax_sigma.legend(handles=patches, fontsize=7, loc="lower right")

    # -- Nu panel (separate x-axis) --
    if has_nu:
        labels_n = df_nu["component"].tolist()
        med_n = df_nu["med"].to_numpy(dtype=float)
        lo_n = df_nu["lo"].to_numpy(dtype=float)
        hi_n = df_nu["hi"].to_numpy(dtype=float)
        y_n = np.arange(len(labels_n))

        for i in range(len(labels_n)):
            ax_nu.plot([lo_n[i], hi_n[i]], [y_n[i], y_n[i]],
                       color=PALETTE[4], lw=1.5, solid_capstyle="round")
        ax_nu.scatter(med_n, y_n, s=40, c=PALETTE[4], zorder=5,
                      edgecolors="white", linewidth=0.5)
        for i in range(len(labels_n)):
            ax_nu.text(hi_n[i] + (hi_n.max() - lo_n.min()) * 0.05, y_n[i],
                       f"{med_n[i]:.1f}", va="center", fontsize=7, color="0.3")

        ax_nu.set_yticks(y_n)
        ax_nu.set_yticklabels(labels_n, fontsize=7)
        ax_nu.invert_yaxis()
        ax_nu.set_xlabel("Posterior value")
        ax_nu.set_title(r"$\nu$ (DoF)")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, axes if has_nu else ax_sigma


# ── Variance components: grouped bar (legacy, fixed) ─────────────────────

def plot_variance_components(
    comp_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "variance_components.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    """Lollipop chart of scale parameters — delegates to lollipop plot."""
    return plot_variance_components_lollipop(
        comp_df, figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
    )


# ── Axiswise decomposition bars ──────────────────────────────────────────

def plot_decomposition_bars(
    decomp_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "decomposition_bars.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    """Bar chart of factorial decomposition (SSI, Case, interaction).

    Expects ``decomp_df`` from ``axiswise_decomposition_table()``
    with columns: component, pct_med, pct_lo, pct_hi.
    """
    out_dir = ensure_dir(out_dir)
    df = decomp_df.copy()

    labels = df["component"].tolist()
    med = df["pct_med"].to_numpy(dtype=float)
    lo = df["pct_lo"].to_numpy(dtype=float) if "pct_lo" in df.columns else med
    hi = df["pct_hi"].to_numpy(dtype=float) if "pct_hi" in df.columns else med
    x = np.arange(len(labels))
    # Map SSI→station color, Nonlinearity→case color, interaction→interaction color
    decomp_colors = []
    for lbl in labels:
        if "interaction" in lbl.lower():
            decomp_colors.append(VARIANCE_COLORS["interaction"])
        elif "SSI" in lbl or "Factor0" in lbl or "Tier" in lbl:
            decomp_colors.append(VARIANCE_COLORS["between_station"])
        else:
            decomp_colors.append(VARIANCE_COLORS["between_case"])
    colors = decomp_colors

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, med, color=colors, edgecolor="white", lw=0.4, alpha=0.85, width=0.6)

    # CI whiskers
    if not np.array_equal(lo, med):
        ax.errorbar(x, med, yerr=[med - lo, hi - med],
                    fmt="none", ecolor="0.3", capsize=3, lw=0.8)

    for i, (v, lbl) in enumerate(zip(med, labels)):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Share of epistemic variance (%)")
    ax.set_title("Factorial decomposition of config-effect variance")
    ax.set_ylim(0, min(max(hi) * 1.25, 110))

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax


# ── Sigma stability summary ─────────────────────────────────────────────

def plot_sigma_stability(
    comp_df: pd.DataFrame,
    *,
    sigma_run_med: float | None = None,
    figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "sigma_stability.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    r"""σ_eps per config as lollipop with σ_run reference line.

    Parameters
    ----------
    comp_df : DataFrame
        Output of ``variance_component_table()`` with columns:
        component, med, lo, hi.
    sigma_run_med : float
        Posterior median of σ_run (drawn as horizontal reference).
    """
    out_dir = ensure_dir(out_dir)
    df = comp_df.copy()

    # Filter to sigma_eps entries only
    eps_mask = df["component"].str.contains("eps")
    df_eps = df[eps_mask].copy()

    # Extract config label from "sigma_eps[4D]" format
    labels = []
    for c in df_eps["component"]:
        if "[" in c and "]" in c:
            labels.append(c.split("[")[1].rstrip("]"))
        else:
            labels.append(c)

    labels = order_config_labels(labels)
    # Reorder df_eps to match
    label_map = {}
    for idx, c in zip(df_eps.index, df_eps["component"]):
        if "[" in c and "]" in c:
            lbl = c.split("[")[1].rstrip("]")
        else:
            lbl = c
        label_map[lbl] = idx
    ordered_idx = [label_map[l] for l in labels]
    df_eps = df_eps.loc[ordered_idx].reset_index(drop=True)

    x = np.arange(len(labels))
    med = df_eps["med"].to_numpy(dtype=float)
    lo = df_eps["lo"].to_numpy(dtype=float)
    hi = df_eps["hi"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    # CI bars
    for i in range(len(labels)):
        ax.plot([x[i], x[i]], [lo[i], hi[i]], color=PALETTE[1], lw=2, solid_capstyle="round")
    ax.scatter(x, med, s=35, color=PALETTE[1], zorder=5, edgecolors="white", lw=0.5,
               label=r"$\sigma_{\varepsilon}$ per config")

    if sigma_run_med is not None:
        ax.axhline(sigma_run_med, color=PALETTE[0], lw=1.3, ls="--",
                   label=rf"$\sigma_{{\mathrm{{run}}}} = {sigma_run_med:.3f}$")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Scale parameter (log EDP)")
    ax.legend(fontsize=7)
    ax.set_title(r"$\sigma_{\varepsilon}$ stability across configurations")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax
