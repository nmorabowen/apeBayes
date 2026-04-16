"""Variance-budget visualizations.

Waterfall, lollipop components, decomposition bars, sigma stability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .helpers import ensure_dir, order_config_labels, savefig
from .style import FULL_WIDTH, HALF_WIDTH, PALETTE, VARIANCE_COLORS

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

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
                short = prefix_str.replace(" spread", "").replace(" dispersion", "")
                short = short.replace(" disp.", "")
                short_names.append(short)
                break
        else:
            short_names.append(n[:12])

    fig, ax = plt.subplots(figsize=figsize)

    bottom = 0.0
    x = np.arange(len(names))
    total = float(vals.sum())

    for i, (_name, val, pct) in enumerate(zip(short_names, vals, pcts, strict=False)):
        ax.bar(i, val, bottom=bottom, color=colors[i], edgecolor="white",
               lw=0.5, width=0.65)
        mid_y = bottom + val / 2
        # Percentage inside bar (only if bar is tall enough)
        bar_frac = val / total if total > 0 else 0
        if bar_frac > 0.08:
            ax.text(i, mid_y, f"{pct:.0f}%", ha="center", va="center",
                    fontweight="bold", color="white")
        else:
            # Small bar: put percentage to the right
            ax.text(i + 0.38, mid_y, f"{pct:.0f}%", ha="left", va="center",
                    color=colors[i])
        bottom += val

    # Total annotation (top-left, away from bars)
    ax.text(0.02, 0.97, f"Total = {bottom:.4f}",
            transform=ax.transAxes, ha="left", va="top", color="0.3")

    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.set_ylabel("Variance (log EDP)\u00b2")
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
    for i, (comp, pct) in enumerate(zip(components, pcts, strict=False)):
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
    figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
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
    df["pct_lo"].to_numpy(dtype=float) if "pct_lo" in df.columns else med
    df["pct_hi"].to_numpy(dtype=float) if "pct_hi" in df.columns else med
    np.arange(len(labels))
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

    # Horizontal waterfall: stacked left-to-right
    left = 0.0
    y_pos = 0
    for i, (_lbl, val, _pct) in enumerate(zip(labels, med, med, strict=False)):
        ax.barh(y_pos, val, left=left, color=colors[i], edgecolor="white",
                lw=0.5, height=0.55)
        # Label inside if wide enough, outside if not
        if val > 8:
            ax.text(left + val / 2, y_pos, f"{val:.1f}%",
                    ha="center", va="center", fontweight="bold", color="white")
        else:
            ax.text(left + val + 1.0, y_pos, f"{val:.1f}%",
                    ha="left", va="center", color=colors[i])
        left += val

    # Short labels on y-axis
    short = []
    for lbl in labels:
        if "interaction" in lbl.lower():
            short.append("Interaction")
        elif "SSI" in lbl or "Factor0" in lbl:
            short.append("SSI")
        elif "Nonlinearity" in lbl or "Factor1" in lbl:
            short.append("Nonlinearity")
        else:
            short.append(lbl[:12])

    ax.set_yticks([0])
    ax.set_yticklabels([""])
    ax.set_xlabel("Share of epistemic variance (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Factorial decomposition of config-effect variance")

    # Legend from labels
    from matplotlib.patches import Patch
    ax.legend(
        [Patch(facecolor=c, edgecolor="white") for c in colors],
        short, loc="upper right",
    )

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
    for idx, c in zip(df_eps.index, df_eps["component"], strict=False):
        lbl = c.split("[")[1].rstrip("]") if "[" in c and "]" in c else c
        label_map[lbl] = idx
    ordered_idx = [label_map[l] for l in labels]
    df_eps = df_eps.loc[ordered_idx].reset_index(drop=True)

    np.arange(len(labels))
    med = df_eps["med"].to_numpy(dtype=float)
    lo = df_eps["lo"].to_numpy(dtype=float)
    hi = df_eps["hi"].to_numpy(dtype=float)

    n = len(labels)
    if figsize == (HALF_WIDTH, 3.0):
        figsize = (HALF_WIDTH, max(0.25 * n + 1.0, 3.0))

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(n)

    # Horizontal CI stems
    from .helpers import case_color as _cc
    colors = [_cc(lbl) for lbl in labels]
    for i in range(n):
        ax.plot([lo[i], hi[i]], [y[i], y[i]], color=colors[i], lw=1.5, solid_capstyle="round")
    ax.scatter(med, y, s=35, c=colors, zorder=5, edgecolors="white", lw=0.5,
               label=r"$\sigma_{\varepsilon}$ per config")

    # Value annotations
    x_max = float(hi.max())
    for i in range(n):
        ax.text(hi[i] + x_max * 0.03, y[i], f"{med[i]:.3f}",
                va="center", color="0.3")

    if sigma_run_med is not None:
        ax.axvline(sigma_run_med, color=VARIANCE_COLORS["between_rupture"], lw=1.3, ls="--",
                   label=rf"$\sigma_{{\mathrm{{run}}}} = {sigma_run_med:.3f}$")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Scale parameter (log EDP)")
    ax.legend(loc="lower right")
    ax.set_title(r"$\sigma_{\varepsilon}$ stability across configurations")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax


# ── Variance ratio: σ_eps / σ_run ─────────────────────────────────────

def plot_variance_ratio(
    sigma_run: np.ndarray,
    sigma_eps: np.ndarray,
    labels: list[str],
    ci: tuple[float, float] = (0.05, 0.95),
    *,
    nu: np.ndarray | None = None,
    figsize: tuple[float, float] = (HALF_WIDTH, 3.5),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "variance_ratio.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    r"""σ_eps^eff / σ_run per config — residual vs aleatory ratio."""
    from .helpers import case_colors_for_labels
    out_dir = ensure_dir(out_dir)

    t_factor = np.ones_like(sigma_run)
    if nu is not None:
        t_factor = np.sqrt(nu / np.maximum(nu - 2.0, 1e-12))

    if sigma_eps.ndim == 1:
        ratio = np.repeat((sigma_eps * t_factor)[:, None], len(labels), axis=1)
    else:
        ratio = (sigma_eps * t_factor[:, None]) / sigma_run[:, None]

    labels = order_config_labels(labels)
    ci_lo, ci_hi = ci
    med = np.median(ratio, axis=0)
    lo = np.quantile(ratio, ci_lo, axis=0)
    hi = np.quantile(ratio, ci_hi, axis=0)

    y = np.arange(len(labels))
    colors = case_colors_for_labels(labels)

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(len(labels)):
        ax.plot([lo[i], hi[i]], [y[i], y[i]], color=colors[i], lw=1.5,
                solid_capstyle="round")
    ax.scatter(med, y, s=35, c=colors, zorder=5, edgecolors="white", lw=0.5)
    ax.axvline(1.0, color="0.3", lw=1.2, ls="-", zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\sigma_{\varepsilon}^{\mathrm{eff}} / \sigma_{\mathrm{run}}$")
    ax.set_title("Residual vs run variability ratio")
    ax.grid(True, axis="x", alpha=0.3, lw=0.4)

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax


# ── Level rankings: Tier + Case energy shares ─────────────────────────

def plot_level_rankings(
    tier_table: pd.DataFrame,
    case_table: pd.DataFrame,
    *,
    factor_names: tuple[str, str] = ("SSI", "Nonlinearity"),
    figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "level_rankings.pdf",
) -> tuple[plt.Figure, np.ndarray]:
    """Two-panel horizontal bar chart of per-level energy shares."""
    out_dir = ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    f0_col = [c for c in tier_table.columns if c in (factor_names[0], "SSI", "Tier", "Factor0")]
    f0_col = f0_col[0] if f0_col else tier_table.columns[0]
    tv = tier_table["share_pct_med"].to_numpy(dtype=float)
    tl = tier_table["share_pct_lo"].to_numpy(dtype=float)
    th = tier_table["share_pct_hi"].to_numpy(dtype=float)
    tlabels = [f"Tier {v}" for v in tier_table[f0_col].tolist()]
    y0 = np.arange(len(tlabels))

    axes[0].barh(y0, tv, color=VARIANCE_COLORS["between_station"], alpha=0.8, height=0.6)
    axes[0].errorbar(tv, y0, xerr=[tv - tl, th - tv], fmt="none", capsize=3, color="0.2", lw=0.8)
    for yi, val in enumerate(tv):
        axes[0].text(min(val + 1.5, 95), yi, f"{val:.1f}%", va="center")
    axes[0].set_yticks(y0)
    axes[0].set_yticklabels(tlabels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel(f"Share of {factor_names[0]} main-effect energy (%)")
    axes[0].set_title(f"{factor_names[0]} ranking")

    f1_candidates = (factor_names[1], "Nonlinearity", "Case", "Factor1")
    f1_col = [c for c in case_table.columns if c in f1_candidates]
    f1_col = f1_col[0] if f1_col else case_table.columns[0]
    cv = case_table["share_pct_med"].to_numpy(dtype=float)
    cl = case_table["share_pct_lo"].to_numpy(dtype=float)
    ch = case_table["share_pct_hi"].to_numpy(dtype=float)
    clabels = [f"Case {v}" for v in case_table[f1_col].tolist()]
    y1 = np.arange(len(clabels))

    axes[1].barh(y1, cv, color=VARIANCE_COLORS["between_case"], alpha=0.8, height=0.6)
    axes[1].errorbar(cv, y1, xerr=[cv - cl, ch - cv], fmt="none", capsize=3, color="0.2", lw=0.8)
    for yi, val in enumerate(cv):
        axes[1].text(min(val + 1.5, 95), yi, f"{val:.1f}%", va="center")
    axes[1].set_yticks(y1)
    axes[1].set_yticklabels(clabels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel(f"Share of {factor_names[1]} main-effect energy (%)")
    axes[1].set_title(f"{factor_names[1]} ranking")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, axes


# ── Sigma stability triptych ──────────────────────────────────────────

def plot_sigma_stability_triptych(
    sigma_eps: np.ndarray,
    sigma_run: np.ndarray,
    labels: list[str],
    ref_idx: int,
    ci: tuple[float, float] = (0.05, 0.95),
    *,
    nu: np.ndarray | None = None,
    order_by: str = "stability",
    figsize: tuple[float, float] = (FULL_WIDTH, 4.5),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "sigma_stability_triptych.pdf",
) -> tuple[plt.Figure, np.ndarray]:
    r"""Three-panel sigma stability diagnostic.

    Panel 1: Posterior σ_ε violins + CI per config.
    Panel 2: Relative CI width = (hi − lo) / median.
    Panel 3: P(σ_ij ≤ σ_ref) — stability probability vs reference.
    """
    from .helpers import case_colors_for_labels
    out_dir = ensure_dir(out_dir)
    ci_lo, ci_hi = ci

    sig = sigma_eps.copy()
    if nu is not None:
        t_factor = np.sqrt(nu / np.maximum(nu - 2.0, 1e-12))
        sig = sig * t_factor[:, None] if sig.ndim == 2 else sig * t_factor
    if sig.ndim == 1:
        sig = np.repeat(sig[:, None], len(labels), axis=1)

    n = sig.shape[1]
    med = np.median(sig, axis=0)
    lo = np.quantile(sig, ci_lo, axis=0)
    hi = np.quantile(sig, ci_hi, axis=0)
    rel_ci_w = (hi - lo) / np.maximum(med, 1e-12)

    sig_ref = sig[:, ref_idx]
    p_le_ref = np.mean(sig <= sig_ref[:, None], axis=0)

    ord_idx = np.argsort(med) if order_by == "stability" else np.arange(n)

    labels_ord = [labels[i] for i in ord_idx]
    sig_ord = sig[:, ord_idx]
    med_ord, lo_ord, hi_ord = med[ord_idx], lo[ord_idx], hi[ord_idx]
    rel_ci_w_ord = rel_ci_w[ord_idx]
    p_le_ref_ord = p_le_ref[ord_idx]

    colors = case_colors_for_labels(labels_ord)
    y = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True,
                             gridspec_kw={"width_ratios": [2, 1, 1]}, sharey=True)
    ax0, ax1, ax2 = axes

    # Panel 1: violins + CI
    data = [sig_ord[:, j].ravel() for j in range(n)]
    parts = ax0.violinplot(data, positions=y, vert=False, widths=0.8,
                           showmeans=False, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.3)
    for i in range(n):
        ax0.plot([lo_ord[i], hi_ord[i]], [y[i], y[i]], color=colors[i],
                 lw=1.5, solid_capstyle="round")
    ax0.scatter(med_ord, y, s=30, c=colors, zorder=5, edgecolors="white", lw=0.4)
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels_ord)
    ax0.set_xlabel(r"$\sigma_{\varepsilon}^{\mathrm{eff}}$")
    ax0.set_title(r"Posterior $\sigma$ by config")
    ax0.grid(True, axis="x", alpha=0.3, lw=0.4)

    # Panel 2: CI width index
    ax1.barh(y, rel_ci_w_ord, color=colors, alpha=0.8, height=0.6)
    ax1.set_xlabel(r"$(\mathrm{CI}_{hi}-\mathrm{CI}_{lo})/\mathrm{median}$")
    ax1.set_title("CI width index")
    ax1.grid(True, axis="x", alpha=0.3, lw=0.4)

    # Panel 3: stability probability
    ref_label = labels[ref_idx]
    ax2.barh(y, p_le_ref_ord, color=colors, alpha=0.8, height=0.6)
    ax2.axvline(0.5, color="0.3", lw=0.8, ls="--")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel(rf"$P(\sigma_{{ij}} \leq \sigma_{{{ref_label}}})$")
    ax2.set_title(f"Stability vs {ref_label}")
    ax2.grid(True, axis="x", alpha=0.3, lw=0.4)

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, axes
