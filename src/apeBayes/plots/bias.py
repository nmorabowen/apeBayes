"""Bias visualizations: triptych heatmaps, forest plots, probability bars."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .helpers import (
    case_color,
    case_colors_for_labels,
    ensure_dir,
    order_config_labels,
    pivot_tier_case,
    safe_log_pos,
    savefig,
    symmetric_bounds,
)
from .style import CMAP_DIV, FULL_WIDTH, HALF_WIDTH, PALETTE, fs

if TYPE_CHECKING:
    from pathlib import Path

# -- mu triptych -------------------------------------------------------------

def plot_mu_triptych(
    mu_hat_df: pd.DataFrame,
    bias_df: pd.DataFrame,
    *,
    ref: str,
    original_edp_scale: bool = False,
    figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
    annot: bool = True,
    fmt_mu: str = ".2f",
    fmt_dmu: str = ".2f",
    fmt_ratio: str = ".2f",
    cmap: str = CMAP_DIV,
    out_dir: str | Path | None = None,
    prefix: str = "",
) -> tuple[plt.Figure, np.ndarray]:
    """Three-panel heatmap: absolute mu, dmu vs ref, median ratio."""
    out_dir = ensure_dir(out_dir)

    pivot_mu = pivot_tier_case(mu_hat_df, "mu_med")
    pivot_dmu = pivot_tier_case(bias_df, "dmu_med")
    pivot_ratio = pivot_tier_case(bias_df, "mult_med")

    cols = sorted(
        set(pivot_mu.columns) | set(pivot_dmu.columns) | set(pivot_ratio.columns)
    )
    pivot_mu = pivot_mu.reindex(columns=cols)
    pivot_dmu = pivot_dmu.reindex(columns=cols)
    pivot_ratio = pivot_ratio.reindex(columns=cols)

    if original_edp_scale:
        ref_mu = float(
            mu_hat_df.loc[mu_hat_df["Config"].astype(str) == str(ref), "mu_med"].iloc[0]
        )
        pivot_mu = np.exp(pivot_mu)
        pivot_dmu = pivot_mu - np.exp(ref_mu)

    vmin_dmu, vmax_dmu = symmetric_bounds(pivot_dmu.to_numpy())

    color_ratio = safe_log_pos(pivot_ratio.to_numpy(dtype=float))
    pivot_ratio_color = pd.DataFrame(
        color_ratio, index=pivot_ratio.index, columns=pivot_ratio.columns
    )
    vmin_r, vmax_r = symmetric_bounds(pivot_ratio_color.to_numpy(dtype=float))

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True,
                             constrained_layout=True)

    _annot_kws = {"size": 7}

    sns.heatmap(
        pivot_mu, annot=annot, fmt=fmt_mu, ax=axes[0], cmap=cmap,
        annot_kws=_annot_kws,
        cbar_kws={"label": (
            "posterior median of EDP" if original_edp_scale
            else r"posterior median of $\mu$ (log scale)"
        )},
    )
    axes[0].set_title(
        "Mean EDP" if original_edp_scale else r"Mean $\mu_{ij}$"
    )
    axes[0].set_xlabel("Case")
    axes[0].set_ylabel("Tier")

    sns.heatmap(
        pivot_dmu, annot=annot, fmt=fmt_dmu,
        vmin=vmin_dmu, vmax=vmax_dmu, center=0.0,
        ax=axes[1], cmap=cmap, annot_kws=_annot_kws,
        cbar_kws={"label": (
            f"median dEDP vs {ref}" if original_edp_scale
            else rf"median $\Delta\mu$ vs {ref}"
        )},
    )
    axes[1].set_title(
        rf"$\Delta$EDP vs {ref}" if original_edp_scale
        else rf"$\Delta\mu_{{ij}}$ vs {ref}"
    )
    axes[1].set_xlabel("Case")
    axes[1].set_ylabel("")

    sns.heatmap(
        pivot_ratio_color,
        annot=(pivot_ratio.round(2) if annot else False),
        fmt=fmt_ratio,
        vmin=vmin_r, vmax=vmax_r, center=0.0,
        ax=axes[2], cmap=cmap, annot_kws=_annot_kws,
        cbar_kws={"label": r"color: $\log$ ratio (= $\Delta\mu$)"},
    )
    axes[2].set_title(rf"Median ratio $\exp(\Delta\mu_{{ij}})$ vs {ref}")
    axes[2].set_xlabel("Case")
    axes[2].set_ylabel("")

    savefig(fig, out_dir, f"mu_triptych_ref_{ref}.pdf", prefix=prefix)
    return fig, axes


# -- Standardized bias forest plot -------------------------------------------

def plot_bias_forest(
    bias_df: pd.DataFrame,
    *,
    col_med: str = "std_bias_med",
    col_lo: str = "std_bias_lo",
    col_hi: str = "std_bias_hi",
    label_col: str = "Config",
    ref_label: str | None = None,
    figsize: tuple[float, float] = (HALF_WIDTH, 4.3),
    color: str | None = None,
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "bias_forest.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    """Horizontal forest plot of standardized bias beta with CI whiskers."""
    out_dir = ensure_dir(out_dir)
    df = bias_df.copy()

    labels = df[label_col].astype(str).tolist()
    y = np.arange(len(labels))
    med = df[col_med].to_numpy(dtype=float)
    lo = df[col_lo].to_numpy(dtype=float)
    hi = df[col_hi].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0, color="0.5", lw=0.6, ls="--", zorder=0)

    c = color or PALETTE[0]
    ax.errorbar(
        med, y, xerr=[med - lo, hi - med],
        fmt="o", ms=4, lw=1.2, color=c, capsize=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"Standardized bias $\beta = \Delta\mu / \sigma_{\mathrm{run}}$")
    ax.grid(True, axis="x", alpha=0.25, lw=0.5, zorder=0)

    if ref_label is not None:
        ax.set_title(f"Standardized bias vs {ref_label}")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax


# -- Standardized bias ridgeplot ----------------------------------------------

def _draw_ridges(
    ax: plt.Axes,
    mu_config: np.ndarray,
    sigma_denom: np.ndarray,
    config_labels: list[str],
    ref_idx: int,
    configs_plot: list[str],
    label_to_idx: dict[str, int],
    *,
    overlap: float = 0.6,
    bw_adjust: float = 0.4,
    spacing: float = 1.0,
) -> None:
    """Draw KDE ridges of β = dmu / sigma_denom on a single Axes."""
    from scipy.stats import gaussian_kde

    ref_draws = mu_config[:, ref_idx]
    n = len(configs_plot)

    for row, lbl in enumerate(configs_plot):
        k = label_to_idx[lbl]
        beta_draws = (mu_config[:, k] - ref_draws) / sigma_denom

        try:
            kde = gaussian_kde(beta_draws, bw_method=bw_adjust)
        except np.linalg.LinAlgError:
            continue

        x_grid = np.linspace(
            float(np.percentile(beta_draws, 0.5)),
            float(np.percentile(beta_draws, 99.5)),
            300,
        )
        density = kde(x_grid)
        density = density / density.max() * spacing * (1 + overlap)

        baseline = row * spacing
        color = case_color(lbl)

        ax.fill_between(x_grid, baseline, baseline + density,
                        alpha=0.35, color=color, zorder=n - row)
        ax.plot(x_grid, baseline + density, lw=1.0, color=color,
                zorder=n - row)

        med = float(np.median(beta_draws))
        ax.plot([med, med], [baseline, baseline + spacing * 0.15],
                color="0.2", lw=1.0, zorder=n - row + 1)

    ax.axvline(0, color="0.4", lw=0.8, ls="--", zorder=0)
    ax.set_yticks([i * spacing for i in range(n)])
    ax.set_yticklabels(configs_plot, fontsize=fs(-1))
    ax.set_ylim(-0.3 * spacing, n * spacing)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


_DENOM_LATEX = {
    "GM": r"\sigma_{\mathrm{GM}}",
    "src": r"\sigma_{\mathrm{src}}",
    "pred": r"\sigma_{\mathrm{pred}}",
}


def _denom_math(denom_name: str) -> str:
    """Return a LaTeX string for the given denominator tag (falls back gracefully)."""
    return _DENOM_LATEX.get(denom_name, rf"\sigma_{{\mathrm{{{denom_name}}}}}")


def plot_bias_ridgeplot(
    mu_config: np.ndarray,
    sigma_denom: np.ndarray,
    config_labels: list[str],
    ref_idx: int,
    *,
    y_obs: np.ndarray | None = None,
    config_idx: np.ndarray | None = None,
    station_idx: np.ndarray | None = None,
    station_labels: list[str] | None = None,
    denom_name: str = "GM",
    overlap: float = 0.6,
    figsize: tuple[float, float] | None = None,
    bw_adjust: float = 0.4,
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "bias_ridgeplot.pdf",
) -> tuple[plt.Figure, np.ndarray]:
    r"""Ridgeplot of posterior β = Δμ / σ_denom distributions.

    When station data is provided, draws one panel per station with the
    observed data overlaid as dots. The ``denom_name`` label (``"GM"``,
    ``"src"``, ``"pred"``) is baked into the x-axis label so the figure
    self-documents which β variant it shows.
    """
    out_dir = ensure_dir(out_dir)
    denom_latex = _denom_math(denom_name)
    x_label = rf"$\beta = \Delta\mu\,/\,{denom_latex}$"

    labels_sorted = order_config_labels(config_labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(config_labels)}
    configs_plot = [lbl for lbl in labels_sorted
                    if label_to_idx[lbl] != ref_idx]
    n = len(configs_plot)
    spacing = 1.0

    has_station = (y_obs is not None and config_idx is not None
                   and station_idx is not None and station_labels is not None)

    if has_station:
        n_sta = len(station_labels)
        if figsize is None:
            figsize = (FULL_WIDTH, 4.3)
        fig, axes = plt.subplots(
            1, n_sta, figsize=figsize, sharey=True,
            constrained_layout=True, squeeze=False,
        )
        axes = axes.ravel()

        denom_med = float(np.median(sigma_denom))

        for s, sta in enumerate(station_labels):
            ax = axes[s]
            _draw_ridges(
                ax, mu_config, sigma_denom, config_labels, ref_idx,
                configs_plot, label_to_idx,
                overlap=overlap, bw_adjust=bw_adjust, spacing=spacing,
            )

            sta_mask = station_idx == s
            ref_mask = sta_mask & (config_idx == ref_idx)
            y_ref_mean = float(np.mean(y_obs[ref_mask])) if ref_mask.any() else 0.0

            for row, lbl in enumerate(configs_plot):
                k = label_to_idx[lbl]
                obs_mask = sta_mask & (config_idx == k)
                if not obs_mask.any():
                    continue
                beta_obs = (y_obs[obs_mask] - y_ref_mean) / denom_med
                baseline = row * spacing
                jitter = np.random.default_rng(42 + s).uniform(
                    0.05 * spacing, 0.25 * spacing, size=len(beta_obs),
                )
                ax.scatter(
                    beta_obs, baseline + jitter,
                    s=18, marker="d", edgecolors="0.2", linewidths=0.4,
                    facecolors=case_color(lbl), alpha=0.8,
                    zorder=n + 10,
                )

            ax.set_xlabel(x_label)
            ax.set_title(str(sta), fontsize=fs(+1))
            if s > 0:
                ax.set_yticklabels([])

        fig.suptitle(
            f"Posterior bias vs {config_labels[ref_idx]}  "
            r"($\diamondsuit$ = observed)",
            fontsize=fs(+1),
        )
    else:
        if figsize is None:
            figsize = (HALF_WIDTH, 4.3)
        fig, ax = plt.subplots(figsize=figsize)
        _draw_ridges(
            ax, mu_config, sigma_denom, config_labels, ref_idx,
            configs_plot, label_to_idx,
            overlap=overlap, bw_adjust=bw_adjust, spacing=spacing,
        )
        ax.set_xlabel(x_label)
        ax.set_title(f"Posterior bias distributions vs {config_labels[ref_idx]}")
        axes = np.array([ax])

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, axes


# -- Standardized bias scatter (with optional station subplots) ---------------

def plot_standardized_bias(
    bias_df: pd.DataFrame,
    *,
    beta_draws: np.ndarray | None = None,
    beta_labels: list[str] | None = None,
    raw_dots: np.ndarray | None = None,
    raw_means: np.ndarray | None = None,
    raw_labels: list[str] | None = None,
    station_subplots: bool = False,
    station_col: str = "station",
    label_col: str = "Config",
    col_med: str = "std_bias_med",
    col_lo: str = "std_bias_lo",
    col_hi: str = "std_bias_hi",
    ref_label: str | None = None,
    denom_name: str = "GM",
    alpha_eq: float = 0.4,
    dot_alpha: float = 0.28,
    posterior_style: str = "violin",
    figsize: tuple[float, float] | None = None,
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "standardized_bias.pdf",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot layered standardised-bias with posterior density and raw dots.

    Includes X markers and credible intervals.

    Parameters
    ----------
    bias_df : DataFrame
        Output of ``standardized_bias_table()``.
    posterior_style : {"violin", "ridge", "none"}
        How to visualise the full posterior of beta.
        ``"violin"``: symmetric KDE mirrored about the row centre.
        ``"ridge"``:  one-sided KDE fill above the row baseline,
        all ridges normalised to a shared global maximum so widths
        are comparable across configs.
        ``"none"``:   no posterior density overlay, only error bars.
    beta_draws : (S, n_configs) array, optional
        Full posterior draws of standardised bias per config.
        When provided, posterior KDEs are drawn behind the
        error bars to show the full posterior shape.
    beta_labels : list[str], optional
        Config labels matching columns of *beta_draws*.
    raw_dots : (n_configs, n_runkeys) array, optional
        Per-runkey standardised bias values.
    raw_means : (n_configs,) array, optional
        Per-config raw-data mean standardised bias.
    raw_labels : list[str], optional
        Config labels matching rows of *raw_dots* / *raw_means*.
    station_subplots : bool
        If True and *station_col* is in the dataframe, draw one
        panel per station.
    """
    out_dir = ensure_dir(out_dir)

    if station_subplots and station_col in bias_df.columns:
        stations = sorted(bias_df[station_col].unique())
        n_panels = len(stations)
        if figsize is None:
            figsize = (FULL_WIDTH, HALF_WIDTH)
        fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=True,
                                 constrained_layout=True, squeeze=False)
        axes = axes.ravel()
        for idx, sta in enumerate(stations):
            sub = bias_df[bias_df[station_col] == sta].copy()
            _draw_bias_panel(axes[idx], sub, label_col, col_med, col_lo, col_hi,
                             beta_draws=beta_draws, beta_labels=beta_labels,
                             raw_dots=raw_dots, raw_means=raw_means,
                             raw_labels=raw_labels, dot_alpha=dot_alpha,
                             posterior_style=posterior_style,
                             denom_name=denom_name, alpha_eq=alpha_eq)
            axes[idx].set_title(str(sta))
            if idx > 0:
                axes[idx].set_ylabel("")
        xmins = [ax.get_xlim()[0] for ax in axes]
        xmaxs = [ax.get_xlim()[1] for ax in axes]
        for ax in axes:
            ax.set_xlim(min(xmins), max(xmaxs))
    else:
        if figsize is None:
            figsize = (HALF_WIDTH, HALF_WIDTH)
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        _draw_bias_panel(ax, bias_df, label_col, col_med, col_lo, col_hi,
                         beta_draws=beta_draws, beta_labels=beta_labels,
                         raw_dots=raw_dots, raw_means=raw_means,
                         raw_labels=raw_labels, dot_alpha=dot_alpha,
                         posterior_style=posterior_style,
                         denom_name=denom_name, alpha_eq=alpha_eq)
        axes = np.array([ax])

    if ref_label is not None:
        fig.suptitle(rf"Standardised bias $\beta$ vs {ref_label}", fontsize=fs(+1))

    handles, labs = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labs, loc="upper center",
                   bbox_to_anchor=(0.5, -0.01), ncol=3,
                   fontsize=fs(-2), frameon=True)

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, axes


def _draw_bias_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    label_col: str,
    col_med: str,
    col_lo: str,
    col_hi: str,
    *,
    beta_draws: np.ndarray | None = None,
    beta_labels: list[str] | None = None,
    raw_dots: np.ndarray | None = None,
    raw_means: np.ndarray | None = None,
    raw_labels: list[str] | None = None,
    dot_alpha: float = 0.28,
    posterior_style: str = "violin",
    denom_name: str = "GM",
    alpha_eq: float = 0.4,
) -> None:
    """Draw one bias panel with layered elements.

    Layers (back to front):
      0  Grey equivalence band |β| < α_eq with ±1 aleatory reference lines
      1  Posterior density (violin, ridge, or none)
      2  Raw per-runkey dots (steel, semi-transparent)
      3  Raw TierCase-mean X markers (amber)
      4  Posterior CI error bars with cap handles (navy)

    posterior_style : {"violin", "ridge", "none"}
    """
    from scipy.stats import gaussian_kde

    labels = order_config_labels(df[label_col].astype(str).tolist())
    df_ord = df.set_index(label_col).loc[labels].reset_index()

    y = np.arange(len(labels))
    med = df_ord[col_med].to_numpy(dtype=float)
    lo = df_ord[col_lo].to_numpy(dtype=float)
    hi = df_ord[col_hi].to_numpy(dtype=float)

    # Layer 0: α_eq equivalence band; ±1 dashed lines kept as aleatory reference
    ax.axvspan(-alpha_eq, alpha_eq, color="grey", alpha=0.12, zorder=0,
               label=rf"$|\beta| < \alpha_{{\mathrm{{eq}}}} = {alpha_eq:g}$")
    ax.axvline(0, color="black", lw=0.8, alpha=0.6, zorder=1)
    ax.axvline(-1.0, color="0.4", lw=0.6, ls="--", alpha=0.6, zorder=0)
    ax.axvline(1.0, color="0.4", lw=0.6, ls="--", alpha=0.6, zorder=0)

    # Layer 1: posterior density overlay
    if posterior_style != "none" and beta_draws is not None and beta_labels is not None:
        label_to_y = {lbl: yi for lbl, yi in zip(labels, y, strict=False)}

        # Pre-compute all KDEs so we can normalise consistently
        kdes: list[tuple[int, float, np.ndarray, np.ndarray]] = []
        global_peak = 0.0
        for i, bl in enumerate(beta_labels):
            if bl not in label_to_y:
                continue
            draws_i = beta_draws[:, i]
            draws_i = draws_i[np.isfinite(draws_i)]
            if len(draws_i) < 10:
                continue
            try:
                kde = gaussian_kde(draws_i, bw_method=0.3)
            except np.linalg.LinAlgError:
                continue
            x_grid = np.linspace(
                float(np.percentile(draws_i, 0.5)),
                float(np.percentile(draws_i, 99.5)),
                200,
            )
            density = kde(x_grid)
            peak = float(density.max())
            if peak > global_peak:
                global_peak = peak
            kdes.append((i, label_to_y[bl], x_grid, density))

        if global_peak <= 0:
            global_peak = 1.0

        hw = 0.38  # max half-width in y-axis units

        for idx, (i, yi, x_grid, density) in enumerate(kdes):
            color = case_color(beta_labels[i])
            # Normalise to shared global peak
            normed = density / global_peak * hw

            if posterior_style == "violin":
                ax.fill_between(
                    x_grid, yi - normed, yi + normed,
                    alpha=0.18, color=color, zorder=1,
                    label=("posterior density" if idx == 0 else None),
                )
                ax.plot(x_grid, yi - normed, lw=0.4, color=color, alpha=0.4, zorder=1)
                ax.plot(x_grid, yi + normed, lw=0.4, color=color, alpha=0.4, zorder=1)
            elif posterior_style == "ridge":
                ax.fill_between(
                    x_grid, yi, yi - normed,
                    alpha=0.25, color=color, zorder=1,
                    label=("posterior density" if idx == 0 else None),
                )
                ax.plot(x_grid, yi - normed, lw=0.6, color=color, alpha=0.6, zorder=1)

    # Layer 2: raw per-runkey dots
    if raw_dots is not None and raw_labels is not None:
        label_to_y = {lbl: yi for lbl, yi in zip(labels, y, strict=False)}
        for i, rl in enumerate(raw_labels):
            if rl in label_to_y:
                yi = label_to_y[rl]
                vals = raw_dots[i]
                vals = vals[np.isfinite(vals)]
                jitter = np.random.default_rng(42 + i).uniform(
                    -0.18, 0.18, size=len(vals))
                ax.scatter(
                    vals, yi + jitter, s=18,
                    alpha=dot_alpha, color=PALETTE[2], edgecolors="none",
                    zorder=2,
                    label=("raw per-runkey" if i == 0 else None),
                )

    # Layer 3: raw TierCase mean (removed — redundant with posterior median)

    # Layer 4: posterior credible interval + median (error bars with caps)
    ax.errorbar(
        med, y, xerr=[med - lo, hi - med],
        fmt="o", ms=4.5, lw=1.6, capsize=3, capthick=1.2,
        color=PALETTE[0], markeredgecolor="white",
        markeredgewidth=0.4, zorder=6,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fs(-1))
    ax.invert_yaxis()
    ax.set_xlabel(
        rf"$\beta_{{ij}} = \Delta\mu_{{ij}}\,/\,{_denom_math(denom_name)}$"
    )
    ax.grid(True, axis="x", alpha=0.25, lw=0.5, zorder=0)


# -- Radar / spider chart of bias probability ---------------------------------

def plot_radar_bias_probability(
    prob_df: pd.DataFrame,
    *,
    prob_col: str = "P_equiv",
    label_col: str = "Config",
    alpha: float = 0.5,
    ref: str = "4D",
    figsize: tuple[float, float] = (HALF_WIDTH, HALF_WIDTH),
    fill_alpha: float = 0.15,
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "radar_bias_probability.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    r"""Radar (spider) chart of P(|beta| < alpha) per configuration."""
    out_dir = ensure_dir(out_dir)
    df = prob_df[prob_df[label_col] != ref].copy()
    labels = order_config_labels(df[label_col].astype(str).tolist())
    df = df.set_index(label_col).loc[labels].reset_index()

    vals = df[prob_col].to_numpy(dtype=float)
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals_closed = np.concatenate([vals, vals[:1]])
    angles_closed = [*angles, angles[0]]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    spoke_colors = case_colors_for_labels(labels)
    ax.plot(angles_closed, vals_closed, "-", lw=1.2, color=PALETTE[0], alpha=0.5)
    ax.fill(angles_closed, vals_closed, alpha=fill_alpha, color=PALETTE[0])
    ax.scatter(angles, vals, c=spoke_colors, s=25, zorder=5, edgecolors="white", linewidths=0.4)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=fs(-1))
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=fs(-2))
    ax.set_title(
        rf"$P(|\beta| < {alpha}\,\sigma_{{\mathrm{{run}}}})$ vs {ref}",
        pad=15, fontsize=fs(),
    )

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax


# -- Bias probability bar chart -----------------------------------------------

def plot_bias_probability(
    prob_df: pd.DataFrame,
    *,
    prob_col: str = "P_exceed",
    label_col: str = "Config",
    threshold_label: str = "",
    figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "bias_probability.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    """Bar chart of bias exceedance (or equivalence) probabilities."""
    out_dir = ensure_dir(out_dir)

    labels = prob_df[label_col].astype(str).tolist()
    vals = prob_df[prob_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    ax.bar(x, vals, color=PALETTE[0], edgecolor="white", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(f"P({threshold_label})" if threshold_label else "Probability")
    ax.set_ylim(0, 1.05)

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax

