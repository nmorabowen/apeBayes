"""
Posterior density / CDF overlays, PPC, and trace diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .helpers import (
    savefig,
    ensure_dir,
    to_edp_scale,
    edp_axis_label,
    order_config_labels,
)
from .style import PALETTE, FULL_WIDTH, HALF_WIDTH


# ── HDI helper ────────────────────────────────────────────────────────

def _hdi(samples: np.ndarray, prob: float = 0.94) -> tuple[float, float]:
    """Highest Density Interval for a 1-D sample array.

    Finds the shortest interval containing *prob* fraction of the mass.
    This is the standard "shortest interval" algorithm: sort the draws,
    slide a window of width ``n_keep`` across them, pick the window
    with the smallest span.
    """
    x = np.sort(samples)
    n = len(x)
    n_keep = int(np.ceil(prob * n))
    if n_keep >= n:
        return float(x[0]), float(x[-1])
    widths = x[n_keep:] - x[: n - n_keep]
    idx = int(np.argmin(widths))
    return float(x[idx]), float(x[idx + n_keep])


def _hdi_vec(flat: np.ndarray, prob: float = 0.94) -> tuple[np.ndarray, np.ndarray]:
    """HDI for each column of a (S, K) array.  Returns (lo, hi) arrays."""
    lo = np.empty(flat.shape[1])
    hi = np.empty(flat.shape[1])
    for k in range(flat.shape[1]):
        lo[k], hi[k] = _hdi(flat[:, k], prob)
    return lo, hi


# ── Posterior density / CDF overlay ─────────────────────────────────────

def plot_mu_density(
    mu0: np.ndarray,
    mu_config: np.ndarray,
    config_labels: list[str],
    *,
    configs: list[str] | None = None,
    delta_st: np.ndarray | None = None,
    station_idx: int | None = None,
    b_run: np.ndarray | None = None,
    run_idx: int | None = None,
    original_edp_scale: bool = False,
    kind: Literal["density", "cdf"] = "density",
    normalize: bool = False,
    figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
    n_cols: int = 4,
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "mu_posterior_density.pdf",
) -> tuple[plt.Figure, np.ndarray]:
    """Posterior density or CDF of modelled mean μ for selected configs.

    Parameters
    ----------
    mu0 : (S,)
    mu_config : (S, K)
    config_labels : length-K list
    configs : subset of labels to plot (default: all)
    delta_st, station_idx : add station effect if both provided
    b_run, run_idx : add run effect if both provided
    original_edp_scale : exponentiate to natural EDP
    kind : "density" or "cdf"
    normalize : rescale KDE peak to 1
    """
    out_dir = ensure_dir(out_dir)
    if configs is None:
        configs = config_labels

    configs = order_config_labels(configs)
    label_to_idx = {lbl: i for i, lbl in enumerate(config_labels)}

    n = len(configs)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows),
                             squeeze=False, constrained_layout=True)

    for i, cfg in enumerate(configs):
        ax = axes.flat[i]
        k = label_to_idx[cfg]
        draws = mu0 + mu_config[:, k]

        if delta_st is not None and station_idx is not None:
            draws = draws + delta_st[:, station_idx]
        if b_run is not None and run_idx is not None:
            draws = draws + b_run[:, run_idx]

        draws_plot = to_edp_scale(draws, original_edp_scale)

        if kind == "cdf":
            sns.ecdfplot(draws_plot, ax=ax, lw=1.4, color=PALETTE[i % len(PALETTE)])
        elif normalize:
            _plot_normalized_kde(ax, draws_plot, color=PALETTE[i % len(PALETTE)])
        else:
            sns.kdeplot(draws_plot, fill=True, alpha=0.25, lw=1.4, ax=ax,
                        color=PALETTE[i % len(PALETTE)])

        ax.set_title(cfg, fontsize=7)
        ax.set_xlabel(edp_axis_label(original_edp_scale))

    # Hide unused axes
    for j in range(n, n_rows * n_cols):
        axes.flat[j].set_visible(False)

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, axes


def _plot_normalized_kde(
    ax: plt.Axes,
    data: np.ndarray,
    color: str = "#4477AA",
) -> None:
    """KDE rescaled so the peak = 1."""
    sns.kdeplot(data, fill=False, lw=1.4, ax=ax, color=color)
    if not ax.lines:
        return
    line = ax.lines[-1]
    ydata = np.asarray(line.get_ydata(), dtype=float)
    ymax = float(np.max(ydata))
    if np.isfinite(ymax) and ymax > 0:
        y_norm = ydata / ymax
        line.set_ydata(y_norm)
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ax.fill_between(xdata, 0, y_norm, alpha=0.25, color=color)


# ── Posterior predictive check ──────────────────────────────────────────

def plot_ppc(
    ppc_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "ppc_check.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    """Horizontal bar chart of PPC p-values (mean and sd) per group.

    Flipped axis layout: configurations on y, p-values on x.
    Fits naturally in a half-width column; auto-sizes height to
    the number of groups.
    """
    out_dir = ensure_dir(out_dir)

    groups = ppc_df["group"].tolist()
    n = len(groups)
    y = np.arange(n)

    if figsize is None:
        figsize = (HALF_WIDTH, max(0.35 * n + 0.8, 3.0))

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    w = 0.35
    ax.barh(y + w / 2, ppc_df["p_value_sd"], w,
            label="p(sd)", color=PALETTE[1])
    ax.barh(y - w / 2, ppc_df["p_value_mean"], w,
            label="p(mean)", color=PALETTE[0])

    ax.axvline(0.5, ls="--", lw=0.6, color="0.5")
    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.set_xlabel("Posterior predictive p-value")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Posterior predictive check")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax


# ── Residual plots ──────────────────────────────────────────────────────

def plot_residuals(
    fitted_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "residuals.pdf",
) -> tuple[plt.Figure, np.ndarray]:
    """Two-panel residual diagnostic: fitted vs residual, QQ."""
    out_dir = ensure_dir(out_dir)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Fitted vs residual
    ax = axes[0]
    ax.scatter(fitted_df["yhat_with_run"], fitted_df["resid"],
               s=8, alpha=0.4, color=PALETTE[0], edgecolors="none")
    ax.axhline(0, ls="--", lw=0.6, color="0.5")
    ax.set_xlabel(r"Fitted $\hat{y}$")
    ax.set_ylabel("Residual")
    ax.set_title("Fitted vs residual")

    # QQ plot
    ax = axes[1]
    from scipy import stats
    resid = fitted_df["resid"].dropna().to_numpy()
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    ax.scatter(osm, osr, s=8, alpha=0.4, color=PALETTE[0], edgecolors="none")
    ax.plot(osm, slope * np.asarray(osm) + intercept, "k--", lw=0.8)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.set_title("Normal Q-Q")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, axes


# ── Station posterior distributions ────────────────────────────────────

def plot_station_posteriors(
    delta_st: np.ndarray,
    station_labels: list[str],
    *,
    figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "station_posteriors.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    r"""KDE overlays of δ_station posterior draws.

    Parameters
    ----------
    delta_st : (S, n_stations)
        Posterior draws of station effects.
    station_labels : length-n_stations list
    """
    out_dir = ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0, color="0.5", lw=0.6, ls="--", zorder=0)

    for i, sta in enumerate(station_labels):
        sns.kdeplot(delta_st[:, i], fill=True, alpha=0.2, lw=1.4,
                    ax=ax, color=PALETTE[i % len(PALETTE)], label=sta)

    ax.set_xlabel(r"$\delta_{\mathrm{station}}$ (log EDP)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.set_title("Station effect posteriors")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax


# ── Observed vs posterior mean ─────────────────────────────────────────

def plot_observed_vs_predicted(
    y_obs: np.ndarray,
    y_hat: np.ndarray,
    *,
    config_idx: np.ndarray | None = None,
    config_labels: list[str] | None = None,
    figsize: tuple[float, float] = (HALF_WIDTH, HALF_WIDTH),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "observed_vs_predicted.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter of observed y vs posterior mean ŷ with 1:1 line.

    Parameters
    ----------
    y_obs : (N,) observed values
    y_hat : (N,) posterior mean of linear predictor
    config_idx : (N,) optional config index for colouring
    config_labels : optional labels for legend
    """
    out_dir = ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=figsize)

    if config_idx is not None and config_labels is not None:
        for k, lbl in enumerate(config_labels):
            mask = config_idx == k
            ax.scatter(y_hat[mask], y_obs[mask], s=12, alpha=0.6,
                       color=PALETTE[k % len(PALETTE)], label=lbl,
                       edgecolors="none")
        ax.legend(fontsize=5, ncol=2, loc="upper left", markerscale=0.8)
    else:
        ax.scatter(y_hat, y_obs, s=12, alpha=0.5, color=PALETTE[0],
                   edgecolors="none")

    lo = min(float(np.nanmin(y_obs)), float(np.nanmin(y_hat)))
    hi = max(float(np.nanmax(y_obs)), float(np.nanmax(y_hat)))
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.7, zorder=0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(r"Posterior mean $\hat{y}$")
    ax.set_ylabel("Observed $y$")
    ax.set_title("Observed vs predicted")
    ax.set_aspect("equal")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax


# ── Raw data exploration ──────────────────────────────────────────────

def plot_raw_data(
    y: np.ndarray,
    config_idx: np.ndarray,
    config_labels: list[str],
    station_idx: np.ndarray,
    station_labels: list[str],
    *,
    figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "raw_data.pdf",
) -> tuple[plt.Figure, np.ndarray]:
    """Two-panel exploration of raw EDP data before modeling.

    Left:  stripplot of log-EDP by config, colored by station.
    Right: boxplot of log-EDP by config.
    """
    out_dir = ensure_dir(out_dir)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    df = pd.DataFrame({
        "y": y,
        "Config": [config_labels[i] for i in config_idx],
        "Station": [station_labels[i] for i in station_idx],
    })
    label_order = order_config_labels(config_labels)

    # Left: stripplot colored by station
    ax = axes[0]
    for i, sta in enumerate(station_labels):
        sub = df[df["Station"] == sta]
        x_pos = [label_order.index(c) for c in sub["Config"]]
        jitter = (i - len(station_labels) / 2) * 0.12
        ax.scatter(np.array(x_pos) + jitter, sub["y"], s=15, alpha=0.7,
                   color=PALETTE[i % len(PALETTE)], label=sta, edgecolors="none")
    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels(label_order, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("log EDP")
    ax.legend(fontsize=7, title="Station", title_fontsize=7)
    ax.set_title("Raw data by config (colored by station)")

    # Right: boxplot
    ax2 = axes[1]
    data_by_config = [df[df["Config"] == c]["y"].to_numpy() for c in label_order]
    bp = ax2.boxplot(data_by_config, labels=label_order, patch_artist=True,
                     widths=0.5, medianprops={"color": "0.2", "lw": 1.2})
    for patch, color_idx in zip(bp["boxes"], range(len(label_order))):
        patch.set_facecolor(PALETTE[color_idx % len(PALETTE)])
        patch.set_alpha(0.5)
    ax2.set_xticklabels(label_order, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("log EDP")
    ax2.set_title("Distribution by config")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, axes


# ── Posterior predictive density overlay ─────────────────────────────

def plot_ppc_density(
    y_obs: np.ndarray,
    y_rep: np.ndarray,
    *,
    n_rep_draws: int = 50,
    figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "ppc_density.pdf",
) -> tuple[plt.Figure, plt.Axes]:
    """Posterior predictive check: observed density vs replicated draws.

    Parameters
    ----------
    y_obs : (N,) observed
    y_rep : (S, N) posterior predictive draws
    n_rep_draws : int
        Number of replicated draws to overlay (thin for readability).
    """
    out_dir = ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=figsize)

    S = y_rep.shape[0]
    idx = np.random.default_rng(42).choice(S, size=min(n_rep_draws, S), replace=False)

    for i in idx:
        sns.kdeplot(y_rep[i], ax=ax, color=PALETTE[0], alpha=0.08, lw=0.5)

    sns.kdeplot(y_obs, ax=ax, color="0.1", lw=2.0, label="Observed")

    ax.set_xlabel("log EDP")
    ax.set_ylabel("Density")
    ax.set_title("Posterior predictive check")
    ax.legend(fontsize=7)

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, ax


# ── ArviZ diagnostic wrappers ────────────────────────────────────────

def plot_trace(
    idata,
    *,
    var_names: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "traceplot.pdf",
) -> plt.Figure:
    """ArviZ traceplot wrapper."""
    import arviz as az
    out_dir = ensure_dir(out_dir)

    if var_names is None:
        var_names = ["mu0", "sigma_run", "nu_minus2"]

    axes = az.plot_trace(idata, var_names=var_names, figsize=figsize, compact=True)
    fig = axes.ravel()[0].get_figure()

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig


def plot_pair(
    idata,
    *,
    var_names: list[str] | None = None,
    figsize: tuple[float, float] | None = (HALF_WIDTH, HALF_WIDTH),
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "pairplot.pdf",
) -> plt.Figure:
    """ArviZ pair plot for key parameters."""
    import arviz as az
    out_dir = ensure_dir(out_dir)

    if var_names is None:
        var_names = ["mu0", "sigma_run"]

    axes = az.plot_pair(
        idata, var_names=var_names,
        kind=["scatter", "kde"],
        marginals=True,
        figsize=figsize,
        scatter_kwargs={"alpha": 0.05, "s": 2},
    )
    fig = axes.ravel()[0].get_figure() if hasattr(axes, "ravel") else plt.gcf()

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig


def plot_forest_arviz(
    idata,
    *,
    var_names: list[str] | None = None,
    observed_y: np.ndarray | None = None,
    config_idx: np.ndarray | None = None,
    station_idx: np.ndarray | None = None,
    run_idx: np.ndarray | None = None,
    config_labels: list[str] | None = None,
    station_labels: list[str] | None = None,
    ci: float = 0.94,
    dot_alpha: float = 0.30,
    figsize: tuple[float, float] | None = None,
    out_dir: str | Path | None = None,
    prefix: str = "",
    filename: str = "forest_arviz.pdf",
) -> tuple[plt.Figure, np.ndarray]:
    """Custom forest plot with observed-data overlay.

    Draws posterior HDI bars and medians for each element of
    ``var_names``, with optional raw-data scatter behind them.

    Parameters
    ----------
    idata : InferenceData
        Fitted ArviZ InferenceData with ``posterior`` group.
    var_names : list[str]
        Variables to plot.  Default ``["mu_config", "delta_st"]``.
    observed_y : (N,) array, optional
        Observed log-EDP values.  When provided together with the
        index arrays, individual observations are shown as semi-
        transparent dots behind the posterior intervals.
    config_idx, station_idx, run_idx : (N,) int arrays, optional
        Factor indices aligned with *observed_y*.
    config_labels, station_labels : list[str], optional
        Human-readable labels for the factor levels.
    ci : float
        Credible interval width (default 0.94 = 94% HDI).
    """
    out_dir = ensure_dir(out_dir)

    if var_names is None:
        var_names = ["mu_config", "delta_st"]

    post = idata.posterior

    # Collect all parameter blocks
    blocks = []  # list of (labels, medians, lo, hi, raw_points)
    for vn in var_names:
        if vn not in post:
            continue
        draws = post[vn].values  # (chains, draws, n_levels) or (chains, draws)
        if draws.ndim == 2:
            draws = draws[:, :, np.newaxis]
        flat = draws.reshape(-1, draws.shape[-1])  # (S, n_levels)

        n_levels = flat.shape[1]
        med = np.median(flat, axis=0)
        lo, hi = _hdi_vec(flat, prob=ci)

        # Determine labels for this variable
        if vn.startswith("mu_config") and config_labels is not None:
            # mu_config_free has n_configs-1 levels
            lbls = config_labels[:n_levels]
        elif vn.startswith("delta_st") and station_labels is not None:
            lbls = station_labels[:n_levels]
        else:
            dims = post[vn].dims
            coord_name = dims[-1] if len(dims) > 2 else None
            if coord_name and coord_name in post.coords:
                lbls = [str(c) for c in post.coords[coord_name].values]
            else:
                lbls = [f"{vn}[{i}]" for i in range(n_levels)]

        # Compute raw data points per level
        raw_per_level = [None] * n_levels
        if observed_y is not None:
            global_mean = float(np.mean(observed_y))
            if vn.startswith("mu_config") and config_idx is not None:
                # For each config, show per-(station,run) mean minus global mean
                for k in range(n_levels):
                    mask = config_idx == k
                    if not np.any(mask):
                        continue
                    y_k = observed_y[mask]
                    # Group by (station, run) to get cell means
                    if station_idx is not None and run_idx is not None:
                        st_k = station_idx[mask]
                        rn_k = run_idx[mask]
                        cells = {}
                        for yi, si, ri in zip(y_k, st_k, rn_k):
                            key = (si, ri)
                            cells.setdefault(key, []).append(yi)
                        raw_per_level[k] = np.array(
                            [np.mean(v) for v in cells.values()]
                        ) - global_mean
                    else:
                        raw_per_level[k] = y_k - global_mean

            elif vn.startswith("delta_st") and station_idx is not None:
                # For each station, show per-(config,run) mean minus global mean
                for k in range(n_levels):
                    mask = station_idx == k
                    if not np.any(mask):
                        continue
                    y_k = observed_y[mask]
                    if config_idx is not None and run_idx is not None:
                        cf_k = config_idx[mask]
                        rn_k = run_idx[mask]
                        cells = {}
                        for yi, ci_val, ri in zip(y_k, cf_k, rn_k):
                            key = (ci_val, ri)
                            cells.setdefault(key, []).append(yi)
                        raw_per_level[k] = np.array(
                            [np.mean(v) for v in cells.values()]
                        ) - global_mean
                    else:
                        raw_per_level[k] = y_k - global_mean

        blocks.append((vn, lbls, med, lo, hi, raw_per_level))

    # Layout: one row per variable block, separated by a gap
    total_rows = sum(len(b[1]) for b in blocks)
    gap = 1.0
    total_h = total_rows + gap * (len(blocks) - 1)

    if figsize is None:
        figsize = (HALF_WIDTH, HALF_WIDTH)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    y_pos = 0.0
    yticks = []
    ytick_labels = []

    for b_idx, (vn, lbls, med, lo, hi, raw) in enumerate(blocks):
        n = len(lbls)
        positions = np.arange(n) + y_pos

        # Layer 0: raw data dots
        for k in range(n):
            if raw[k] is not None:
                pts = raw[k]
                jitter = np.random.default_rng(42 + k).uniform(
                    -0.2, 0.2, size=len(pts))
                ax.scatter(pts, positions[k] + jitter, s=12,
                           alpha=dot_alpha, color=PALETTE[2],
                           edgecolors="none", zorder=1,
                           label=("observed" if b_idx == 0 and k == 0
                                  else None))

        # Layer 1: HDI bars with caplines
        ax.errorbar(med, positions, xerr=[med - lo, hi - med],
                    fmt="o", ms=4.5, lw=1.6, capsize=3, capthick=1.2,
                    color=PALETTE[0], markeredgecolor="white",
                    markeredgewidth=0.4, zorder=4)

        yticks.extend(positions.tolist())
        ytick_labels.extend(lbls)

        y_pos += n + gap

    ax.axvline(0, color="0.4", lw=0.7, ls="--", zorder=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("posterior value (log EDP)")
    ax.grid(True, axis="x", alpha=0.25, lw=0.5, zorder=0)

    ci_pct = int(ci * 100)
    ax.set_title(f"{ci_pct}% HDI", fontsize=9)

    handles, labs = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=6, loc="lower right")

    savefig(fig, out_dir, filename, prefix=prefix)
    return fig, np.array([ax])
