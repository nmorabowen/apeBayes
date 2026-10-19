r"""
Variance budget and component analysis.

Decomposes the variance of a single observation into its components, per
posterior draw and on one consistent footing (see :func:`variance_budget`):

  1. Spread across configuration means  Var(μ_config)    (finite population)
  2. Spread across station means        Var(δ_s)         (finite population)
  3. Rupture term                       E[λ²]·σ_run²     (superpopulation)
  4. Station×rupture interaction        E[ξ²]·σ_inter²   (superpopulation, v8+)
  5. Case×record terms                  E[σ_u²], E[σ_v²] (v10)
  6. Residual                           E[σ_eps²]·ν/(ν−2)

The fixed factors (configs, stations) can only be summarised by the
variance of their realised levels; the random factors are summarised by
their model scale (the variance a *new* rupture / cell would carry), with
the per-Case loadings averaged over the observation mix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import EPS, student_t_var_factor


def budget_from_components(
    components: list[tuple[str, np.ndarray]],
    ci: tuple[float, float] = (0.05, 0.95),
) -> pd.DataFrame:
    """Summarise per-draw variance components into a budget table.

    Parameters
    ----------
    components : list of (name, (S,) array)
        Variance of each component, per posterior draw, all on the same
        footing (see :func:`variance_budget`).
    ci : tuple
        Credible-interval quantiles.

    Returns
    -------
    pd.DataFrame
        Columns ``component``, ``var_med``, ``var_lo``, ``var_hi`` and the
        share of the total computed **per draw** and then summarised:
        ``pct_med``, ``pct_lo``, ``pct_hi`` (median and interval of the
        per-draw share) and ``pct_mean`` (posterior-mean share; these sum
        to exactly 100 and are what the stacked-bar plots draw).
    """
    ci_lo, ci_hi = ci
    V = np.column_stack([
        np.broadcast_to(np.asarray(v, dtype=float), np.shape(components[0][1]))
        for _, v in components
    ])
    total = np.maximum(np.nansum(V, axis=1, keepdims=True), EPS)
    share = 100.0 * V / total
    return pd.DataFrame({
        "component": [n for n, _ in components],
        "var_med": np.nanmedian(V, axis=0),
        "var_lo": np.nanquantile(V, ci_lo, axis=0),
        "var_hi": np.nanquantile(V, ci_hi, axis=0),
        "pct_med": np.nanmedian(share, axis=0),
        "pct_lo": np.nanquantile(share, ci_lo, axis=0),
        "pct_hi": np.nanquantile(share, ci_hi, axis=0),
        "pct_mean": np.nanmean(share, axis=0),
    })


def _obs_var(effect_obs: np.ndarray) -> np.ndarray:
    """Variance over observations, per draw, of an ``(S, N)`` effect."""
    return np.var(effect_obs, axis=1, ddof=0)


def variance_budget(
    mu_config: np.ndarray,
    delta_st: np.ndarray,
    sigma_run: np.ndarray,
    sigma_eps: np.ndarray,
    config_idx: np.ndarray,
    ci: tuple[float, float] = (0.05, 0.95),
    *,
    nu: np.ndarray | None = None,
    gamma_sr: np.ndarray | None = None,
    station_idx: np.ndarray | None = None,
    run_loading_obs: np.ndarray | None = None,
    sigma_inter: np.ndarray | None = None,
    inter_loading_obs: np.ndarray | None = None,
    sigma_u_obs: np.ndarray | None = None,
    sigma_v_obs: np.ndarray | None = None,
) -> pd.DataFrame:
    r"""Posterior variance budget on a consistent per-observation footing.

    Every component is the variance it contributes to one observation,
    averaged over the observation mix of the design, per posterior draw:

    - **Config spread**: :math:`\mathrm{Var}_i(\mu_{k(i)})`, the
      finite-population variance of the configuration effects over the
      observations (configs are the fixed factor under study).
    - **Station spread**: :math:`\mathrm{Var}_i(\delta_{s(i)})`, the same
      for the fixed stations. Without ``station_idx`` the unweighted
      variance across station levels is used (identical when balanced).
    - **Run dispersion**: superpopulation variance of the loaded rupture
      term, :math:`\overline{\lambda_{c(i)}^2}\,\sigma_{run}^2` (λ ≡ 1 when
      ``run_loading_obs`` is None, i.e. v1–v3).
    - **Interaction**: :math:`\overline{\xi_{c(i)}^2}\,\sigma_{inter}^2`
      when ``sigma_inter`` is given (superpopulation, ξ ≡ 1 for v8).
      Legacy call: with only ``gamma_sr``, the finite-population variance
      of the realised γ cells is used instead.
    - **Case×rupture / Case×station×rupture** (v10):
      :math:`\overline{\sigma_{u,c(i)}^2}`, :math:`\overline{\sigma_{v,c(i)}^2}`.
    - **Residual**: :math:`\overline{\sigma_{\varepsilon,k(i)}^2}` times the
      Student-t variance factor ν/(ν−2).

    Shares are computed per draw and then summarised (see
    :func:`budget_from_components`); they are fractions of each draw's own
    total, not ratios of posterior medians.

    Parameters
    ----------
    mu_config : (S, K) configuration effects.
    delta_st : (S, J) station effects.
    sigma_run : (S,) run-level SD.
    sigma_eps : (S,) or (S, K) residual SD.
    config_idx : (N,) per-observation config index.
    ci : credible-interval quantiles.
    nu : (S,) Student-t degrees of freedom, or None.
    gamma_sr : (S, n_inter), optional
        Legacy: realised interaction cells (used only without ``sigma_inter``).
    station_idx : (N,), optional
        Per-observation station index.
    run_loading_obs : (S, N), optional
        λ_{c(i)} per draw and observation.
    sigma_inter : (S,), optional
        Interaction scale (preferred over ``gamma_sr``).
    inter_loading_obs : (S, N), optional
        ξ_{c(i)} (v9).
    sigma_u_obs, sigma_v_obs : (S, N), optional
        v10 Case-level SDs per observation.
    """
    if station_idx is None:
        var_station = np.var(delta_st, axis=1, ddof=0)
    else:
        var_station = _obs_var(delta_st[:, station_idx])

    lam2 = 1.0 if run_loading_obs is None else np.mean(run_loading_obs ** 2, axis=1)
    var_run = lam2 * sigma_run ** 2

    t_factor = np.ones_like(sigma_run, dtype=float)
    if nu is not None:
        t_factor = student_t_var_factor(nu)
    if sigma_eps.ndim == 2:
        var_eps = np.mean(sigma_eps[:, config_idx] ** 2, axis=1) * t_factor
    else:
        var_eps = sigma_eps ** 2 * t_factor

    comps: list[tuple[str, np.ndarray]] = [
        ("Config spread  Var(μ_config)", _obs_var(mu_config[:, config_idx])),
        ("Station spread Var(δ_s)", var_station),
        ("Run dispersion λ²σ²_run", var_run),
    ]
    if sigma_inter is not None:
        xi2 = 1.0 if inter_loading_obs is None else np.mean(inter_loading_obs ** 2, axis=1)
        comps.append(("Interaction   ξ²σ²_inter", xi2 * sigma_inter ** 2))
    elif gamma_sr is not None:
        comps.append(("Interaction   Var(γ_sr)", np.var(gamma_sr, axis=1, ddof=0)))
    if sigma_u_obs is not None:
        comps.append(("Case×rupture  σ²_u", np.mean(sigma_u_obs ** 2, axis=1)))
    if sigma_v_obs is not None:
        comps.append(("Case×sta×rup  σ²_v", np.mean(sigma_v_obs ** 2, axis=1)))
    comps.append(("Residual disp. σ²_ε", var_eps))
    return budget_from_components(comps, ci=ci)


def variance_components(
    sigma_run: np.ndarray,
    sigma_eps: np.ndarray,
    labels: list[str],
    ci: tuple[float, float] = (0.05, 0.95),
    *,
    nu: np.ndarray | None = None,
    subset_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    """Posterior summaries of scale parameters.

    Returns one row per component: sigma_run, sigma_eps (or
    sigma_eps_config[k] for each label), and optionally nu.

    Parameters
    ----------
    labels : list[str]
        Config labels of the rows to report.
    subset_idx : (M,) int array, optional
        Column of ``sigma_eps`` belonging to each label. Required when
        ``labels`` is a subset or reordering of the configs; when None,
        ``labels[i]`` is taken to be column ``i``.
    """
    ci_lo, ci_hi = ci

    def _row(name: str, v: np.ndarray) -> dict:
        return {
            "component": name,
            "med": float(np.median(v)),
            "lo": float(np.quantile(v, ci_lo)),
            "hi": float(np.quantile(v, ci_hi)),
        }

    rows = [_row("sigma_run", sigma_run)]

    if sigma_eps.ndim == 1:
        rows.append(_row("sigma_eps", sigma_eps))
    else:
        idx = np.arange(len(labels)) if subset_idx is None else np.asarray(subset_idx)
        if len(idx) != len(labels):
            raise ValueError("subset_idx and labels must have the same length.")
        for k, lbl in zip(idx, labels, strict=True):
            rows.append(_row(f"sigma_eps[{lbl}]", sigma_eps[:, k]))

    if nu is not None:
        rows.append(_row("nu", nu))

    return pd.DataFrame(rows)
