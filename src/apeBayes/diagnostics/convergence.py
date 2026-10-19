"""MCMC convergence diagnostics: R-hat, ESS, divergences."""

from __future__ import annotations

from typing import Literal

import arviz as az
import pandas as pd
from arviz import InferenceData


def diagnostics_summary(
    idata: InferenceData,
    var_names: list[str] | None = None,
) -> pd.DataFrame:
    """ArviZ summary table for key model parameters."""
    if var_names is None:
        var_names = list(idata.posterior.data_vars)
    return az.summary(idata, var_names=var_names, round_to=4)


def divergences_count(idata: InferenceData) -> int:
    """Count divergent transitions."""
    if not hasattr(idata, "sample_stats"):
        return 0
    if "diverging" not in idata.sample_stats:
        return 0
    return int(idata.sample_stats["diverging"].values.sum())


def rhat_table(
    idata: InferenceData,
    *,
    var_names: list[str] | None = None,
    aggregate: bool = True,
    sort_desc: bool = True,
) -> pd.DataFrame:
    """R-hat diagnostics as a tidy DataFrame.

    Parameters
    ----------
    aggregate : bool
        If True, collapse indexed parameters (mu_config[0], …) into
        one row per base parameter with min/mean/max R-hat.
    """
    if var_names is None:
        var_names = list(idata.posterior.data_vars)

    summ = az.summary(idata, var_names=var_names, kind="diagnostics", round_to=6)

    if "r_hat" not in summ.columns:
        raise RuntimeError("ArviZ summary missing 'r_hat' column.")

    elements = summ.index.astype(str).to_numpy()
    rhat_vals = summ["r_hat"].to_numpy(dtype=float)

    if not aggregate:
        out = pd.DataFrame({"parameter": elements, "r_hat": rhat_vals})
        if sort_desc:
            out = out.sort_values("r_hat", ascending=False)
        return out.reset_index(drop=True)

    base_names = pd.Series(elements).str.replace(r"\[.*\]$", "", regex=True)
    df = pd.DataFrame({"parameter": base_names.to_numpy(), "r_hat": rhat_vals})
    out = df.groupby("parameter", as_index=False).agg(
        n_elements=("r_hat", "size"),
        r_hat_min=("r_hat", "min"),
        r_hat_mean=("r_hat", "mean"),
        r_hat_max=("r_hat", "max"),
    )
    if sort_desc:
        out = out.sort_values("r_hat_max", ascending=False)
    return out.reset_index(drop=True)


def ess_table(
    idata: InferenceData,
    *,
    var_names: list[str] | None = None,
    aggregate: bool = True,
    sort_by: Literal["ess_bulk_min", "ess_tail_min", "parameter"] = "ess_bulk_min",
) -> pd.DataFrame:
    """Effective sample size diagnostics as a tidy DataFrame."""
    if var_names is None:
        var_names = list(idata.posterior.data_vars)

    summ = az.summary(idata, var_names=var_names, kind="diagnostics", round_to=6)

    for col in ("ess_bulk", "ess_tail"):
        if col not in summ.columns:
            raise RuntimeError(f"ArviZ summary missing '{col}' column.")

    elements = summ.index.astype(str).to_numpy()
    ess_bulk = summ["ess_bulk"].to_numpy(dtype=float)
    ess_tail = summ["ess_tail"].to_numpy(dtype=float)

    if not aggregate:
        out = pd.DataFrame({
            "parameter": elements,
            "ess_bulk": ess_bulk,
            "ess_tail": ess_tail,
        })
        return out.sort_values("ess_bulk", ascending=True).reset_index(drop=True)

    base_names = pd.Series(elements).str.replace(r"\[.*\]$", "", regex=True)
    df = pd.DataFrame({
        "parameter": base_names.to_numpy(),
        "ess_bulk": ess_bulk,
        "ess_tail": ess_tail,
    })
    out = df.groupby("parameter", as_index=False).agg(
        n_elements=("ess_bulk", "size"),
        ess_bulk_min=("ess_bulk", "min"),
        ess_bulk_mean=("ess_bulk", "mean"),
        ess_bulk_max=("ess_bulk", "max"),
        ess_tail_min=("ess_tail", "min"),
        ess_tail_mean=("ess_tail", "mean"),
        ess_tail_max=("ess_tail", "max"),
    )

    sort_map = {
        "ess_bulk_min": ["ess_bulk_min"],
        "ess_tail_min": ["ess_tail_min"],
        "parameter": ["parameter"],
    }
    out = out.sort_values(sort_map.get(sort_by, ["ess_bulk_min"]), ascending=True)
    return out.reset_index(drop=True)


# ── Convergence gate ────────────────────────────────────────────────────────

RHAT_MAX = 1.01
ESS_MIN = 400
DIVERGENCES_MAX = 0

_ATTR_PREFIX = "convergence_"


class ConvergenceWarning(UserWarning):
    """The posterior failed the convergence gate."""


def convergence_report(
    idata: InferenceData,
    *,
    var_names: list[str] | None = None,
    rhat_max: float = RHAT_MAX,
    ess_min: float = ESS_MIN,
    divergences_max: int = DIVERGENCES_MAX,
) -> dict:
    """Worst-case convergence statistics over the posterior and a verdict.

    Parameters
    ----------
    idata : InferenceData
    var_names : list[str], optional
        Variables to check. Defaults to every posterior variable.
        Elements with undefined R-hat (constants, e.g. the anchored
        λ_ref = 1 or the last element of a sum-to-zero vector that is
        exactly determined) are ignored.
    rhat_max, ess_min, divergences_max
        Thresholds: converged ⇔ max R-hat ≤ ``rhat_max`` and min bulk and
        tail ESS ≥ ``ess_min`` and divergences ≤ ``divergences_max``.

    Returns
    -------
    dict
        ``max_rhat``, ``min_ess_bulk``, ``min_ess_tail``, ``divergences``,
        ``converged`` (bool), the thresholds, and ``worst_rhat_var`` /
        ``worst_ess_var`` naming the offending parameter.
    """
    import numpy as np

    post = idata.posterior
    if var_names is None:
        var_names = list(post.data_vars)
    import warnings

    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        # constant elements (anchored λ_ref, zero arrays) give 0/0 here
        warnings.simplefilter("ignore", RuntimeWarning)
        rhat = az.rhat(post, var_names=var_names)
        ess_b = az.ess(post, var_names=var_names, method="bulk")
        ess_t = az.ess(post, var_names=var_names, method="tail")

    def _worst(ds: object, fn: object) -> tuple[float, str]:
        best_val, best_var = float("nan"), ""
        for v in ds.data_vars:
            vals = np.asarray(ds[v].values, dtype=float).ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            x = float(fn(vals))
            if np.isnan(best_val) or fn([x, best_val]) == x:
                best_val, best_var = x, str(v)
        return best_val, best_var

    max_rhat, worst_rhat_var = _worst(rhat, np.max)
    min_bulk, worst_ess_var = _worst(ess_b, np.min)
    min_tail, worst_tail_var = _worst(ess_t, np.min)
    n_div = divergences_count(idata)
    converged = bool(
        (np.isnan(max_rhat) or max_rhat <= rhat_max)
        and (np.isnan(min_bulk) or min_bulk >= ess_min)
        and (np.isnan(min_tail) or min_tail >= ess_min)
        and n_div <= divergences_max
    )
    if not np.isnan(min_tail) and (np.isnan(min_bulk) or min_tail < min_bulk):
        worst_ess_var = worst_tail_var
    return {
        "max_rhat": max_rhat,
        "min_ess_bulk": min_bulk,
        "min_ess_tail": min_tail,
        "divergences": n_div,
        "converged": converged,
        "worst_rhat_var": worst_rhat_var,
        "worst_ess_var": worst_ess_var,
        "rhat_threshold": float(rhat_max),
        "ess_threshold": float(ess_min),
        "divergences_threshold": int(divergences_max),
    }


def attach_convergence(idata: InferenceData, **kwargs: float) -> dict:
    """Compute :func:`convergence_report` and store it in ``idata.attrs``.

    Keys are prefixed ``convergence_``; booleans are stored as 0/1 so the
    attrs survive NetCDF.
    """
    rep = convergence_report(idata, **kwargs)
    for k, v in rep.items():
        idata.attrs[_ATTR_PREFIX + k] = int(v) if isinstance(v, bool) else v
    return rep


def stored_convergence(idata: InferenceData) -> dict | None:
    """Read a report written by :func:`attach_convergence`, or None."""
    attrs = getattr(idata, "attrs", {}) or {}
    if _ATTR_PREFIX + "converged" not in attrs:
        return None
    rep = {
        k[len(_ATTR_PREFIX):]: v for k, v in attrs.items() if k.startswith(_ATTR_PREFIX)
    }
    rep["converged"] = bool(rep["converged"])
    return rep


def convergence_message(rep: dict) -> str:
    """One-line human-readable summary of a convergence report."""
    return (
        f"max R-hat {rep['max_rhat']:.3f} ({rep.get('worst_rhat_var', '')}; "
        f"gate ≤ {rep['rhat_threshold']:g}), min ESS bulk/tail "
        f"{rep['min_ess_bulk']:.0f}/{rep['min_ess_tail']:.0f} "
        f"({rep.get('worst_ess_var', '')}; gate ≥ {rep['ess_threshold']:g}), "
        f"divergences {rep['divergences']} (gate ≤ {rep['divergences_threshold']})"
    )
