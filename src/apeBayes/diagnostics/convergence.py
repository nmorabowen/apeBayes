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
