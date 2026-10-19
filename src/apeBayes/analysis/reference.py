r"""
Record-to-record SD of the reference configuration, estimated from its own data.

σ_GM in v8 is the record variance *shared across configurations*, weighted
by every config's residual scale; it is not the reference configuration's
own record-to-record SD, and on the Paper 1 PFA-roof data the two differ by
a factor 1.5 (0.24 vs 0.36). This module estimates the latter directly from
the reference configuration's n_stations × n_runs values (2026-10-19
critique, ``paired.py``):

.. math::
    y_{ref,sr} = m_s + b_r + e_{sr},\quad b_r \sim N(0, \sigma_b),\
    e_{sr} \sim N(0, \sigma_e),\quad
    \sigma_{RTR,ref} = \sqrt{\sigma_b^2 + \sigma_e^2},

with m_s ~ N(0, 10), σ_b, σ_e ~ HalfNormal(1). It is a model-light
cross-check of the β denominator, not a replacement for σ_GM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pymc as pm

if TYPE_CHECKING:
    from ..config import SamplingConfig
    from ..data import EpistemicDataset


def reference_record_model(data: EpistemicDataset, *, prior_sd: float = 1.0) -> pm.Model:
    """PyMC model for the reference config's record-to-record SD."""
    sel = data.config_idx == data.ref_idx
    if sel.sum() < 4:
        raise ValueError("The reference configuration has fewer than 4 observations.")
    s = data.station_idx[sel]
    r = data.run_idx[sel]
    y = data.y[sel]
    with pm.Model() as m:
        m_s = pm.Normal("m_s", 0.0, 10.0, shape=data.n_stations)
        sb = pm.HalfNormal("sigma_b_ref", prior_sd)
        se = pm.HalfNormal("sigma_e_ref", prior_sd)
        b = pm.Normal("z_b", 0.0, 1.0, shape=data.n_runs) * sb
        pm.Normal("y_ref", m_s[s] + b[r], se, observed=y)
        pm.Deterministic("sigma_rtr_ref", pm.math.sqrt(sb ** 2 + se ** 2))
    return m


def reference_record_sd(
    data: EpistemicDataset,
    sampling: SamplingConfig,
    *,
    prior_sd: float = 1.0,
) -> np.ndarray:
    """Posterior draws of σ_RTR,ref (flattened over chains)."""
    from ..model.sampling import sample_model

    idata = sample_model(
        reference_record_model(data, prior_sd=prior_sd), sampling,
        prior_predictive=False, posterior_predictive=False,
    )
    return idata.posterior["sigma_rtr_ref"].values.reshape(-1)


def reference_record_sd_moment(data: EpistemicDataset) -> float:
    """Moment estimate: SD of the reference values after removing station means.

    Uses ``ddof = n_stations`` (one mean per station), i.e. the pooled
    within-station SD over ruptures. Quick and prior-free; no interval.
    """
    sel = data.config_idx == data.ref_idx
    y, s = data.y[sel], data.station_idx[sel]
    resid = y - np.array([y[s == j].mean() for j in s])
    dof = len(y) - len(np.unique(s))
    return float(np.sqrt(np.sum(resid ** 2) / max(dof, 1)))
