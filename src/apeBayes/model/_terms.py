"""Shared PyMC building blocks for the model builders.

Sum-to-zero parameterisations
-----------------------------
``zero_sum="last"`` (the v4–v9 default, kept so published fits reproduce)
draws the first n−1 levels freely and sets the last level to minus their
sum. Its prior is asymmetric: the last level's prior variance is (n−1)
times that of the others (√15 ≈ 3.9× in SD for the 16th config, √5× for
the 6th rupture), and the scale is learnt from n−1 levels, so results
depend on which level happens to be sorted last.

``zero_sum="zsn"`` uses ``pm.ZeroSumNormal``: every level has the same
marginal prior, the constraint is imposed symmetrically, and the result
does not depend on level order. For the station×rupture interaction it
imposes the zero sum on **both** axes (each station's and each rupture's
interaction sums to zero), so γ cannot absorb a station or rupture main
effect.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pymc as pm

if TYPE_CHECKING:
    import pytensor.tensor as pt

ZeroSum = Literal["last", "zsn"]


def check_zero_sum(zero_sum: str) -> None:
    """Validate a ``zero_sum`` option."""
    if zero_sum not in ("last", "zsn"):
        raise ValueError(f"zero_sum must be 'last' or 'zsn', got {zero_sum!r}")


def hand_rolled_zero_sum(free: pt.TensorVariable) -> pt.TensorVariable:
    """Append ``-sum(free)`` so the vector sums to zero ("last" scheme)."""
    return pm.math.concatenate([free, -pm.math.sum(free)[None]])


def anchored_log_loading(
    name: str, sigma: float, n_levels: int, ref_idx: int, free_dims: str,
) -> pt.TensorVariable:
    """exp(log-loading) per level with the reference level fixed at 1."""
    free = pm.Normal(name, mu=0.0, sigma=sigma, dims=free_dims)
    zero = pm.math.zeros((1,))
    if ref_idx == 0:
        log = pm.math.concatenate([zero, free])
    elif ref_idx == n_levels - 1:
        log = pm.math.concatenate([free, zero])
    else:
        log = pm.math.concatenate([free[:ref_idx], zero, free[ref_idx:]])
    return pm.math.exp(log)


def residual_scales(
    *,
    pooling: str,
    tau: float,
    tau_dist: str,
    log_mean: float,
    log_sd: float,
    sigma_eps_prior: float,
) -> pt.TensorVariable:
    """Per-config residual SD ``sigma_eps_config`` (dims "Config")."""
    if pooling == "partial":
        log_bar = pm.Normal("log_sigma_eps_bar", mu=log_mean, sigma=log_sd)
        if tau_dist == "halfcauchy":
            tau_rv = pm.HalfCauchy("tau_sigma_eps", beta=tau)
        else:
            tau_rv = pm.HalfNormal("tau_sigma_eps", sigma=tau)
        z = pm.Normal("z_sigma_eps", mu=0.0, sigma=1.0, dims="Config")
        return pm.Deterministic(
            "sigma_eps_config", pm.math.exp(log_bar + tau_rv * z), dims="Config",
        )
    return pm.HalfNormal("sigma_eps_config", sigma=sigma_eps_prior, dims="Config")


def likelihood_term(
    likelihood: str,
    mu: pt.TensorVariable,
    sigma: pt.TensorVariable,
    y: np.ndarray,
    nu_prior_lambda: float,
) -> None:
    """Student-t (ν = 2 + Exp) or Gaussian observation model on ``y_obs``."""
    if likelihood == "student_t":
        nu_minus2 = pm.Exponential("nu_minus2", lam=1.0 / nu_prior_lambda)
        pm.StudentT("y_obs", nu=nu_minus2 + 2.0, mu=mu, sigma=sigma,
                    observed=y, dims="obs_id")
    elif likelihood == "gaussian":
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y, dims="obs_id")
    else:
        raise ValueError(f"Unknown likelihood: {likelihood!r}")
