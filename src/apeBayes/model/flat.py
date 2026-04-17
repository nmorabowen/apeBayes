"""
Flat-configuration hierarchical model.

This is the "Path A" model: one effect per configuration (TierCase),
with no factorial structure built into the model itself.  The factorial
decomposition happens post-hoc in the analysis layer.

Supports:
  - Student-t or Gaussian likelihood
  - Heteroskedastic or homoskedastic residuals
  - Non-centered parameterization for run effects (fixes the funnel)
  - Sum-to-zero constraints via N-1 free parameters
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pymc as pm

if TYPE_CHECKING:
    import numpy as np

    from ..config import ModelConfig
    from ..data import EpistemicDataset
    from ..posterior import PosteriorAccessor


class FlatConfigModel:
    """Build a flat-configuration hierarchical model.

    Parameters
    ----------
    likelihood : {"student_t", "gaussian"}, optional
        Override ``cfg.likelihood`` if provided.
    heteroskedastic : bool, optional
        Override ``cfg.heteroskedastic`` if provided.
    """

    def __init__(
        self,
        *,
        likelihood: str | None = None,
        heteroskedastic: bool | None = None,
    ) -> None:
        self._likelihood_override = likelihood
        self._hetero_override = heteroskedastic

    @property
    def description(self) -> str:
        """Return a human-readable description of the model variant."""
        parts = []
        lik = self._likelihood_override or "config"
        het = self._hetero_override
        if lik != "config":
            parts.append(lik)
        if het is not None:
            parts.append("hetero" if het else "homo")
        return f"Flat({', '.join(parts)})" if parts else "Flat(default)"

    def sigma_GM(self, post: PosteriorAccessor) -> np.ndarray:
        """σ_GM = σ_src for flat-configuration models (v1–v3).

        No station×rupture interaction is modelled, so the single-station
        aleatory SD reduces to pure source variability. See
        ``uncertanty_measures.md`` §7.
        """
        return post.sigma_src()

    def build(
        self,
        data: EpistemicDataset,
        cfg: ModelConfig,
    ) -> pm.Model:
        """Construct the PyMC model."""
        p = cfg.priors
        likelihood = self._likelihood_override or cfg.likelihood
        hetero = self._hetero_override if self._hetero_override is not None else cfg.heteroskedastic

        coords = data.coords_dict()

        with pm.Model(coords=coords) as model:

            # ── Intercept ────────────────────────────────────────────────
            mu0 = pm.Normal("mu0", mu=0.0, sigma=p.sigma_intercept)

            # ── Configuration effects (sum-to-zero via N-1 free) ─────────
            n_cfg = data.n_configs
            if n_cfg > 1:
                mu_cfg_free = pm.Normal(
                    "mu_config_free",
                    mu=0.0,
                    sigma=p.sigma_config,
                    dims="Config_free",
                )
                mu_cfg_last = -pm.math.sum(mu_cfg_free)
                mu_config = pm.Deterministic(
                    "mu_config",
                    pm.math.concatenate([mu_cfg_free, mu_cfg_last[None]]),
                    dims="Config",
                )
            else:
                mu_config = pm.Deterministic(
                    "mu_config",
                    pm.math.zeros((1,)),
                    dims="Config",
                )

            # ── Station effects (sum-to-zero via N-1 free) ───────────────
            n_st = data.n_stations
            if n_st > 1:
                delta_free = pm.Normal(
                    "delta_st_free",
                    mu=0.0,
                    sigma=p.sigma_station,
                    dims="Station_free",
                )
                delta_last = -pm.math.sum(delta_free)
                delta_st = pm.Deterministic(
                    "delta_st",
                    pm.math.concatenate([delta_free, delta_last[None]]),
                    dims="Station",
                )
            else:
                delta_st = pm.Deterministic(
                    "delta_st",
                    pm.math.zeros((1,)),
                    dims="Station",
                )

            # ── Run random effects (NON-CENTERED parameterization) ───────
            sigma_run = pm.HalfNormal("sigma_run", sigma=p.sigma_src)
            n_rk = data.n_runs
            if n_rk > 1:
                # Non-centered: sample z ~ N(0,1), then b = sigma_run * z
                z_run_free = pm.Normal(
                    "z_run_free",
                    mu=0.0,
                    sigma=1.0,
                    dims="Run_free",
                )
                # Sum-to-zero: last element = -sum(free)
                z_run_last = -pm.math.sum(z_run_free)
                z_run = pm.math.concatenate([z_run_free, z_run_last[None]])
                b_run = pm.Deterministic(
                    "b_run",
                    sigma_run * z_run,
                    dims="Run",
                )
            else:
                b_run = pm.Deterministic(
                    "b_run",
                    pm.math.zeros((1,)),
                    dims="Run",
                )

            # ── Residual SD ──────────────────────────────────────────────
            if hetero:
                sigma_eps_config = pm.HalfNormal(
                    "sigma_eps_config",
                    sigma=p.sigma_eps,
                    dims="Config",
                )
                sigma_obs = sigma_eps_config[data.config_idx]
            else:
                sigma_eps = pm.HalfNormal("sigma_eps", sigma=p.sigma_eps)
                sigma_obs = sigma_eps

            # ── Linear predictor ─────────────────────────────────────────
            mu = (
                mu0
                + mu_config[data.config_idx]
                + delta_st[data.station_idx]
                + b_run[data.run_idx]
            )

            # ── Likelihood ───────────────────────────────────────────────
            if likelihood == "student_t":
                nu_minus2 = pm.Exponential(
                    "nu_minus2",
                    lam=1.0 / p.nu_prior_lambda,
                )
                nu = nu_minus2 + 2.0
                pm.StudentT(
                    "y_obs",
                    nu=nu,
                    mu=mu,
                    sigma=sigma_obs,
                    observed=data.y,
                    dims="obs_id",
                )
            elif likelihood == "gaussian":
                pm.Normal(
                    "y_obs",
                    mu=mu,
                    sigma=sigma_obs,
                    observed=data.y,
                    dims="obs_id",
                )
            else:
                raise ValueError(f"Unknown likelihood: {likelihood!r}")

        return model
