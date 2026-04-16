"""
Hierarchical-configuration model with τ_config shrinkage.

Extends the flat model by learning the config-effect scale τ_config from
data rather than fixing it to the prior σ_config.  This adds partial
pooling across configurations, which:

  1. Regularizes individual μ_config[k] estimates toward zero
  2. Improves MCMC mixing (less freedom → fewer ridges in posterior)
  3. Yields τ_config as a direct measure of total config-level epistemic
     variability — useful for reporting

Model:
  μ_config[k] ~ Normal(0, τ_config)
  τ_config    ~ HalfNormal(σ_τ)        # σ_τ from PriorConfig.sigma_config

Everything else is identical to FlatConfigModel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pymc as pm

if TYPE_CHECKING:
    from ..config import ModelConfig
    from ..data import EpistemicDataset


class HierarchicalConfigModel:
    """Build a hierarchical-configuration model with learned τ_config.

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
        return f"Hierarchical({', '.join(parts)})" if parts else "Hierarchical(default)"

    def build(
        self,
        data: EpistemicDataset,
        cfg: ModelConfig,
    ) -> pm.Model:
        """Construct the PyMC model with hierarchical config shrinkage."""
        p = cfg.priors
        likelihood = self._likelihood_override or cfg.likelihood
        hetero = (
            self._hetero_override
            if self._hetero_override is not None
            else cfg.heteroskedastic
        )

        coords = data.coords_dict()

        with pm.Model(coords=coords) as model:

            # ── Intercept ────────────────────────────────────────────────
            mu0 = pm.Normal("mu0", mu=0.0, sigma=p.sigma_intercept)

            # ── Hierarchical config-effect scale ─────────────────────────
            # τ_config is *learned* from data — the key difference vs flat
            tau_config = pm.HalfNormal("tau_config", sigma=p.sigma_config)

            # ── Configuration effects (sum-to-zero via N-1 free) ─────────
            n_cfg = data.n_configs
            if n_cfg > 1:
                # Non-centered parameterization for config effects too
                z_config_free = pm.Normal(
                    "z_config_free",
                    mu=0.0,
                    sigma=1.0,
                    dims="Config_free",
                )
                z_config_last = -pm.math.sum(z_config_free)
                z_config = pm.math.concatenate([z_config_free, z_config_last[None]])
                mu_config = pm.Deterministic(
                    "mu_config",
                    tau_config * z_config,
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
            sigma_run = pm.HalfNormal("sigma_run", sigma=p.sigma_run)
            n_rk = data.n_runs
            if n_rk > 1:
                z_run_free = pm.Normal(
                    "z_run_free",
                    mu=0.0,
                    sigma=1.0,
                    dims="Run_free",
                )
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
