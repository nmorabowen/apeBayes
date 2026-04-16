"""
Random-slopes model with explicit station×rupture interaction.

Extends RandomSlopesModel (v4–v7) by adding a crossed interaction term:
  y_i = μ₀ + μ_config[k] + δ_station[j] + γ_sr[j,r] + λ_case[c(k)] · b_rupture[r] + ε_i

The interaction γ_sr captures station-specific rupture responses that
cannot be explained by additive station + rupture effects. In v7, this
variance was absorbed into the heteroskedastic residual σ_ε. In v8,
it is explicitly modeled as a random effect with shared scale σ_inter.

v9 extension (interaction_loading=True):
  y_i = μ₀ + μ_config[k] + δ_station[j]
        + ξ_case[c(k)] · γ_sr[j,r]
        + λ_case[c(k)] · b_rupture[r] + ε_i

  ξ_case is the per-Case loading on the station×rupture interaction,
  analogous to how λ_case modulates the run (rupture) effect. A linear
  model (Case A) has narrow spectral bandwidth and should see weaker
  station×rupture interaction than a nonlinear model (Case D).

  Parameterization: log(ξ) ~ N(0, σ_ξ), with ξ_ref = 1 (log ξ_ref = 0).
  This mirrors the λ_case parameterization exactly.

Identifiability:
  - 3 stations × 6 ruptures = 18 cells
  - γ must sum to zero within each station and within each rupture
  - That's (3 + 6 - 1) = 8 constraints → 10 free parameters
  - With 16 obs per cell, identifiability is adequate
  - ξ_case adds 3 free parameters (A, B, C relative to D)
  - The within-config interaction spread identifies ξ (analogous to
    how within-config run spread identifies λ)

Implementation:
  Rather than enforcing double sum-to-zero constraints (which creates
  a complex parameterization), we use a simpler approach:
  - Create an 18-level random effect indexed by (station, rupture) pairs
  - Apply a single sum-to-zero constraint (last = -sum of rest)
  - The main station and rupture effects absorb the marginal means,
    so γ estimates the residual interaction in practice
  - σ_inter ~ HalfNormal controls shrinkage toward zero
  - When interaction_loading=True, ξ_case scales γ_sr per Case
"""

from __future__ import annotations

import numpy as np
import pymc as pm

from ..config import ModelConfig
from ..data import EpistemicDataset
from .random_slopes import RandomSlopesModel


class RandomSlopesInteractionModel(RandomSlopesModel):
    """Random-slopes model with station×rupture interaction.

    Parameters
    ----------
    sigma_lambda : float
        Prior SD for log(λ_case). Default 0.5.
    sigma_inter : float
        Prior SD for the station×rupture interaction scale.
        Default 0.3 — moderately informative, expecting the
        interaction to be smaller than main effects.
    centered_runs : bool
        If True, center the run parameterization.
    interaction_loading : bool
        If True, add per-Case loading ξ_case on the interaction
        γ_sr, analogous to λ_case on b_run. Default False (v8).
        Set True for v9.
    sigma_xi : float
        Prior SD for log(ξ_case). Default 0.5. Only used when
        interaction_loading=True.
    """

    def __init__(
        self,
        *,
        sigma_lambda: float = 0.5,
        sigma_inter: float = 0.3,
        centered_runs: bool = False,
        likelihood: str | None = None,
        heteroskedastic: bool | None = None,
        interaction_loading: bool = False,
        sigma_xi: float = 0.5,
    ):
        super().__init__(
            sigma_lambda=sigma_lambda,
            centered_runs=centered_runs,
            likelihood=likelihood,
            heteroskedastic=heteroskedastic,
        )
        self._sigma_inter = sigma_inter
        self._interaction_loading = interaction_loading
        self._sigma_xi = sigma_xi

    @property
    def description(self) -> str:
        base = super().description
        tag = f"RandomSlopesInteraction(σ_γ={self._sigma_inter}"
        if self._interaction_loading:
            tag += f", ξ_loading=True, σ_ξ={self._sigma_xi}"
        tag += ", "
        return base.replace("RandomSlopes(", tag)

    def build(
        self,
        data: EpistemicDataset,
        cfg: ModelConfig,
    ) -> pm.Model:
        """Construct the PyMC model with station×rupture interaction."""
        p = cfg.priors
        likelihood = self._likelihood_override or cfg.likelihood
        hetero = (
            self._hetero_override
            if self._hetero_override is not None
            else cfg.heteroskedastic
        )

        # ── Case-level mapping ──────────────────────────────────────────
        case_levels, config_to_case, case_idx_obs, ref_case_idx = (
            self._build_case_mapping(data, cfg)
        )
        n_cases = len(case_levels)

        # ── Station × Rupture interaction indices ────────────────────────
        # Create unique (station, run) pairs and map each observation
        n_st = data.n_stations
        n_rk = data.n_runs
        n_inter = n_st * n_rk  # 3 × 6 = 18

        # Interaction index: station_idx * n_runs + run_idx
        inter_idx = data.station_idx * n_rk + data.run_idx

        # Labels for the interaction coordinates
        inter_labels = []
        for s in range(n_st):
            for r in range(n_rk):
                slabel = data.station_labels[s]
                rlabel = data.run_labels[r]
                inter_labels.append(f"{slabel}|{rlabel}")
        inter_labels = np.array(inter_labels, dtype=object)

        coords = data.coords_dict()
        coords["Case"] = case_levels
        free_case_labels = [c for i, c in enumerate(case_levels) if i != ref_case_idx]
        coords["Case_free"] = free_case_labels
        coords["StationRun"] = inter_labels
        coords["StationRun_free"] = inter_labels[:-1]

        with pm.Model(coords=coords) as model:

            # ── Intercept ────────────────────────────────────────────────
            mu0 = pm.Normal("mu0", mu=0.0, sigma=p.sigma_intercept)

            # ── Hierarchical config-effect scale ─────────────────────────
            tau_config = pm.HalfNormal("tau_config", sigma=p.sigma_config)

            # ── Configuration effects (sum-to-zero, non-centered) ────────
            n_cfg = data.n_configs
            if n_cfg > 1:
                z_config_free = pm.Normal(
                    "z_config_free", mu=0.0, sigma=1.0, dims="Config_free",
                )
                z_config_last = -pm.math.sum(z_config_free)
                z_config = pm.math.concatenate([z_config_free, z_config_last[None]])
                mu_config = pm.Deterministic(
                    "mu_config", tau_config * z_config, dims="Config",
                )
            else:
                mu_config = pm.Deterministic(
                    "mu_config", pm.math.zeros((1,)), dims="Config",
                )

            # ── Station effects (sum-to-zero) ────────────────────────────
            if n_st > 1:
                delta_free = pm.Normal(
                    "delta_st_free", mu=0.0, sigma=p.sigma_station, dims="Station_free",
                )
                delta_last = -pm.math.sum(delta_free)
                delta_st = pm.Deterministic(
                    "delta_st",
                    pm.math.concatenate([delta_free, delta_last[None]]),
                    dims="Station",
                )
            else:
                delta_st = pm.Deterministic(
                    "delta_st", pm.math.zeros((1,)), dims="Station",
                )

            # ── Run random effects (sum-to-zero) ────────────────────────
            sigma_run = pm.HalfNormal("sigma_run", sigma=p.sigma_run)
            if n_rk > 1:
                if self._centered_runs:
                    b_run_free = pm.Normal(
                        "b_run_free", mu=0.0, sigma=sigma_run, dims="Run_free",
                    )
                    b_run_last = -pm.math.sum(b_run_free)
                    b_run = pm.Deterministic(
                        "b_run",
                        pm.math.concatenate([b_run_free, b_run_last[None]]),
                        dims="Run",
                    )
                else:
                    z_run_free = pm.Normal(
                        "z_run_free", mu=0.0, sigma=1.0, dims="Run_free",
                    )
                    z_run_last = -pm.math.sum(z_run_free)
                    z_run = pm.math.concatenate([z_run_free, z_run_last[None]])
                    b_run = pm.Deterministic(
                        "b_run", sigma_run * z_run, dims="Run",
                    )
            else:
                b_run = pm.Deterministic(
                    "b_run", pm.math.zeros((1,)), dims="Run",
                )

            # ── Station × Rupture interaction (sum-to-zero, non-centered) ─
            sigma_inter = pm.HalfNormal("sigma_inter", sigma=self._sigma_inter)
            n_inter_free = n_inter - 1
            z_inter_free = pm.Normal(
                "z_inter_free", mu=0.0, sigma=1.0, dims="StationRun_free",
            )
            z_inter_last = -pm.math.sum(z_inter_free)
            z_inter = pm.math.concatenate([z_inter_free, z_inter_last[None]])
            gamma_sr = pm.Deterministic(
                "gamma_sr", sigma_inter * z_inter, dims="StationRun",
            )

            # ── Per-Case run-sensitivity loadings (random slopes) ────────
            if n_cases > 1:
                log_lambda_free = pm.Normal(
                    "log_lambda_free", mu=0.0, sigma=self._sigma_lambda,
                    dims="Case_free",
                )
                if ref_case_idx == 0:
                    log_lambda = pm.math.concatenate(
                        [pm.math.zeros((1,)), log_lambda_free]
                    )
                elif ref_case_idx == n_cases - 1:
                    log_lambda = pm.math.concatenate(
                        [log_lambda_free, pm.math.zeros((1,))]
                    )
                else:
                    log_lambda = pm.math.concatenate([
                        log_lambda_free[:ref_case_idx],
                        pm.math.zeros((1,)),
                        log_lambda_free[ref_case_idx:],
                    ])
                lambda_case = pm.Deterministic(
                    "lambda_case", pm.math.exp(log_lambda), dims="Case",
                )
            else:
                lambda_case = pm.Deterministic(
                    "lambda_case", pm.math.ones((1,)), dims="Case",
                )

            # ── Per-Case interaction-loading ξ (v9, optional) ────────────
            if self._interaction_loading and n_cases > 1:
                log_xi_free = pm.Normal(
                    "log_xi_free", mu=0.0, sigma=self._sigma_xi,
                    dims="Case_free",
                )
                if ref_case_idx == 0:
                    log_xi = pm.math.concatenate(
                        [pm.math.zeros((1,)), log_xi_free]
                    )
                elif ref_case_idx == n_cases - 1:
                    log_xi = pm.math.concatenate(
                        [log_xi_free, pm.math.zeros((1,))]
                    )
                else:
                    log_xi = pm.math.concatenate([
                        log_xi_free[:ref_case_idx],
                        pm.math.zeros((1,)),
                        log_xi_free[ref_case_idx:],
                    ])
                xi_case = pm.Deterministic(
                    "xi_case", pm.math.exp(log_xi), dims="Case",
                )
            else:
                xi_case = None

            # ── Residual SD ──────────────────────────────────────────────
            if hetero:
                sigma_eps_config = pm.HalfNormal(
                    "sigma_eps_config", sigma=p.sigma_eps, dims="Config",
                )
                sigma_obs = sigma_eps_config[data.config_idx]
            else:
                sigma_eps = pm.HalfNormal("sigma_eps", sigma=p.sigma_eps)
                sigma_obs = sigma_eps

            # ── Linear predictor ─────────────────────────────────────────
            inter_term = gamma_sr[inter_idx]
            if xi_case is not None:
                inter_term = xi_case[case_idx_obs] * inter_term

            mu = (
                mu0
                + mu_config[data.config_idx]
                + delta_st[data.station_idx]
                + inter_term
                + lambda_case[case_idx_obs] * b_run[data.run_idx]
            )

            # ── Likelihood ───────────────────────────────────────────────
            if likelihood == "student_t":
                nu_minus2 = pm.Exponential(
                    "nu_minus2", lam=1.0 / p.nu_prior_lambda,
                )
                nu = nu_minus2 + 2.0
                pm.StudentT(
                    "y_obs", nu=nu, mu=mu, sigma=sigma_obs,
                    observed=data.y, dims="obs_id",
                )
            elif likelihood == "gaussian":
                pm.Normal(
                    "y_obs", mu=mu, sigma=sigma_obs,
                    observed=data.y, dims="obs_id",
                )
            else:
                raise ValueError(f"Unknown likelihood: {likelihood!r}")

        return model
