"""
v10 "vc4": Case-specific record effects (Case×rupture and Case×station×rupture).

Proposed by the 2026-10-19 model critique (MEMO §3). On the Paper 1 data
the rank-1 record structure of v8 (λ_c·b_r plus one γ_sr shared by every
configuration) loses leave-one-rupture-out prediction by 56–75 nats; this
nested extension repairs it. v8 is the special case σ_u = σ_v = 0 with
per-config residual scales.

Model
-----
::

    y_ksr ~ StudentT(ν, m_ksr, σ_ε,c(k)),       ν − 2 ~ Exp(1/10)
    m_ksr = μ₀ + μ_k + δ_s + λ_c b_r + γ_sr + u_cr + v_csr

    μ_k   ~ ZeroSumNormal(σ_config)           unpooled config effects (Δ_k = μ_k − μ_ref)
    δ_s   ~ ZeroSumNormal(σ_station)
    b_r   = σ_run · ZSN_r(1),                 σ_run ~ HalfNormal(σ_src)
    log λ_c ~ N(0, σ_λ),  λ_ref = 1
    γ_sr  = σ_inter · ZSN_{s,r}(1)            zero-sum on both axes, σ_inter ~ HalfNormal(0.3)
    u_cr  = σ_u,c · ZSN_r(1)                  Case×rupture,          σ_u,c ~ HalfNormal(0.3)
    v_csr = σ_v,c · ZSN_{s,r}(1)              Case×station×rupture,  σ_v,c ~ HalfNormal(0.3)
    σ_ε,c ~ HalfNormal(σ_eps)                 Case-level residual (= Tier×record interaction)

with σ_config, σ_station, σ_src, σ_eps and the ν prior taken from
``cfg.priors`` (defaults 5, 3, 5, 5, 10 as in the critique).

σ_GM (record-to-record SD shared by the reference Case) is

    σ_GM = √(σ_run² + σ_inter² + σ_u,ref² + σ_v,ref²).

Posterior variable names follow v8 where the term exists (``mu_config``,
``delta_st``, ``b_run``, ``lambda_case``, ``gamma_sr`` flattened over
``StationRun``), plus ``sigma_u_case``, ``u_case_run`` (Case, Run),
``sigma_v_case``, ``v_case_sr`` (Case, Station, Run), ``sigma_eps_case``
and the per-config view ``sigma_eps_config`` so every v8 post-processing
routine applies unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pymc as pm

from ._terms import anchored_log_loading, likelihood_term

if TYPE_CHECKING:
    from ..config import ModelConfig
    from ..data import EpistemicDataset
    from ..posterior import PosteriorAccessor


class CaseRecordModel:
    """v10 ("vc4") builder: v8 + Case×rupture and Case×station×rupture terms.

    Parameters
    ----------
    sigma_lambda : float
        Prior SD of log λ_c. Default 0.5.
    sigma_inter : float
        HalfNormal scale of σ_inter. Default 0.3.
    sigma_u, sigma_v : float
        HalfNormal scales of the per-Case σ_u,c and σ_v,c. Default 0.3.
    case_residual : bool
        True (default): one residual SD per Case. False: one per config
        (v8-style), for sensitivity.
    likelihood : {"student_t", "gaussian"}, optional
        Override ``cfg.likelihood``.
    """

    def __init__(
        self,
        *,
        sigma_lambda: float = 0.5,
        sigma_inter: float = 0.3,
        sigma_u: float = 0.3,
        sigma_v: float = 0.3,
        case_residual: bool = True,
        likelihood: str | None = None,
    ) -> None:
        for name, v in (("sigma_lambda", sigma_lambda), ("sigma_inter", sigma_inter),
                        ("sigma_u", sigma_u), ("sigma_v", sigma_v)):
            if v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")
        self._sigma_lambda = sigma_lambda
        self._sigma_inter = sigma_inter
        self._sigma_u = sigma_u
        self._sigma_v = sigma_v
        self._case_residual = case_residual
        self._likelihood_override = likelihood

    def options(self) -> dict:
        """Constructor keyword arguments that reproduce this builder."""
        return {
            "sigma_lambda": self._sigma_lambda,
            "sigma_inter": self._sigma_inter,
            "sigma_u": self._sigma_u,
            "sigma_v": self._sigma_v,
            "case_residual": self._case_residual,
            "likelihood": self._likelihood_override,
        }

    @property
    def description(self) -> str:
        """Return a human-readable description of the model variant."""
        parts = [f"σ_λ={self._sigma_lambda}", f"σ_γ={self._sigma_inter}",
                 f"σ_u={self._sigma_u}", f"σ_v={self._sigma_v}"]
        if not self._case_residual:
            parts.append("config_residual")
        if self._likelihood_override:
            parts.append(self._likelihood_override)
        return f"CaseRecord[v10]({', '.join(parts)})"

    def sigma_GM(self, post: PosteriorAccessor) -> np.ndarray:
        """σ_GM = √(σ_run² + σ_inter² + σ_u,ref² + σ_v,ref²) (λ_ref = 1)."""
        ref_c = post.data.case_mapping()[3]
        return np.sqrt(
            post.sigma_run() ** 2
            + post.sigma_inter() ** 2
            + post.sigma_u_case()[:, ref_c] ** 2
            + post.sigma_v_case()[:, ref_c] ** 2
        )

    def build(self, data: EpistemicDataset, cfg: ModelConfig) -> pm.Model:
        """Construct the v10 PyMC model."""
        p = cfg.priors
        likelihood = self._likelihood_override or cfg.likelihood
        case_levels, config_to_case, case_obs, ref_case = data.case_mapping()
        n_cases = len(case_levels)
        if data.n_stations < 2 or data.n_runs < 2 or data.n_configs < 2:
            raise ValueError(
                "CaseRecordModel needs at least 2 configs, 2 stations and 2 runs."
            )

        coords = data.coords_dict()
        coords["Case"] = case_levels
        coords["Case_free"] = [c for i, c in enumerate(case_levels) if i != ref_case]
        coords["StationRun"] = np.array(
            [f"{s}|{r}" for s in data.station_labels for r in data.run_labels],
            dtype=object,
        )
        s_idx, r_idx, k_idx = data.station_idx, data.run_idx, data.config_idx

        with pm.Model(coords=coords) as model:
            mu0 = pm.Normal("mu0", mu=0.0, sigma=p.sigma_intercept)
            mu_config = pm.ZeroSumNormal("mu_config", sigma=p.sigma_config, dims="Config")
            delta_st = pm.ZeroSumNormal("delta_st", sigma=p.sigma_station, dims="Station")

            sigma_run = pm.HalfNormal("sigma_run", sigma=p.sigma_src)
            z_run = pm.ZeroSumNormal("z_run", sigma=1.0, dims="Run")
            b_run = pm.Deterministic("b_run", sigma_run * z_run, dims="Run")

            sigma_inter = pm.HalfNormal("sigma_inter", sigma=self._sigma_inter)
            z_inter = pm.ZeroSumNormal(
                "z_inter", sigma=1.0, dims=("Station", "Run"), n_zerosum_axes=2,
            )
            gamma = sigma_inter * z_inter
            pm.Deterministic("gamma_sr", gamma.flatten(), dims="StationRun")

            if n_cases > 1:
                lambda_case = pm.Deterministic(
                    "lambda_case",
                    anchored_log_loading(
                        "log_lambda_free", self._sigma_lambda, n_cases, ref_case, "Case_free",
                    ),
                    dims="Case",
                )
            else:
                lambda_case = pm.Deterministic("lambda_case", pm.math.ones((1,)), dims="Case")

            sigma_u = pm.HalfNormal("sigma_u_case", sigma=self._sigma_u, dims="Case")
            z_u = pm.ZeroSumNormal(
                "z_u", sigma=1.0, dims=("Case", "Run"), n_zerosum_axes=1,
            )
            u = pm.Deterministic("u_case_run", sigma_u[:, None] * z_u, dims=("Case", "Run"))
            sigma_v = pm.HalfNormal("sigma_v_case", sigma=self._sigma_v, dims="Case")
            z_v = pm.ZeroSumNormal(
                "z_v", sigma=1.0, dims=("Case", "Station", "Run"), n_zerosum_axes=2,
            )
            v = pm.Deterministic(
                "v_case_sr", sigma_v[:, None, None] * z_v, dims=("Case", "Station", "Run"),
            )

            if self._case_residual:
                sigma_case = pm.HalfNormal("sigma_eps_case", sigma=p.sigma_eps, dims="Case")
                sigma_cfg = pm.Deterministic(
                    "sigma_eps_config", sigma_case[config_to_case], dims="Config",
                )
            else:
                sigma_cfg = pm.HalfNormal("sigma_eps_config", sigma=p.sigma_eps, dims="Config")

            mu = (
                mu0
                + mu_config[k_idx]
                + delta_st[s_idx]
                + lambda_case[case_obs] * b_run[r_idx]
                + gamma[s_idx, r_idx]
                + u[case_obs, r_idx]
                + v[case_obs, s_idx, r_idx]
            )
            likelihood_term(likelihood, mu, sigma_cfg[k_idx], data.y, p.nu_prior_lambda)

        return model
