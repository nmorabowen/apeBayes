"""Synthetic crossed 16 config × 3 station × 6 rupture design with a
"posterior" concentrated at known parameters (no MCMC).

Used by the 2026-10-19 review regression tests: a point-mass posterior at
the truth makes every post-processing quantity exactly computable.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd

from apeBayes import BayesEpistemicModel, FactorSpec, ModelConfig
from apeBayes.model import RandomSlopesInteractionModel

TIERS = ["1", "2", "3", "4"]
CASES = ["A", "B", "C", "D"]
STATIONS = ["sta_2", "sta_5", "sta_8"]
RUPTURES = [f"rup_{r}" for r in range(1, 7)]


def crossed_config() -> ModelConfig:
    return ModelConfig(
        factors=[
            FactorSpec("SSI", "Tier", TIERS),
            FactorSpec("Nonlinearity", "Case", CASES),
        ],
        run_col="rupture",
        ref_config="4D",
    )


def truth(seed: int = 0, *, case_terms: bool = False, station_terms: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    t = {
        "mu0": -2.0,
        "mu_config": rng.normal(0, 0.3, 16),
        "delta_st": np.array([-0.3, 0.1, 0.2]),
        "lambda_case": np.array([0.4, 1.0, 1.2, 1.0]),
        "sigma_run": 0.5,
        "sigma_inter": 0.2,
        "sigma_eps": np.linspace(0.05, 0.2, 16),
    }
    t["mu_config"] -= t["mu_config"].mean()
    b = rng.normal(0, 0.5, 6)
    t["b_run"] = b - b.mean()
    g = rng.normal(0, 0.2, (3, 6))
    g -= g.mean(0)
    g -= g.mean(1, keepdims=True)
    t["gamma_sr"] = g.reshape(-1)
    if case_terms:
        t["sigma_u_case"] = np.array([0.3, 0.05, 0.1, 0.15])
        t["sigma_v_case"] = np.array([0.2, 0.05, 0.05, 0.1])
        u = rng.normal(0, 1, (4, 6)) * t["sigma_u_case"][:, None]
        t["u_case_run"] = u - u.mean(1, keepdims=True)
        v = rng.normal(0, 1, (4, 3, 6)) * t["sigma_v_case"][:, None, None]
        v -= v.mean(2, keepdims=True)
        v -= v.mean(1, keepdims=True)
        t["v_case_sr"] = v
    if station_terms:  # v11: zero-sum over stations, per Case and per Tier
        t["sigma_case_station"] = 0.15
        t["sigma_tier_station"] = 0.05
        w = rng.normal(0, 0.15, (4, 3))
        t["case_station"] = w - w.mean(1, keepdims=True)
        kappa = rng.normal(0, 0.05, (4, 3))
        t["tier_station"] = kappa - kappa.mean(1, keepdims=True)
    return t


def crossed_df(t: dict, *, noise: bool = False, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    k = 0
    for tier in TIERS:
        for ci, case in enumerate(CASES):
            for si, sta in enumerate(STATIONS):
                for ri, rup in enumerate(RUPTURES):
                    y = (
                        t["mu0"] + t["mu_config"][k] + t["delta_st"][si]
                        + t["lambda_case"][ci] * t["b_run"][ri]
                        + t["gamma_sr"][si * 6 + ri]
                    )
                    if "u_case_run" in t:
                        y += t["u_case_run"][ci, ri] + t["v_case_sr"][ci, si, ri]
                    if "case_station" in t:
                        y += t["case_station"][ci, si]
                    if "tier_station" in t:
                        y += t["tier_station"][TIERS.index(tier), si]
                    if noise:
                        y += rng.normal(0, t["sigma_eps"][k])
                    rows.append(dict(Tier=tier, Case=case, TierCase=tier + case,
                                     sta=sta, rupture=rup, edp=y))
            k += 1
    return pd.DataFrame(rows)


def point_mass_idata(t: dict, n_draws: int = 50, jitter: float = 0.0, seed: int = 2):
    """Posterior with every draw equal to the truth (+ optional jitter)."""
    rng = np.random.default_rng(seed)
    configs = [a + b for a in TIERS for b in CASES]
    dims = {
        "mu_config": ["Config"], "delta_st": ["Station"], "b_run": ["Run"],
        "lambda_case": ["Case"], "gamma_sr": ["StationRun"],
        "sigma_eps_config": ["Config"], "sigma_u_case": ["Case"],
        "sigma_v_case": ["Case"], "u_case_run": ["Case", "Run"],
        "v_case_sr": ["Case", "Station", "Run"],
        "case_station": ["Case", "Station"], "tier_station": ["Tier", "Station"],
    }
    names = {"sigma_eps": "sigma_eps_config"}
    post = {}
    for key, val in t.items():
        arr = np.asarray(val, dtype=float)
        draws = np.broadcast_to(arr, (1, n_draws, *arr.shape)).copy()
        if jitter:
            draws = draws + jitter * rng.standard_normal(draws.shape)
        post[names.get(key, key)] = draws
    coords = {
        "Config": configs, "Station": STATIONS, "Run": RUPTURES, "Case": CASES, "Tier": TIERS,
        "StationRun": [f"{s}|{r}" for s in STATIONS for r in RUPTURES],
    }
    return az.from_dict(
        posterior=post, coords=coords,
        dims={k: v for k, v in dims.items() if k in post},
    )


def point_mass_model(t: dict, *, noise: bool = False, **kw) -> BayesEpistemicModel:
    m = BayesEpistemicModel(crossed_df(t, noise=noise), crossed_config())
    from apeBayes.posterior import PosteriorAccessor
    m._posterior = PosteriorAccessor(point_mass_idata(t, **kw), m.data)
    m._builder = RandomSlopesInteractionModel()
    return m
