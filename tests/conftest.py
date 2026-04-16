"""
Shared pytest fixtures for apeBayes tests.

Provides synthetic data that mirrors the real 4×4 Tier×Case epistemic
experiment (4 tiers × 4 cases = 16 configs, 18 stations, 3 runs per
station-config).  All random data, no real ground-motion files needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apeBayes.config import ModelConfig, FactorSpec, PriorConfig, SamplingConfig


# ── Constants ───────────────────────────────────────────────────────────

TIERS = [1, 2, 3, 4]
CASES = ["A", "B", "C", "D"]
N_STATIONS = 18
N_RUNS = 3
RNG_SEED = 42


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


@pytest.fixture(scope="session")
def synthetic_long_df(rng: np.random.Generator) -> pd.DataFrame:
    """Simulate a long-format DataFrame with log-EDP observations.

    Structure:  16 configs × 18 stations × 3 runs = 864 observations.
    True generative process (matches the model spec):
        y = mu0 + mu_config[k] + delta_st[j] + b_run[r] + eps
    """
    configs = [f"{t}{c}" for t in TIERS for c in CASES]
    stations = [f"sta_{i}" for i in range(N_STATIONS)]

    # True parameters
    mu0 = -2.0
    mu_config = rng.normal(0, 0.3, size=len(configs))
    mu_config -= mu_config.mean()  # sum-to-zero

    delta_st = rng.normal(0, 0.15, size=N_STATIONS)
    delta_st -= delta_st.mean()

    sigma_run = 0.20
    sigma_eps = 0.10

    rows = []
    run_counter = 0
    for k, cfg in enumerate(configs):
        tier = cfg[0]       # keep as string to match FactorSpec levels
        case = cfg[1]
        for j, sta in enumerate(stations):
            for r in range(N_RUNS):
                b_run_val = rng.normal(0, sigma_run)
                eps = rng.normal(0, sigma_eps)
                y = mu0 + mu_config[k] + delta_st[j] + b_run_val + eps
                rows.append({
                    "Tier": tier,
                    "Case": case,
                    "TierCase": cfg,
                    "sta": sta,
                    "runkey": f"run_{run_counter}",
                    "edp": y,
                })
                run_counter += 1

    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def default_config() -> ModelConfig:
    """A ModelConfig matching the synthetic data columns."""
    return ModelConfig(
        factors=[
            FactorSpec(name="SSI", column="Tier", levels=["1", "2", "3", "4"]),
            FactorSpec(name="Nonlinearity", column="Case", levels=["A", "B", "C", "D"]),
        ],
        edp_col="edp",
        station_col="sta",
        run_col="runkey",
        ref_config="4D",
        likelihood="student_t",
        heteroskedastic=True,
        priors=PriorConfig(),
        sampling=SamplingConfig(draws=200, tune=200, chains=2, seed=RNG_SEED),
    )


@pytest.fixture(scope="session")
def fast_config() -> ModelConfig:
    """Minimal sampling config for fast CI tests (not real inference)."""
    return ModelConfig(
        factors=[
            FactorSpec(name="SSI", column="Tier", levels=["1", "2", "3", "4"]),
            FactorSpec(name="Nonlinearity", column="Case", levels=["A", "B", "C", "D"]),
        ],
        edp_col="edp",
        station_col="sta",
        run_col="runkey",
        ref_config="4D",
        likelihood="student_t",
        heteroskedastic=True,
        priors=PriorConfig(),
        sampling=SamplingConfig(draws=50, tune=50, chains=2, seed=RNG_SEED),
    )
