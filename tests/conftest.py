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
from apeBayes.data import encode_dataset
from apeBayes.posterior import PosteriorAccessor


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


def _make_idata(
    *,
    sigma_src: np.ndarray,
    sigma_inter: np.ndarray | None = None,
    gamma_sr: np.ndarray | None = None,
    xi_case: np.ndarray | None = None,
    mu0: np.ndarray | None = None,
    mu_config: np.ndarray | None = None,
    sigma_eps: np.ndarray | None = None,
    nu: np.ndarray | None = None,
    case_labels: tuple[str, ...] = ("A", "B", "C", "D"),
    config_labels: tuple[str, ...] = tuple(
        f"{t}{c}" for t in (1, 2, 3, 4) for c in ("A", "B", "C", "D")
    ),
):
    """Return a minimal InferenceData-like posterior wrapped for PosteriorAccessor.

    Uses xarray to build a real Dataset under ``posterior`` so
    ``flatten_posterior`` works unchanged.
    """
    import arviz as az
    import xarray as xr

    S = len(sigma_src)
    # Pretend single chain: reshape (S,) -> (1, S) for (chain, draw)
    data_vars: dict = {
        "sigma_run": (("chain", "draw"), sigma_src.reshape(1, S)),
    }
    if sigma_inter is not None:
        data_vars["sigma_inter"] = (("chain", "draw"), sigma_inter.reshape(1, S))
    if gamma_sr is not None:
        data_vars["gamma_sr"] = (
            ("chain", "draw", "StationRun"),
            gamma_sr.reshape(1, S, -1),
        )
    if xi_case is not None:
        data_vars["xi_case"] = (
            ("chain", "draw", "Case"),
            xi_case.reshape(1, S, -1),
        )
    if mu0 is not None:
        data_vars["mu0"] = (("chain", "draw"), mu0.reshape(1, S))
    if mu_config is not None:
        data_vars["mu_config"] = (
            ("chain", "draw", "Config"),
            mu_config.reshape(1, S, -1),
        )
    if sigma_eps is not None:
        if sigma_eps.ndim == 1:
            data_vars["sigma_eps"] = (("chain", "draw"), sigma_eps.reshape(1, S))
        else:
            data_vars["sigma_eps_config"] = (
                ("chain", "draw", "Config"),
                sigma_eps.reshape(1, S, -1),
            )
    if nu is not None:
        data_vars["nu_minus2"] = (("chain", "draw"), (nu - 2.0).reshape(1, S))

    coords: dict = {"chain": [0], "draw": np.arange(S)}
    all_dims = {d for dims, _ in data_vars.values() for d in dims}  # type: ignore[misc]
    if "Case" in all_dims:
        coords["Case"] = list(case_labels)
    if "Config" in all_dims:
        coords["Config"] = list(config_labels)
    if "StationRun" in all_dims:
        n_sr = next(
            arr.shape[2] for dims, arr in data_vars.values()  # type: ignore[misc]
            if "StationRun" in dims
        )
        coords["StationRun"] = [f"sr_{i}" for i in range(n_sr)]

    post = xr.Dataset(data_vars, coords=coords)
    return az.InferenceData(posterior=post)


@pytest.fixture
def stub_dataset(synthetic_long_df, default_config):
    """EpistemicDataset for the 16-config synthetic data."""
    return encode_dataset(synthetic_long_df, default_config)


@pytest.fixture
def stub_posterior_v8(stub_dataset):
    """Lightweight PosteriorAccessor for a v8-shaped model (src + inter + gamma).

    50 draws, no xi_case. Provides deterministic σ_src, σ_inter, γ_sr, σ_eps_config, ν.
    """
    rng = np.random.default_rng(0)
    S = 50
    sigma_src = np.abs(rng.normal(0.55, 0.05, size=S))
    sigma_inter = np.abs(rng.normal(0.30, 0.04, size=S))
    # 3 stations × 6 runs = 18 interaction cells (matches the paper suite shape).
    gamma_sr = rng.normal(0.0, 0.30, size=(S, 18))
    mu0 = rng.normal(-2.0, 0.05, size=S)
    mu_config = rng.normal(0.0, 0.2, size=(S, 16))
    sigma_eps = np.abs(rng.normal(0.10, 0.02, size=(S, 16)))
    nu = np.abs(rng.normal(25.0, 3.0, size=S)) + 3.0
    idata = _make_idata(
        sigma_src=sigma_src,
        sigma_inter=sigma_inter,
        gamma_sr=gamma_sr,
        mu0=mu0,
        mu_config=mu_config,
        sigma_eps=sigma_eps,
        nu=nu,
    )
    return PosteriorAccessor(idata, stub_dataset)


@pytest.fixture
def stub_posterior_v9(stub_dataset):
    """Lightweight v9-shaped posterior with xi_case present (per-case interaction loading)."""
    rng = np.random.default_rng(1)
    S = 30
    sigma_src = np.abs(rng.normal(0.55, 0.05, size=S))
    sigma_inter = np.abs(rng.normal(0.30, 0.04, size=S))
    gamma_sr = rng.normal(0.0, 0.30, size=(S, 18))
    xi_case = np.abs(rng.normal(1.0, 0.1, size=(S, 4)))
    idata = _make_idata(
        sigma_src=sigma_src,
        sigma_inter=sigma_inter,
        gamma_sr=gamma_sr,
        xi_case=xi_case,
    )
    return PosteriorAccessor(idata, stub_dataset)


@pytest.fixture
def stub_posterior_v3(stub_dataset):
    """Lightweight v1/v2/v3-shaped posterior: source only, no interaction."""
    rng = np.random.default_rng(2)
    S = 40
    sigma_src = np.abs(rng.normal(0.55, 0.05, size=S))
    idata = _make_idata(sigma_src=sigma_src)
    return PosteriorAccessor(idata, stub_dataset)


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
