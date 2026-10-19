"""Reproduce a published Paper 1 number with the unchanged v8 builder.

Fits v8 (``RandomSlopesInteractionModel()``, default ``zero_sum="last"``)
to the San Ramon PFA-mid data and compares the posterior median of σ_GM
with the canonical export (``canonical/sigma/sigma_components_per_edp.csv``,
0.2691). The 2026-10-19 critique's independent re-implementation matched
it within 0.003. Read-only on the Paper 1 folder; skipped where it is
absent.

Run with ``pytest -m slow`` (several minutes).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apeBayes import BayesEpistemicModel, FactorSpec, ModelConfig, SamplingConfig
from apeBayes.model import RandomSlopesInteractionModel

pytestmark = pytest.mark.slow

PAPER1 = Path(
    r"C:/Users/nmb/Documents/gitAPE/Epistemic Uncertanty/results/sanramon-tiers/"
    r"paper1-calculations/selected_statistical_model"
)
CANONICAL = PAPER1 / "canonical" / "sigma" / "sigma_components_per_edp.csv"
DATA = PAPER1 / "data_csv" / "pfa_mid.parquet"


@pytest.mark.skipif(not (CANONICAL.exists() and DATA.exists()),
                    reason="Paper 1 canonical data not available")
def test_v8_reproduces_canonical_sigma_gm_pfa_mid():
    df = pd.read_parquet(DATA).rename(columns={
        "tier": "Tier", "case": "Case", "tiercase": "TierCase", "station": "sta",
        "log_pfa_g": "edp",
    })
    for c in ("Tier", "Case", "TierCase", "sta", "rupture"):
        df[c] = df[c].astype(str)
    cfg = ModelConfig(
        factors=[FactorSpec("SSI", "Tier", ["1", "2", "3", "4"]),
                 FactorSpec("Nonlinearity", "Case", ["A", "B", "C", "D"])],
        config_col="TierCase", edp_col="edp", station_col="sta", run_col="rupture",
        ref_config="4D", likelihood="student_t", heteroskedastic=True,
        # canonical: 4 x 10000 draws, tune 5000, target 0.985, seed 123; nutpie
        # then ran at its default depth 10 (max_treedepth was not forwarded)
        sampling=SamplingConfig(draws=5000, tune=3000, chains=4, target_accept=0.985,
                                seed=123, max_treedepth=10, store_log_likelihood=False),
    )
    m = BayesEpistemicModel(df[["Tier", "Case", "TierCase", "sta", "rupture", "edp"]], cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(builder=RandomSlopesInteractionModel(sigma_lambda=0.5, sigma_inter=0.3),
              prior_predictive=False, posterior_predictive=False)
    canon = pd.read_csv(CANONICAL).set_index("EDP").loc["PFA Mid"]
    gm = float(np.median(m.sigma_GM_draws()))
    print(f"sigma_GM median: refit {gm:.4f} vs canonical {canon['sigma_GM_med']:.4f}; "
          f"converged={m.converged}")
    assert gm == pytest.approx(canon["sigma_GM_med"], abs=0.005)
    assert float(np.median(m.posterior.sigma_inter())) == pytest.approx(
        canon["sigma_inter_med"], abs=0.005)
