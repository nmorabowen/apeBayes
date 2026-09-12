"""Tests for apeBayes.diagnostics.validation.separability_check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apeBayes.config import FactorSpec, ModelConfig
from apeBayes.diagnostics import separability_check


def _make_pattern_df(case_patterns: dict[str, list[float]]) -> pd.DataFrame:
    """Build a tiny long-format frame with a deterministic run pattern per Case.

    Each Case gets its own single Config (so the (config, station) mean
    subtraction only ever centres over runs), replicated across three
    stations with an arbitrary per-station offset that cancels out of the
    Case-by-run residual.
    """
    runs = ["r1", "r2", "r3"]
    station_offset = {"s1": 0.0, "s2": 1.0, "s3": 2.0}

    rows = []
    for case, pattern in case_patterns.items():
        for sta, offset in station_offset.items():
            for run, val in zip(runs, pattern, strict=True):
                rows.append({
                    "Config": case,
                    "sta": sta,
                    "run": run,
                    "Case": case,
                    "edp": offset + val,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def pattern_cfg() -> ModelConfig:
    """ModelConfig matching the columns produced by ``_make_pattern_df``."""
    return ModelConfig(
        factors=[FactorSpec(name="Case", column="Case")],
        config_col="Config",
        edp_col="edp",
        station_col="sta",
        run_col="run",
        ref_config="A",
    )


class TestSeparabilityCheckShape:
    def test_square_symmetric_unit_diagonal(self, synthetic_long_df, default_config):
        result = separability_check(synthetic_long_df, default_config)

        assert list(result.index) == list(result.columns)
        np.testing.assert_allclose(np.diag(result.to_numpy()), 1.0)
        np.testing.assert_allclose(result.to_numpy(), result.to_numpy().T)

    def test_default_case_factor_is_last_factor(self, synthetic_long_df, default_config):
        result = separability_check(synthetic_long_df, default_config)
        assert list(result.index) == default_config.factors[-1].levels

    def test_attrs_present(self, synthetic_long_df, default_config):
        # NOTE: synthetic_long_df assigns a globally unique runkey per row
        # (no run is shared across Case levels), so the pairwise
        # correlations are undefined (NaN) here; the pattern_cfg-based
        # tests below exercise the actual numeric signal.
        result = separability_check(synthetic_long_df, default_config)
        assert "min_pairwise_corr" in result.attrs
        assert "all_near_one" in result.attrs
        assert isinstance(result.attrs["min_pairwise_corr"], float)
        assert isinstance(result.attrs["all_near_one"], bool)

    def test_explicit_case_factor_by_name(self, synthetic_long_df, default_config):
        by_name = separability_check(synthetic_long_df, default_config, case_factor="Nonlinearity")
        by_default = separability_check(synthetic_long_df, default_config)
        pd.testing.assert_frame_equal(by_name, by_default)

    def test_unknown_case_factor_raises(self, synthetic_long_df, default_config):
        with pytest.raises(ValueError, match="case_factor"):
            separability_check(synthetic_long_df, default_config, case_factor="nope")


class TestSeparabilityCheckSignal:
    def test_negated_pattern_gives_negative_min_corr(self, pattern_cfg):
        normal = [1.0, 0.0, -1.0]
        negated = [-1.0, 0.0, 1.0]
        df = _make_pattern_df({"A": normal, "B": normal, "C": normal, "D": negated})

        result = separability_check(df, pattern_cfg)

        assert result.loc["D", "A"] < 0
        assert result.attrs["min_pairwise_corr"] < 0
        assert result.attrs["all_near_one"] is False

    def test_identical_patterns_give_all_near_one(self, pattern_cfg):
        pattern = [1.0, 0.0, -1.0]
        df = _make_pattern_df({"A": pattern, "B": pattern, "C": pattern, "D": pattern})

        result = separability_check(df, pattern_cfg)

        assert result.attrs["all_near_one"] is True
        assert result.attrs["min_pairwise_corr"] >= 0.98

    def test_missing_column_raises(self, pattern_cfg):
        pattern = [1.0, 0.0, -1.0]
        df = _make_pattern_df({"A": pattern, "B": pattern})
        df = df.drop(columns=["sta"])
        with pytest.raises(ValueError, match="missing columns"):
            separability_check(df, pattern_cfg)
