"""
Tests for the data encoding module.

Verifies that encode_dataset correctly validates, encodes, and freezes
a long-format DataFrame into an EpistemicDataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apeBayes.config import ModelConfig, FactorSpec
from apeBayes.data import EpistemicDataset, encode_dataset


class TestEncodeDataset:
    """Core encoding logic."""

    def test_basic_encoding(self, synthetic_long_df, fast_config):
        ds = encode_dataset(synthetic_long_df, fast_config)
        assert isinstance(ds, EpistemicDataset)
        assert ds.n_configs == 16
        assert ds.n_stations == 18
        assert ds.n_obs == len(synthetic_long_df)

    def test_config_labels_sorted(self, synthetic_long_df, fast_config):
        ds = encode_dataset(synthetic_long_df, fast_config)
        labels = ds.config_labels.tolist()
        # tier-major order: 1A, 1B, 1C, 1D, 2A, ...
        assert labels[0] == "1A"
        assert labels[-1] == "4D"

    def test_ref_config(self, synthetic_long_df, fast_config):
        ds = encode_dataset(synthetic_long_df, fast_config)
        assert ds.ref_label == "4D"
        assert ds.config_labels[ds.ref_idx] == "4D"

    def test_config_label_to_idx(self, synthetic_long_df, fast_config):
        ds = encode_dataset(synthetic_long_df, fast_config)
        idx = ds.config_label_to_idx("2C")
        assert ds.config_labels[idx] == "2C"

    def test_config_label_to_idx_missing(self, synthetic_long_df, fast_config):
        ds = encode_dataset(synthetic_long_df, fast_config)
        with pytest.raises(KeyError, match="not found"):
            ds.config_label_to_idx("9Z")

    def test_factor_index_grid(self, synthetic_long_df, fast_config):
        ds = encode_dataset(synthetic_long_df, fast_config)
        f0, f1, grid = ds.factor_index_grid()
        assert len(f0) == 4  # tiers
        assert len(f1) == 4  # cases
        assert len(grid) == 16
        # grid maps (tier, case) → flat idx
        assert grid[("4", "D")] == ds.ref_idx

    def test_y_is_float(self, synthetic_long_df, fast_config):
        ds = encode_dataset(synthetic_long_df, fast_config)
        assert ds.y.dtype == float
        assert np.all(np.isfinite(ds.y))

    def test_subset_config_indices(self, synthetic_long_df, fast_config):
        ds = encode_dataset(synthetic_long_df, fast_config)
        labels, idx = ds.subset_config_indices(["1A", "4D"])
        assert labels == ["1A", "4D"]
        assert len(idx) == 2

    def test_subset_none_returns_all(self, synthetic_long_df, fast_config):
        ds = encode_dataset(synthetic_long_df, fast_config)
        labels, idx = ds.subset_config_indices(None)
        assert len(labels) == 16
        assert len(idx) == 16


class TestEncodeDatasetValidation:
    """Error handling and edge cases."""

    def test_missing_column_raises(self, fast_config):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing columns"):
            encode_dataset(df, fast_config)

    def test_bad_ref_config_raises(self, synthetic_long_df):
        cfg = ModelConfig(ref_config="99Z")
        with pytest.raises(ValueError, match="ref_config"):
            encode_dataset(synthetic_long_df, cfg)

    def test_nan_edp_dropped(self, synthetic_long_df, fast_config):
        df = synthetic_long_df.copy()
        df.loc[0, "edp"] = float("nan")
        df.loc[1, "edp"] = float("nan")
        with pytest.warns(UserWarning, match="Dropped 2 rows"):
            ds = encode_dataset(df, fast_config)
        assert ds.n_obs == len(synthetic_long_df) - 2


class TestDelimiterLabels:
    """Test config_sep for multi-word factor levels."""

    def test_underscore_separator(self):
        df = pd.DataFrame({
            "soil": ["PM4Sand", "PM4Sand", "elastic", "elastic"],
            "ssi": ["fixed", "fixed", "fixed", "fixed"],
            "sta": ["s1", "s1", "s1", "s1"],
            "run": ["r1", "r2", "r1", "r2"],
            "edp": [1.0, 1.1, 0.9, 1.0],
        })
        cfg = ModelConfig(
            factors=[
                FactorSpec(name="SSI", column="ssi", levels=["fixed"]),
                FactorSpec(name="Soil", column="soil", levels=["PM4Sand", "elastic"]),
            ],
            config_col="config_auto",  # not in df → will be built from factors
            edp_col="edp",
            station_col="sta",
            run_col="run",
            ref_config="fixed_PM4Sand",
            config_sep="_",
        )
        ds = encode_dataset(df, cfg)
        assert "fixed_PM4Sand" in ds.config_labels.tolist()
        assert "fixed_elastic" in ds.config_labels.tolist()
        assert ds.n_configs == 2
