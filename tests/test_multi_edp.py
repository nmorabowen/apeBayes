"""
Tests for the MultiEDPModel wrapper (Stage 1 machinery).

These tests verify the orchestration logic — creating configs per EDP,
collecting results across EDPs, and producing cross-EDP comparison tables.
No MCMC sampling is performed; we test the wiring, not the inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apeBayes import MultiEDPModel, EDPSpec, ModelConfig, FactorSpec
from apeBayes.config import SamplingConfig, PriorConfig


class TestEDPSpec:
    def test_basic_creation(self):
        spec = EDPSpec("IDR_x_1", "idr_x_1", category="IDR", story=1, direction="x")
        assert spec.name == "IDR_x_1"
        assert spec.column == "idr_x_1"
        assert spec.category == "IDR"
        assert spec.story == 1
        assert spec.direction == "x"

    def test_repr(self):
        spec = EDPSpec("PFA_y_3", "pfa_y_3", story=3, direction="y")
        r = repr(spec)
        assert "PFA_y_3" in r
        assert "story=3" in r


class TestMultiEDPModelConstruction:
    """Test MultiEDPModel setup without fitting."""

    @pytest.fixture()
    def multi_df(self, synthetic_long_df):
        """Add a second EDP column (fake PFA) to the synthetic data."""
        df = synthetic_long_df.copy()
        rng = np.random.default_rng(123)
        df["pfa"] = df["edp"] + rng.normal(0, 0.05, len(df))
        return df

    @pytest.fixture()
    def edp_specs(self):
        return [
            EDPSpec("IDR", "edp", category="drift"),
            EDPSpec("PFA", "pfa", category="acceleration"),
        ]

    def test_creation(self, multi_df, edp_specs, fast_config):
        multi = MultiEDPModel(multi_df, edp_specs, fast_config)
        assert len(multi.edp_specs) == 2
        assert multi.edp_names == ["IDR", "PFA"]
        assert not multi.is_fitted

    def test_empty_specs_raises(self, multi_df, fast_config):
        with pytest.raises(ValueError, match="at least one"):
            MultiEDPModel(multi_df, [], fast_config)

    def test_repr(self, multi_df, edp_specs, fast_config):
        multi = MultiEDPModel(multi_df, edp_specs, fast_config, name="test")
        r = repr(multi)
        assert "test" in r
        assert "2 defined" in r
        assert "0 fitted" in r

    def test_getitem_unfitted_raises(self, multi_df, edp_specs, fast_config):
        multi = MultiEDPModel(multi_df, edp_specs, fast_config)
        with pytest.raises(KeyError, match="not fitted"):
            _ = multi["IDR"]

    def test_check_fitted_raises(self, multi_df, edp_specs, fast_config):
        multi = MultiEDPModel(multi_df, edp_specs, fast_config)
        with pytest.raises(RuntimeError, match="Not all EDPs"):
            multi.compare_bias()

    def test_cfg_per_edp(self, multi_df, edp_specs, fast_config):
        multi = MultiEDPModel(multi_df, edp_specs, fast_config)
        cfg_idr = multi._make_cfg_for_edp(edp_specs[0])
        cfg_pfa = multi._make_cfg_for_edp(edp_specs[1])
        assert cfg_idr.edp_col == "edp"
        assert cfg_pfa.edp_col == "pfa"
        # Everything else should be the same
        assert cfg_idr.ref_config == cfg_pfa.ref_config
        assert cfg_idr.likelihood == cfg_pfa.likelihood
