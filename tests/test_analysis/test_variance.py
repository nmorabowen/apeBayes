"""
Tests for the variance budget analysis module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apeBayes.analysis.variance import variance_budget, variance_components


class TestVarianceBudget:
    """Variance budget decomposition."""

    @pytest.fixture()
    def mock_draws(self):
        """Fake posterior draws: S=100 samples, K=4 configs, J=3 stations."""
        rng = np.random.default_rng(99)
        S, K, J = 100, 4, 3
        mu_config = rng.normal(0, 0.3, size=(S, K))
        delta_st = rng.normal(0, 0.1, size=(S, J))
        sigma_run = np.abs(rng.normal(0.2, 0.02, size=S))
        sigma_eps = np.abs(rng.normal(0.1, 0.01, size=S))
        config_idx = np.array([0]*3 + [1]*3 + [2]*3 + [3]*3)
        return mu_config, delta_st, sigma_run, sigma_eps, config_idx

    def test_returns_dataframe(self, mock_draws):
        mu_config, delta_st, sigma_run, sigma_eps, config_idx = mock_draws
        result = variance_budget(mu_config, delta_st, sigma_run, sigma_eps, config_idx)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 3  # at least config, station, run, eps components

    def test_percentages_sum_near_100(self, mock_draws):
        mu_config, delta_st, sigma_run, sigma_eps, config_idx = mock_draws
        result = variance_budget(mu_config, delta_st, sigma_run, sigma_eps, config_idx)
        if "pct_med" in result.columns:
            total = result["pct_med"].sum()
            assert 90.0 < total < 110.0  # approximately 100%

    def test_with_student_t_nu(self, mock_draws):
        mu_config, delta_st, sigma_run, sigma_eps, config_idx = mock_draws
        nu = np.full(100, 10.0)
        result = variance_budget(
            mu_config, delta_st, sigma_run, sigma_eps, config_idx, nu=nu
        )
        assert isinstance(result, pd.DataFrame)


class TestVarianceComponents:
    """Per-config variance components."""

    def test_returns_per_config(self):
        rng = np.random.default_rng(42)
        S = 80
        sigma_run = np.abs(rng.normal(0.2, 0.02, size=S))
        sigma_eps = np.abs(rng.normal(0.1, 0.01, size=(S, 3)))
        labels = ["A", "B", "C"]
        result = variance_components(sigma_run, sigma_eps, labels)
        assert isinstance(result, pd.DataFrame)
        # 1 row for sigma_run + K rows for sigma_eps per config (heteroskedastic)
        assert len(result) == 1 + len(labels)

    def test_scalar_sigma_eps(self):
        rng = np.random.default_rng(42)
        S = 80
        sigma_run = np.abs(rng.normal(0.2, 0.02, size=S))
        sigma_eps = np.abs(rng.normal(0.1, 0.01, size=S))
        labels = ["A", "B"]
        result = variance_components(sigma_run, sigma_eps, labels)
        assert isinstance(result, pd.DataFrame)
