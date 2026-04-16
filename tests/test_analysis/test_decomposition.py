"""
Tests for the factorial decomposition module.
"""

from __future__ import annotations

import numpy as np
import pytest

from apeBayes.analysis.decomposition import (
    axiswise_decomposition_draws,
    axiswise_table,
    level_ranking_tables,
)


class TestAxiswiseDecomposition:
    """2-factor decomposition of configuration effects."""

    @pytest.fixture()
    def mock_decomp_inputs(self):
        """4 tiers × 3 cases = 12 configs, 200 posterior draws."""
        rng = np.random.default_rng(77)
        S, I, J = 200, 4, 3
        K = I * J
        mu_config = rng.normal(0, 0.5, size=(S, K))
        sigma_run = np.abs(rng.normal(0.2, 0.02, size=S))

        f0 = [str(i + 1) for i in range(I)]
        f1 = [chr(ord("A") + j) for j in range(J)]
        grid = {}
        idx = 0
        for i, a in enumerate(f0):
            for j, b in enumerate(f1):
                grid[(a, b)] = idx
                idx += 1

        return mu_config, f0, f1, grid, sigma_run

    def test_decomposition_shapes(self, mock_decomp_inputs):
        mu_config, f0, f1, grid, sigma_run = mock_decomp_inputs
        decomp = axiswise_decomposition_draws(mu_config, f0, f1, grid, sigma_run)

        S = mu_config.shape[0]
        I, J = len(f0), len(f1)
        assert decomp["mu_ij"].shape == (S, I, J)
        assert decomp["mu_bar"].shape == (S,)
        assert decomp["tau"].shape == (S, I)
        assert decomp["kappa"].shape == (S, J)
        assert decomp["gamma"].shape == (S, I, J)

    def test_decomposition_sums_to_original(self, mock_decomp_inputs):
        mu_config, f0, f1, grid, sigma_run = mock_decomp_inputs
        decomp = axiswise_decomposition_draws(mu_config, f0, f1, grid, sigma_run)

        # Reconstruct: mu_ij = mu_bar + tau_i + kappa_j + gamma_ij
        recon = (
            decomp["mu_bar"][:, None, None]
            + decomp["tau"][:, :, None]
            + decomp["kappa"][:, None, :]
            + decomp["gamma"]
        )
        np.testing.assert_allclose(recon, decomp["mu_ij"], atol=1e-12)

    def test_main_effects_sum_to_zero(self, mock_decomp_inputs):
        mu_config, f0, f1, grid, sigma_run = mock_decomp_inputs
        decomp = axiswise_decomposition_draws(mu_config, f0, f1, grid, sigma_run)

        # τ sums to zero across tiers per draw
        np.testing.assert_allclose(
            decomp["tau"].sum(axis=1), 0.0, atol=1e-12
        )
        # κ sums to zero across cases per draw
        np.testing.assert_allclose(
            decomp["kappa"].sum(axis=1), 0.0, atol=1e-12
        )

    def test_interaction_sums_to_zero(self, mock_decomp_inputs):
        mu_config, f0, f1, grid, sigma_run = mock_decomp_inputs
        decomp = axiswise_decomposition_draws(mu_config, f0, f1, grid, sigma_run)

        # γ sums to zero along both axes
        np.testing.assert_allclose(
            decomp["gamma"].sum(axis=1), 0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            decomp["gamma"].sum(axis=2), 0.0, atol=1e-12
        )

    def test_incomplete_grid_raises(self):
        rng = np.random.default_rng(1)
        mu_config = rng.normal(0, 1, size=(10, 4))
        sigma_run = np.abs(rng.normal(0.2, 0.02, size=10))
        # Missing cell (2, B)
        grid = {("1", "A"): 0, ("1", "B"): 1, ("2", "A"): 2}
        with pytest.raises(ValueError, match="complete factor grid"):
            axiswise_decomposition_draws(
                mu_config, ["1", "2"], ["A", "B"], grid, sigma_run
            )


class TestAxiswiseTable:
    """Summary table from decomposition."""

    def test_table_has_three_rows(self):
        rng = np.random.default_rng(77)
        S, I, J = 200, 4, 3
        K = I * J
        mu_config = rng.normal(0, 0.5, size=(S, K))
        sigma_run = np.abs(rng.normal(0.2, 0.02, size=S))

        f0 = [str(i + 1) for i in range(I)]
        f1 = [chr(ord("A") + j) for j in range(J)]
        grid = {}
        idx = 0
        for a in f0:
            for b in f1:
                grid[(a, b)] = idx
                idx += 1

        decomp = axiswise_decomposition_draws(mu_config, f0, f1, grid, sigma_run)
        table = axiswise_table(decomp, factor_names=("Tier", "Case"))
        assert len(table) == 3
        assert "component" in table.columns
        assert "var_med" in table.columns
        assert "pct_med" in table.columns


class TestLevelRankingTables:
    """Per-level ranking tables."""

    def test_returns_three_tables(self):
        rng = np.random.default_rng(77)
        S, I, J = 200, 4, 3
        K = I * J
        mu_config = rng.normal(0, 0.5, size=(S, K))
        sigma_run = np.abs(rng.normal(0.2, 0.02, size=S))

        f0 = [str(i + 1) for i in range(I)]
        f1 = [chr(ord("A") + j) for j in range(J)]
        grid = {}
        idx = 0
        for a in f0:
            for b in f1:
                grid[(a, b)] = idx
                idx += 1

        decomp = axiswise_decomposition_draws(mu_config, f0, f1, grid, sigma_run)
        t0, t1, spread = level_ranking_tables(decomp, factor_names=("Tier", "Case"))
        assert len(t0) == I
        assert len(t1) == J
        assert len(spread) == 2  # one row per axis
