"""
Tests for the equivalence analysis module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apeBayes.analysis.equivalence import (
    equivalence_probability,
    equivalence_sweep,
    epistemic_distance_matrix,
    epistemic_equivalence_matrix,
    epistemic_clusters,
)


@pytest.fixture()
def mock_posterior():
    """4 configs, 200 draws. Config 0 and 1 are near-identical, 2 and 3 diverge."""
    rng = np.random.default_rng(88)
    S = 200
    mu_config = np.zeros((S, 4))
    mu_config[:, 0] = rng.normal(0.0, 0.01, S)     # near zero
    mu_config[:, 1] = rng.normal(0.02, 0.01, S)     # near zero (equivalent to 0)
    mu_config[:, 2] = rng.normal(1.0, 0.05, S)      # far from 0
    mu_config[:, 3] = rng.normal(-0.8, 0.05, S)     # far from 0
    sigma_run = np.abs(rng.normal(0.2, 0.02, S))
    labels = ["1A", "1B", "2A", "2B"]
    return mu_config, sigma_run, labels


class TestEquivalenceDefaults:
    def test_alpha_default_is_paper_spec(self):
        """alpha defaults to 0.4 per uncertanty_measures.md \u00a75."""
        import inspect
        assert inspect.signature(equivalence_probability).parameters["alpha"].default == 0.4
        assert inspect.signature(epistemic_distance_matrix).parameters["alpha"].default == 0.4
        assert inspect.signature(epistemic_equivalence_matrix).parameters["alpha"].default == 0.4
        assert inspect.signature(epistemic_clusters).parameters["alpha"].default == 0.4


class TestEquivalenceProbability:
    def test_shape(self, mock_posterior):
        mu, sigma, labels = mock_posterior
        result = equivalence_probability(mu, sigma, ref_idx=0, labels=labels)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4

    def test_ref_has_high_equiv(self, mock_posterior):
        mu, sigma, labels = mock_posterior
        result = equivalence_probability(mu, sigma, ref_idx=0, labels=labels)
        ref_row = result[result["Config"] == "1A"]
        assert float(ref_row["P_equiv"].iloc[0]) > 0.9

    def test_near_config_higher_than_far(self, mock_posterior):
        mu, sigma, labels = mock_posterior
        result = equivalence_probability(mu, sigma, ref_idx=0, labels=labels)
        p_near = float(result[result["Config"] == "1B"]["P_equiv"].iloc[0])
        p_far = float(result[result["Config"] == "2A"]["P_equiv"].iloc[0])
        assert p_near > p_far


class TestEquivalenceSweep:
    def test_returns_long_df(self, mock_posterior):
        mu, sigma, labels = mock_posterior
        alphas = np.array([0.1, 0.2, 0.5])
        result = equivalence_sweep(mu, sigma, ref_idx=0, labels=labels, alphas=alphas)
        assert isinstance(result, pd.DataFrame)
        assert "alpha" in result.columns
        assert len(result) == len(alphas) * len(labels)

    def test_monotone_in_alpha(self, mock_posterior):
        mu, sigma, labels = mock_posterior
        alphas = np.linspace(0.01, 1.0, 20)
        result = equivalence_sweep(mu, sigma, ref_idx=0, labels=labels, alphas=alphas)
        for lbl in labels:
            sub = result[result["Config"] == lbl].sort_values("alpha")
            p = sub["P_equiv"].values
            # P_equiv should be non-decreasing in alpha
            assert np.all(np.diff(p) >= -1e-10)


class TestDistanceMatrix:
    def test_symmetric(self, mock_posterior):
        mu, sigma, labels = mock_posterior
        _, D = epistemic_distance_matrix(mu, sigma, labels=labels)
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    def test_diagonal_zero(self, mock_posterior):
        mu, sigma, labels = mock_posterior
        _, D = epistemic_distance_matrix(mu, sigma, labels=labels)
        np.testing.assert_allclose(np.diag(D), 0.0)


class TestEquivalenceMatrix:
    def test_values_in_01(self, mock_posterior):
        mu, sigma, labels = mock_posterior
        _, P = epistemic_equivalence_matrix(mu, sigma, labels=labels)
        assert np.all(P >= 0.0)
        assert np.all(P <= 1.0)

    def test_diagonal_one(self, mock_posterior):
        mu, sigma, labels = mock_posterior
        _, P = epistemic_equivalence_matrix(mu, sigma, labels=labels)
        np.testing.assert_allclose(np.diag(P), 1.0)


class TestClusters:
    def test_returns_cluster_ids(self, mock_posterior):
        mu, sigma, labels = mock_posterior
        result = epistemic_clusters(mu, sigma, labels=labels)
        assert "cluster" in result.columns
        assert len(result) == 4

    def test_near_configs_same_cluster(self, mock_posterior):
        mu, sigma, labels = mock_posterior
        result = epistemic_clusters(mu, sigma, labels=labels, alpha=0.5)
        c1a = int(result[result["Config"] == "1A"]["cluster"].iloc[0])
        c1b = int(result[result["Config"] == "1B"]["cluster"].iloc[0])
        # Configs 0 and 1 are close — should cluster together
        assert c1a == c1b
