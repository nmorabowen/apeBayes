"""Unit tests for apeBayes.analysis.bias — pure-function tests on synthetic draws."""

import numpy as np
import pandas as pd
import pytest

from apeBayes.analysis.bias import standardized_bias, standardized_bias_total, bias_probability


@pytest.fixture
def draws():
    """Fake posterior draws: 500 samples, 4 configs."""
    rng = np.random.default_rng(99)
    mu_config = rng.normal(0, 0.3, size=(500, 4))
    mu_config -= mu_config.mean(axis=1, keepdims=True)
    sigma_run = np.abs(rng.normal(0.2, 0.02, size=500))
    sigma_eps = np.abs(rng.normal(0.1, 0.01, size=500))
    labels = ["1A", "1B", "2A", "2B"]
    return mu_config, sigma_run, sigma_eps, labels


class TestStandardizedBias:
    def test_shape(self, draws):
        mu_config, sigma_run, _, labels = draws
        df = standardized_bias(mu_config, sigma_run, ref_idx=0, labels=labels)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(labels)

    def test_ref_has_zero_bias(self, draws):
        mu_config, sigma_run, _, labels = draws
        df = standardized_bias(mu_config, sigma_run, ref_idx=0, labels=labels)
        ref_row = df[df["Config"] == labels[0]]
        assert abs(float(ref_row["std_bias_med"].iloc[0])) < 1e-10

    def test_subset(self, draws):
        mu_config, sigma_run, _, labels = draws
        subset_idx = np.array([0, 2])
        df = standardized_bias(
            mu_config, sigma_run, ref_idx=0,
            labels=["1A", "2A"], subset_idx=subset_idx,
        )
        assert len(df) == 2

    def test_columns_present(self, draws):
        mu_config, sigma_run, _, labels = draws
        df = standardized_bias(mu_config, sigma_run, ref_idx=0, labels=labels)
        expected = {"Config", "std_bias_med", "std_bias_lo", "std_bias_hi",
                    "dmu_med", "dmu_lo", "dmu_hi", "mult_med", "mult_lo", "mult_hi"}
        assert expected == set(df.columns)


class TestStandardizedBiasTotal:
    def test_includes_beta_total(self, draws):
        mu_config, sigma_run, sigma_eps, labels = draws
        rng = np.random.default_rng(42)
        nu = rng.exponential(10, size=500) + 2
        df = standardized_bias_total(
            mu_config, sigma_run, sigma_eps, ref_idx=0, labels=labels, nu=nu,
        )
        assert "beta_tot_med" in df.columns
        assert "sigma_tot_med" in df.columns

    def test_without_nu(self, draws):
        mu_config, sigma_run, sigma_eps, labels = draws
        df = standardized_bias_total(
            mu_config, sigma_run, sigma_eps, ref_idx=0, labels=labels,
        )
        assert len(df) == len(labels)


class TestBiasProbability:
    def test_exceed_mode(self, draws):
        mu_config, sigma_run, _, labels = draws
        df = bias_probability(
            mu_config, sigma_run, ref_idx=0, labels=labels,
            mode="exceed_band", band=1.0,
        )
        assert "prob" in df.columns
        assert df["prob"].between(0, 1).all()

    def test_within_equiv_mode(self, draws):
        mu_config, sigma_run, _, labels = draws
        df = bias_probability(
            mu_config, sigma_run, ref_idx=0, labels=labels,
            mode="within_equiv", alpha_equiv=0.5,
        )
        assert "prob" in df.columns

    def test_positive_mode(self, draws):
        mu_config, sigma_run, _, labels = draws
        df = bias_probability(
            mu_config, sigma_run, ref_idx=0, labels=labels,
            mode="positive",
        )
        # Reference config has Δμ = exactly 0 → P(β>0) = 0 (strict inequality)
        ref_p = float(df[df["Config"] == labels[0]]["prob"].iloc[0])
        assert ref_p == 0.0
        # Non-ref configs should have reasonable probabilities
        assert df["prob"].between(0, 1).all()
