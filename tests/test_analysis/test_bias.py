"""Unit tests for apeBayes.analysis.bias \u2014 pure-function tests on synthetic draws."""

import numpy as np
import pandas as pd
import pytest

from apeBayes.analysis.bias import bias_probability, standardized_bias


@pytest.fixture
def draws():
    """Fake posterior draws: 500 samples, 4 configs, with \u03c3_src and \u03c3_inter."""
    rng = np.random.default_rng(99)
    mu_config = rng.normal(0, 0.3, size=(500, 4))
    mu_config -= mu_config.mean(axis=1, keepdims=True)
    sigma_src = np.abs(rng.normal(0.20, 0.02, size=500))
    sigma_inter = np.abs(rng.normal(0.10, 0.01, size=500))
    sigma_eps = np.abs(rng.normal(0.05, 0.01, size=500))
    labels = ["1A", "1B", "2A", "2B"]
    return mu_config, sigma_src, sigma_inter, sigma_eps, labels


class TestStandardizedBias:
    def test_shape(self, draws):
        mu_config, sigma_src, *_, labels = draws
        df = standardized_bias(mu_config, sigma_src, ref_idx=0, labels=labels)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(labels)

    def test_ref_has_zero_bias(self, draws):
        mu_config, sigma_src, *_, labels = draws
        df = standardized_bias(mu_config, sigma_src, ref_idx=0, labels=labels)
        ref_row = df[df["Config"] == labels[0]]
        assert abs(float(ref_row["std_bias_med"].iloc[0])) < 1e-10

    def test_subset(self, draws):
        mu_config, sigma_src, *_ = draws
        subset_idx = np.array([0, 2])
        df = standardized_bias(
            mu_config, sigma_src, ref_idx=0,
            labels=["1A", "2A"], subset_idx=subset_idx,
        )
        assert len(df) == 2

    def test_columns_present(self, draws):
        mu_config, sigma_src, *_, labels = draws
        df = standardized_bias(mu_config, sigma_src, ref_idx=0, labels=labels)
        expected = {"Config", "std_bias_med", "std_bias_lo", "std_bias_hi",
                    "dmu_med", "dmu_lo", "dmu_hi", "mult_med", "mult_lo", "mult_hi"}
        assert expected == set(df.columns)

    def test_sigma_denom_is_respected(self, draws):
        """Different denominators produce different \u03b2 magnitudes but the same rank."""
        mu_config, sigma_src, sigma_inter, *_, labels = draws
        sigma_gm = np.sqrt(sigma_src ** 2 + sigma_inter ** 2)

        df_src = standardized_bias(mu_config, sigma_src, ref_idx=0, labels=labels)
        df_gm = standardized_bias(mu_config, sigma_gm, ref_idx=0, labels=labels)

        # \u03c3_GM > \u03c3_src pointwise \u2192 |\u03b2_GM| < |\u03b2_src| for every non-ref row
        for i in range(1, len(labels)):
            b_src = abs(df_src.iloc[i]["std_bias_med"])
            b_gm = abs(df_gm.iloc[i]["std_bias_med"])
            assert b_gm < b_src, (i, b_src, b_gm)

        # Rank order of magnitudes preserved (ignore the ref row which is 0)
        rank_src = np.argsort(np.abs(df_src["std_bias_med"].to_numpy()))
        rank_gm = np.argsort(np.abs(df_gm["std_bias_med"].to_numpy()))
        np.testing.assert_array_equal(rank_src, rank_gm)

    def test_2d_sigma_denom_for_sigma_pred(self, draws):
        """(S, K) denominator works for per-config \u03c3_pred."""
        mu_config, sigma_src, sigma_inter, sigma_eps, labels = draws
        sigma_gm = np.sqrt(sigma_src ** 2 + sigma_inter ** 2)
        # Heteroskedastic \u03c3_eps: broadcast to (S, K) with small per-config jitter.
        rng = np.random.default_rng(7)
        sigma_eps_k = sigma_eps[:, None] + rng.normal(0, 0.005, size=mu_config.shape)
        sigma_pred = np.sqrt(sigma_gm[:, None] ** 2 + sigma_eps_k ** 2)
        df = standardized_bias(mu_config, sigma_pred, ref_idx=0, labels=labels)
        assert len(df) == len(labels)
        # ref row still exactly 0
        assert abs(float(df.iloc[0]["std_bias_med"])) < 1e-10

    def test_bad_denom_shape_raises(self, draws):
        mu_config, *_, labels = draws
        bad = np.ones((500, 5))  # wrong K
        with pytest.raises(ValueError, match="incompatible"):
            standardized_bias(mu_config, bad, ref_idx=0, labels=labels)


class TestBiasProbability:
    def test_default_alpha_is_paper_spec(self):
        """alpha_equiv defaults to 0.4 per uncertanty_measures.md \u00a75."""
        import inspect
        sig = inspect.signature(bias_probability)
        assert sig.parameters["alpha_equiv"].default == 0.4
        assert sig.parameters["band"].default == 1.1
        assert sig.parameters["mode"].default == "within_equiv"

    def test_exceed_mode(self, draws):
        mu_config, sigma_src, *_, labels = draws
        df = bias_probability(
            mu_config, sigma_src, ref_idx=0, labels=labels,
            mode="exceed_band", band=1.1,
        )
        assert "prob" in df.columns
        assert df["prob"].between(0, 1).all()

    def test_within_equiv_mode(self, draws):
        mu_config, sigma_src, *_, labels = draws
        df = bias_probability(
            mu_config, sigma_src, ref_idx=0, labels=labels,
            mode="within_equiv", alpha_equiv=0.4,
        )
        assert "prob" in df.columns
        # Reference config: \u0394\u03bc = 0 \u2192 \u03b2 = 0 \u2192 P(|\u03b2| < 0.4) = 1
        ref_p = float(df[df["Config"] == labels[0]]["prob"].iloc[0])
        assert ref_p == 1.0

    def test_positive_mode(self, draws):
        mu_config, sigma_src, *_, labels = draws
        df = bias_probability(
            mu_config, sigma_src, ref_idx=0, labels=labels,
            mode="positive",
        )
        ref_p = float(df[df["Config"] == labels[0]]["prob"].iloc[0])
        assert ref_p == 0.0
        assert df["prob"].between(0, 1).all()

    def test_within_equiv_with_sigma_gm(self, draws):
        """Probability increases when the denominator grows (\u03c3_src \u2192 \u03c3_GM)."""
        mu_config, sigma_src, sigma_inter, *_, labels = draws
        sigma_gm = np.sqrt(sigma_src ** 2 + sigma_inter ** 2)
        df_src = bias_probability(
            mu_config, sigma_src, ref_idx=0, labels=labels,
            mode="within_equiv", alpha_equiv=0.4,
        )
        df_gm = bias_probability(
            mu_config, sigma_gm, ref_idx=0, labels=labels,
            mode="within_equiv", alpha_equiv=0.4,
        )
        # Larger denominator \u2192 |\u03b2| smaller \u2192 more mass inside the band
        for i in range(1, len(labels)):
            assert df_gm.iloc[i]["prob"] >= df_src.iloc[i]["prob"]
