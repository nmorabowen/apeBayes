"""Unit tests for apeBayes.analysis.validation — pure-function tests on synthetic draws."""

import numpy as np
import pandas as pd
import pytest

from apeBayes.analysis.validation import validation_decision, validation_parameter


@pytest.fixture
def draws():
    """Fake posterior: 500 samples, 4 configs. Config 1 intentionally high."""
    rng = np.random.default_rng(7)
    mu_config = rng.normal(0, 0.05, size=(500, 4))
    mu_config[:, 1] += np.log(2.0)     # ~2x ref
    mu_config[:, 2] += np.log(1.05)    # ~1.05x ref
    mu_config[:, 3] += np.log(0.8)     # ~0.8x ref
    labels = ["1A", "1B", "2A", "2B"]
    return mu_config, labels


class TestValidationParameter:
    def test_returns_tuple(self, draws):
        mu_config, labels = draws
        edp, summary = validation_parameter(
            mu_config, u0=np.log(0.01), ref_idx=0, labels=labels,
        )
        assert isinstance(edp, np.ndarray)
        assert edp.shape == (500, 4)
        assert isinstance(summary, pd.DataFrame)
        assert set(summary.columns) == {"Config", "edp_med", "edp_lo", "edp_hi"}

    def test_ref_row_equals_exp_u0(self, draws):
        mu_config, labels = draws
        u0 = np.log(0.01)
        edp, summary = validation_parameter(
            mu_config, u0=u0, ref_idx=0, labels=labels,
        )
        # Reference config draws differ only by the tiny zero-centred noise
        # added in the fixture → exp(u0 + Δμ) fluctuates mildly around exp(u0).
        np.testing.assert_allclose(
            float(summary.iloc[0]["edp_med"]), np.exp(u0), rtol=0.02,
        )

    def test_doubling_config_is_doubled(self, draws):
        mu_config, labels = draws
        u0 = np.log(0.01)
        _, summary = validation_parameter(mu_config, u0, 0, labels)
        ratio = float(summary.iloc[1]["edp_med"]) / float(summary.iloc[0]["edp_med"])
        np.testing.assert_allclose(ratio, 2.0, rtol=0.05)


class TestValidationDecisionThreshold:
    def test_pass_below_threshold(self, draws):
        mu_config, labels = draws
        # Threshold at 0.02, reference sits near 0.01 ⇒ most configs pass,
        # the 2× config (index 1) is borderline.
        df = validation_decision(
            mu_config, u0=np.log(0.01), ref_idx=0, labels=labels,
            rule="threshold", theta=0.02, direction="below",
        )
        assert set(df.columns) >= {
            "Config", "edp_med", "edp_lo", "edp_hi",
            "posterior_mass", "decision", "rule",
        }
        assert (df["rule"] == "threshold").all()
        # Ref row: ~all mass below 0.02 ⇒ pass
        assert df.iloc[0]["decision"] == "pass"
        assert df.iloc[0]["posterior_mass"] > 0.95

    def test_above_direction(self, draws):
        mu_config, labels = draws
        df = validation_decision(
            mu_config, u0=np.log(0.01), ref_idx=0, labels=labels,
            rule="threshold", theta=0.005, direction="above",
        )
        # Ref row sits near 0.01 > 0.005 ⇒ pass under "above"
        assert df.iloc[0]["decision"] == "pass"

    def test_theta_required(self, draws):
        mu_config, labels = draws
        with pytest.raises(ValueError, match="theta"):
            validation_decision(
                mu_config, u0=0.0, ref_idx=0, labels=labels, rule="threshold",
            )

    def test_bad_direction_raises(self, draws):
        mu_config, labels = draws
        with pytest.raises(ValueError, match="direction"):
            validation_decision(
                mu_config, u0=0.0, ref_idx=0, labels=labels,
                rule="threshold", theta=0.02, direction="sideways",
            )


class TestValidationDecisionFractional:
    def test_ref_inside_tolerance(self, draws):
        mu_config, labels = draws
        u0 = np.log(0.01)
        df = validation_decision(
            mu_config, u0=u0, ref_idx=0, labels=labels,
            rule="fractional", target=0.01, tau_val=0.30,
        )
        assert df.iloc[0]["decision"] == "pass"
        # 2x config way outside a 30% band
        assert df.iloc[1]["decision"] == "fail"

    def test_required_args(self, draws):
        mu_config, labels = draws
        with pytest.raises(ValueError, match="target and tau_val"):
            validation_decision(
                mu_config, u0=0.0, ref_idx=0, labels=labels,
                rule="fractional", tau_val=0.1,
            )

    def test_zero_target_raises(self, draws):
        mu_config, labels = draws
        with pytest.raises(ValueError, match="nonzero"):
            validation_decision(
                mu_config, u0=0.0, ref_idx=0, labels=labels,
                rule="fractional", target=0.0, tau_val=0.1,
            )


class TestValidationDecisionCIOverlap:
    def test_overlap_pass_fail(self, draws):
        mu_config, labels = draws
        u0 = np.log(0.01)
        df = validation_decision(
            mu_config, u0=u0, ref_idx=0, labels=labels,
            rule="ci_overlap", target_hdi=(0.009, 0.011),
        )
        # ref row overlaps 0.009–0.011; 2x config (~0.02) does not
        assert df.iloc[0]["decision"] == "pass"
        assert df.iloc[1]["decision"] == "fail"
        # Mass is 0/1 for the CI-overlap rule
        assert set(df["posterior_mass"].unique()).issubset({0.0, 1.0})

    def test_target_hdi_required(self, draws):
        mu_config, labels = draws
        with pytest.raises(ValueError, match="target_hdi"):
            validation_decision(
                mu_config, u0=0.0, ref_idx=0, labels=labels, rule="ci_overlap",
            )


def test_unknown_rule_raises(draws):
    mu_config, labels = draws
    with pytest.raises(ValueError, match="unknown rule"):
        validation_decision(
            mu_config, u0=0.0, ref_idx=0, labels=labels, rule="nope",
        )


def test_p_star_val_is_independent_of_paper_p_star(draws):
    """p_star_val default is 0.95 but is declared per use case (§7.4)."""
    import inspect
    sig = inspect.signature(validation_decision)
    assert sig.parameters["p_star_val"].default == 0.95
    # Must not be coupled to cfg.decision.p_star — function takes no cfg.
    assert "cfg" not in sig.parameters
