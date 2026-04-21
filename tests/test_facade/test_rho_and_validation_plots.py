"""Smoke tests for BayesEpistemicModel.plot_epistemic_median_ratio (§6) and
plot_validation_decision (§7 of uncertanty_measures.md)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from apeBayes import BayesEpistemicModel
from apeBayes.model import RandomSlopesInteractionModel


@pytest.fixture
def model_v8(synthetic_long_df, default_config, stub_posterior_v8):
    m = BayesEpistemicModel(synthetic_long_df, cfg=default_config)
    m._posterior = stub_posterior_v8
    m._builder = RandomSlopesInteractionModel()
    return m


# ── plot_epistemic_median_ratio ──────────────────────────────────────────

class TestPlotEpistemicMedianRatio:
    def test_returns_fig_ax(self, model_v8):
        fig, ax = model_v8.plot_epistemic_median_ratio()
        assert fig is not None and ax is not None
        plt.close(fig)

    def test_reference_row_dropped(self, model_v8):
        """The reference config (ρ=1, Δμ=0 by construction) is not plotted."""
        fig, ax = model_v8.plot_epistemic_median_ratio()
        labels = [t.get_text() for t in ax.get_yticklabels()]
        ref = model_v8.data.ref_label
        assert ref not in labels
        assert len(labels) == 15  # 4 tiers x 4 cases - 1 ref
        plt.close(fig)

    def test_default_order_is_tier(self, model_v8):
        """Default order matches order_config_labels(by='tier') — 1A, 1B, ..."""
        fig, ax = model_v8.plot_epistemic_median_ratio()
        labels = [t.get_text() for t in ax.get_yticklabels()]
        # Yticks are top-to-bottom, reversed from the sort order of the table.
        # The first ytick should be '1A'.
        assert labels[0] == "1A"
        plt.close(fig)

    def test_case_major_order(self, model_v8):
        fig, ax = model_v8.plot_epistemic_median_ratio(order="case")
        labels = [t.get_text() for t in ax.get_yticklabels()]
        # Case-major: 1A, 2A, 3A, 4A, 1B, ...
        assert labels[:4] == ["1A", "2A", "3A", "4A"]
        plt.close(fig)

    def test_log_scale_default(self, model_v8):
        fig, ax = model_v8.plot_epistemic_median_ratio()
        assert "Delta" in ax.get_xlabel() or "Δ" in ax.get_xlabel() or \
               r"\log \rho" in ax.get_xlabel()
        plt.close(fig)

    def test_linear_scale(self, model_v8):
        fig, ax = model_v8.plot_epistemic_median_ratio(log_scale=False)
        assert r"\rho_{\mathrm{epi}" in ax.get_xlabel()
        # No alpha-ladder labels in the legend on linear scale
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert not any("severe" in t for t in legend_texts)
        plt.close(fig)

    def test_show_alpha_ladder_false(self, model_v8):
        fig, ax = model_v8.plot_epistemic_median_ratio(show_alpha_ladder=False)
        legend = ax.get_legend()
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert not any("severe" in t for t in legend_texts)
        plt.close(fig)


# ── plot_validation_decision ─────────────────────────────────────────────

class TestPlotValidationDecision:
    def test_threshold_returns_fig_ax(self, model_v8):
        fig, ax = model_v8.plot_validation_decision(
            u0=np.log(0.01), rule="threshold", theta=0.02, direction="below",
        )
        assert fig is not None and ax is not None
        plt.close(fig)

    def test_threshold_reference_row_dropped(self, model_v8):
        fig, ax = model_v8.plot_validation_decision(
            u0=np.log(0.01), rule="threshold", theta=0.02,
        )
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert model_v8.data.ref_label not in labels
        assert len(labels) == 15
        plt.close(fig)

    def test_threshold_draws_theta_line(self, model_v8):
        fig, ax = model_v8.plot_validation_decision(
            u0=np.log(0.01), rule="threshold", theta=0.025, direction="below",
        )
        # Expect one vertical line at theta plus horizontal error-bar lines;
        # look for the theta legend entry.
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any(r"\theta" in t for t in legend_texts)
        plt.close(fig)

    def test_fractional_draws_target_band(self, model_v8):
        fig, ax = model_v8.plot_validation_decision(
            u0=np.log(0.01), rule="fractional", target=0.01, tau_val=0.30,
        )
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("target" in t for t in legend_texts)
        assert any(r"\tau" in t or "%" in t for t in legend_texts)
        plt.close(fig)

    def test_ci_overlap_draws_hdi_band(self, model_v8):
        fig, ax = model_v8.plot_validation_decision(
            u0=np.log(0.01), rule="ci_overlap", target_hdi=(0.008, 0.012),
        )
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("HDI" in t or "target" in t for t in legend_texts)
        plt.close(fig)

    def test_required_args_enforced(self, model_v8):
        # Threshold requires theta.
        with pytest.raises(ValueError, match="theta"):
            model_v8.plot_validation_decision(
                u0=0.0, rule="threshold",
            )

    def test_case_major_order(self, model_v8):
        fig, ax = model_v8.plot_validation_decision(
            u0=np.log(0.01), rule="threshold", theta=0.02,
            order="case",
        )
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels[:4] == ["1A", "2A", "3A", "4A"]
        plt.close(fig)
