"""Smoke tests: verify that all public modules import cleanly."""

import pytest


def test_top_level_import():
    import apeBayes
    assert hasattr(apeBayes, "__version__")
    assert hasattr(apeBayes, "BayesEpistemicModel")


def test_config_imports():
    from apeBayes.config import ModelConfig, PriorConfig, SamplingConfig, FactorSpec
    cfg = ModelConfig()
    assert cfg.likelihood == "student_t"


def test_data_imports():
    from apeBayes.data import EpistemicDataset, encode_dataset


def test_model_imports():
    from apeBayes.model import FlatConfigModel, ModelBuilder, sample_model, compare_models


def test_posterior_imports():
    from apeBayes.posterior import PosteriorAccessor


def test_analysis_imports():
    from apeBayes.analysis import bias, variance, decomposition, equivalence, fitted


def test_diagnostics_imports():
    from apeBayes.diagnostics import (
        rhat_table, ess_table, diagnostics_summary, divergences_count,
        posterior_predictive_check,
    )


def test_plots_imports():
    from apeBayes.plots import apply_style, PALETTE, savefig
    from apeBayes.plots.bias import plot_mu_triptych, plot_bias_forest
    from apeBayes.plots.variance import plot_variance_budget_bars
    from apeBayes.plots.equivalence import plot_equivalence_matrix, plot_equivalence_dendrogram
    from apeBayes.plots.posterior import plot_mu_density, plot_ppc, plot_residuals


def test_facade_import():
    from apeBayes.facade import BayesEpistemicModel
