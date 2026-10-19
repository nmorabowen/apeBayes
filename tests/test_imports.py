"""Smoke tests: verify that all public modules import cleanly.

The imports are the assertions, so they go through ``importlib`` rather than
``from ... import ...`` statements. ``ruff check --fix`` (F401) saw those
statements as unused imports and deleted them in f60272d, leaving ``pass``.
"""

import importlib


def _assert_exports(module_name, *names):
    """Import ``module_name`` and assert that it exposes every name in ``names``."""
    module = importlib.import_module(module_name)
    missing = [name for name in names if not hasattr(module, name)]
    assert not missing, f"{module_name} does not expose {missing}"


def test_top_level_import():
    import apeBayes
    assert hasattr(apeBayes, "__version__")
    assert hasattr(apeBayes, "BayesEpistemicModel")


def test_config_imports():
    from apeBayes.config import ModelConfig
    _assert_exports("apeBayes.config", "PriorConfig", "SamplingConfig", "FactorSpec")
    cfg = ModelConfig()
    assert cfg.likelihood == "student_t"


def test_data_imports():
    _assert_exports("apeBayes.data", "EpistemicDataset", "encode_dataset")


def test_model_imports():
    _assert_exports(
        "apeBayes.model", "FlatConfigModel", "ModelBuilder", "sample_model", "compare_models",
    )


def test_posterior_imports():
    _assert_exports("apeBayes.posterior", "PosteriorAccessor")


def test_analysis_imports():
    _assert_exports(
        "apeBayes.analysis", "bias", "variance", "decomposition", "equivalence", "fitted",
    )


def test_diagnostics_imports():
    _assert_exports(
        "apeBayes.diagnostics",
        "rhat_table", "ess_table", "diagnostics_summary", "divergences_count",
        "posterior_predictive_check",
    )


def test_plots_imports():
    _assert_exports("apeBayes.plots", "apply_style", "PALETTE", "savefig")
    _assert_exports("apeBayes.plots.bias", "plot_mu_triptych", "plot_bias_forest")
    _assert_exports("apeBayes.plots.variance", "plot_variance_budget_bars")
    _assert_exports(
        "apeBayes.plots.equivalence", "plot_equivalence_matrix", "plot_equivalence_dendrogram",
    )
    _assert_exports("apeBayes.plots.posterior", "plot_mu_density", "plot_ppc", "plot_residuals")


def test_facade_import():
    _assert_exports("apeBayes.facade", "BayesEpistemicModel")
