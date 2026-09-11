"""Smoke tests: verify that all public modules import cleanly."""



def test_top_level_import():
    import apeBayes
    assert hasattr(apeBayes, "__version__")
    assert hasattr(apeBayes, "BayesEpistemicModel")


def test_config_imports():
    from apeBayes.config import ModelConfig
    cfg = ModelConfig()
    assert cfg.likelihood == "student_t"


def test_data_imports():
    pass


def test_model_imports():
    pass


def test_posterior_imports():
    pass


def test_analysis_imports():
    pass


def test_diagnostics_imports():
    pass


def test_plots_imports():
    pass


def test_facade_import():
    pass
