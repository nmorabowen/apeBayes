"""
Tests for model construction (no MCMC sampling).

Verifies that FlatConfigModel.build() produces a valid PyMC model
with the expected free random variables, deterministics, and observed data.
"""

from __future__ import annotations

import pytest
import pymc as pm

from apeBayes.data import encode_dataset
from apeBayes.model.flat import FlatConfigModel


class TestFlatConfigModelBuild:
    """Build the PyMC model on synthetic data, verify structure."""

    @pytest.fixture()
    def dataset(self, synthetic_long_df, fast_config):
        return encode_dataset(synthetic_long_df, fast_config)

    def test_build_returns_pymc_model(self, dataset, fast_config):
        builder = FlatConfigModel()
        model = builder.build(dataset, fast_config)
        assert isinstance(model, pm.Model)

    def test_student_t_hetero_has_expected_vars(self, dataset, fast_config):
        builder = FlatConfigModel(likelihood="student_t", heteroskedastic=True)
        model = builder.build(dataset, fast_config)
        rv_names = {rv.name for rv in model.free_RVs}
        # Must have these free RVs
        assert "mu0" in rv_names
        assert "mu_config_free" in rv_names
        assert "delta_st_free" in rv_names
        assert "z_run_free" in rv_names
        assert "sigma_run" in rv_names
        assert "sigma_eps_config" in rv_names
        assert "nu_minus2" in rv_names

    def test_gaussian_homo_lacks_nu_and_hetero(self, dataset, fast_config):
        builder = FlatConfigModel(likelihood="gaussian", heteroskedastic=False)
        model = builder.build(dataset, fast_config)
        rv_names = {rv.name for rv in model.free_RVs}
        assert "nu_minus2" not in rv_names
        assert "sigma_eps_config" not in rv_names
        assert "sigma_eps" in rv_names

    def test_deterministics_present(self, dataset, fast_config):
        builder = FlatConfigModel()
        model = builder.build(dataset, fast_config)
        det_names = {det.name for det in model.deterministics}
        assert "mu_config" in det_names
        assert "delta_st" in det_names
        assert "b_run" in det_names

    def test_observed_data_shape(self, dataset, fast_config):
        builder = FlatConfigModel()
        model = builder.build(dataset, fast_config)
        obs_rv = model.observed_RVs[0]
        assert obs_rv.name == "y_obs"

    def test_coords_have_config_labels(self, dataset, fast_config):
        builder = FlatConfigModel()
        model = builder.build(dataset, fast_config)
        assert "Config" in model.coords
        assert len(model.coords["Config"]) == 16

    def test_description_property(self):
        b1 = FlatConfigModel()
        assert "Flat" in b1.description
        b2 = FlatConfigModel(likelihood="gaussian", heteroskedastic=False)
        assert "gaussian" in b2.description
        assert "homo" in b2.description
