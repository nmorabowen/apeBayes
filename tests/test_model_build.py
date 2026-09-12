"""
Tests for model construction (no MCMC sampling).

Verifies that FlatConfigModel.build() produces a valid PyMC model
with the expected free random variables, deterministics, and observed data.
"""

from __future__ import annotations

import pymc as pm
import pytest

from apeBayes.data import encode_dataset
from apeBayes.model.flat import FlatConfigModel
from apeBayes.model.random_slopes import RandomSlopesModel
from apeBayes.model.random_slopes_interaction import RandomSlopesInteractionModel


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


_POOLED_RV_NAMES = {"log_sigma_eps_bar", "tau_sigma_eps", "z_sigma_eps"}


class TestResidualPooling:
    """residual_pooling='partial' on RandomSlopesModel / RandomSlopesInteractionModel.

    Verifies the opt-in partially-pooled prior on the heteroskedastic
    residual scales builds the expected variables, that the default
    ("none") path is unchanged, and that an invalid value is rejected.
    """

    @pytest.fixture()
    def dataset(self, synthetic_long_df, fast_config):
        return encode_dataset(synthetic_long_df, fast_config)

    def test_random_slopes_partial_pooling_has_expected_vars(self, dataset, fast_config):
        builder = RandomSlopesModel(residual_pooling="partial")
        model = builder.build(dataset, fast_config)
        rv_names = {rv.name for rv in model.free_RVs}
        det_names = {det.name for det in model.deterministics}
        assert rv_names >= _POOLED_RV_NAMES
        assert "sigma_eps_config" not in rv_names
        assert "sigma_eps_config" in det_names
        sigma_eps_config = model["sigma_eps_config"]
        assert model.named_vars_to_dims[sigma_eps_config.name] == ("Config",)

    def test_random_slopes_interaction_partial_pooling_has_expected_vars(
        self, dataset, fast_config
    ):
        builder = RandomSlopesInteractionModel(residual_pooling="partial")
        model = builder.build(dataset, fast_config)
        rv_names = {rv.name for rv in model.free_RVs}
        det_names = {det.name for det in model.deterministics}
        assert rv_names >= _POOLED_RV_NAMES
        assert "sigma_eps_config" not in rv_names
        assert "sigma_eps_config" in det_names
        sigma_eps_config = model["sigma_eps_config"]
        assert model.named_vars_to_dims[sigma_eps_config.name] == ("Config",)

    def test_random_slopes_default_pooling_is_none(self, dataset, fast_config):
        builder = RandomSlopesModel()
        model = builder.build(dataset, fast_config)
        rv_names = {rv.name for rv in model.free_RVs}
        det_names = {det.name for det in model.deterministics}
        assert _POOLED_RV_NAMES.isdisjoint(rv_names)
        assert _POOLED_RV_NAMES.isdisjoint(det_names)
        assert "sigma_eps_config" in rv_names
        assert "sigma_eps_config" not in det_names

    def test_random_slopes_interaction_default_pooling_is_none(self, dataset, fast_config):
        builder = RandomSlopesInteractionModel()
        model = builder.build(dataset, fast_config)
        rv_names = {rv.name for rv in model.free_RVs}
        det_names = {det.name for det in model.deterministics}
        assert _POOLED_RV_NAMES.isdisjoint(rv_names)
        assert _POOLED_RV_NAMES.isdisjoint(det_names)
        assert "sigma_eps_config" in rv_names
        assert "sigma_eps_config" not in det_names

    def test_random_slopes_invalid_residual_pooling_raises(self):
        with pytest.raises(ValueError, match="residual_pooling"):
            RandomSlopesModel(residual_pooling="bogus")  # type: ignore[arg-type]

    def test_random_slopes_interaction_invalid_residual_pooling_raises(self):
        with pytest.raises(ValueError, match="residual_pooling"):
            RandomSlopesInteractionModel(residual_pooling="bogus")  # type: ignore[arg-type]

    def test_description_mentions_residual_pooling(self):
        builder = RandomSlopesModel(residual_pooling="partial")
        assert "residual_pooling=partial" in builder.description
        inter_builder = RandomSlopesInteractionModel(residual_pooling="partial")
        assert "residual_pooling=partial" in inter_builder.description
        assert "residual_pooling=partial" not in RandomSlopesModel().description

    def test_interaction_partial_pooling_custom_tau_is_halfnormal(self, dataset, fast_config):
        builder = RandomSlopesInteractionModel(residual_pooling="partial", residual_tau=1.5)
        model = builder.build(dataset, fast_config)
        op_class_name = type(model["tau_sigma_eps"].owner.op).__name__
        assert "HalfNormal" in op_class_name

    def test_interaction_partial_pooling_halfcauchy(self, dataset, fast_config):
        builder = RandomSlopesInteractionModel(
            residual_pooling="partial", residual_tau=1.5, residual_tau_dist="halfcauchy"
        )
        model = builder.build(dataset, fast_config)
        op_class_name = type(model["tau_sigma_eps"].owner.op).__name__
        assert "HalfCauchy" in op_class_name

    def test_invalid_residual_tau_raises(self):
        with pytest.raises(ValueError, match="residual_tau"):
            RandomSlopesModel(residual_tau=0)
        with pytest.raises(ValueError, match="residual_tau"):
            RandomSlopesInteractionModel(residual_tau=0)

    def test_invalid_residual_tau_dist_raises(self):
        with pytest.raises(ValueError, match="residual_tau_dist"):
            RandomSlopesModel(residual_tau_dist="bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="residual_tau_dist"):
            RandomSlopesInteractionModel(residual_tau_dist="bogus")  # type: ignore[arg-type]

    def test_description_mentions_default_residual_tau(self):
        builder = RandomSlopesModel(residual_pooling="partial")
        assert "tau~HalfNormal(0.5)" in builder.description
        inter_builder = RandomSlopesInteractionModel(residual_pooling="partial")
        assert "tau~HalfNormal(0.5)" in inter_builder.description
