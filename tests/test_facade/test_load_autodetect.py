"""Tests for auto-detection of ModelBuilder in BayesEpistemicModel.load()."""
from __future__ import annotations

import numpy as np
import pytest

from apeBayes import BayesEpistemicModel
from apeBayes.facade import _detect_builder, _variant_tag
from apeBayes.model import (
    FlatConfigModel,
    HierarchicalConfigModel,
    RandomSlopesInteractionModel,
    RandomSlopesModel,
)


class TestDetectBuilder:
    def test_v8_detected_from_gamma_sr(self, stub_posterior_v8):
        builder, tag = _detect_builder(stub_posterior_v8)
        assert tag == "v8"
        assert isinstance(builder, RandomSlopesInteractionModel)
        assert not builder._interaction_loading

    def test_v9_detected_from_xi_case(self, stub_posterior_v9):
        builder, tag = _detect_builder(stub_posterior_v9)
        assert tag == "v9"
        assert isinstance(builder, RandomSlopesInteractionModel)
        assert builder._interaction_loading

    def test_v1_to_v3_detected_when_only_sigma_run(self, stub_posterior_v3):
        builder, tag = _detect_builder(stub_posterior_v3)
        assert tag == "v1-v3"
        assert isinstance(builder, FlatConfigModel)

    def test_v4_to_v7_detected_from_lambda_case(self, stub_dataset):
        """Posterior with lambda_case but no gamma_sr / xi_case \u2192 RandomSlopesModel."""
        from apeBayes.posterior import PosteriorAccessor
        from tests.conftest import _make_idata  # type: ignore[import-not-found]

        rng = np.random.default_rng(99)
        S = 30
        sigma_src = np.abs(rng.normal(0.55, 0.05, size=S))
        # Manually splice lambda_case into a v3 idata.
        idata = _make_idata(sigma_src=sigma_src)
        import xarray as xr
        lam = rng.normal(1.0, 0.1, size=(1, S, 4))
        idata.posterior["lambda_case"] = xr.DataArray(
            lam, dims=("chain", "draw", "Case"),
            coords={"Case": ["A", "B", "C", "D"]},
        )
        post = PosteriorAccessor(idata, stub_dataset)
        builder, tag = _detect_builder(post)
        assert tag == "v4-v7"
        assert isinstance(builder, RandomSlopesModel)
        # And not the v8 subclass
        assert not isinstance(builder, RandomSlopesInteractionModel)


class TestVariantTag:
    def test_flat_is_v1_v3(self):
        assert _variant_tag(FlatConfigModel()) == "v1-v3"

    def test_hierarchical_is_v1_v3(self):
        """HierarchicalConfigModel has the same \u03c3_GM formula as Flat."""
        assert _variant_tag(HierarchicalConfigModel()) == "v1-v3"

    def test_random_slopes_is_v4_v7(self):
        assert _variant_tag(RandomSlopesModel()) == "v4-v7"

    def test_rsi_no_loading_is_v8(self):
        assert _variant_tag(RandomSlopesInteractionModel()) == "v8"

    def test_rsi_with_loading_is_v9(self):
        assert _variant_tag(RandomSlopesInteractionModel(interaction_loading=True)) == "v9"


class TestLoadRoundTrip:
    def test_autodetect_v8_from_netcdf(
        self, tmp_path, synthetic_long_df, default_config, stub_posterior_v8,
    ):
        """save()/load() a v8-stub posterior picks up RandomSlopesInteractionModel."""
        m = BayesEpistemicModel(synthetic_long_df, cfg=default_config)
        m._posterior = stub_posterior_v8
        m._builder = RandomSlopesInteractionModel()

        nc_path = tmp_path / "v8_stub.nc"
        m.save(nc_path)

        reloaded = BayesEpistemicModel.load(nc_path, synthetic_long_df, cfg=default_config)
        assert isinstance(reloaded.builder, RandomSlopesInteractionModel)
        assert not reloaded.builder._interaction_loading
        # And \u03c3_GM dispatch returns the v8 formula, not \u03c3_src.
        gm = reloaded.sigma_GM_draws()
        src = reloaded.sigma_src_draws()
        # Every draw: \u03c3_GM \u2265 \u03c3_src, and strictly greater on average.
        assert np.all(gm >= src - 1e-12)
        assert float(np.mean(gm)) > float(np.mean(src))

    def test_autodetect_v3(
        self, tmp_path, synthetic_long_df, default_config, stub_posterior_v3,
    ):
        """v1-v3 posterior autodetects to FlatConfigModel; \u03c3_GM == \u03c3_src."""
        m = BayesEpistemicModel(synthetic_long_df, cfg=default_config)
        m._posterior = stub_posterior_v3
        m._builder = FlatConfigModel()

        nc_path = tmp_path / "v3_stub.nc"
        m.save(nc_path)

        reloaded = BayesEpistemicModel.load(nc_path, synthetic_long_df, cfg=default_config)
        assert isinstance(reloaded.builder, FlatConfigModel)
        np.testing.assert_array_equal(
            reloaded.sigma_GM_draws(), reloaded.sigma_src_draws(),
        )

    def test_explicit_builder_matching_detected_accepted(
        self, tmp_path, synthetic_long_df, default_config, stub_posterior_v8,
    ):
        """User-supplied builder of the right variant is honored (preserves priors etc.)."""
        m = BayesEpistemicModel(synthetic_long_df, cfg=default_config)
        m._posterior = stub_posterior_v8
        m._builder = RandomSlopesInteractionModel()

        nc_path = tmp_path / "v8.nc"
        m.save(nc_path)

        # Pass an explicit v8 builder with a custom sigma_inter prior:
        custom = RandomSlopesInteractionModel(sigma_inter=0.2)
        reloaded = BayesEpistemicModel.load(
            nc_path, synthetic_long_df, cfg=default_config, builder=custom,
        )
        assert reloaded.builder is custom

    def test_explicit_builder_mismatch_raises(
        self, tmp_path, synthetic_long_df, default_config, stub_posterior_v8,
    ):
        """User passing FlatConfigModel for a v8 posterior is the old foot-gun; now raises."""
        m = BayesEpistemicModel(synthetic_long_df, cfg=default_config)
        m._posterior = stub_posterior_v8
        m._builder = RandomSlopesInteractionModel()

        nc_path = tmp_path / "v8_mismatch.nc"
        m.save(nc_path)

        with pytest.raises(ValueError, match=r"v1-v3.*v8|v8.*v1-v3"):
            BayesEpistemicModel.load(
                nc_path, synthetic_long_df, cfg=default_config,
                builder=FlatConfigModel(),
            )

    def test_explicit_v8_builder_against_v9_posterior_raises(
        self, tmp_path, synthetic_long_df, default_config, stub_posterior_v9,
    ):
        """Passing v8 builder for a v9 (xi_case present) posterior raises."""
        m = BayesEpistemicModel(synthetic_long_df, cfg=default_config)
        m._posterior = stub_posterior_v9
        m._builder = RandomSlopesInteractionModel(interaction_loading=True)

        nc_path = tmp_path / "v9.nc"
        m.save(nc_path)

        with pytest.raises(ValueError, match=r"v8.*v9|v9.*v8"):
            BayesEpistemicModel.load(
                nc_path, synthetic_long_df, cfg=default_config,
                builder=RandomSlopesInteractionModel(),  # v8, no interaction_loading
            )
