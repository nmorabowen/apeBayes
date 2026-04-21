"""Round-trip tests for the self-contained apeBayes bundle format.

Covers:
- Directory bundle save/load auto-rehydrates df and cfg.
- Zip bundle save/load does the same.
- Legacy plain ``.nc`` save still works and still requires df on load.
- Overrides: user-supplied cfg on load beats the packed one.
- Schema drift: unknown schema_version raises a clear error.
- Tuple fields (ci, alpha_ladder, denominators) round-trip as tuples.
- FactorSpec.levels=None round-trips cleanly.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from apeBayes import BayesEpistemicModel
from apeBayes.config import (
    DecisionConfig,
    FactorSpec,
    ModelConfig,
    PriorConfig,
    SamplingConfig,
)
from apeBayes.facade import (
    _BUNDLE_CONFIG_NAME,
    _BUNDLE_DATA_NAME,
    _BUNDLE_IDATA_NAME,
    _deserialize_config,
    _serialize_config,
)
from apeBayes.model import FlatConfigModel


class TestConfigSerialization:
    def test_round_trip_default(self):
        cfg = ModelConfig(
            factors=[
                FactorSpec(name="SSI", column="Tier", levels=["1", "2", "3", "4"]),
                FactorSpec(name="Nonlinearity", column="Case", levels=["A", "B", "C", "D"]),
            ],
        )
        payload = _serialize_config(cfg)
        # JSON-serializable sanity check.
        round_tripped = _deserialize_config(json.loads(json.dumps(payload)))
        assert round_tripped == cfg

    def test_round_trip_preserves_tuples(self):
        cfg = ModelConfig(
            factors=[FactorSpec(name="SSI", column="Tier")],
            ci=(0.1, 0.9),
            decision=DecisionConfig(
                alpha_eq=0.5,
                alpha_ladder=(0.5, 1.0),
                p_star=0.9,
                denominators=("src", "gm"),
            ),
        )
        payload = _serialize_config(cfg)
        restored = _deserialize_config(json.loads(json.dumps(payload)))
        assert isinstance(restored.ci, tuple)
        assert restored.ci == (0.1, 0.9)
        assert isinstance(restored.decision.alpha_ladder, tuple)
        assert restored.decision.alpha_ladder == (0.5, 1.0)
        assert isinstance(restored.decision.denominators, tuple)
        assert restored.decision.denominators == ("src", "gm")

    def test_factor_levels_none_round_trips(self):
        cfg = ModelConfig(
            factors=[FactorSpec(name="SSI", column="Tier", levels=None)],
        )
        restored = _deserialize_config(
            json.loads(json.dumps(_serialize_config(cfg)))
        )
        assert restored.factors[0].levels is None

    def test_unknown_schema_version_raises(self):
        payload = _serialize_config(ModelConfig(
            factors=[FactorSpec(name="SSI", column="Tier")]
        ))
        payload["schema_version"] = 999
        with pytest.raises(ValueError, match="schema_version"):
            _deserialize_config(payload)


class TestBundleRoundTrip:
    def test_directory_bundle_round_trip(
        self, tmp_path, synthetic_long_df, default_config, stub_posterior_v3,
    ):
        m = BayesEpistemicModel(synthetic_long_df, cfg=default_config)
        m._posterior = stub_posterior_v3
        m._builder = FlatConfigModel()

        bundle = m.save(tmp_path / "roof")
        assert bundle.is_dir()
        assert bundle.name == "roof.apebayes"
        assert (bundle / _BUNDLE_IDATA_NAME).exists()
        assert (bundle / _BUNDLE_CONFIG_NAME).exists()
        assert (bundle / _BUNDLE_DATA_NAME).exists()

        # Bundle load needs only the path.
        reloaded = BayesEpistemicModel.load(bundle)
        assert reloaded.cfg == default_config
        pd.testing.assert_frame_equal(
            reloaded._df.reset_index(drop=True),
            synthetic_long_df.reset_index(drop=True),
            check_like=True,
        )
        assert isinstance(reloaded.builder, FlatConfigModel)

    def test_explicit_apebayes_suffix_not_duplicated(
        self, tmp_path, synthetic_long_df, default_config, stub_posterior_v3,
    ):
        m = BayesEpistemicModel(synthetic_long_df, cfg=default_config)
        m._posterior = stub_posterior_v3
        m._builder = FlatConfigModel()

        bundle = m.save(tmp_path / "roof.apebayes")
        assert bundle.name == "roof.apebayes"
        assert not bundle.name.endswith(".apebayes.apebayes")

    def test_zip_bundle_round_trip(
        self, tmp_path, synthetic_long_df, default_config, stub_posterior_v3,
    ):
        m = BayesEpistemicModel(synthetic_long_df, cfg=default_config)
        m._posterior = stub_posterior_v3
        m._builder = FlatConfigModel()

        zip_path = tmp_path / "roof.apebayes.zip"
        out = m.save(zip_path)
        assert out == zip_path
        assert zip_path.is_file()

        reloaded = BayesEpistemicModel.load(zip_path)
        assert reloaded.cfg == default_config
        pd.testing.assert_frame_equal(
            reloaded._df.reset_index(drop=True),
            synthetic_long_df.reset_index(drop=True),
            check_like=True,
        )

    def test_legacy_nc_still_requires_df(
        self, tmp_path, synthetic_long_df, default_config, stub_posterior_v3,
    ):
        """Plain-.nc save (pre-bundle era) still loads with explicit df/cfg."""
        m = BayesEpistemicModel(synthetic_long_df, cfg=default_config)
        m._posterior = stub_posterior_v3
        m._builder = FlatConfigModel()

        nc_path = tmp_path / "legacy.nc"
        m.save(nc_path)
        assert nc_path.is_file()

        # Missing df → clear error.
        with pytest.raises(ValueError, match="df="):
            BayesEpistemicModel.load(nc_path)

        # Supplying df still works.
        reloaded = BayesEpistemicModel.load(
            nc_path, synthetic_long_df, cfg=default_config,
        )
        assert reloaded.is_fitted

    def test_load_cfg_override_wins(
        self, tmp_path, synthetic_long_df, default_config, stub_posterior_v3,
    ):
        """Passing cfg= on load overrides the packed one (e.g. new DecisionConfig)."""
        m = BayesEpistemicModel(synthetic_long_df, cfg=default_config)
        m._posterior = stub_posterior_v3
        m._builder = FlatConfigModel()

        bundle = m.save(tmp_path / "roof")

        override = ModelConfig(
            factors=default_config.factors,
            edp_col=default_config.edp_col,
            station_col=default_config.station_col,
            run_col=default_config.run_col,
            ref_config=default_config.ref_config,
            likelihood=default_config.likelihood,
            heteroskedastic=default_config.heteroskedastic,
            priors=default_config.priors,
            sampling=default_config.sampling,
            decision=DecisionConfig(
                alpha_eq=0.7,
                alpha_ladder=(0.4, 0.7, 1.1),
                p_star=0.9,
                denominators=("gm",),
            ),
        )
        reloaded = BayesEpistemicModel.load(bundle, cfg=override)
        assert reloaded.cfg.decision.alpha_eq == 0.7
        assert reloaded.cfg.decision.denominators == ("gm",)
