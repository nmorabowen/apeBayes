"""
BayesEpistemicModel — convenience facade.

Wires together data encoding, model building, sampling, posterior access,
and analysis into one object.  Notebooks call this; internals are
independently usable.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
import pandas as pd

from .analysis import bias, decomposition, equivalence, fitted, validation, variance
from .config import (
    DecisionConfig,
    FactorSpec,
    ModelConfig,
    PriorConfig,
    SamplingConfig,
)
from .data import EpistemicDataset, encode_dataset
from .diagnostics.convergence import (
    diagnostics_summary,
    divergences_count,
    ess_table,
    rhat_table,
)
from .diagnostics.validation import posterior_predictive_check
from .model.comparison import FittedVariant, compare_models
from .model.flat import FlatConfigModel
from .model.hierarchical import HierarchicalConfigModel
from .model.random_slopes import RandomSlopesModel
from .model.random_slopes_interaction import RandomSlopesInteractionModel
from .model.sampling import sample_model
from .plots.style import CMAP_DIV, CMAP_SEQ, FULL_WIDTH, HALF_WIDTH
from .posterior.accessor import PosteriorAccessor

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from arviz import InferenceData

    from .model.base import ModelBuilder


def _fmt(a: float) -> str:
    """Format an α value into a stable column-label suffix (e.g. 0.4 -> '0.4')."""
    # Strip trailing zeros, keep at least one decimal for floats that look int
    s = f"{a:g}"
    return s


# ── Bundle format (self-contained save/load) ──────────────────────────────
#
# A fitted model is persisted as a directory bundle with the ``.apebayes``
# suffix, containing three files:
#
#   <name>.apebayes/
#   ├── idata.nc       InferenceData (NetCDF)
#   ├── config.json    ModelConfig with schema_version (see _serialize_config)
#   └── data.parquet   Source long-format DataFrame
#
# Zipped bundles use the ``.apebayes.zip`` suffix and contain the same
# three members. Legacy plain ``.nc`` save/load still works for back-compat
# but requires the caller to supply ``df`` (and usually ``cfg``) at load
# time, because the plain NetCDF does not carry that metadata.

_BUNDLE_IDATA_NAME = "idata.nc"
_BUNDLE_CONFIG_NAME = "config.json"
_BUNDLE_DATA_NAME = "data.parquet"
_BUNDLE_SUFFIX = ".apebayes"
_BUNDLE_SCHEMA_VERSION = 1


def _serialize_config(cfg: ModelConfig) -> dict[str, Any]:
    """JSON-safe round-trip representation of a ``ModelConfig`` (schema v1).

    Tuples (``ci``, ``alpha_ladder``, ``denominators``) are emitted as JSON
    arrays; :func:`_deserialize_config` restores them as tuples. Literal-
    typed fields (``likelihood``, ``sampler``) go through as plain strings
    because the dataclass ``__post_init__`` validates them on reconstruction.
    """
    return {
        "schema_version": _BUNDLE_SCHEMA_VERSION,
        "config": {
            "factors": [
                {
                    "name": f.name,
                    "column": f.column,
                    "levels": (list(f.levels) if f.levels is not None else None),
                }
                for f in cfg.factors
            ],
            "config_col": cfg.config_col,
            "edp_col": cfg.edp_col,
            "station_col": cfg.station_col,
            "run_col": cfg.run_col,
            "ref_config": cfg.ref_config,
            "likelihood": cfg.likelihood,
            "heteroskedastic": cfg.heteroskedastic,
            "ci": list(cfg.ci),
            "config_sep": cfg.config_sep,
            "priors": {
                "sigma_intercept": cfg.priors.sigma_intercept,
                "sigma_config": cfg.priors.sigma_config,
                "sigma_station": cfg.priors.sigma_station,
                "sigma_src": cfg.priors.sigma_src,
                "sigma_eps": cfg.priors.sigma_eps,
                "nu_prior_lambda": cfg.priors.nu_prior_lambda,
            },
            "sampling": {
                "draws": cfg.sampling.draws,
                "tune": cfg.sampling.tune,
                "chains": cfg.sampling.chains,
                "target_accept": cfg.sampling.target_accept,
                "max_treedepth": cfg.sampling.max_treedepth,
                "seed": cfg.sampling.seed,
                "sampler": cfg.sampling.sampler,
            },
            "decision": {
                "alpha_eq": cfg.decision.alpha_eq,
                "alpha_ladder": list(cfg.decision.alpha_ladder),
                "p_star": cfg.decision.p_star,
                "denominators": list(cfg.decision.denominators),
            },
        },
    }


def _deserialize_config(payload: dict[str, Any]) -> ModelConfig:
    """Inverse of :func:`_serialize_config`. Raises on unknown schema_version."""
    sv = payload.get("schema_version")
    if sv != _BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Unknown bundle schema_version={sv!r}; this apeBayes build "
            f"understands schema_version={_BUNDLE_SCHEMA_VERSION}. Upgrade "
            f"or downgrade the library to match the bundle."
        )
    c = payload["config"]
    return ModelConfig(
        factors=[
            FactorSpec(
                name=f["name"],
                column=f["column"],
                levels=(list(f["levels"]) if f.get("levels") is not None else None),
            )
            for f in c["factors"]
        ],
        config_col=c["config_col"],
        edp_col=c["edp_col"],
        station_col=c["station_col"],
        run_col=c["run_col"],
        ref_config=c["ref_config"],
        likelihood=c["likelihood"],
        heteroskedastic=c["heteroskedastic"],
        ci=tuple(c["ci"]),
        config_sep=c.get("config_sep", ""),
        priors=PriorConfig(**c["priors"]),
        sampling=SamplingConfig(**c["sampling"]),
        decision=DecisionConfig(
            alpha_eq=c["decision"]["alpha_eq"],
            alpha_ladder=tuple(c["decision"]["alpha_ladder"]),
            p_star=c["decision"]["p_star"],
            denominators=tuple(c["decision"]["denominators"]),
        ),
    )


def _is_zip_bundle(path: Path) -> bool:
    """True iff the path ends in ``.apebayes.zip``."""
    s = str(path)
    return s.endswith(_BUNDLE_SUFFIX + ".zip")


def _is_nc_file(path: Path) -> bool:
    """True iff the path points to a plain ``.nc`` file (legacy format)."""
    return path.is_file() and path.suffix == ".nc"


def _resolve_bundle_dir_path(path: Path) -> Path:
    """Normalise a directory-bundle save path.

    - ``foo``           → ``foo.apebayes``
    - ``foo.apebayes``  → ``foo.apebayes``  (no double-suffix)
    """
    if path.suffix == _BUNDLE_SUFFIX:
        return path
    return path.with_name(path.name + _BUNDLE_SUFFIX)


# The 32 plot methods return some mix of plt.Figure, (Figure, Axes),
# (Figure, ndarray), etc. ``_stamp`` is generic over that return shape
# so wrapping a plot call preserves the precise type for mypy callers.
_R = TypeVar("_R")


def _variant_tag(builder: ModelBuilder) -> str:
    """Return a short tag describing the variant class.

    Used by :func:`_detect_builder` to compare a user-supplied builder
    against the variant inferred from the posterior.
    """
    if isinstance(builder, RandomSlopesInteractionModel):
        # pylint: disable=protected-access
        return "v9" if getattr(builder, "_interaction_loading", False) else "v8"
    if isinstance(builder, RandomSlopesModel):
        return "v4-v7"
    if isinstance(builder, (HierarchicalConfigModel, FlatConfigModel)):
        return "v1-v3"
    return type(builder).__name__


def _detect_builder(post: PosteriorAccessor) -> tuple[ModelBuilder, str]:
    """Infer which ModelBuilder produced a posterior from its variable set.

    Returns
    -------
    (builder, tag)
        A default-constructed builder matching the posterior's variant and
        the variant tag string (``'v9'`` / ``'v8'`` / ``'v4-v7'`` /
        ``'v1-v3'``).

    Notes
    -----
    Probes used (see ``uncertanty_measures.md`` §7 for the σ_GM formula
    associated with each variant):

    - ``xi_case``     present → v9 RandomSlopesInteractionModel (guard:
      σ_GM raises NotImplementedError, but we still want to route there
      so the error is correct rather than silently returning σ_src).
    - ``gamma_sr``    present → v8 RandomSlopesInteractionModel.
    - ``lambda_case`` present → v4-v7 RandomSlopesModel.
    - otherwise                → v1-v3; σ_GM is σ_src either way so we
      return FlatConfigModel (HierarchicalConfigModel has identical
      σ_GM, so the dispatch is equivalent).
    """
    if post.has_var("xi_case"):
        return RandomSlopesInteractionModel(interaction_loading=True), "v9"
    if post.has_var("gamma_sr"):
        return RandomSlopesInteractionModel(), "v8"
    if post.has_var("lambda_case"):
        return RandomSlopesModel(), "v4-v7"
    return FlatConfigModel(), "v1-v3"


class BayesEpistemicModel:
    """High-level API for Bayesian epistemic uncertainty analysis.

    Usage
    -----
    >>> model = BayesEpistemicModel(df_long, cfg=my_config)
    >>> model.fit()
    >>> model.standardized_bias_table()
    >>> model.variance_budget_table()
    >>> model.equivalence_probability_table(alpha=0.4)
    """

    # ── Construction & fitting ───────────────────────────────────────────

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: ModelConfig | None = None,
        *,
        name: str | None = None,
    ) -> None:
        self.cfg = cfg or ModelConfig()
        self.name = name
        # Retain the source DataFrame so ``.save()`` can pack it into the
        # bundle and so downstream consumers (tests, notebooks) can reach
        # the original frame via ``model._df`` without keeping a separate
        # reference. EpistemicDataset only keeps the encoded arrays.
        self._df: pd.DataFrame = df
        self.data: EpistemicDataset = encode_dataset(df, self.cfg)
        self._posterior: PosteriorAccessor | None = None
        self._builder: ModelBuilder | None = None
        self._sigma_gm_cache: np.ndarray | None = None
        # Plot behaviour: when self.name is set, every plot method stamps
        # a bottom-right "name: {self.name}" watermark on the figure so
        # printed/shared figures self-identify which model produced them.
        # Set to False to suppress globally for this model instance.
        self.show_model_name_on_plots: bool = True

    def fit(
        self,
        *,
        builder: ModelBuilder | None = None,
        prior_predictive: bool = True,
        posterior_predictive: bool = True,
    ) -> BayesEpistemicModel:
        """Build and sample the model.

        Parameters
        ----------
        builder : ModelBuilder, optional
            Custom model builder.  Defaults to FlatConfigModel with
            settings from self.cfg.
        prior_predictive : bool
            Sample prior predictive before MCMC.
        posterior_predictive : bool
            Sample posterior predictive after MCMC.

        Returns self for chaining.
        """
        if builder is None:
            builder = FlatConfigModel()
        self._builder = builder

        pm_model = builder.build(self.data, self.cfg)
        idata = sample_model(
            pm_model,
            self.cfg.sampling,
            prior_predictive=prior_predictive,
            posterior_predictive=posterior_predictive,
        )
        self._posterior = PosteriorAccessor(idata, self.data)
        self._sigma_gm_cache = None
        return self

    @property
    def posterior(self) -> PosteriorAccessor:
        """Return the posterior accessor; raise if not yet fitted."""
        if self._posterior is None:
            raise RuntimeError("Call fit() before accessing the posterior.")
        return self._posterior

    @property
    def idata(self) -> InferenceData:
        """Return the underlying ArviZ InferenceData object."""
        return self.posterior.idata

    @property
    def is_fitted(self) -> bool:
        """Return True if the model has been fitted."""
        return self._posterior is not None

    @property
    def builder(self) -> ModelBuilder:
        """Return the ModelBuilder that produced the posterior.

        Raises
        ------
        RuntimeError
            If the model has not been fitted or loaded with a builder.
        """
        if self._builder is None:
            raise RuntimeError(
                "No ModelBuilder registered. Call fit() or load(builder=...) first."
            )
        return self._builder

    # ── Canonical aleatory denominators ──────────────────────────────────

    def sigma_src_draws(self) -> np.ndarray:
        """(S,) posterior draws of σ_src. Alias for ``posterior.sigma_src()``."""
        return self.posterior.sigma_src()

    def sigma_GM_draws(self) -> np.ndarray:
        """(S,) posterior draws of σ_GM via per-variant dispatch.

        Dispatches to the fitted model's ``sigma_GM(post)`` method per
        ``uncertanty_measures.md`` §7. Cached after the first call.
        """
        if self._sigma_gm_cache is None:
            self._sigma_gm_cache = self.builder.sigma_GM(self.posterior)
        return self._sigma_gm_cache

    def sigma_pred_draws(self, ref_idx: int | None = None) -> np.ndarray:
        """(S,) posterior draws of σ_pred = √(σ_GM² + σ_eps_eff²).

        Parameters
        ----------
        ref_idx : int | None
            Residual to fold in. When ``None``, uses the reference config
            index from the dataset (matches the spec's default).
        """
        p = self.posterior
        if ref_idx is None:
            ref_idx = p.ref_idx
        return p.sigma_pred(self.sigma_GM_draws(), ref_idx=ref_idx)

    # ── Serialization ───────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Save the fitted model as a self-contained bundle.

        Three output formats, dispatched by the path suffix:

        - ``path.apebayes/``  (**default**) — directory bundle with
          ``idata.nc`` + ``config.json`` + ``data.parquet``. Adds the
          ``.apebayes`` suffix automatically if absent (e.g. passing
          ``"roof"`` writes ``roof.apebayes/``).
        - ``path.apebayes.zip`` — same three members inside a zip archive.
        - ``path.nc``  (**legacy**) — plain NetCDF of the posterior only.
          ``load()`` then requires the caller to supply ``df`` and
          ``cfg`` explicitly.

        Bundles round-trip the full ``ModelConfig`` (with ``schema_version``
        for future compat) and the source DataFrame, so
        ``BayesEpistemicModel.load(bundle_path)`` needs nothing but the
        path.

        Parameters
        ----------
        path : str or Path
            Output path. Suffix selects the format as above.

        Returns
        -------
        Path to the saved bundle/file.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        path = Path(path)
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model. Call fit() first.")

        # Legacy NetCDF path — unchanged behaviour.
        if path.suffix == ".nc":
            self.idata.to_netcdf(str(path))
            return path

        # Zip bundle — build in a temp dir then zip the three members.
        if _is_zip_bundle(path):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                self.idata.to_netcdf(str(tmp / _BUNDLE_IDATA_NAME))
                (tmp / _BUNDLE_CONFIG_NAME).write_text(
                    json.dumps(_serialize_config(self.cfg), indent=2),
                    encoding="utf-8",
                )
                self._df.to_parquet(
                    tmp / _BUNDLE_DATA_NAME, engine="pyarrow", compression="snappy",
                )
                with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for name in (
                        _BUNDLE_IDATA_NAME, _BUNDLE_CONFIG_NAME, _BUNDLE_DATA_NAME,
                    ):
                        zf.write(tmp / name, arcname=name)
            return path

        # Directory bundle.
        bundle_dir = _resolve_bundle_dir_path(path)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        self.idata.to_netcdf(str(bundle_dir / _BUNDLE_IDATA_NAME))
        (bundle_dir / _BUNDLE_CONFIG_NAME).write_text(
            json.dumps(_serialize_config(self.cfg), indent=2),
            encoding="utf-8",
        )
        self._df.to_parquet(
            bundle_dir / _BUNDLE_DATA_NAME, engine="pyarrow", compression="snappy",
        )
        return bundle_dir

    @classmethod
    def load(
        cls,
        path: str | Path,
        df: pd.DataFrame | None = None,
        cfg: ModelConfig | None = None,
        *,
        name: str | None = None,
        builder: ModelBuilder | None = None,
    ) -> BayesEpistemicModel:
        """Reload a fitted model from a bundle, zip, or legacy NetCDF.

        Three accepted inputs, dispatched by what ``path`` points at:

        - **Directory bundle** (``*.apebayes/``) — reads ``idata.nc`` +
          ``config.json`` + ``data.parquet``. ``df`` and ``cfg`` are
          auto-rehydrated; passing them as kwargs overrides the packed
          versions (useful when migrating or tweaking ``DecisionConfig``).
        - **Zip bundle** (``*.apebayes.zip``) — same as above, extracted
          to a temp directory for the load.
        - **Legacy NetCDF** (``*.nc``) — plain posterior only. Requires
          the caller to supply ``df``; raises ``ValueError`` otherwise.

        The ``ModelBuilder`` is auto-detected from the posterior's variable
        set so σ_GM dispatch routes to the correct variant without user
        input. Probes:

        - ``xi_case``     present → v9 ``RandomSlopesInteractionModel``
        - ``gamma_sr``    present → v8 ``RandomSlopesInteractionModel``
        - ``lambda_case`` present → v4-v7 ``RandomSlopesModel``
        - otherwise                → v1-v3 ``FlatConfigModel``

        Parameters
        ----------
        path : str or Path
            Bundle directory, ``.apebayes.zip`` archive, or ``.nc`` file.
        df : pd.DataFrame, optional
            Override the bundled DataFrame. **Required** for legacy
            ``.nc`` inputs. For bundles, omit unless you know what you
            are doing.
        cfg : ModelConfig, optional
            Override the bundled configuration (e.g. swap in a new
            ``DecisionConfig`` to re-interpret an old fit under a
            different α ladder).
        name : str, optional
            Human-readable model name.
        builder : ModelBuilder, optional
            Override the auto-detected variant. A mismatch raises
            ``ValueError`` rather than silently returning the wrong σ_GM.

        Returns
        -------
        BayesEpistemicModel
            Fully reconstructed fitted model.

        Raises
        ------
        ValueError
            - If ``path`` is a plain ``.nc`` file and ``df`` is not
              supplied.
            - If the bundle's ``schema_version`` is not understood.
            - If a supplied ``builder`` disagrees with the variant
              implied by the posterior's variables.
        """
        import arviz as az

        path = Path(path)

        # Route 1 — zip bundle. Extract to a tempdir and recurse with the
        # directory path; the tempdir is cleaned up after the model is
        # constructed (idata is already loaded in memory by then).
        if _is_zip_bundle(path):
            with tempfile.TemporaryDirectory() as tmp:
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(tmp)
                return cls.load(
                    Path(tmp), df=df, cfg=cfg, name=name, builder=builder,
                )

        # Route 2 — legacy plain .nc. Requires df (and usually cfg).
        if _is_nc_file(path):
            if df is None:
                raise ValueError(
                    f"Legacy .nc file {path.name!r} carries no source "
                    f"DataFrame. Pass df=... explicitly, or save the "
                    f"model as an .apebayes bundle to round-trip df/cfg."
                )
            idata = az.from_netcdf(str(path))
            obj = cls(df, cfg=cfg, name=name)

        else:
            # Route 3 — directory bundle. Path may or may not have the
            # .apebayes suffix; we just need the three members inside.
            if not path.is_dir():
                raise ValueError(
                    f"Path {path!r} is not an .apebayes directory bundle, "
                    f"not an .apebayes.zip archive, and not a .nc file."
                )
            idata_file = path / _BUNDLE_IDATA_NAME
            cfg_file = path / _BUNDLE_CONFIG_NAME
            data_file = path / _BUNDLE_DATA_NAME
            for member, fp in (
                ("idata.nc", idata_file),
                ("config.json", cfg_file),
                ("data.parquet", data_file),
            ):
                if not fp.exists():
                    raise ValueError(
                        f"Bundle at {path!r} is missing required member "
                        f"{member!r}."
                    )

            if cfg is None:
                cfg = _deserialize_config(
                    json.loads(cfg_file.read_text(encoding="utf-8")),
                )
            if df is None:
                df = pd.read_parquet(data_file, engine="pyarrow")

            idata = az.from_netcdf(str(idata_file))
            obj = cls(df, cfg=cfg, name=name)

        post = PosteriorAccessor(idata, obj.data)
        obj._posterior = post

        detected, detected_tag = _detect_builder(post)
        if builder is None:
            obj._builder = detected
        else:
            user_tag = _variant_tag(builder)
            if user_tag != detected_tag:
                raise ValueError(
                    f"Supplied builder is {user_tag} "
                    f"({type(builder).__name__}) but the posterior's "
                    f"variable set implies {detected_tag}. Drop the "
                    f"builder kwarg to accept the detected variant, "
                    f"or re-check the source bundle."
                )
            obj._builder = builder
        return obj

    # ── Analysis: bias ───────────────────────────────────────────────────

    def standardized_bias_table(
        self,
        ref: str | None = None,
        configs: list[str] | None = None,
        *,
        denominator: Literal["src", "gm", "pred"] = "gm",
    ) -> pd.DataFrame:
        """Standardised epistemic bias β = Δμ / σ_denominator.

        Parameters
        ----------
        ref : str, optional
            Reference configuration label. Defaults to ``self.data.ref_label``.
        configs : list[str], optional
            Subset of configuration labels to report. Defaults to all.
        denominator : {"src", "gm", "pred"}, default "gm"
            Aleatory SD used in the β denominator; see
            ``uncertanty_measures.md`` §4 for the full definition.

            - ``"src"`` — σ_src alone (pure source variability). Conservative:
              smaller denominator ⇒ larger |β|. Matches the v6-era paper
              convention; call it β_src in the sensitivity appendix.
            - ``"gm"`` — canonical σ_GM = √(σ_src² + σ_inter²). Station-
              specific aleatory spread under the DRM suite. Paper default;
              the P* = 0.95 gate is defined on this denominator.
            - ``"pred"`` — σ_pred = √(σ_GM² + σ_eps_eff²) with the
              Student-t SD correction √(ν/(ν-2)) folded into σ_eps_eff.
              Generous: absorbs the configuration-specific residual into
              the aleatory pool, so |β_pred| < |β|.

        Returns
        -------
        pd.DataFrame
            One row per configuration. Columns: ``Config``,
            ``std_bias_{med,lo,hi}``, ``dmu_{med,lo,hi}``,
            ``mult_{med,lo,hi}``.
        """
        p = self.posterior
        ref_idx = self.data.config_label_to_idx(ref) if ref else p.ref_idx
        labels, subset_idx = self.data.subset_config_indices(configs)
        sigma_denom = self._sigma_denom_for(denominator, ref_idx)
        return bias.standardized_bias(
            p.mu_config(), sigma_denom, ref_idx, labels,
            ci=self.cfg.ci, subset_idx=subset_idx,
        )

    def bias_probability_table(
        self,
        *,
        mode: Literal["exceed_band", "within_equiv", "positive"] = "within_equiv",
        band: float | None = None,
        alpha_equiv: float | None = None,
        ref: str | None = None,
        configs: list[str] | None = None,
        denominator: Literal["src", "gm", "pred"] = "gm",
    ) -> pd.DataFrame:
        """Posterior probability summaries of β under one of three modes.

        Parameters
        ----------
        mode : {"exceed_band", "within_equiv", "positive"}, default "within_equiv"
            Which probability to compute per configuration:

            - ``"exceed_band"`` — P(|β| > ``band``). Reads as "severely
              biased at this α level"; use with ``band = α_ladder[-1]``
              (paper default 1.1) for the top-rung "severely biased"
              reference.
            - ``"within_equiv"`` — P(|β| < ``alpha_equiv``). This is the
              paper's headline P_eq (ROPE mass inside the equivalence
              band). Use with ``alpha_equiv = α_eq`` (paper default 0.4);
              the P* = 0.95 gate is applied to this column in
              ``decision_report``.
            - ``"positive"`` — P(β > 0). Direction-of-bias probability;
              useful for answering "does configuration (t, c) over- or
              under-predict relative to the reference?"
        band : float, optional
            Threshold for the ``"exceed_band"`` mode. Defaults to
            ``cfg.decision.alpha_ladder[-1]`` (paper default 1.1).
        alpha_equiv : float, optional
            Threshold for the ``"within_equiv"`` mode. Defaults to
            ``cfg.decision.alpha_eq`` (paper default 0.4).
        ref : str, optional
            Reference configuration label. Defaults to
            ``self.data.ref_label``.
        configs : list[str], optional
            Subset of configuration labels to report. Defaults to all.
        denominator : {"src", "gm", "pred"}, default "gm"
            Aleatory SD used as β denominator. Same semantics as in
            :meth:`standardized_bias_table` — see that docstring for
            the full "src / gm / pred" definitions.

        Returns
        -------
        pd.DataFrame
            Columns: ``Config``, ``beta_med``, ``prob``, ``prob_label``.
        """
        p = self.posterior
        ref_idx = self.data.config_label_to_idx(ref) if ref else p.ref_idx
        labels, subset_idx = self.data.subset_config_indices(configs)
        dec = self.cfg.decision
        if alpha_equiv is None:
            alpha_equiv = dec.alpha_eq
        if band is None:
            band = dec.alpha_ladder[-1]
        sigma_denom = self._sigma_denom_for(denominator, ref_idx)
        return bias.bias_probability(
            p.mu_config(), sigma_denom, ref_idx, labels,
            mode=mode, band=band, alpha_equiv=alpha_equiv,
            subset_idx=subset_idx,
        )

    def epistemic_median_ratio_table(
        self,
        ref: str | None = None,
        configs: list[str] | None = None,
        *,
        r_band: float | None = 1.25,
    ) -> pd.DataFrame:
        """Epistemic Median Ratio ρ_epi = exp(Δμ) per ``uncertanty_measures.md`` §6.

        Physical-unit companion to β. Computed from the mean structure of the
        posterior only — no σ_GM dependency — so the CI stays bounded in the
        regime where β is lever-arm inflated (§6.3). Report **alongside** β,
        not as a replacement; the equivalence decision gate remains P_eq on β
        (§5, §8).

        Parameters
        ----------
        ref : str, optional
            Reference configuration label. Defaults to
            ``self.data.ref_label``.
        configs : list[str], optional
            Subset of configuration labels. Defaults to all.
        r_band : float or None, default 1.25
            Engineering ratio for the post-hoc P_rho = P(ρ_epi ∈ [1/r, r])
            column (§6.5). ``None`` omits the column. **Not** a decision
            gate; the gate is P_eq on β.

        Returns
        -------
        pd.DataFrame
            Columns: ``Config``, ``rho_{med,lo,hi}``, ``dmu_{med,lo,hi}``,
            and ``P_rho`` when ``r_band`` is given.
        """
        p = self.posterior
        ref_idx = self.data.config_label_to_idx(ref) if ref else p.ref_idx
        labels, subset_idx = self.data.subset_config_indices(configs)
        return bias.epistemic_median_ratio(
            p.mu_config(), ref_idx, labels, ci=self.cfg.ci,
            r_band=r_band, subset_idx=subset_idx,
        )

    def _sigma_denom_for(
        self, denominator: Literal["src", "gm", "pred"], ref_idx: int,
    ) -> np.ndarray:
        """Resolve the requested aleatory denominator to an (S,) array."""
        if denominator == "src":
            return self.sigma_src_draws()
        if denominator == "gm":
            return self.sigma_GM_draws()
        if denominator == "pred":
            return self.sigma_pred_draws(ref_idx=ref_idx)
        raise ValueError(f"Unknown denominator {denominator!r}")

    # ── Analysis: headline decision report ───────────────────────────────

    def decision_report(
        self,
        *,
        ref: str | None = None,
        configs: list[str] | None = None,
        denominators: tuple[str, ...] | None = None,
        alpha_eq: float | None = None,
        alpha_ladder: tuple[float, ...] | None = None,
        p_star: float | None = None,
    ) -> pd.DataFrame:
        """Single-shot equivalence decision table per ``uncertanty_measures.md``.

        Computes β_src, β_GM, β_pred (whichever ``denominators`` selects)
        and the α-ladder P_eq table, then applies the P* gate at α_eq
        under the canonical σ_GM to label each configuration as
        ``equivalent`` / ``inequivalent`` / ``undecided``. The
        ``denominator_robust`` column flags configurations where the
        three denominators agree on the label.

        Parameters
        ----------
        ref : str, optional
            Reference configuration label. Defaults to
            ``self.data.ref_label``.
        configs : list[str], optional
            Subset of configurations. Defaults to all.
        denominators : tuple[str, ...], optional
            Subset of ``{"src", "gm", "pred"}``. Defaults to
            ``cfg.decision.denominators`` (the full triple).

            - ``"src"`` — σ_src only. β_src columns will appear.
            - ``"gm"`` — σ_GM = √(σ_src² + σ_inter²). **Required.** The P*
              gate and the headline ``decision`` column are computed on
              this denominator; dropping it raises ``ValueError``.
            - ``"pred"`` — σ_pred with residual folded in. β_pred columns
              will appear.

            Drop a denominator only for a reduced summary table; the paper's
            Results section requires the full triple to demonstrate
            denominator robustness.
        alpha_eq : float, optional
            Equivalence threshold for the decision column. Defaults to
            ``cfg.decision.alpha_eq`` (0.4). Must be a member of
            ``alpha_ladder``.
        alpha_ladder : tuple[float, ...], optional
            α values reported as P_eq columns (under σ_GM). Defaults to
            ``cfg.decision.alpha_ladder`` (0.4, 0.7, 1.1).
        p_star : float, optional
            Probability gate: ``P_eq ≥ p_star`` ⇒ ``"equivalent"``,
            ``P_eq ≤ 1 − p_star`` ⇒ ``"inequivalent"``, else
            ``"undecided"``. Defaults to ``cfg.decision.p_star`` (0.95).

        Returns
        -------
        pd.DataFrame
            One row per configuration (ordered by label). Columns:

            - ``Config``
            - ``beta_{src,gm,pred}_{med,lo,hi}`` for every requested denom
            - ``P_eq_{\u03b1}_gm`` for each α in the ladder
            - ``P_eq_{\u03b1_eq}_{src,pred}`` (sensitivity at α_eq)
            - ``decision`` (``equivalent`` | ``inequivalent`` | ``undecided``)
            - ``denominator_robust`` (bool)
        """
        dec = self.cfg.decision
        if alpha_eq is None:
            alpha_eq = dec.alpha_eq
        if alpha_ladder is None:
            alpha_ladder = dec.alpha_ladder
        if p_star is None:
            p_star = dec.p_star
        if denominators is None:
            denominators = dec.denominators

        if alpha_eq not in alpha_ladder:
            raise ValueError(
                f"alpha_eq ({alpha_eq}) must be in alpha_ladder {alpha_ladder}."
            )
        for d in denominators:
            if d not in ("src", "gm", "pred"):
                raise ValueError(f"Unknown denominator {d!r}")
        if "gm" not in denominators:
            raise ValueError(
                "decision_report requires 'gm' among denominators "
                "(it is the canonical denominator for the P* gate)."
            )

        p = self.posterior
        ref_idx = self.data.config_label_to_idx(ref) if ref else p.ref_idx
        labels, subset_idx = self.data.subset_config_indices(configs)
        mu_config = p.mu_config()
        mu_sel = mu_config[:, subset_idx] if subset_idx is not None else mu_config
        dmu = mu_sel - mu_config[:, [ref_idx]]  # (S, M)
        ci_lo, ci_hi = self.cfg.ci

        out = pd.DataFrame({"Config": labels})

        # Pre-compute each requested denominator once.
        denom_arrays: dict[str, np.ndarray] = {}
        for d in denominators:
            if d not in ("src", "gm", "pred"):
                raise ValueError(f"Unknown denominator {d!r}")
            denom_arrays[d] = self._sigma_denom_for(d, ref_idx)  # type: ignore[arg-type]

        # β summaries per denominator
        for d in denominators:
            beta = dmu / denom_arrays[d][:, None]
            out[f"beta_{d}_med"] = np.median(beta, axis=0)
            out[f"beta_{d}_lo"] = np.quantile(beta, ci_lo, axis=0)
            out[f"beta_{d}_hi"] = np.quantile(beta, ci_hi, axis=0)

        # P_eq ladder under canonical σ_GM
        gm_denom = denom_arrays["gm"]
        beta_gm = dmu / gm_denom[:, None]
        abs_beta_gm = np.abs(beta_gm)
        for a in alpha_ladder:
            out[f"P_eq_{_fmt(a)}_gm"] = np.mean(abs_beta_gm < a, axis=0)

        # P_eq at α_eq under the sensitivity denominators (if requested)
        for d in denominators:
            if d == "gm":
                continue
            beta_d = dmu / denom_arrays[d][:, None]
            out[f"P_eq_{_fmt(alpha_eq)}_{d}"] = np.mean(np.abs(beta_d) < alpha_eq, axis=0)

        # Decision column under σ_GM
        p_eq_col = out[f"P_eq_{_fmt(alpha_eq)}_gm"].to_numpy()
        decision = np.where(
            p_eq_col >= p_star, "equivalent",
            np.where(p_eq_col <= 1.0 - p_star, "inequivalent", "undecided"),
        )
        out["decision"] = decision

        # Denominator-robust = all requested denoms give the same decision
        decisions_by_denom = {}
        for d in denominators:
            col = out[f"P_eq_{_fmt(alpha_eq)}_{d}"] if d != "gm" else p_eq_col
            col_arr = col.to_numpy() if hasattr(col, "to_numpy") else col
            decisions_by_denom[d] = np.where(
                col_arr >= p_star, "equivalent",
                np.where(col_arr <= 1.0 - p_star, "inequivalent", "undecided"),
            )
        first_d = denominators[0]
        robust = np.ones(len(labels), dtype=bool)
        for d in denominators[1:]:
            robust &= decisions_by_denom[d] == decisions_by_denom[first_d]
        out["denominator_robust"] = robust

        return out

    # ── Analysis: variance ───────────────────────────────────────────────

    def variance_budget_table(
        self,
        *,
        include_interaction: bool | None = None,
    ) -> pd.DataFrame:
        """Posterior variance budget.

        Parameters
        ----------
        include_interaction : bool, optional
            If True, add a fifth "Interaction Var(γ_sr)" row (requires
            a v8+ interaction model).  If None (default), include it
            automatically when the posterior contains ``gamma_sr``.
            If False, always return the legacy 4-row budget for
            backward compatibility with v1–v7 pipelines.
        """
        p = self.posterior
        if include_interaction is None:
            include_interaction = p.has_interaction
        gamma = p.gamma_sr() if include_interaction else None
        return variance.variance_budget(
            p.mu_config(), p.delta_st(), p.sigma_run(), p.sigma_eps(),
            self.data.config_idx, ci=self.cfg.ci, nu=p.nu(),
            gamma_sr=gamma,
        )

    def variance_component_table(
        self,
        configs: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute per-configuration variance components."""
        p = self.posterior
        labels = configs or p.config_labels
        return variance.variance_components(
            p.sigma_run(), p.sigma_eps(), labels,
            ci=self.cfg.ci, nu=p.nu(),
        )

    # ── Analysis: random-slopes parameters (v4+ λ, v8+ σ_inter, v9+ ξ) ──

    def lambda_case_table(self) -> pd.DataFrame:
        """Posterior summary of per-Case run-sensitivity loadings λ."""
        p = self.posterior
        lam = p.lambda_case()
        if lam is None:
            msg = "No lambda_case in posterior. Fit with RandomSlopesModel."
            raise RuntimeError(msg)
        case_labels = p.case_labels or [f"Case{i}" for i in range(lam.shape[1])]
        ci_lo, ci_hi = self.cfg.ci
        return pd.DataFrame({
            "Case": case_labels,
            "lambda_med": np.median(lam, axis=0),
            "lambda_lo": np.quantile(lam, ci_lo, axis=0),
            "lambda_hi": np.quantile(lam, ci_hi, axis=0),
        })

    def xi_case_table(self) -> pd.DataFrame:
        """Posterior summary of per-Case interaction-sensitivity loadings ξ."""
        p = self.posterior
        xi = p.xi_case()
        if xi is None:
            msg = (
                "No xi_case in posterior. Fit with "
                "RandomSlopesInteractionModel(interaction_loading=True)."
            )
            raise RuntimeError(msg)
        case_labels = p.case_labels or [f"Case{i}" for i in range(xi.shape[1])]
        ci_lo, ci_hi = self.cfg.ci
        return pd.DataFrame({
            "Case": case_labels,
            "xi_med": np.median(xi, axis=0),
            "xi_lo": np.quantile(xi, ci_lo, axis=0),
            "xi_hi": np.quantile(xi, ci_hi, axis=0),
        })

    def sigma_inter_table(self) -> pd.DataFrame:
        """Posterior summary of the station×rupture interaction scale σ_inter."""
        p = self.posterior
        sig = p.sigma_inter()
        if sig is None:
            msg = "No sigma_inter in posterior. Fit with RandomSlopesInteractionModel."
            raise RuntimeError(msg)
        ci_lo, ci_hi = self.cfg.ci
        return pd.DataFrame({
            "component": ["sigma_inter"],
            "med": [float(np.median(sig))],
            "lo": [float(np.quantile(sig, ci_lo))],
            "hi": [float(np.quantile(sig, ci_hi))],
        })

    def sigma_rupture_table(self) -> pd.DataFrame:
        """Posterior summary of σ_run (named sigma_rupture for paper clarity)."""
        p = self.posterior
        sig = p.sigma_run()
        ci_lo, ci_hi = self.cfg.ci
        return pd.DataFrame({
            "component": ["sigma_run"],
            "med": [float(np.median(sig))],
            "lo": [float(np.quantile(sig, ci_lo))],
            "hi": [float(np.quantile(sig, ci_hi))],
        })

    # ── Analysis: decomposition ──────────────────────────────────────────

    def axiswise_decomposition_table(
        self,
        *,
        ratio_mode: Literal["var_over_sigma", "var_over_sigma2"] = "var_over_sigma",
    ) -> pd.DataFrame:
        """Decompose the μ surface into tier, case, and interaction effects.

        Parameters
        ----------
        ratio_mode : {"var_over_sigma", "var_over_sigma2"}, default "var_over_sigma"
            How to scale each axis-wise variance contribution when reporting
            it as an "energy share". Both use σ_run (source variability) as
            the reference scale.

            - ``"var_over_sigma"`` — contribution scaled by ``σ_run``
              (standard-deviation units). Matches the scale on which β is
              reported, so numbers read as "shift in standardised-bias
              units per axis". Default.
            - ``"var_over_sigma2"`` — contribution scaled by ``σ_run²``
              (variance units). Dimensionally consistent with a classical
              ANOVA variance-share table; percentages add up to the
              total explained variance more cleanly.
        """
        p = self.posterior
        f0_levels, f1_levels, grid = self.data.factor_index_grid()
        decomp = decomposition.axiswise_decomposition_draws(
            p.mu_config(), f0_levels, f1_levels, grid, p.sigma_run(),
        )
        return decomposition.axiswise_table(
            decomp, ci=self.cfg.ci, ratio_mode=ratio_mode,
            factor_names=(self.cfg.factors[0].name, self.cfg.factors[1].name),
        )

    def level_ranking_tables(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute tier- and case-level ranking tables with interaction."""
        p = self.posterior
        f0_levels, f1_levels, grid = self.data.factor_index_grid()
        decomp = decomposition.axiswise_decomposition_draws(
            p.mu_config(), f0_levels, f1_levels, grid, p.sigma_run(),
        )
        return decomposition.level_ranking_tables(
            decomp, ci=self.cfg.ci,
            factor_names=(self.cfg.factors[0].name, self.cfg.factors[1].name),
        )

    # ── Analysis: equivalence ────────────────────────────────────────────

    def equivalence_probability_table(
        self,
        *,
        alpha: float | None = None,
        ref: str | None = None,
        configs: list[str] | None = None,
        denominator: Literal["src", "gm", "pred"] = "gm",
    ) -> pd.DataFrame:
        """P_eq(t, c; α) = P(|β_{t,c}| < α) per configuration vs reference.

        Parameters
        ----------
        alpha : float, optional
            Equivalence radius on the β scale. Defaults to
            ``cfg.decision.alpha_eq`` (paper default 0.4).
        ref : str, optional
            Reference configuration label. Defaults to
            ``self.data.ref_label``.
        configs : list[str], optional
            Subset of configurations. Defaults to all.
        denominator : {"src", "gm", "pred"}, default "gm"
            Aleatory SD used as β denominator. Same semantics as in
            :meth:`standardized_bias_table`. Paper uses ``"gm"`` for the
            headline P_eq and reports ``"src"`` / ``"pred"`` in the
            sensitivity appendix.
        """
        p = self.posterior
        ref_idx = self.data.config_label_to_idx(ref) if ref else p.ref_idx
        labels, subset_idx = self.data.subset_config_indices(configs)
        if alpha is None:
            alpha = self.cfg.decision.alpha_eq
        sigma_denom = self._sigma_denom_for(denominator, ref_idx)
        return equivalence.equivalence_probability(
            p.mu_config(), sigma_denom, ref_idx, labels,
            alpha=alpha, ci=self.cfg.ci, subset_idx=subset_idx,
        )

    def equivalence_sweep_table(
        self,
        *,
        alphas: np.ndarray | None = None,
        ref: str | None = None,
        configs: list[str] | None = None,
        denominator: Literal["src", "gm", "pred"] = "gm",
    ) -> pd.DataFrame:
        """Sweep P_eq across a range of α thresholds (long-format DataFrame).

        Parameters
        ----------
        alphas : np.ndarray, optional
            α values to sweep. If None, uses a default grid covering the
            α ladder.
        ref, configs : optional
            Same as :meth:`equivalence_probability_table`.
        denominator : {"src", "gm", "pred"}, default "gm"
            Aleatory SD used as β denominator. See
            :meth:`standardized_bias_table` for the three choices'
            semantics.
        """
        p = self.posterior
        ref_idx = self.data.config_label_to_idx(ref) if ref else p.ref_idx
        labels, subset_idx = self.data.subset_config_indices(configs)
        sigma_denom = self._sigma_denom_for(denominator, ref_idx)
        return equivalence.equivalence_sweep(
            p.mu_config(), sigma_denom, ref_idx, labels,
            alphas=alphas, subset_idx=subset_idx,
        )

    def epistemic_equivalence_matrix(
        self,
        *,
        alpha: float | None = None,
        configs: list[str] | None = None,
        denominator: Literal["src", "gm"] = "gm",
    ) -> tuple[list[str], np.ndarray]:
        """Pairwise P(|μ_i − μ_j| < α·σ_denom) matrix across all configs.

        Parameters
        ----------
        alpha : float, optional
            Equivalence radius. Defaults to ``cfg.decision.alpha_eq``.
        configs : list[str], optional
            Subset of configurations to include in the matrix. Defaults to all.
        denominator : {"src", "gm"}, default "gm"
            Aleatory SD used as β denominator. ``"pred"`` is **not
            supported** here — pairwise comparisons share a single
            station-level aleatory scale per draw, so a per-configuration
            σ_pred would have no unambiguous "which config's residual"
            answer. Use ``"gm"`` for the paper's canonical pairwise matrix
            or ``"src"`` for the v6-era sensitivity view.

        Returns
        -------
        (labels, matrix) : (list[str], np.ndarray)
            Ordered config labels and an ``(M, M)`` symmetric probability
            matrix with ones on the diagonal.
        """
        p = self.posterior
        labels, subset_idx = self.data.subset_config_indices(configs)
        if alpha is None:
            alpha = self.cfg.decision.alpha_eq
        sigma_denom = self._sigma_denom_for(denominator, p.ref_idx)
        return equivalence.epistemic_equivalence_matrix(
            p.mu_config(), sigma_denom,
            alpha=alpha, labels=labels, subset_idx=subset_idx,
        )

    def epistemic_clusters_table(
        self,
        *,
        alpha: float | None = None,
        configs: list[str] | None = None,
        method: str = "average",
        threshold: float | None = None,
        n_clusters: int | None = None,
        denominator: Literal["src", "gm"] = "gm",
    ) -> pd.DataFrame:
        r"""Hierarchical clustering of configurations by epistemic distance.

        Builds a distance matrix ``D = 1 − P_eq`` and runs SciPy linkage +
        flat-cluster extraction.

        Parameters
        ----------
        alpha : float, optional
            Equivalence radius. Defaults to ``cfg.decision.alpha_eq``.
        configs : list[str], optional
            Subset of configurations. Defaults to all.
        method : {"single", "complete", "average", "weighted", \\
                  "centroid", "median", "ward"}, default "average"
            Linkage method passed to
            ``scipy.cluster.hierarchy.linkage``.

            - ``"single"`` — nearest-neighbour; chain-prone.
            - ``"complete"`` — furthest-neighbour; compact clusters.
            - ``"average"`` — UPGMA, balances chaining vs compactness.
              Good general default for 16-config grids.
            - ``"weighted"`` — WPGMA (average with equal branch weight).
            - ``"centroid"``, ``"median"``, ``"ward"`` — centroid-based;
              assume Euclidean geometry, less natural on probability
              distances.
        threshold : float, optional
            Distance cutoff for flat clusters. Mutually exclusive with
            ``n_clusters``. If both are None, defaults to ``1 − α`` so
            that two configs with ``P_eq < α`` land in the same cluster.
        n_clusters : int, optional
            If set, cut the linkage tree to produce exactly this many
            flat clusters (``criterion="maxclust"``).
        denominator : {"src", "gm"}, default "gm"
            Aleatory SD for β. ``"pred"`` is not supported (same pairwise
            ambiguity as :meth:`epistemic_equivalence_matrix`).

        Returns
        -------
        pd.DataFrame
            Columns: ``Config``, ``cluster``, ``leaf_order``; sorted by
            cluster then leaf order.
        """
        p = self.posterior
        labels, subset_idx = self.data.subset_config_indices(configs)
        if alpha is None:
            alpha = self.cfg.decision.alpha_eq
        sigma_denom = self._sigma_denom_for(denominator, p.ref_idx)
        return equivalence.epistemic_clusters(
            p.mu_config(), sigma_denom,
            alpha=alpha, labels=labels, subset_idx=subset_idx,
            method=method, threshold=threshold, n_clusters=n_clusters,
        )

    # ── Analysis: validation parameter ───────────────────────────────────

    def validation_decision_table(
        self,
        u0: float,
        *,
        rule: Literal["threshold", "fractional", "ci_overlap"],
        theta: float | None = None,
        direction: Literal["below", "above"] = "below",
        target: float | None = None,
        tau_val: float | None = None,
        target_hdi: tuple[float, float] | None = None,
        p_star_val: float = 0.95,
        ref: str | None = None,
        configs: list[str] | None = None,
    ) -> pd.DataFrame:
        """Case-conditional validation decision per ``uncertanty_measures.md`` §7.

        Computes ÊDP^med = exp(u_0 + Δμ) on the posterior and applies one of
        the three §7.4 rules (``threshold`` / ``fractional`` / ``ci_overlap``).

        ``u_0`` MUST be **externally anchored** (§7.3): experimental benchmark,
        higher-fidelity reference simulation, code-specified capacity
        threshold, or a specified null such as linear-elastic. An internally
        computed cross-configuration mean of ``mu_config`` is NOT admissible
        — that question is already answered by :meth:`epistemic_median_ratio_table`.

        ``tau_val`` and ``p_star_val`` are per use case and are NOT inherited
        from ``cfg.decision.alpha_eq`` / ``cfg.decision.p_star`` (§7.4, §7.5).

        Parameters
        ----------
        u0 : float
            Externally anchored log-scale reference for the case.
        rule : {"threshold", "fractional", "ci_overlap"}
            §7.4 rule selection.
        theta : float, optional
            Threshold on ÊDP^med (required for ``rule='threshold'``).
        direction : {"below", "above"}, default "below"
            Passing side of ``theta``.
        target : float, optional
            External benchmark EDP in physical units (required for
            ``rule='fractional'``).
        tau_val : float, optional
            Fractional tolerance (required for ``rule='fractional'``).
        target_hdi : tuple[float, float], optional
            Physical-unit interval on the benchmark (required for
            ``rule='ci_overlap'``).
        p_star_val : float, default 0.95
            Posterior-mass gate for pass/fail. Per use case.
        ref : str, optional
            Reference configuration label. Defaults to
            ``self.data.ref_label``.
        configs : list[str], optional
            Subset of configurations. Defaults to all.

        Returns
        -------
        pd.DataFrame
            One row per configuration. Columns: ``Config``,
            ``edp_{med,lo,hi}``, ``posterior_mass``, ``decision``, ``rule``.
        """
        p = self.posterior
        ref_idx = self.data.config_label_to_idx(ref) if ref else p.ref_idx
        labels, subset_idx = self.data.subset_config_indices(configs)
        return validation.validation_decision(
            p.mu_config(), u0, ref_idx, labels,
            rule=rule, theta=theta, direction=direction,
            target=target, tau_val=tau_val, target_hdi=target_hdi,
            p_star_val=p_star_val, ci=self.cfg.ci, subset_idx=subset_idx,
        )

    # ── Analysis: fitted values ──────────────────────────────────────────

    def fitted_values(self) -> pd.DataFrame:
        """Compute posterior-mean fitted values and residuals."""
        p = self.posterior
        return fitted.posterior_mean_fitted(
            self.data.y, p.mu0(), p.mu_config(), p.delta_st(), p.b_run(),
            self.data.config_idx, self.data.station_idx, self.data.run_idx,
        )

    def r2_table(self) -> pd.DataFrame:
        """Compute posterior R-squared summary."""
        p = self.posterior
        return fitted.posterior_r2(
            self.data.y, p.mu0(), p.mu_config(), p.delta_st(), p.b_run(),
            self.data.config_idx, self.data.station_idx, self.data.run_idx,
            ci=self.cfg.ci,
        )

    def mu_hat_table(self, configs: list[str] | None = None) -> pd.DataFrame:
        """Compute configuration intercepts with posterior summaries."""
        p = self.posterior
        labels, subset_idx = self.data.subset_config_indices(configs)
        return fitted.mu_hat_table(
            p.mu0(), p.mu_config(), labels, ci=self.cfg.ci, subset_idx=subset_idx,
        )

    def delta_hat_table(self) -> pd.DataFrame:
        """Compute station random-effect posterior summaries."""
        p = self.posterior
        return fitted.delta_hat_table(p.delta_st(), p.station_labels, ci=self.cfg.ci)

    # ── Diagnostics ──────────────────────────────────────────────────────

    def diagnostics_summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """Return convergence diagnostics for all or selected parameters."""
        return diagnostics_summary(self.idata, var_names)

    def divergences_count(self) -> int:
        """Return the number of divergent transitions."""
        return divergences_count(self.idata)

    def rhat_table(self, **kwargs: Any) -> pd.DataFrame:
        """Return R-hat convergence statistics."""
        return rhat_table(self.idata, **kwargs)

    def ess_table(self, **kwargs: Any) -> pd.DataFrame:
        """Return effective sample size statistics."""
        return ess_table(self.idata, **kwargs)

    def posterior_predictive_check(self) -> pd.DataFrame:
        """Run posterior predictive check and return summary statistics."""
        p = self.posterior
        y_rep = p.y_rep()
        if y_rep is None:
            raise RuntimeError(
                "No posterior predictive samples. "
                "Refit with posterior_predictive=True."
            )
        return posterior_predictive_check(
            self.data.y, y_rep,
            config_idx=self.data.config_idx,
            config_labels=p.config_labels,
        )

    # ── Plots ───────────────────────────────────────────────────────────
    #
    # Every plot method exposes the underlying plot function's user-facing
    # kwargs explicitly. Plumbing kwargs (draw arrays, indices, column
    # names) are computed by the facade and not surfaced. No **kwargs
    # escape hatch — IDE autocomplete and the type checker see the real
    # parameter surface.
    #
    # Every plot return is threaded through ``self._stamp`` which, when
    # ``self.name`` is set and ``self.show_model_name_on_plots`` is True,
    # adds a small ``"name: <self.name>"`` watermark at the bottom-right
    # of the figure. This makes printed/shared plots self-identify the
    # model that produced them without touching titles or layout.

    def _stamp(self, result: _R) -> _R:
        """Annotate the figure in ``result`` with ``self.name``; return unchanged.

        Draws the bare model name as a small grey line at the top-centre
        of the figure (figure coords, above any existing suptitle or axes
        title). No-op when ``self.name`` is None or
        ``self.show_model_name_on_plots`` is False.

        ``result`` is either a bare ``plt.Figure`` or a tuple whose first
        element is a figure. Generic over the result shape so wrapping a
        plot call preserves the precise return type for the caller (e.g.,
        a ``tuple[plt.Figure, plt.Axes]`` stays that exact type).

        See Also
        --------
        :meth:`_stamp_and_save` — use this instead at plot-method call
        sites so that the stamp runs **before** ``savefig`` and the
        on-disk PDF includes the name. This bare ``_stamp`` is kept for
        the one plot (``plot_mu_triptych``) whose underlying function
        saves multiple PDFs with computed filenames we can't intercept.
        """
        if not self.name or not self.show_model_name_on_plots:
            return result
        # Treat fig as Any for the annotation call — the generic _R
        # makes mypy lose sight of the matplotlib Figure attributes.
        fig: Any = result[0] if isinstance(result, tuple) else result
        # A plot backend that lacks fig.text (vanishingly unlikely, but
        # e.g. a mocked Figure in tests) shouldn't break the plot.
        with contextlib.suppress(Exception):
            fig.text(
                0.5, 0.995, self.name,
                ha="center", va="top",
                fontsize=8, color="0.35", alpha=0.85,
                transform=fig.transFigure,
            )
        return result

    def _stamp_and_save(
        self,
        result: _R,
        *,
        out_dir: str | Path | None,
        prefix: str,
        filename: str | None,
    ) -> _R:
        """Stamp the figure with ``self.name`` then save to disk.

        Drop-in replacement for the old pattern
        ``self._stamp(_plot(..., out_dir=out_dir, ...))`` which had the
        save/stamp order reversed (the inner ``savefig`` ran **before**
        the stamp, so the on-disk PDF never included the name). Plot
        methods now pass ``out_dir=None`` to the inner ``_plot`` to
        suppress the internal save, then call this method which stamps
        first and saves second.
        """
        self._stamp(result)
        if out_dir is not None and filename is not None:
            from .plots.helpers import savefig

            fig: Any = result[0] if isinstance(result, tuple) else result
            savefig(fig, out_dir, filename, prefix=prefix)
        return result

    # Diagnostics ........................................................

    def plot_rhat_bar(
        self,
        *,
        top_n: int | None = 10,
        threshold: float = 1.01,
        figsize: tuple[float, float] = (HALF_WIDTH, HALF_WIDTH),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "rhat_bar.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot R-hat convergence bar chart."""
        from .plots.diagnostics import plot_rhat_bar as _plot
        return self._stamp_and_save(
            _plot(
                self.rhat_table(),
                top_n=top_n, threshold=threshold,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_ess_bar(
        self,
        *,
        kind: str = "bulk",
        top_n: int | None = 10,
        threshold: float = 400.0,
        figsize: tuple[float, float] = (HALF_WIDTH, HALF_WIDTH),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot effective sample size bar chart.

        Parameters
        ----------
        kind : {"bulk", "tail"}, default "bulk"
            Which ESS flavour to plot.

            - ``"bulk"`` — ESS in the centre of the posterior (rank-based).
              Check this first; usually the binding constraint for reliable
              posterior means and medians.
            - ``"tail"`` — ESS in the tails (quantile-based). Check when
              the 5th/95th CI is what matters (e.g., β decision intervals).
              Typically 2× smaller than bulk ESS for heavy-tailed posteriors.
        top_n : int, optional
            If set, show only the ``top_n`` worst-ESS parameter groups.
        threshold : float, default 400.0
            ESS line drawn on the plot as a "good enough" marker. ESS
            ≥ 400 is the rule-of-thumb floor for reliable quantiles from
            a 4-chain run.
        figsize, out_dir, prefix, filename
            Standard figure-save quartet. ``filename=None`` yields the
            default ``f"ess_{kind}_bar.pdf"``.
        """
        from .plots.diagnostics import plot_ess_bar as _plot
        return self._stamp_and_save(
            _plot(
                self.ess_table(),
                kind=kind, top_n=top_n, threshold=threshold,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    # Bias .............................................................

    def plot_mu_triptych(
        self,
        *,
        ref: str | None = None,
        original_edp_scale: bool = False,
        annot: bool = True,
        fmt_mu: str = ".2f",
        fmt_dmu: str = ".2f",
        fmt_ratio: str = ".2f",
        cmap: str = CMAP_DIV,
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Three-panel heatmap: absolute μ, Δμ vs ref, median EDP ratio.

        Parameters
        ----------
        ref : str, optional
            Reference configuration label for the Δμ and ratio panels.
            Defaults to ``self.data.ref_label``.
        original_edp_scale : bool, default False
            If True, exponentiate μ values back to the native EDP scale
            (``exp(μ)``). Leave False for the log-EDP scale used
            throughout the paper.
        annot : bool, default True
            Annotate each heatmap cell with its numeric value.
        fmt_mu, fmt_dmu, fmt_ratio : str, default ".2f"
            Python format specs (mini-language) for the three panels'
            cell annotations. Examples: ``".2f"`` (default), ``".0%"``
            (percent, no decimals), ``".3g"`` (3-significant-figure
            general), ``".2e"`` (scientific notation). The full format
            spec reference is at
            https://docs.python.org/3/library/string.html#format-specification-mini-language.
        cmap : str, default ``CMAP_DIV``
            Matplotlib colormap name used for all three panels. Use a
            diverging palette so Δμ reads correctly around zero (the
            package default ``CMAP_DIV`` already is). Common built-in
            choices: ``"RdBu_r"``, ``"coolwarm"``, ``"PuOr"``.
        figsize, out_dir, prefix
            Standard figure-save triplet (no ``filename`` — this plot
            writes three PDFs with fixed suffixes).
        """
        ref = ref or self.data.ref_label
        mu_hat = self.mu_hat_table()
        bias_df = self.standardized_bias_table()
        from .plots.bias import plot_mu_triptych as _plot
        return self._stamp(_plot(
            mu_hat, bias_df,
            ref=ref, original_edp_scale=original_edp_scale,
            annot=annot, fmt_mu=fmt_mu, fmt_dmu=fmt_dmu, fmt_ratio=fmt_ratio,
            cmap=cmap, figsize=figsize, out_dir=out_dir, prefix=prefix,
        ))

    def plot_standardized_bias(
        self,
        *,
        station_subplots: bool = False,
        dot_alpha: float = 0.28,
        posterior_style: str = "violin",
        figsize: tuple[float, float] | None = None,
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "standardized_bias.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot layered standardized-bias figure with density and raw data.

        Uses σ_GM (canonical denominator) for the standardised axis.
        The β draws, raw-data dots, and reference label are computed
        from the fit and injected automatically.

        Parameters
        ----------
        station_subplots : bool, default False
            If True, produce one panel per station instead of one combined
            panel — useful for visualising between-station heterogeneity.
        dot_alpha : float, default 0.28
            Alpha for the raw per-observation dots layered under the
            posterior density. Drop lower (~0.1) if you have many runs
            per configuration.
        posterior_style : {"violin", "ridge", "none"}, default "violin"
            How to render the posterior β density per configuration.

            - ``"violin"`` — symmetric violin centred on the β median,
              with matplotlib's default smoothing. Best for the main-text
              figure where vertical space is at a premium.
            - ``"ridge"`` — KDE ridge baseline with the density stacked
              above. Reads more naturally when configurations are compared
              by shape as well as centre; uses more vertical room.
            - ``"none"`` — no density layer; only the median/CI whiskers
              and the raw-data dots. Use when overlaying the figure on
              another panel or when the density adds noise to a compact
              layout.
        figsize, out_dir, prefix, filename
            Standard figure-save quartet. ``figsize=None`` auto-selects
            based on whether ``station_subplots`` is set.
        """
        bias_df = self.standardized_bias_table()
        from .plots.bias import plot_standardized_bias as _plot

        d = self.data
        p = self.posterior
        mu_config = p.mu_config()                 # (S, K)
        sigma_denom = self.sigma_GM_draws()       # (S,)
        ref_idx = list(d.config_labels).index(d.ref_label)
        ref_draws = mu_config[:, ref_idx]         # (S,)
        denom_med = float(np.median(sigma_denom))

        # Posterior β draws (for violins) — divide by per-draw σ_GM so the
        # violin is the same β posterior the CI whiskers summarise. Both
        # therefore include denominator (σ_GM) uncertainty, not just Δμ.
        beta_draws = (mu_config - ref_draws[:, None]) / sigma_denom[:, None]
        beta_labels = list(d.config_labels)

        # Raw per-observation dots: one per (station, rupture) pair — no
        # cross-station averaging. Paired subtraction at the same (s, r)
        # cancels μ₀, δ_s, γ_{s,r} and λ_c·b_r at the linear-predictor
        # level, so each raw dot is a realised standardised bias plus ε
        # noise (and the λ_c ≠ 1 residual on b_r for non-reference Cases).
        n_configs = d.n_configs
        n_stations = d.n_stations
        n_runs = d.n_runs
        n_pairs = n_stations * n_runs
        raw_dots = np.full((n_configs, n_pairs), np.nan)

        ref_mask = d.config_idx == ref_idx
        ref_y_map: dict[tuple[int, int], float] = {
            (int(d.station_idx[i]), int(d.run_idx[i])): float(d.y[i])
            for i in np.where(ref_mask)[0]
        }

        for k in range(n_configs):
            for i in np.where(d.config_idx == k)[0]:
                s = int(d.station_idx[i])
                r = int(d.run_idx[i])
                ref_val = ref_y_map.get((s, r))
                if ref_val is None:
                    continue
                col = s * n_runs + r
                raw_dots[k, col] = (float(d.y[i]) - ref_val) / denom_med
        raw_means = np.nanmean(raw_dots, axis=1)

        return self._stamp_and_save(
            _plot(
                bias_df,
                beta_draws=beta_draws, beta_labels=beta_labels,
                raw_dots=raw_dots, raw_means=raw_means, raw_labels=beta_labels,
                station_subplots=station_subplots,
                ref_label=d.ref_label,
                denom_name="GM",
                alpha_eq=self.cfg.decision.alpha_eq,
                dot_alpha=dot_alpha, posterior_style=posterior_style,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_radar_bias_probability(
        self,
        *,
        alpha: float | None = None,
        order: Literal["tier", "case", "input"] = "tier",
        fill_alpha: float = 0.15,
        figsize: tuple[float, float] = (HALF_WIDTH, HALF_WIDTH),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "radar_bias_probability.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot radar chart of equivalence probabilities per configuration.

        Parameters
        ----------
        alpha : float, optional
            Equivalence radius on the β scale. Defaults to
            ``cfg.decision.alpha_eq``.
        order : {"tier", "case", "input"}, default "tier"
            Angular layout of the spokes around the polar plot.

            - ``"tier"`` — tier-major: ``1A, 1B, 1C, 1D, 2A, 2B, …``.
              Groups all nonlinearity cases of each SSI tier together.
              Paper default.
            - ``"case"`` — case-major: ``1A, 2A, 3A, 4A, 1B, 2B, …``.
              Groups each nonlinearity case across tiers.
            - ``"input"`` — keep the order returned by
              :meth:`equivalence_probability_table` (sorted by P_eq).

        fill_alpha, figsize, out_dir, prefix, filename
            Standard plot / save parameters.
        """
        if alpha is None:
            alpha = self.cfg.decision.alpha_eq
        equiv_df = self.equivalence_probability_table(alpha=alpha)
        from .plots.bias import plot_radar_bias_probability as _plot
        return self._stamp_and_save(
            _plot(
                equiv_df, alpha=alpha, ref=self.data.ref_label,
                order=order, fill_alpha=fill_alpha,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_bias_ridgeplot(
        self,
        *,
        station_subplots: bool = False,
        denom_name: Literal["GM", "src", "pred"] = "GM",
        overlap: float = 0.6,
        bw_adjust: float = 0.4,
        figsize: tuple[float, float] | None = None,
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "bias_ridgeplot.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot ridgeplot of posterior β densities under the chosen denominator.

        ``denom_name`` picks **both** the denominator draws used to build β
        **and** the LaTeX label on the x-axis. The two stay in sync
        automatically — don't rely on the underlying plot function to infer
        the denominator from the label.

        Parameters
        ----------
        station_subplots : bool, default False
            One panel per station with observed-data dots overlaid, instead
            of a single combined panel.
        denom_name : {"GM", "src", "pred"}, default "GM"
            Aleatory SD used as β denominator and reflected in the axis
            label.

            - ``"GM"`` — canonical σ_GM = √(σ_src² + σ_inter²). Paper
              default; use for the main bias figure.
            - ``"src"`` — σ_src only. Conservative sensitivity view.
            - ``"pred"`` — σ_pred with residual folded in (evaluated at the
              reference configuration). Generous sensitivity view.
        overlap : float, default 0.6
            Ridge stacking overlap (0 = disjoint, 1 = full overlap).
        bw_adjust : float, default 0.4
            Bandwidth multiplier for the KDE. Lower ⇒ spikier densities.
        figsize, out_dir, prefix, filename
            Standard figure-save quartet.
        """
        p = self.posterior
        if denom_name == "GM":
            sigma_denom = self.sigma_GM_draws()
        elif denom_name == "src":
            sigma_denom = self.sigma_src_draws()
        elif denom_name == "pred":
            sigma_denom = self.sigma_pred_draws()
        else:  # pragma: no cover — Literal guard would catch at type-check time
            raise ValueError(
                f"denom_name must be one of 'GM'/'src'/'pred', got {denom_name!r}"
            )

        from .plots.bias import plot_bias_ridgeplot as _plot
        extra: dict[str, Any] = {}
        if station_subplots:
            extra = dict(
                y_obs=self.data.y,
                config_idx=self.data.config_idx,
                station_idx=self.data.station_idx,
                station_labels=list(self.data.station_labels),
            )
        return self._stamp_and_save(
            _plot(
                p.mu_config(), sigma_denom,
                p.config_labels, p.ref_idx,
                denom_name=denom_name, overlap=overlap, bw_adjust=bw_adjust,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
                **extra,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_bias_probability(
        self,
        *,
        mode: Literal["exceed_band", "within_equiv", "positive"] = "within_equiv",
        alpha_equiv: float | None = None,
        band: float | None = None,
        ref: str | None = None,
        configs: list[str] | None = None,
        denominator: Literal["src", "gm", "pred"] = "gm",
        threshold_label: str = "",
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "bias_probability.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot bar chart of β probability under the requested mode.

        Parameters
        ----------
        mode : {"exceed_band", "within_equiv", "positive"}, default "within_equiv"
            Which probability to bar-chart; same three choices as in
            :meth:`bias_probability_table`:

            - ``"exceed_band"`` — P(|β| > ``band``) per config.
            - ``"within_equiv"`` — P(|β| < ``alpha_equiv``); the headline
              P_eq plot.
            - ``"positive"`` — P(β > 0); bias-direction plot.
        alpha_equiv : float, optional
            Threshold for ``"within_equiv"``. Defaults to
            ``cfg.decision.alpha_eq``.
        band : float, optional
            Threshold for ``"exceed_band"``. Defaults to
            ``cfg.decision.alpha_ladder[-1]``.
        ref : str, optional
            Reference configuration. Defaults to ``self.data.ref_label``.
        configs : list[str], optional
            Subset of configurations. Defaults to all.
        denominator : {"src", "gm", "pred"}, default "gm"
            Aleatory SD used as β denominator. Same semantics as in
            :meth:`standardized_bias_table`.
        threshold_label : str, default ""
            Optional text label drawn at the threshold line on the x-axis
            (e.g. ``"α_eq"`` or ``"3×"``). Empty string suppresses the
            annotation.
        figsize, out_dir, prefix, filename
            Standard figure-save quartet.

        Notes
        -----
        ``prob_col='prob'`` is pinned internally to match the column name
        emitted by :meth:`bias_probability_table` (previously a split-kwargs
        hazard that silently picked the wrong column).
        """
        prob_df = self.bias_probability_table(
            mode=mode, alpha_equiv=alpha_equiv, band=band,
            ref=ref, configs=configs, denominator=denominator,
        )
        from .plots.bias import plot_bias_probability as _plot
        return self._stamp_and_save(
            _plot(
                prob_df,
                prob_col="prob", label_col="Config",
                threshold_label=threshold_label,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    # Variance ..........................................................

    def plot_variance_budget(
        self,
        *,
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "variance_budget.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot variance-budget bar chart."""
        vb = self.variance_budget_table()
        from .plots.variance import plot_variance_budget_bars as _plot
        return self._stamp_and_save(
            _plot(
                vb, figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_variance_budget_waterfall(
        self,
        *,
        figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "variance_budget_waterfall.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot variance-budget waterfall chart."""
        vb = self.variance_budget_table()
        from .plots.variance import plot_variance_budget_waterfall as _plot
        return self._stamp_and_save(
            _plot(
                vb, figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_decomposition_bars(
        self,
        *,
        figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "decomposition_bars.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot axis-wise decomposition bar chart."""
        decomp = self.axiswise_decomposition_table()
        from .plots.variance import plot_decomposition_bars as _plot
        return self._stamp_and_save(
            _plot(
                decomp, figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_sigma_stability(
        self,
        *,
        figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "sigma_stability.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot per-configuration residual-scale stability."""
        vc = self.variance_component_table()
        sigma_src_med = float(np.median(self.posterior.sigma_src()))
        from .plots.variance import plot_sigma_stability as _plot
        return self._stamp_and_save(
            _plot(
                vc, sigma_src_med=sigma_src_med,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_variance_ratio(
        self,
        *,
        figsize: tuple[float, float] = (HALF_WIDTH, 3.5),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "variance_ratio.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot variance-ratio forest across configurations."""
        p = self.posterior
        from .plots.variance import plot_variance_ratio as _plot
        return self._stamp_and_save(
            _plot(
                p.sigma_src(), p.sigma_eps(),
                p.config_labels, ci=self.cfg.ci, nu=p.nu(),
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_level_rankings(
        self,
        *,
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "level_rankings.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot tier and case level rankings."""
        tier_tbl, case_tbl, _ = self.level_ranking_tables()
        factor_names = (self.cfg.factors[0].name, self.cfg.factors[1].name)
        from .plots.variance import plot_level_rankings as _plot
        return self._stamp_and_save(
            _plot(
                tier_tbl, case_tbl, factor_names=factor_names,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_sigma_stability_triptych(
        self,
        *,
        order_by: Literal["stability", "config"] = "stability",
        figsize: tuple[float, float] = (FULL_WIDTH, 4.5),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "sigma_stability_triptych.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot three-panel σ_ε stability diagnostic.

        Panels: posterior σ_ε violins + CI, relative CI width, and
        P(σ_ij ≤ σ_ref) per configuration.

        Parameters
        ----------
        order_by : {"stability", "config"}, default "stability"
            Row-ordering on all three panels.

            - ``"stability"`` — configurations sorted ascending by median
              σ_ε (most stable on top). Best for spotting which configs
              are the noisy ones.
            - ``"config"`` — original configuration-label order from
              ``self.data.config_labels`` (i.e., the Tier×Case grid as
              fitted). Use when you want to read the figure alongside
              other plots that use the same order.

            Any other string falls back to input order, same as
            ``"config"``.
        figsize, out_dir, prefix, filename
            Standard figure-save quartet.
        """
        p = self.posterior
        from .plots.variance import plot_sigma_stability_triptych as _plot
        return self._stamp_and_save(
            _plot(
                p.sigma_eps(), p.sigma_src(), p.config_labels, p.ref_idx,
                ci=self.cfg.ci, nu=p.nu(),
                order_by=order_by,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    # Equivalence .......................................................

    def plot_equivalence_matrix(
        self,
        *,
        alpha: float | None = None,
        annot: bool = True,
        fmt: str = ".2f",
        cmap: str = CMAP_SEQ,
        figsize: tuple[float, float] = (HALF_WIDTH, HALF_WIDTH),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "equivalence_matrix.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot equivalence probability heatmap."""
        if alpha is None:
            alpha = self.cfg.decision.alpha_eq
        labels, P_mat = self.epistemic_equivalence_matrix(alpha=alpha)
        from .plots.equivalence import plot_equivalence_matrix as _plot
        return self._stamp_and_save(
            _plot(
                labels, P_mat, alpha=alpha,
                annot=annot, fmt=fmt, cmap=cmap,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_equivalence_matrix_with_dendrogram(
        self,
        *,
        alpha: float | None = None,
        method: str = "average",
        cluster_order: bool = True,
        annot: bool = True,
        fmt: str = ".2f",
        cmap: str = CMAP_SEQ,
        figsize: tuple[float, float] = (FULL_WIDTH, 5.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "equivalence_matrix_dendro.pdf",
    ) -> tuple[plt.Figure, np.ndarray, np.ndarray]:
        r"""Equivalence heatmap with a clustering dendrogram overlay.

        Parameters
        ----------
        alpha : float, optional
            Equivalence radius. Defaults to ``cfg.decision.alpha_eq``.
        method : {"single", "complete", "average", "weighted", \\
                  "centroid", "median", "ward"}, default "average"
            Linkage method passed to
            ``scipy.cluster.hierarchy.linkage``. ``"average"`` (UPGMA) is
            the natural choice on a probability-distance matrix; see
            :meth:`epistemic_clusters_table` for how the seven options
            differ.
        cluster_order : bool, default True
            If True, reorder rows/columns so cluster-adjacent configs are
            visually adjacent in the heatmap. Set False to keep the
            original configuration-label order (useful when comparing
            multiple figures that should share an axis).
        annot : bool, default True
            Annotate each cell with its P_eq value.
        fmt : str, default ".2f"
            Format spec for the cell annotations (e.g. ``".0%"`` for
            percentages).
        cmap : str, default ``CMAP_SEQ``
            Matplotlib colormap name for the probability heat layer.
        figsize, out_dir, prefix, filename
            Standard figure-save quartet.
        """
        if alpha is None:
            alpha = self.cfg.decision.alpha_eq
        labels, P_mat = self.epistemic_equivalence_matrix(alpha=alpha)
        from .plots.equivalence import (
            plot_equivalence_matrix_with_dendrogram as _plot,
        )
        return self._stamp_and_save(
            _plot(
                labels, P_mat, alpha=alpha,
                method=method, cluster_order=cluster_order,
                annot=annot, fmt=fmt, cmap=cmap,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_equivalence_matrix_comparison(
        self,
        *,
        alpha: float | None = None,
        orders: tuple[str, ...] | list[str] = ("similarity", "tier", "case"),
        method: str = "average",
        annot: bool = False,
        fmt: str = ".2f",
        cmap: str = CMAP_SEQ,
        figsize: tuple[float, float] | None = None,
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "equivalence_matrix_comparison.pdf",
    ) -> tuple[plt.Figure, np.ndarray, dict[str, pd.DataFrame]]:
        r"""Side-by-side equivalence heatmaps under different label orderings.

        Renders one panel per ordering in ``orders``, sharing a single
        colourbar. Supported orderings are ``"similarity"`` (hierarchical
        clustering), ``"tier"`` (1A, 1B, …, 2A, …), and ``"case"``
        (1A, 2A, …, 1B, …). Useful for side-by-side comparison of how
        different sort orders expose clusters in the same P_eq matrix.

        Parameters
        ----------
        alpha : float, optional
            Equivalence radius. Defaults to ``cfg.decision.alpha_eq``.
        orders : sequence of {"similarity", "tier", "case"}
            One panel per entry.
        method : str, default "average"
            Linkage method for ``"similarity"`` ordering.
        annot : bool, default False
            Annotate each cell with its P_eq value.
        fmt : str, default ".2f"
            Annotation format spec.
        cmap : str, default ``CMAP_SEQ``
            Heatmap colormap.
        figsize, out_dir, prefix, filename
            Standard figure-save quartet.

        Returns
        -------
        fig, axes, P_dfs
            ``P_dfs`` maps each ordering key to the reordered DataFrame.
        """
        if alpha is None:
            alpha = self.cfg.decision.alpha_eq
        labels, P_mat = self.epistemic_equivalence_matrix(alpha=alpha)
        from .plots.equivalence import (
            plot_equivalence_matrix_comparison as _plot,
        )
        return self._stamp_and_save(
            _plot(
                labels, P_mat, alpha=alpha,
                orders=orders, method=method,
                annot=annot, fmt=fmt, cmap=cmap,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_equivalence_dendrogram(
        self,
        *,
        alpha: float | None = None,
        method: str = "average",
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "equivalence_dendrogram.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        r"""Dendrogram built from the epistemic distance matrix D = 1 − P_eq.

        Parameters
        ----------
        alpha : float, optional
            Equivalence radius used to build D. Defaults to
            ``cfg.decision.alpha_eq``.
        method : {"single", "complete", "average", "weighted", \\
                  "centroid", "median", "ward"}, default "average"
            Linkage method passed to
            ``scipy.cluster.hierarchy.linkage``. See
            :meth:`epistemic_clusters_table` for what each option means.
        figsize, out_dir, prefix, filename
            Standard figure-save quartet.
        """
        if alpha is None:
            alpha = self.cfg.decision.alpha_eq
        labels, P_mat = self.epistemic_equivalence_matrix(alpha=alpha)
        from .plots.equivalence import plot_equivalence_dendrogram as _plot
        return self._stamp_and_save(
            _plot(
                labels, P_mat, alpha=alpha, method=method,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_equivalence_sweep(
        self,
        *,
        figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "equivalence_sweep.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot equivalence probability sweep across alpha values."""
        sweep = self.equivalence_sweep_table()
        from .plots.equivalence import plot_equivalence_sweep as _plot
        return self._stamp_and_save(
            _plot(
                sweep, figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_equivalence_bars_plus_sweep(
        self,
        *,
        alpha: float | None = None,
        alphas: np.ndarray | None = None,
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "equivalence_bars_sweep.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot combined equivalence bars and sweep lines."""
        if alpha is None:
            alpha = self.cfg.decision.alpha_eq
        equiv = self.equivalence_probability_table(alpha=alpha)
        sweep = self.equivalence_sweep_table(alphas=alphas)
        from .plots.equivalence import (
            plot_equivalence_bars_plus_sweep as _plot,
        )
        return self._stamp_and_save(
            _plot(
                equiv, sweep, alpha=alpha,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    # Interaction (v8+, v9+) ............................................

    def plot_interaction_heatmap(
        self,
        *,
        annotate: bool = True,
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "interaction_heatmap.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot heatmap of the shared station-rupture interaction.

        Requires a v8+ interaction model in the posterior.
        """
        p = self.posterior
        if not p.has_interaction:
            raise RuntimeError(
                "Posterior has no 'gamma_sr'. Fit with "
                "RandomSlopesInteractionModel (v8+) to use this plot."
            )
        gamma = p.gamma_sr()
        assert gamma is not None
        from .plots.interaction import plot_interaction_heatmap as _plot
        return self._stamp_and_save(
            _plot(
                gamma,
                list(self.data.station_labels),
                list(self.data.run_labels),
                annotate=annotate,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_interaction_by_case(
        self,
        *,
        annotate: bool = True,
        share_scale: bool = True,
        ncols: int | None = None,
        figsize: tuple[float, float] | None = None,
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "interaction_by_case.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot per-Case panel of effective interactions.

        Works for both v8 and v9 interaction models. ``xi_case`` is
        auto-detected from the posterior when present.
        """
        p = self.posterior
        if not p.has_interaction:
            raise RuntimeError(
                "Posterior has no 'gamma_sr'. Fit with "
                "RandomSlopesInteractionModel (v8+) to use this plot."
            )
        gamma = p.gamma_sr()
        assert gamma is not None
        case_labels = p.case_labels
        if case_labels is None:
            case_labels = list(self.data.factor_levels[
                self.cfg.factors[1].name
            ])
        from .plots.interaction import plot_interaction_by_case as _plot
        return self._stamp_and_save(
            _plot(
                gamma,
                case_labels,
                list(self.data.station_labels),
                list(self.data.run_labels),
                xi_case_draws=p.xi_case(),
                annotate=annotate, share_scale=share_scale, ncols=ncols,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_interaction_forest(
        self,
        *,
        ci: tuple[float, float] | None = None,
        sort: bool = True,
        max_cells: int | None = None,
        figsize: tuple[float, float] = (HALF_WIDTH, 4.3),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "interaction_forest.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot forest plot of interaction cells with credible intervals.

        Requires a v8+ interaction model in the posterior. ``ci`` defaults
        to ``cfg.ci``.
        """
        p = self.posterior
        if not p.has_interaction:
            raise RuntimeError(
                "Posterior has no 'gamma_sr'. Fit with "
                "RandomSlopesInteractionModel (v8+) to use this plot."
            )
        gamma = p.gamma_sr()
        assert gamma is not None
        labels = p.station_run_labels
        if labels is None:
            labels = [
                f"{s}|{r}"
                for s in self.data.station_labels
                for r in self.data.run_labels
            ]
        if ci is None:
            ci = self.cfg.ci
        from .plots.interaction import plot_interaction_forest as _plot
        return self._stamp_and_save(
            _plot(
                gamma, labels, ci=ci, sort=sort, max_cells=max_cells,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    # Posterior .........................................................

    def plot_station_posteriors(
        self,
        *,
        figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "station_posteriors.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot station random-effect posterior densities."""
        p = self.posterior
        from .plots.posterior import plot_station_posteriors as _plot
        return self._stamp_and_save(
            _plot(
                p.delta_st(), p.station_labels,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_observed_vs_predicted(
        self,
        *,
        figsize: tuple[float, float] = (HALF_WIDTH, HALF_WIDTH),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "observed_vs_predicted.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot observed vs. posterior-mean predicted values."""
        p = self.posterior
        fv = self.fitted_values()
        from .plots.posterior import plot_observed_vs_predicted as _plot
        return self._stamp_and_save(
            _plot(
                self.data.y, fv["yhat_with_run"].to_numpy(),
                config_idx=self.data.config_idx,
                config_labels=p.config_labels,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_mu_density(
        self,
        *,
        configs: list[str] | None = None,
        station_idx: int | None = None,
        run_idx: int | None = None,
        original_edp_scale: bool = False,
        kind: Literal["density", "cdf"] = "density",
        normalize: bool = False,
        n_cols: int = 4,
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "mu_posterior_density.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Posterior density (or CDF) of the modelled mean μ per configuration.

        Parameters
        ----------
        configs : list[str], optional
            Subset of configuration labels to include. Defaults to all.
        station_idx : int, optional
            If set, evaluate μ at this specific station (adds δ_{station}
            to μ₀ + μ_config). If None, μ is the population-level mean.
        run_idx : int, optional
            If set, evaluate μ at this specific rupture (adds b_run).
            Stacks with ``station_idx``.
        original_edp_scale : bool, default False
            If True, exponentiate back to the native EDP scale
            (``exp(μ)``). Leave False for the log-EDP scale used
            throughout the paper.
        kind : {"density", "cdf"}, default "density"
            What to plot per configuration:

            - ``"density"`` — kernel-density estimate of the posterior.
              Shows shape, multimodality, and tails. Natural for comparing
              configuration centres visually.
            - ``"cdf"`` — empirical cumulative distribution of the
              posterior draws. Reads crossings easily; best when you want
              to compare specific quantiles (e.g. median, 90%) across
              configs.
        normalize : bool, default False
            For ``kind="density"``: if True, scale each config's density
            to peak at 1 (visual only — destroys integral=1). Useful when
            configurations have very different variances and the larger-
            variance ones would otherwise disappear under a common y-axis.
        n_cols : int, default 4
            Number of subplot columns in the panel grid.
        figsize, out_dir, prefix, filename
            Standard figure-save quartet.
        """
        p = self.posterior
        from .plots.posterior import plot_mu_density as _plot
        return self._stamp_and_save(
            _plot(
                p.mu0(), p.mu_config(), p.config_labels,
                configs=configs,
                delta_st=p.delta_st() if station_idx is not None else None,
                station_idx=station_idx,
                b_run=p.b_run() if run_idx is not None else None,
                run_idx=run_idx,
                original_edp_scale=original_edp_scale,
                kind=kind, normalize=normalize, n_cols=n_cols,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_ppc(
        self,
        *,
        figsize: tuple[float, float] | None = None,
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "ppc_check.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot posterior predictive check summary."""
        ppc_df = self.posterior_predictive_check()
        from .plots.posterior import plot_ppc as _plot
        return self._stamp_and_save(
            _plot(
                ppc_df, figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_residuals(
        self,
        *,
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "residuals.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot residual diagnostics."""
        fv = self.fitted_values()
        from .plots.posterior import plot_residuals as _plot
        return self._stamp_and_save(
            _plot(
                fv, figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_raw_data(
        self,
        *,
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "raw_data.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot raw EDP data by configuration and station."""
        from .plots.posterior import plot_raw_data as _plot
        return self._stamp_and_save(
            _plot(
                self.data.y,
                self.data.config_idx,
                list(self.data.config_labels),
                self.data.station_idx,
                list(self.data.station_labels),
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_ppc_density(
        self,
        *,
        n_rep_draws: int = 50,
        figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "ppc_density.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot posterior predictive density overlay."""
        p = self.posterior
        y_rep = p.y_rep()
        if y_rep is None:
            raise RuntimeError(
                "No posterior predictive samples. "
                "Refit with posterior_predictive=True."
            )
        from .plots.posterior import plot_ppc_density as _plot
        return self._stamp_and_save(
            _plot(
                self.data.y, y_rep, n_rep_draws=n_rep_draws,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_trace(
        self,
        *,
        var_names: list[str] | None = None,
        figsize: tuple[float, float] | None = None,
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "traceplot.pdf",
    ) -> plt.Figure:
        """Plot ArviZ traceplot for key parameters."""
        from .plots.posterior import plot_trace as _plot
        return self._stamp_and_save(
            _plot(
                self.idata, var_names=var_names,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_pair(
        self,
        *,
        var_names: list[str] | None = None,
        figsize: tuple[float, float] = (HALF_WIDTH, HALF_WIDTH),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "pairplot.pdf",
    ) -> plt.Figure:
        """Plot ArviZ pair plot for key parameters."""
        from .plots.posterior import plot_pair as _plot
        return self._stamp_and_save(
            _plot(
                self.idata, var_names=var_names,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_forest_arviz(
        self,
        *,
        var_names: list[str] | None = None,
        ci: float = 0.94,
        dot_alpha: float = 0.30,
        figsize: tuple[float, float] | None = None,
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "forest_arviz.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot ArviZ forest plot with observed data overlay."""
        d = self.data
        from .plots.posterior import plot_forest_arviz as _plot
        return self._stamp_and_save(
            _plot(
                self.idata, var_names=var_names,
                observed_y=d.y,
                config_idx=d.config_idx,
                station_idx=d.station_idx,
                run_idx=d.run_idx,
                config_labels=list(d.config_labels),
                station_labels=list(d.station_labels),
                ci=ci, dot_alpha=dot_alpha,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    # ── Plots: ρ_epi & validation (§6, §7 of uncertanty_measures.md) ─────

    def plot_epistemic_median_ratio(
        self,
        *,
        ref: str | None = None,
        configs: list[str] | None = None,
        log_scale: bool = True,
        order: Literal["tier", "case", "input"] = "tier",
        show_alpha_ladder: bool = True,
        figsize: tuple[float, float] = (HALF_WIDTH, 4.5),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "epistemic_median_ratio.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        r"""Forest plot of the Epistemic Median Ratio ρ_epi — §6.

        Physical-unit companion to :meth:`plot_standardized_bias`. Shows
        Δμ (= log ρ_epi) per configuration with HDI whiskers, the α-ladder
        imaged into log-EDP units as shaded reference bands, and the
        ρ_epi = 1 reference line. Preferred visualisation when ``|Δμ|`` is
        large (§6.3 lever-arm regime), where β's CI is inflated but
        ρ_epi's CI stays bounded.

        Parameters
        ----------
        ref : str, optional
            Reference configuration label. Defaults to
            ``self.data.ref_label``.
        configs : list[str], optional
            Subset of configuration labels. Defaults to all.
        log_scale : bool, default True
            Plot on the Δμ = log ρ_epi axis (recommended for symmetry).
            Set False to see ρ_epi directly on a linear axis; in that
            case the α-ladder bands are suppressed because they are a
            log-scale construction.
        order : {"tier", "case", "input"}, default "tier"
            Row ordering on the y-axis.
        show_alpha_ladder : bool, default True
            Draw the ``±α·σ_GM`` reference bands using
            ``cfg.decision.alpha_eq`` (inner) and
            ``cfg.decision.alpha_ladder[-1]`` (outer) as β-unit radii,
            multiplied by the posterior-mean σ_GM. Ignored if
            ``log_scale=False``.
        figsize, out_dir, prefix, filename
            Standard figure-save quartet.
        """
        rho_df = self.epistemic_median_ratio_table(
            ref=ref, configs=configs, r_band=None,
        )
        sigma_gm_ref = (
            float(np.mean(self.sigma_GM_draws()))
            if (log_scale and show_alpha_ladder) else None
        )
        dec = self.cfg.decision
        # Drop the reference row (identically ρ=1, Δμ=0) from the forest.
        ref_label = ref if ref is not None else self.data.ref_label
        from .plots.bias import plot_epistemic_median_ratio as _plot
        return self._stamp_and_save(
            _plot(
                rho_df,
                sigma_GM_ref=sigma_gm_ref,
                alpha_eq=dec.alpha_eq,
                alpha_severe=dec.alpha_ladder[-1],
                log_scale=log_scale,
                order=order,
                ref_label=ref_label,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_validation_decision(
        self,
        u0: float,
        *,
        rule: Literal["threshold", "fractional", "ci_overlap"],
        theta: float | None = None,
        direction: Literal["below", "above"] = "below",
        target: float | None = None,
        tau_val: float | None = None,
        target_hdi: tuple[float, float] | None = None,
        p_star_val: float = 0.95,
        ref: str | None = None,
        configs: list[str] | None = None,
        order: Literal["tier", "case", "input"] = "tier",
        figsize: tuple[float, float] = (HALF_WIDTH, 4.5),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "validation_decision.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        r"""Forest-style validation decision plot — §7.

        Renders ÊDP^med with HDI whiskers per configuration, pass/fail
        colour coded (navy/crimson), posterior-mass ``P(rule holds)``
        annotated per row, and a rule-specific reference element:

        - ``"threshold"``  : threshold θ as a dashed line, fail side shaded.
        - ``"fractional"`` : target as a dashed line, ±τ_val band shaded.
        - ``"ci_overlap"`` : benchmark HDI shaded band.

        ``u_0`` must be externally anchored (§7.3). ``tau_val`` and
        ``p_star_val`` are per use case (§7.4), not tied to the paper-level
        ``cfg.decision.alpha_eq`` / ``cfg.decision.p_star``.

        Parameters
        ----------
        u0 : float
            Externally anchored log-scale reference EDP for the case.
        rule, theta, direction, target, tau_val, target_hdi, p_star_val
            See :meth:`validation_decision_table`.
        ref, configs
            Reference label / subset filter passed to
            :meth:`validation_decision_table`.
        order : {"tier", "case", "input"}, default "tier"
            Row ordering on the y-axis.
        figsize, out_dir, prefix, filename
            Standard figure-save quartet.
        """
        val_df = self.validation_decision_table(
            u0=u0,
            rule=rule,
            theta=theta,
            direction=direction,
            target=target,
            tau_val=tau_val,
            target_hdi=target_hdi,
            p_star_val=p_star_val,
            ref=ref,
            configs=configs,
        )
        # Drop the reference row — ÊDP^med at the reference is exactly
        # exp(u₀) by construction, carries no validation information.
        ref_label = ref if ref is not None else self.data.ref_label
        from .plots.validation import plot_validation_decision as _plot
        return self._stamp_and_save(
            _plot(
                val_df,
                rule=rule,
                theta=theta,
                direction=direction,
                target=target,
                tau_val=tau_val,
                target_hdi=target_hdi,
                p_star_val=p_star_val,
                order=order,
                ref_label=ref_label,
                figsize=figsize, out_dir=None, prefix=prefix, filename=filename,
            ),
            out_dir=out_dir, prefix=prefix, filename=filename,
        )

    # ── Model comparison ─────────────────────────────────────────────────

    def compare_variants(
        self,
        *,
        ic: str = "loo",
    ) -> tuple[pd.DataFrame, dict[str, FittedVariant]]:
        """Compare Student-t/Gaussian × hetero/homo variants.

        Returns the ArviZ comparison table and all fitted variants.
        """
        variants = {
            "StudentT_hetero": FlatConfigModel(likelihood="student_t", heteroskedastic=True),
            "Gaussian_hetero": FlatConfigModel(likelihood="gaussian", heteroskedastic=True),
            "StudentT_homo": FlatConfigModel(likelihood="student_t", heteroskedastic=False),
            "Gaussian_homo": FlatConfigModel(likelihood="gaussian", heteroskedastic=False),
        }
        return compare_models(self.data, variants, self.cfg, ic=ic)

    # ── Repr ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        name_str = f" ({self.name})" if self.name else ""
        return (
            f"BayesEpistemicModel{name_str} [{status}]\n"
            f"  configs: {self.data.n_configs}, "
            f"stations: {self.data.n_stations}, "
            f"runs: {self.data.n_runs}, "
            f"obs: {self.data.n_obs}\n"
            f"  factors: {self.cfg.factor_names}\n"
            f"  ref: {self.data.ref_label}\n"
            f"  likelihood: {self.cfg.likelihood}, "
            f"hetero: {self.cfg.heteroskedastic}"
        )
