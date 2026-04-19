"""
MultiEDPModel — run the epistemic analysis pipeline across multiple EDPs.

Stage 1 of the PhD roadmap: expand from a single scalar EDP to a vector
of EDPs (IDR per story, PFA per floor, roof drift, base shear, etc.)
and compare the epistemic bias landscape β(EDP, story, direction).

Usage
-----
>>> from apeBayes import MultiEDPModel, ModelConfig
>>> multi = MultiEDPModel(df_wide, edp_specs={"IDR_x_1": "IDR_x_1", ...}, cfg=cfg)
>>> multi.fit_all()
>>> multi.compare_bias()
>>> multi.compare_equivalence(alpha=0.4)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .config import ModelConfig
from .facade import BayesEpistemicModel

# ── EDP specification ────────────────────────────────────────────────────────

class EDPSpec:
    """Describes one EDP to analyze.

    Parameters
    ----------
    name : str
        Human-readable label (e.g., "IDR_x_story3", "PFA_y_roof").
    column : str
        Column name in the DataFrame holding the (log-transformed) EDP values.
    category : str, optional
        Grouping label for comparison plots (e.g., "IDR", "PFA", "global").
    story : int | None
        Story number, if applicable.
    direction : str | None
        Direction label ("x", "y"), if applicable.
    """

    def __init__(
        self,
        name: str,
        column: str,
        *,
        category: str = "general",
        story: int | None = None,
        direction: str | None = None,
    ) -> None:
        self.name = name
        self.column = column
        self.category = category
        self.story = story
        self.direction = direction

    def __repr__(self) -> str:
        parts = [f"EDPSpec({self.name!r}, col={self.column!r}"]
        if self.story is not None:
            parts.append(f"story={self.story}")
        if self.direction is not None:
            parts.append(f"dir={self.direction!r}")
        return ", ".join(parts) + ")"


# ── MultiEDPModel ────────────────────────────────────────────────────────────

class MultiEDPModel:
    """Run BayesEpistemicModel independently per EDP and compare results.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame.  Must contain columns for all factors,
        station, run, and every EDP listed in *edp_specs*.
    edp_specs : list[EDPSpec]
        Ordered list of EDPs to analyze.
    cfg : ModelConfig
        Base model configuration.  The ``edp_col`` field is overridden
        per EDP during fitting.
    name : str, optional
        Human-readable name for this multi-EDP study.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        edp_specs: list[EDPSpec],
        cfg: ModelConfig,
        *,
        name: str | None = None,
    ) -> None:
        if not edp_specs:
            raise ValueError("edp_specs must contain at least one EDPSpec.")
        self.df = df
        self.edp_specs = list(edp_specs)
        self.cfg = cfg
        self.name = name
        self.models: dict[str, BayesEpistemicModel] = {}

    @property
    def edp_names(self) -> list[str]:
        """Return the list of EDP names."""
        return [s.name for s in self.edp_specs]

    @property
    def is_fitted(self) -> bool:
        """Return True if all EDPs have been fitted."""
        return len(self.models) == len(self.edp_specs) and all(
            m.is_fitted for m in self.models.values()
        )

    # ── Fitting ─────────────────────────────────────────────────────────

    def _make_cfg_for_edp(self, spec: EDPSpec) -> ModelConfig:
        """Clone the base config with edp_col swapped to this EDP's column."""
        # Frozen dataclass — reconstruct
        return ModelConfig(
            factors=self.cfg.factors,
            config_col=self.cfg.config_col,
            edp_col=spec.column,
            station_col=self.cfg.station_col,
            run_col=self.cfg.run_col,
            ref_config=self.cfg.ref_config,
            likelihood=self.cfg.likelihood,
            heteroskedastic=self.cfg.heteroskedastic,
            ci=self.cfg.ci,
            config_sep=self.cfg.config_sep,
            priors=self.cfg.priors,
            sampling=self.cfg.sampling,
            decision=self.cfg.decision,
        )

    def fit(self, edp_name: str, **fit_kwargs: Any) -> BayesEpistemicModel:
        """Fit a single EDP.

        Parameters
        ----------
        edp_name : str
            Must match an EDPSpec.name in self.edp_specs.
        **fit_kwargs
            Forwarded to BayesEpistemicModel.fit().

        Returns
        -------
        The fitted BayesEpistemicModel for this EDP.
        """
        spec = self._get_spec(edp_name)
        edp_cfg = self._make_cfg_for_edp(spec)
        model = BayesEpistemicModel(self.df, cfg=edp_cfg, name=spec.name)
        model.fit(**fit_kwargs)
        self.models[spec.name] = model
        return model

    def fit_all(self, **fit_kwargs: Any) -> MultiEDPModel:
        """Fit every EDP sequentially.  Returns self for chaining."""
        for spec in self.edp_specs:
            if spec.name not in self.models:
                self.fit(spec.name, **fit_kwargs)
        return self

    # ── Cross-EDP comparison tables ─────────────────────────────────────

    def compare_bias(
        self,
        ref: str | None = None,
        configs: list[str] | None = None,
        *,
        denominator: Literal["src", "gm", "pred"] = "gm",
    ) -> pd.DataFrame:
        """Standardised bias table concatenated across all EDPs.

        Thin wrapper that loops over per-EDP
        :meth:`BayesEpistemicModel.standardized_bias_table` and prepends
        ``edp`` and ``category`` (plus ``story`` / ``direction`` when set
        on the :class:`EDPSpec`).

        Parameters
        ----------
        ref : str, optional
            Reference configuration label (same for every EDP). Defaults
            to each model's ``self.data.ref_label``.
        configs : list[str], optional
            Subset of configurations to report. Defaults to all.
        denominator : {"src", "gm", "pred"}, default "gm"
            Aleatory SD used as β denominator.

            - ``"src"`` — σ_src only. Conservative.
            - ``"gm"`` — canonical σ_GM = √(σ_src² + σ_inter²). Paper
              default.
            - ``"pred"`` — σ_pred with residual folded in. Generous.

            See :meth:`BayesEpistemicModel.standardized_bias_table` for
            the full "src / gm / pred" semantics.
        """
        self._check_fitted()
        frames = []
        for spec in self.edp_specs:
            m = self.models[spec.name]
            tbl = m.standardized_bias_table(ref=ref, configs=configs,
                                            denominator=denominator)
            tbl = tbl.copy()
            tbl.insert(0, "edp", spec.name)
            tbl.insert(1, "category", spec.category)
            if spec.story is not None:
                tbl["story"] = spec.story
            if spec.direction is not None:
                tbl["direction"] = spec.direction
            frames.append(tbl)
        return pd.concat(frames, ignore_index=True)  # type: ignore[no-any-return]

    def compare_equivalence(
        self,
        *,
        alpha: float | None = None,
        ref: str | None = None,
        configs: list[str] | None = None,
        denominator: Literal["src", "gm", "pred"] = "gm",
    ) -> pd.DataFrame:
        """Equivalence probability table concatenated across all EDPs.

        Parameters
        ----------
        alpha : float, optional
            Equivalence radius on the β scale. Defaults to
            ``cfg.decision.alpha_eq`` (paper default 0.4).
        ref : str, optional
            Reference configuration label. Defaults to each model's
            ``self.data.ref_label``.
        configs : list[str], optional
            Subset of configurations. Defaults to all.
        denominator : {"src", "gm", "pred"}, default "gm"
            Aleatory SD used as β denominator. Same semantics as in
            :meth:`BayesEpistemicModel.standardized_bias_table` — paper
            uses ``"gm"``, sensitivity appendix also reports ``"src"``
            and ``"pred"``.
        """
        self._check_fitted()
        frames = []
        for spec in self.edp_specs:
            m = self.models[spec.name]
            tbl = m.equivalence_probability_table(
                alpha=alpha, ref=ref, configs=configs, denominator=denominator,
            )
            tbl = tbl.copy()
            tbl.insert(0, "edp", spec.name)
            tbl.insert(1, "category", spec.category)
            frames.append(tbl)
        return pd.concat(frames, ignore_index=True)  # type: ignore[no-any-return]

    def compare_decision_report(
        self,
        *,
        ref: str | None = None,
        configs: list[str] | None = None,
        denominators: tuple[str, ...] | None = None,
        alpha_eq: float | None = None,
        alpha_ladder: tuple[float, ...] | None = None,
        p_star: float | None = None,
    ) -> pd.DataFrame:
        """Headline decision report concatenated across all EDPs (§3.1 of integration plan).

        Concatenates each EDP's
        :meth:`BayesEpistemicModel.decision_report` with ``edp`` and
        ``category`` (and ``story`` / ``direction`` when set on the
        :class:`EDPSpec`) prepended. This is the primary Stage-1
        deliverable: the β field across EDP × config with decisions
        already attached.

        Parameters
        ----------
        ref : str, optional
            Reference configuration label. Defaults to each model's
            ``self.data.ref_label``.
        configs : list[str], optional
            Subset of configurations. Defaults to all.
        denominators : tuple[str, ...], optional
            Subset of ``{"src", "gm", "pred"}`` selecting which β
            variants to compute. ``"gm"`` must be present (the P* gate
            is defined on it). Defaults to ``cfg.decision.denominators``
            (the full triple). See
            :meth:`BayesEpistemicModel.decision_report` for the src /
            gm / pred semantics.
        alpha_eq : float, optional
            Equivalence threshold for the ``decision`` column. Must be a
            member of ``alpha_ladder``. Defaults to
            ``cfg.decision.alpha_eq`` (0.4).
        alpha_ladder : tuple[float, ...], optional
            α values reported as P_eq columns. Defaults to
            ``cfg.decision.alpha_ladder`` = ``(0.4, 0.7, 1.1)``.
        p_star : float, optional
            Probability gate. Defaults to ``cfg.decision.p_star`` = 0.95.
        """
        self._check_fitted()
        frames = []
        for spec in self.edp_specs:
            m = self.models[spec.name]
            tbl = m.decision_report(
                ref=ref, configs=configs, denominators=denominators,
                alpha_eq=alpha_eq, alpha_ladder=alpha_ladder, p_star=p_star,
            )
            tbl = tbl.copy()
            tbl.insert(0, "edp", spec.name)
            tbl.insert(1, "category", spec.category)
            if spec.story is not None:
                tbl["story"] = spec.story
            if spec.direction is not None:
                tbl["direction"] = spec.direction
            frames.append(tbl)
        return pd.concat(frames, ignore_index=True)  # type: ignore[no-any-return]

    def compare_variance_budget(self) -> pd.DataFrame:
        """Variance budget table across all EDPs."""
        self._check_fitted()
        frames = []
        for spec in self.edp_specs:
            m = self.models[spec.name]
            tbl = m.variance_budget_table()
            tbl = tbl.copy()
            tbl.insert(0, "edp", spec.name)
            tbl.insert(1, "category", spec.category)
            frames.append(tbl)
        return pd.concat(frames, ignore_index=True)  # type: ignore[no-any-return]

    def compare_decomposition(
        self,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Axiswise decomposition table across all EDPs."""
        self._check_fitted()
        frames = []
        for spec in self.edp_specs:
            m = self.models[spec.name]
            tbl = m.axiswise_decomposition_table(**kwargs)
            tbl = tbl.copy()
            tbl.insert(0, "edp", spec.name)
            tbl.insert(1, "category", spec.category)
            frames.append(tbl)
        return pd.concat(frames, ignore_index=True)  # type: ignore[no-any-return]

    # ── Summary: the epistemic landscape ────────────────────────────────

    def epistemic_landscape(
        self,
        *,
        alpha: float | None = None,
        ref: str | None = None,
    ) -> pd.DataFrame:
        """One-row-per-(EDP, config) summary combining bias + equivalence.

        ``alpha`` defaults to ``cfg.decision.alpha_eq`` (paper default 0.4).

        This is the key Stage 1 deliverable: the field β(EDP, config)
        that reveals *where in the structure* modeling fidelity matters.
        Both tables use σ_GM as denominator.
        """
        self._check_fitted()
        if alpha is None:
            alpha = self.cfg.decision.alpha_eq
        bias_df = self.compare_bias(ref=ref)
        equiv_df = self.compare_equivalence(alpha=alpha, ref=ref)

        # Merge on (edp, Config)
        merged: pd.DataFrame = bias_df.merge(
            equiv_df[["edp", "Config", "P_equiv"]],
            on=["edp", "Config"],
            how="left",
        )
        return merged

    # ── Serialization ───────────────────────────────────────────────────

    def save_all(self, directory: str | Path) -> list[Path]:
        """Save all fitted models to NetCDF files in *directory*.

        File names: {edp_name}.nc
        """
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for spec in self.edp_specs:
            p = self.models[spec.name].save(directory / f"{spec.name}.nc")
            paths.append(p)
        return paths

    # ── Accessors ───────────────────────────────────────────────────────

    def __getitem__(self, edp_name: str) -> BayesEpistemicModel:
        """Access a fitted model by EDP name."""
        if edp_name not in self.models:
            raise KeyError(
                f"EDP {edp_name!r} not fitted. "
                f"Available: {list(self.models.keys())}"
            )
        return self.models[edp_name]

    def __repr__(self) -> str:
        n_fitted = sum(1 for m in self.models.values() if m.is_fitted)
        name_str = f" ({self.name})" if self.name else ""
        return (
            f"MultiEDPModel{name_str}\n"
            f"  EDPs: {len(self.edp_specs)} defined, {n_fitted} fitted\n"
            f"  Names: {self.edp_names}"
        )

    # ── Internal ────────────────────────────────────────────────────────

    def _get_spec(self, edp_name: str) -> EDPSpec:
        for s in self.edp_specs:
            if s.name == edp_name:
                return s
        raise KeyError(
            f"EDP {edp_name!r} not found. Available: {self.edp_names}"
        )

    def _check_fitted(self) -> None:
        missing = [s.name for s in self.edp_specs if s.name not in self.models]
        if missing:
            raise RuntimeError(
                f"Not all EDPs are fitted. Missing: {missing}. "
                f"Call fit_all() or fit() for each EDP."
            )
