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

import numpy as np
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
    ):
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
    ):
        if not edp_specs:
            raise ValueError("edp_specs must contain at least one EDPSpec.")
        self.df = df
        self.edp_specs = list(edp_specs)
        self.cfg = cfg
        self.name = name
        self.models: dict[str, BayesEpistemicModel] = {}

    @property
    def edp_names(self) -> list[str]:
        return [s.name for s in self.edp_specs]

    @property
    def is_fitted(self) -> bool:
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
        )

    def fit(self, edp_name: str, **fit_kwargs) -> BayesEpistemicModel:
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

    def fit_all(self, **fit_kwargs) -> "MultiEDPModel":
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
    ) -> pd.DataFrame:
        """Standardized bias table across all EDPs.

        Returns a long DataFrame with an extra 'edp' column.
        """
        self._check_fitted()
        frames = []
        for spec in self.edp_specs:
            m = self.models[spec.name]
            tbl = m.standardized_bias_table(ref=ref, configs=configs)
            tbl = tbl.copy()
            tbl.insert(0, "edp", spec.name)
            tbl.insert(1, "category", spec.category)
            if spec.story is not None:
                tbl["story"] = spec.story
            if spec.direction is not None:
                tbl["direction"] = spec.direction
            frames.append(tbl)
        return pd.concat(frames, ignore_index=True)

    def compare_bias_total(
        self,
        ref: str | None = None,
        configs: list[str] | None = None,
    ) -> pd.DataFrame:
        """Standardized bias (total) table across all EDPs."""
        self._check_fitted()
        frames = []
        for spec in self.edp_specs:
            m = self.models[spec.name]
            tbl = m.standardized_bias_total_table(ref=ref, configs=configs)
            tbl = tbl.copy()
            tbl.insert(0, "edp", spec.name)
            tbl.insert(1, "category", spec.category)
            frames.append(tbl)
        return pd.concat(frames, ignore_index=True)

    def compare_equivalence(
        self,
        *,
        alpha: float = 0.5,
        ref: str | None = None,
        configs: list[str] | None = None,
    ) -> pd.DataFrame:
        """Equivalence probability table across all EDPs."""
        self._check_fitted()
        frames = []
        for spec in self.edp_specs:
            m = self.models[spec.name]
            tbl = m.equivalence_probability_table(alpha=alpha, ref=ref, configs=configs)
            tbl = tbl.copy()
            tbl.insert(0, "edp", spec.name)
            tbl.insert(1, "category", spec.category)
            frames.append(tbl)
        return pd.concat(frames, ignore_index=True)

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
        return pd.concat(frames, ignore_index=True)

    def compare_decomposition(
        self,
        **kwargs,
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
        return pd.concat(frames, ignore_index=True)

    # ── Summary: the epistemic landscape ────────────────────────────────

    def epistemic_landscape(
        self,
        *,
        alpha: float = 0.5,
        ref: str | None = None,
    ) -> pd.DataFrame:
        """One-row-per-(EDP, config) summary combining bias + equivalence.

        This is the key Stage 1 deliverable: the field β(EDP, config)
        that reveals *where in the structure* modeling fidelity matters.
        """
        self._check_fitted()
        bias_df = self.compare_bias(ref=ref)
        equiv_df = self.compare_equivalence(alpha=alpha, ref=ref)

        # Merge on (edp, Config)
        merged = bias_df.merge(
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
