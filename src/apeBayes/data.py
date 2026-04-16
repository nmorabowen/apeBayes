"""
Immutable encoded dataset for Bayesian epistemic analysis.

EpistemicDataset takes a long-format DataFrame and a ModelConfig,
validates, encodes categorical factors into integer indices, and
freezes.  It is the *single source of truth* for y, indices, and
level labels consumed by every downstream module.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .utils import order_config_labels

if TYPE_CHECKING:
    from .config import ModelConfig

# ── Public dataclass ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EpistemicDataset:
    """Immutable container holding encoded observations and factor structure.

    Attributes
    ----------
    y : np.ndarray
        Observed (log-transformed) EDP values, shape (N,).
    config_idx : np.ndarray
        Integer index into *config_labels* for each observation, shape (N,).
    station_idx : np.ndarray
        Integer index into *station_labels*, shape (N,).
    run_idx : np.ndarray
        Integer index into *run_labels*, shape (N,).
    config_labels : np.ndarray
        Ordered flat configuration labels (e.g. ["1A", "1B", …, "4D"]).
    station_labels : np.ndarray
        Station-level labels.
    run_labels : np.ndarray
        Run (earthquake realisation) labels.
    factor_levels : dict[str, list[str]]
        Per-factor level lists keyed by factor name.
    factor_grid : dict[tuple[str, ...], int]
        Maps a tuple of per-factor levels to the index in *config_labels*.
    n_obs : int
        Number of observations.
    n_configs : int
        Number of distinct configurations.
    n_stations : int
        Number of stations.
    n_runs : int
        Number of runs.
    ref_idx : int
        Index of the reference configuration in *config_labels*.
    cfg : ModelConfig
        The originating model configuration (for provenance).
    """

    y: np.ndarray
    config_idx: np.ndarray
    station_idx: np.ndarray
    run_idx: np.ndarray
    config_labels: np.ndarray
    station_labels: np.ndarray
    run_labels: np.ndarray
    factor_levels: dict[str, list[str]]
    factor_grid: dict[tuple[str, ...], int]
    n_obs: int
    n_configs: int
    n_stations: int
    n_runs: int
    ref_idx: int
    cfg: ModelConfig

    # ── Convenience ──────────────────────────────────────────────────────

    @property
    def ref_label(self) -> str:
        """Return the label of the reference configuration."""
        return str(self.config_labels[self.ref_idx])

    def config_label_to_idx(self, label: str) -> int:
        """Return the integer index for a config label, or raise KeyError."""
        matches = np.where(self.config_labels == label)[0]
        if len(matches) == 0:
            raise KeyError(
                f"Config label {label!r} not found. "
                f"Available: {self.config_labels.tolist()}"
            )
        return int(matches[0])

    def subset_config_indices(self, labels: list[str] | None) -> tuple[list[str], np.ndarray]:
        """Return (ordered_labels, integer_indices) for a subset of configs.

        If *labels* is None, returns all configs in their canonical order.
        """
        all_labels = self.config_labels.tolist()
        if labels is None:
            return all_labels, np.arange(self.n_configs, dtype=int)
        idx = np.array([self.config_label_to_idx(l) for l in labels], dtype=int)
        return labels, idx

    def factor_index_grid(self) -> tuple[list[str], list[str], dict[tuple[str, str], int]]:
        """For the 2-factor case, return (tier_levels, case_levels, grid_map).

        The grid_map maps (tier, case) → flat config index.
        Raises ValueError if > 2 factors.
        """
        if len(self.cfg.factors) != 2:
            raise ValueError(
                f"factor_index_grid requires exactly 2 factors, got {len(self.cfg.factors)}."
            )
        f0_name = self.cfg.factors[0].name
        f1_name = self.cfg.factors[1].name
        f0_levels = self.factor_levels[f0_name]
        f1_levels = self.factor_levels[f1_name]
        grid = {(str(k[0]), str(k[1])): v for k, v in self.factor_grid.items()}
        return f0_levels, f1_levels, grid

    def coords_dict(self) -> dict[str, Any]:
        """Build a PyMC-compatible coords dictionary."""
        coords: dict[str, Any] = {
            "Config": self.config_labels,
            "Station": self.station_labels,
            "Run": self.run_labels,
            "obs_id": np.arange(self.n_obs),
        }
        if self.n_configs > 1:
            coords["Config_free"] = self.config_labels[:-1]
        if self.n_stations > 1:
            coords["Station_free"] = self.station_labels[:-1]
        if self.n_runs > 1:
            coords["Run_free"] = self.run_labels[:-1]
        return coords


# ── Factory function ────────────────────────────────────────────────────────

def encode_dataset(df: pd.DataFrame, cfg: ModelConfig) -> EpistemicDataset:
    """Validate and encode a long-format DataFrame into an EpistemicDataset.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format observations.  Must contain columns for the EDP,
        every factor, the station, and the run.
    cfg : ModelConfig
        Model configuration specifying column names, factors, reference, etc.

    Returns
    -------
    EpistemicDataset
        Frozen, immutable encoded dataset.

    Raises
    ------
    ValueError
        On missing columns, missing reference, or non-numeric EDP.
    """
    df = df.copy()  # never mutate the caller's frame

    # ── validate required columns ────────────────────────────────────────
    required = [cfg.edp_col, cfg.station_col, cfg.run_col, *cfg.factor_columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    # ── coerce EDP to float, drop NaN ────────────────────────────────────
    df[cfg.edp_col] = pd.to_numeric(df[cfg.edp_col], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=[cfg.edp_col]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        import warnings
        warnings.warn(
            f"Dropped {n_dropped} rows with non-finite EDP values.",
            stacklevel=2,
        )

    y = df[cfg.edp_col].to_numpy(dtype=float)

    # ── encode each factor ───────────────────────────────────────────────
    factor_levels: dict[str, list[str]] = {}

    for fspec in cfg.factors:
        col_vals = df[fspec.column].astype(str)
        if fspec.levels is not None:
            levels = list(fspec.levels)
        else:
            levels = sorted(col_vals.unique().tolist())
        factor_levels[fspec.name] = levels

    # ── build flat configuration labels and index ────────────────────────
    # Cartesian product of factor levels in declaration order
    sep = cfg.config_sep
    all_combos = list(product(*[factor_levels[f.name] for f in cfg.factors]))
    # flat label = joined levels (sep="" → "4D", sep="_" → "4_D_PM4Sand")
    flat_labels = [sep.join(combo) for combo in all_combos]
    flat_labels = order_config_labels(flat_labels, major="tier", sep=sep)

    # rebuild combo → index mapping with the sorted labels
    label_to_combo = {
        sep.join(combo): combo for combo in all_combos
    }
    combo_to_flat_idx = {
        label_to_combo[lbl]: i for i, lbl in enumerate(flat_labels)
    }

    config_labels = np.array(flat_labels, dtype=object)

    # build per-row flat config label
    if cfg.config_col in df.columns:
        row_config = df[cfg.config_col].astype(str).to_numpy()
    else:
        factor_cols = [df[f.column].astype(str) for f in cfg.factors]
        row_config = np.array([sep.join(parts) for parts in zip(*factor_cols, strict=False)])

    label_to_idx = {lbl: i for i, lbl in enumerate(flat_labels)}
    config_idx = np.array([label_to_idx.get(lbl, -1) for lbl in row_config], dtype=int)

    if np.any(config_idx < 0):
        bad = set(row_config[config_idx < 0])
        raise ValueError(
            f"Config labels in data not found in factor grid: {sorted(bad)[:10]}. "
            f"Expected one of: {flat_labels[:10]}…"
        )

    # ── encode station ───────────────────────────────────────────────────
    st_cat = pd.Categorical(df[cfg.station_col].astype(str))
    station_labels = np.array(st_cat.categories.tolist(), dtype=object)
    station_idx = st_cat.codes.astype(int)

    # ── encode run ───────────────────────────────────────────────────────
    rk_cat = pd.Categorical(df[cfg.run_col].astype(str))
    run_labels = np.array(rk_cat.categories.tolist(), dtype=object)
    run_idx = rk_cat.codes.astype(int)

    # ── reference config ─────────────────────────────────────────────────
    if cfg.ref_config not in label_to_idx:
        raise ValueError(
            f"ref_config={cfg.ref_config!r} not found in configurations. "
            f"Available: {flat_labels}"
        )
    ref_idx = label_to_idx[cfg.ref_config]

    # ── factor grid (combo tuple → flat index) ───────────────────────────
    factor_grid = dict(combo_to_flat_idx)

    return EpistemicDataset(
        y=y,
        config_idx=config_idx,
        station_idx=station_idx,
        run_idx=run_idx,
        config_labels=config_labels,
        station_labels=station_labels,
        run_labels=run_labels,
        factor_levels=factor_levels,
        factor_grid=factor_grid,
        n_obs=len(y),
        n_configs=len(config_labels),
        n_stations=len(station_labels),
        n_runs=len(run_labels),
        ref_idx=ref_idx,
        cfg=cfg,
    )
