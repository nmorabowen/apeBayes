"""
Protocol defining the interface for model builders.

Any model builder must implement ``build()`` returning a ``pm.Model``
whose posterior contains the expected variable names.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pymc as pm

from ..config import ModelConfig
from ..data import EpistemicDataset


# ── Expected posterior variable names ───────────────────────────────────────
# Every model builder must produce these in the pm.Model trace.
# Optional variables are listed separately.

REQUIRED_VARS = frozenset({
    "mu0",          # scalar intercept
    "mu_config",    # (n_configs,) configuration effects, sum-to-zero
    "delta_st",     # (n_stations,) station effects, sum-to-zero
    "sigma_run",    # scalar run-level SD
    "b_run",        # (n_runs,) run random effects
})

OPTIONAL_VARS = frozenset({
    "sigma_eps",        # scalar residual SD (homoskedastic)
    "sigma_eps_config", # (n_configs,) per-config residual SD (heteroskedastic)
    "nu_minus2",        # Student-t degrees-of-freedom minus 2
})


# ── Protocol ────────────────────────────────────────────────────────────────

@runtime_checkable
class ModelBuilder(Protocol):
    """Interface that all model builders must satisfy."""

    def build(
        self,
        data: EpistemicDataset,
        cfg: ModelConfig,
    ) -> pm.Model:
        """Construct a PyMC model from encoded data and configuration.

        Returns
        -------
        pm.Model
            A model ready for ``pm.sample()``.  The model's posterior
            must contain at least the variables in ``REQUIRED_VARS``.
        """
        ...

    @property
    def description(self) -> str:
        """Short human-readable description for comparison tables."""
        ...
