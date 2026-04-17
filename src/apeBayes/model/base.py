"""
Protocol defining the interface for model builders.

Any model builder must implement ``build()`` returning a ``pm.Model``
whose posterior contains the expected variable names.

It must also implement ``sigma_GM(post)`` returning the canonical aleatory
SD σ_GM for its variant (see ``uncertanty_measures.md`` §7). σ_GM is a
modelling statement (which variance layers count as aleatory for a single
station under the DRM suite) so the formula lives on the model class,
not on the data accessor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import pymc as pm

    from ..config import ModelConfig
    from ..data import EpistemicDataset
    from ..posterior import PosteriorAccessor

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

    def sigma_GM(self, post: PosteriorAccessor) -> np.ndarray:
        """Return σ_GM draws for this model variant.

        Parameters
        ----------
        post : PosteriorAccessor
            Fitted-model posterior.

        Returns
        -------
        np.ndarray
            ``(S,)`` draws of the canonical aleatory SD σ_GM (the station-
            specific ground-motion variability). Each model variant
            implements its own formula per ``uncertanty_measures.md`` §7.
        """
        ...

    @property
    def description(self) -> str:
        """Short human-readable description for comparison tables."""
        ...
