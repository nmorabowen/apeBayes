"""
MCMC sampling with proper error handling.

No silent exception swallowing — if nutpie is requested but unavailable,
we warn explicitly and fall back to PyMC's default NUTS.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import pymc as pm

if TYPE_CHECKING:
    from arviz import InferenceData

    from ..config import SamplingConfig

logger = logging.getLogger(__name__)


def sample_model(
    model: pm.Model,
    sampling: SamplingConfig,
    *,
    prior_predictive: bool = True,
    posterior_predictive: bool = True,
) -> InferenceData:
    """Run MCMC on a compiled PyMC model.

    Parameters
    ----------
    model : pm.Model
        A built (but not yet sampled) PyMC model.
    sampling : SamplingConfig
        Sampler settings.
    prior_predictive : bool
        If True, draw prior-predictive samples before MCMC.
    posterior_predictive : bool
        If True, draw posterior-predictive samples after MCMC.

    Returns
    -------
    InferenceData
        ArviZ InferenceData with posterior, sample_stats, and optionally
        prior_predictive / posterior_predictive groups.
    """
    base_kwargs = dict(
        draws=sampling.draws,
        tune=sampling.tune,
        chains=sampling.chains,
        target_accept=sampling.target_accept,
        max_treedepth=sampling.max_treedepth,
        random_seed=sampling.seed,
        progressbar=True,
        return_inferencedata=True,
    )

    with model:
        # ── Prior predictive ─────────────────────────────────────────
        if prior_predictive:
            try:
                prior = pm.sample_prior_predictive(
                    samples=500,
                    random_seed=sampling.seed,
                )
            except Exception as exc:
                warnings.warn(
                    f"Prior predictive sampling failed: {exc}",
                    stacklevel=2,
                )
                prior = None
        else:
            prior = None

        # ── Posterior sampling ───────────────────────────────────────
        if sampling.sampler == "nutpie":
            try:
                import nutpie  # noqa: F401
                idata = pm.sample(nuts_sampler="nutpie", **base_kwargs)
                logger.info("Sampled with nutpie.")
            except ImportError:
                warnings.warn(
                    "nutpie not installed — falling back to PyMC default NUTS. "
                    "Install with: pip install nutpie",
                    stacklevel=2,
                )
                idata = pm.sample(**base_kwargs)
            except Exception as exc:
                warnings.warn(
                    f"nutpie sampling failed ({type(exc).__name__}: {exc}). "
                    "Falling back to PyMC default NUTS.",
                    stacklevel=2,
                )
                idata = pm.sample(**base_kwargs)
        else:
            idata = pm.sample(**base_kwargs)

        # ── Posterior predictive ─────────────────────────────────────
        if posterior_predictive:
            try:
                pm.sample_posterior_predictive(
                    idata,
                    extend_inferencedata=True,
                    random_seed=sampling.seed,
                )
            except Exception as exc:
                warnings.warn(
                    f"Posterior predictive sampling failed: {exc}",
                    stacklevel=2,
                )

    # ── Attach prior predictive ──────────────────────────────────────
    if prior is not None:
        idata.extend(prior)

    return idata
