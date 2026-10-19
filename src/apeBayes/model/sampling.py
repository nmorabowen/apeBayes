"""
MCMC sampling with proper error handling.

No silent exception swallowing:

- If nutpie is requested but not installed, we warn and fall back to PyMC's
  NUTS, and the fallback is recorded in ``idata.attrs`` (``sampler_used``,
  ``sampler_fallback``).
- If nutpie is installed but fails, the error is raised. Falling back would
  silently change the sampler (and its tree-depth / adaptation behaviour)
  behind a published number.

The sampler settings actually used are written to ``idata.attrs`` so a
saved posterior documents how it was produced, the pointwise
log-likelihood is stored (``SamplingConfig.store_log_likelihood``; nutpie
does not produce it, so it is computed after sampling) for LOO / WAIC, and
a convergence verdict is attached (see :mod:`..diagnostics.convergence`).
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import pymc as pm

from ..diagnostics.convergence import attach_convergence

if TYPE_CHECKING:
    from arviz import InferenceData

    from ..config import SamplingConfig

logger = logging.getLogger(__name__)


def _nutpie_available() -> bool:
    try:
        import nutpie  # noqa: F401
    except ImportError:
        return False
    return True


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
        Sampler settings. ``max_treedepth`` is honoured by both samplers
        (nutpie receives it as ``maxdepth``; before 0.4.0 nutpie silently
        ran at its default depth 10).
    prior_predictive : bool
        If True, draw prior-predictive samples before MCMC.
    posterior_predictive : bool
        If True, draw posterior-predictive samples after MCMC.

    Returns
    -------
    InferenceData
        ArviZ InferenceData with posterior, sample_stats, log_likelihood
        (unless disabled) and optionally prior_predictive /
        posterior_predictive groups. ``idata.attrs`` carries the sampler
        settings and the convergence verdict.
    """
    common = dict(
        draws=sampling.draws,
        tune=sampling.tune,
        chains=sampling.chains,
        target_accept=sampling.target_accept,
        random_seed=sampling.seed,
        progressbar=True,
        return_inferencedata=True,
    )
    store_ll = getattr(sampling, "store_log_likelihood", True)

    with model:
        # ── Prior predictive ─────────────────────────────────────────
        prior = None
        if prior_predictive:
            try:
                prior = pm.sample_prior_predictive(
                    draws=500,
                    random_seed=sampling.seed,
                )
            except Exception as exc:
                warnings.warn(
                    f"Prior predictive sampling failed: {exc}",
                    stacklevel=2,
                )

        # ── Posterior sampling ───────────────────────────────────────
        sampler_used = sampling.sampler
        fallback = ""
        if sampling.sampler == "nutpie" and not _nutpie_available():
            warnings.warn(
                "nutpie not installed: falling back to PyMC NUTS (recorded in "
                "idata.attrs['sampler_fallback']). Install with: pip install nutpie",
                stacklevel=2,
            )
            sampler_used = "pymc"
            fallback = "nutpie not installed"

        if sampler_used == "nutpie":
            idata = pm.sample(
                nuts_sampler="nutpie",
                nuts_sampler_kwargs={"maxdepth": sampling.max_treedepth},
                **common,
            )
            logger.info("Sampled with nutpie.")
            if store_ll:
                pm.compute_log_likelihood(idata, extend_inferencedata=True, progressbar=False)
        else:
            idata = pm.sample(
                max_treedepth=sampling.max_treedepth,
                idata_kwargs={"log_likelihood": store_ll},
                **common,
            )

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

    idata.attrs.update({
        "sampler_requested": sampling.sampler,
        "sampler_used": sampler_used,
        "sampler_fallback": fallback,
        "draws": int(sampling.draws),
        "tune": int(sampling.tune),
        "chains": int(sampling.chains),
        "target_accept": float(sampling.target_accept),
        "max_treedepth": int(sampling.max_treedepth),
        "seed": int(sampling.seed),
    })
    attach_convergence(idata)
    return idata
