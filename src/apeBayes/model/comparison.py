"""
Model comparison via LOO-CV / WAIC.

Fits multiple model variants to the same EpistemicDataset and returns
a ranked comparison table. Needs the pointwise log-likelihood, which
``sample_model`` stores since 0.4.0 (and :func:`ensure_log_likelihood`
adds to older posteriors).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import arviz as az
from arviz import InferenceData

from .sampling import sample_model

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from ..config import ModelConfig
    from ..data import EpistemicDataset
    from .base import ModelBuilder


@dataclass
class FittedVariant:
    """One fitted model variant for comparison."""

    name: str
    builder: ModelBuilder
    cfg: ModelConfig
    idata: InferenceData


def compare_models(
    data: EpistemicDataset,
    variants: Mapping[str, ModelBuilder],
    cfg: ModelConfig,
    *,
    ic: str = "loo",
    scale: str = "log",
) -> tuple[pd.DataFrame, dict[str, FittedVariant]]:
    """Fit multiple model variants and compare with information criteria.

    Parameters
    ----------
    data : EpistemicDataset
        Shared encoded dataset.
    variants : dict[str, ModelBuilder]
        Named model builders to compare.  E.g.::

            {
                "StudentT_hetero": FlatConfigModel(likelihood="student_t", heteroskedastic=True),
                "Gaussian_hetero": FlatConfigModel(likelihood="gaussian", heteroskedastic=True),
                "StudentT_homo":   FlatConfigModel(likelihood="student_t", heteroskedastic=False),
                "Gaussian_homo":   FlatConfigModel(likelihood="gaussian", heteroskedastic=False),
            }
    cfg : ModelConfig
        Base configuration (priors, sampling, etc.).
    ic : {"loo", "waic"}
        Information criterion for ranking.
    scale : str
        Scale for ``az.compare()`` (default "log").

    Returns
    -------
    comparison_table : pd.DataFrame
        ArviZ comparison table, ranked best-to-worst.
    fitted : dict[str, FittedVariant]
        All fitted variants keyed by name, for further inspection.
    """
    fitted: dict[str, FittedVariant] = {}
    idata_dict: dict[str, InferenceData] = {}

    for name, builder in variants.items():
        # Build variant-specific config overrides if the builder carries them
        model = builder.build(data, cfg)
        idata = sample_model(model, cfg.sampling)
        fitted[name] = FittedVariant(
            name=name,
            builder=builder,
            cfg=cfg,
            idata=idata,
        )
        idata_dict[name] = idata

    comparison = az.compare(idata_dict, ic=ic, scale=scale)

    return comparison, fitted


def ensure_log_likelihood(
    idata: InferenceData,
    builder: ModelBuilder,
    data: EpistemicDataset,
    cfg: ModelConfig,
) -> InferenceData:
    """Add the pointwise ``log_likelihood`` group if the posterior lacks it.

    Posteriors sampled before 0.4.0 (or with nutpie and
    ``store_log_likelihood=False``) have none; the model is rebuilt from
    ``builder`` and the log-likelihood evaluated at every stored draw.
    """
    if "log_likelihood" in idata.groups():
        return idata
    import pymc as pm

    model = builder.build(data, cfg)
    pm.compute_log_likelihood(idata, model=model, extend_inferencedata=True,
                              progressbar=False)
    return idata


def compare_fitted(
    fitted: Mapping[str, Any],
    *,
    ic: str = "loo",
    scale: str = "log",
) -> pd.DataFrame:
    """Compare already-fitted ``BayesEpistemicModel`` objects with LOO / WAIC.

    Each model's log-likelihood is computed on demand if missing (see
    :func:`ensure_log_likelihood`). All models must be fitted to the same
    observations.

    Note: PSIS-LOO leaves out single observations, which share rupture and
    station effects with the rest; to test the record structure use
    :func:`apeBayes.model.cv.leave_one_group_out`.
    """
    idatas = {}
    n_obs = set()
    for name, m in fitted.items():
        idatas[name] = ensure_log_likelihood(m.idata, m.builder, m.data, m.cfg)
        n_obs.add(m.data.n_obs)
    if len(n_obs) > 1:
        raise ValueError(f"Models were fitted to different data sizes: {sorted(n_obs)}")
    return az.compare(idatas, ic=ic, scale=scale)
