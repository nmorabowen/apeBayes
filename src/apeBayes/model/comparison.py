"""
Model comparison via LOO-CV / WAIC.

Fits multiple model variants to the same EpistemicDataset and returns
a ranked comparison table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
