"""
apeBayes — Bayesian epistemic uncertainty analysis for computational models.

Quick start
-----------
>>> from apeBayes import BayesEpistemicModel, ModelConfig
>>> model = BayesEpistemicModel(df_long, cfg=ModelConfig(...))
>>> model.fit()
>>> model.standardized_bias_table()
"""

from __future__ import annotations

from .config import DecisionConfig, FactorSpec, ModelConfig, PriorConfig, SamplingConfig
from .data import EpistemicDataset, encode_dataset
from .facade import BayesEpistemicModel
from .model import (
    FlatConfigModel,
    HierarchicalConfigModel,
    ModelBuilder,
    RandomSlopesInteractionModel,
    RandomSlopesModel,
    compare_models,
    sample_model,
)
from .multi_edp import EDPSpec, MultiEDPModel
from .posterior import PosteriorAccessor

__all__ = [
    # Main entry points
    "BayesEpistemicModel",
    # Configuration
    "DecisionConfig",
    "EDPSpec",
    # Data
    "EpistemicDataset",
    "FactorSpec",
    # Model building
    "FlatConfigModel",
    "HierarchicalConfigModel",
    "ModelBuilder",
    # Configuration
    "ModelConfig",
    "MultiEDPModel",
    # Posterior
    "PosteriorAccessor",
    "PriorConfig",
    "RandomSlopesInteractionModel",
    "RandomSlopesModel",
    "SamplingConfig",
    "compare_models",
    "encode_dataset",
    "sample_model",
]

__version__ = "0.3.0.dev0"
