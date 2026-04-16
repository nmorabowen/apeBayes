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

from .config import ModelConfig, PriorConfig, SamplingConfig, FactorSpec
from .data import EpistemicDataset, encode_dataset
from .facade import BayesEpistemicModel
from .model import FlatConfigModel, HierarchicalConfigModel, RandomSlopesModel, RandomSlopesInteractionModel, ModelBuilder, sample_model, compare_models
from .multi_edp import MultiEDPModel, EDPSpec
from .posterior import PosteriorAccessor

__all__ = [
    # Main entry points
    "BayesEpistemicModel",
    "MultiEDPModel",
    "EDPSpec",
    # Configuration
    "ModelConfig",
    "PriorConfig",
    "SamplingConfig",
    "FactorSpec",
    # Data
    "EpistemicDataset",
    "encode_dataset",
    # Model building
    "FlatConfigModel",
    "HierarchicalConfigModel",
    "RandomSlopesModel",
    "RandomSlopesInteractionModel",
    "ModelBuilder",
    "sample_model",
    "compare_models",
    # Posterior
    "PosteriorAccessor",
]

__version__ = "0.2.0"
