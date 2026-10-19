"""Model specification, sampling, and comparison."""

from .base import ModelBuilder
from .case_record import CaseRecordModel
from .comparison import compare_fitted, compare_models, ensure_log_likelihood
from .cv import compare_logo, leave_one_group_out
from .flat import FlatConfigModel
from .hierarchical import HierarchicalConfigModel
from .random_slopes import RandomSlopesModel
from .random_slopes_interaction import RandomSlopesInteractionModel
from .registry import builder_from_spec, builder_spec, model_version
from .sampling import sample_model

__all__ = [
    "CaseRecordModel",
    "FlatConfigModel",
    "HierarchicalConfigModel",
    "ModelBuilder",
    "RandomSlopesInteractionModel",
    "RandomSlopesModel",
    "builder_from_spec",
    "builder_spec",
    "compare_fitted",
    "compare_logo",
    "compare_models",
    "ensure_log_likelihood",
    "leave_one_group_out",
    "model_version",
    "sample_model",
]
