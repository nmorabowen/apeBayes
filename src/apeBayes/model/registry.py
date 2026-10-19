"""Builder serialisation: builder ⇄ JSON spec, and version labels.

A fitted posterior records the builder that produced it (class name and
constructor options) in ``idata.attrs["apebayes_builder"]`` and in the
bundle's ``config.json``, so ``BayesEpistemicModel.load`` reconstructs the
exact variant (v8 vs v8.1 vs v10, the residual prior, the zero-sum scheme,
σ_inter, σ_λ, …) instead of guessing it from the variable names.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .case_record import CaseRecordModel
from .flat import FlatConfigModel
from .hierarchical import HierarchicalConfigModel
from .random_slopes import RandomSlopesModel
from .random_slopes_interaction import RandomSlopesInteractionModel

if TYPE_CHECKING:
    from .base import ModelBuilder

BUILDERS: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        FlatConfigModel,
        HierarchicalConfigModel,
        RandomSlopesModel,
        RandomSlopesInteractionModel,
        CaseRecordModel,
    )
}

ATTR_KEY = "apebayes_builder"


def builder_options(builder: ModelBuilder) -> dict[str, Any]:
    """Return the constructor kwargs of a builder (empty-safe for v1–v3)."""
    if hasattr(builder, "options"):
        return dict(builder.options())
    opts: dict[str, Any] = {}
    for kw in ("likelihood", "heteroskedastic"):
        attr = {"likelihood": "_likelihood_override",
                "heteroskedastic": "_hetero_override"}[kw]
        if hasattr(builder, attr):
            opts[kw] = getattr(builder, attr)
    return opts


def model_version(builder: ModelBuilder) -> str:
    """Lineage label: v1-v3, v4-v7, v8, v8.1, v9 or v10 (see README)."""
    if isinstance(builder, CaseRecordModel):
        return "v10"
    if isinstance(builder, RandomSlopesInteractionModel):
        o = builder.options()
        if o["interaction_loading"]:
            return "v9"
        return "v8.1" if o["residual_pooling"] == "partial" else "v8"
    if isinstance(builder, RandomSlopesModel):
        return "v4-v7"
    if isinstance(builder, (FlatConfigModel, HierarchicalConfigModel)):
        return "v1-v3"
    return type(builder).__name__


def builder_spec(builder: ModelBuilder) -> dict[str, Any]:
    """JSON-safe description of a builder."""
    return {
        "class": type(builder).__name__,
        "version": model_version(builder),
        "options": builder_options(builder),
    }


def builder_from_spec(spec: dict[str, Any] | str) -> ModelBuilder:
    """Inverse of :func:`builder_spec` (accepts the JSON string too)."""
    if isinstance(spec, str):
        spec = json.loads(spec)
    cls = BUILDERS.get(spec["class"])
    if cls is None:
        raise ValueError(f"Unknown builder class {spec['class']!r} in saved spec.")
    return cls(**spec.get("options", {}))
