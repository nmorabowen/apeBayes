"""
Configuration dataclasses for apeBayes.

Four concerns, four configs:
  - FactorSpec     : describes one axis of the epistemic tensor
  - PriorConfig    : prior hyper-parameters (swappable per model variant)
  - SamplingConfig : MCMC tuning knobs
  - DecisionConfig : α-ladder and P* gate for equivalence decisions
  - ModelConfig    : ties everything together (factors, columns, reference, …)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ── Epistemic factor ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FactorSpec:
    """One axis of the epistemic modelling tensor.

    Parameters
    ----------
    name : str
        Human-readable axis label (e.g. "SSI", "Nonlinearity").
    column : str
        Column name in the long-format dataframe.
    levels : list[str] | None
        Explicit level ordering.  If *None* the ordering is inferred from the
        data (sorted alphanumerically).
    """

    name: str
    column: str
    levels: list[str] | None = None


# ── Prior configuration ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class PriorConfig:
    """Hyper-parameters for the Bayesian model priors.

    All σ-scale parameters are for half-normal or normal widths.
    """

    # Fixed effects
    sigma_intercept: float = 10.0
    sigma_config: float = 5.0       # configuration (TierCase) effects
    sigma_station: float = 3.0      # station fixed effects

    # Random effects
    sigma_run: float = 5.0          # run-level RE scale prior

    # Observation noise
    sigma_eps: float = 5.0          # residual SD prior

    # Student-t degrees of freedom
    nu_prior_lambda: float = 10.0   # Exponential(1/lambda) for (nu - 2)

    def __post_init__(self) -> None:
        for fld_name in [
            "sigma_intercept",
            "sigma_config",
            "sigma_station",
            "sigma_run",
            "sigma_eps",
            "nu_prior_lambda",
        ]:
            v = getattr(self, fld_name)
            if not (isinstance(v, (int, float)) and v > 0):
                raise ValueError(f"PriorConfig.{fld_name} must be positive, got {v!r}")


# ── Sampling configuration ──────────────────────────────────────────────────

@dataclass(frozen=True)
class SamplingConfig:
    """MCMC sampler settings."""

    draws: int = 2000
    tune: int = 2000
    chains: int = 4
    target_accept: float = 0.92
    max_treedepth: int = 12
    seed: int = 123
    sampler: Literal["nutpie", "pymc"] = "nutpie"

    def __post_init__(self) -> None:
        if self.draws < 1:
            raise ValueError(f"draws must be >= 1, got {self.draws}")
        if self.tune < 1:
            raise ValueError(f"tune must be >= 1, got {self.tune}")
        if self.chains < 1:
            raise ValueError(f"chains must be >= 1, got {self.chains}")
        if not (0 < self.target_accept < 1):
            raise ValueError(f"target_accept must be in (0,1), got {self.target_accept}")


# ── Decision configuration ──────────────────────────────────────────────────

# Allowed denominator identifiers for standardised bias β.
#   src  -> σ_src                         (v6-era conservative)
#   gm   -> σ_GM = √(σ_src² + σ_inter²)   (canonical, v8)
#   pred -> σ_pred (adds residual)        (full-predictive generous)
_ALLOWED_DENOMINATORS: tuple[str, ...] = ("src", "gm", "pred")


@dataclass(frozen=True)
class DecisionConfig:
    """Equivalence-decision parameters per ``uncertanty_measures.md`` §5-§6.

    Parameters
    ----------
    alpha_eq : float
        Primary equivalence threshold on the β scale. Paper default 0.4
        corresponds to the "three-halves" code-amplification reading.
    alpha_ladder : tuple[float, ...]
        Reported α values for the P_eq table. Must contain ``alpha_eq``.
        Default ``(0.4, 0.7, 1.1)``: equivalence / clearly biased / severely
        biased rungs.
    p_star : float
        Probability gate for the equivalence call. Must lie in ``(0, 1)``.
        Default 0.95 matches moderate-consequence reliability conventions.
    denominators : tuple[str, ...]
        Subset of ``{"src", "gm", "pred"}`` indicating which β variants to
        compute. ``"gm"`` is canonical; ``"src"`` and ``"pred"`` are the
        sensitivity endpoints.
    """

    alpha_eq: float = 0.4
    alpha_ladder: tuple[float, ...] = (0.4, 0.7, 1.1)
    p_star: float = 0.95
    denominators: tuple[str, ...] = ("src", "gm", "pred")

    def __post_init__(self) -> None:
        if not (isinstance(self.alpha_eq, (int, float)) and self.alpha_eq > 0):
            raise ValueError(f"alpha_eq must be positive, got {self.alpha_eq!r}")
        if len(self.alpha_ladder) == 0:
            raise ValueError("alpha_ladder must be non-empty.")
        for a in self.alpha_ladder:
            if not (isinstance(a, (int, float)) and a > 0):
                raise ValueError(f"alpha_ladder entries must be positive, got {a!r}")
        if self.alpha_eq not in self.alpha_ladder:
            raise ValueError(
                f"alpha_eq ({self.alpha_eq}) must be a member of alpha_ladder "
                f"({self.alpha_ladder})."
            )
        if not (isinstance(self.p_star, (int, float)) and 0 < self.p_star < 1):
            raise ValueError(f"p_star must lie in (0, 1), got {self.p_star!r}")
        if len(self.denominators) == 0:
            raise ValueError("denominators must be non-empty.")
        for d in self.denominators:
            if d not in _ALLOWED_DENOMINATORS:
                raise ValueError(
                    f"denominator {d!r} not in {_ALLOWED_DENOMINATORS}."
                )
        if len(set(self.denominators)) != len(self.denominators):
            raise ValueError(f"denominators must be unique, got {self.denominators}")


# ── Model configuration ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelConfig:
    """Full model specification.

    Parameters
    ----------
    factors : list[FactorSpec]
        Ordered list of epistemic modelling dimensions.  The Cartesian product
        of their levels forms the configuration space.
    config_col : str
        Name of the (possibly derived) flat configuration column.
    edp_col : str
        Column holding the (already log-transformed) EDP values.
    station_col : str
        Column identifying the recording station.
    run_col : str
        Column identifying the earthquake realisation (run key).
    ref_config : str
        Label of the reference configuration (e.g. "4D").
    likelihood : {"student_t", "gaussian"}
        Observation likelihood family.
    heteroskedastic : bool
        If True, estimate a separate σ_ε per configuration.
    ci : tuple[float, float]
        Credible-interval quantiles for posterior summaries.
    priors : PriorConfig
        Prior hyper-parameters.
    sampling : SamplingConfig
        MCMC sampler settings.
    decision : DecisionConfig
        Equivalence-decision parameters (α-ladder, P* gate, denominator set).
    """

    factors: list[FactorSpec] = field(default_factory=lambda: [
        FactorSpec(name="SSI", column="Tier"),
        FactorSpec(name="Nonlinearity", column="Case"),
    ])
    config_col: str = "TierCase"
    edp_col: str = "edp"
    station_col: str = "sta"
    run_col: str = "runkey"
    ref_config: str = "4D"

    # Model variant switches
    likelihood: Literal["student_t", "gaussian"] = "student_t"
    heteroskedastic: bool = True

    # Posterior summary
    ci: tuple[float, float] = (0.05, 0.95)

    # Label generation
    config_sep: str = ""  # separator for flat config labels; "" gives "4D", "_" gives "4_D"

    # Sub-configs
    priors: PriorConfig = field(default_factory=PriorConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    @property
    def ci_lo(self) -> float:
        """Return the lower credible-interval quantile."""
        return self.ci[0]

    @property
    def ci_hi(self) -> float:
        """Return the upper credible-interval quantile."""
        return self.ci[1]

    @property
    def factor_columns(self) -> list[str]:
        """Column names for all epistemic factors."""
        return [f.column for f in self.factors]

    @property
    def factor_names(self) -> list[str]:
        """Human-readable names for all epistemic factors."""
        return [f.name for f in self.factors]

    def __post_init__(self) -> None:
        if len(self.factors) == 0:
            raise ValueError("At least one FactorSpec is required.")
        if not (0 < self.ci[0] < self.ci[1] < 1):
            raise ValueError(f"ci must satisfy 0 < lo < hi < 1, got {self.ci}")
        if self.likelihood not in ("student_t", "gaussian"):
            msg = f"likelihood must be 'student_t' or 'gaussian', got {self.likelihood!r}"
            raise ValueError(msg)
