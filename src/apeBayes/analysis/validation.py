"""
Validation-parameter analysis — ``uncertanty_measures.md`` §7.

The validation parameter

    ÊDP^med_{t,c}(case)^(s) = exp(u_0^(s)(case) + Δμ_{t,c}^(s)(case))

is the posterior-predictive median EDP implied by modelling configuration
(t, c) at one specific case (structure × IM level × EDP × station),
expressed in physical units. It is a *case-conditional* object (§7.2) —
not comparable across cases — and exists to sit next to an **externally
anchored** log-scale reference ``u_0`` (§7.3):

- experimental benchmark,
- higher-fidelity reference simulation,
- code-specified capacity threshold, or
- a specified null (linear-elastic, code-default, etc.).

``u_0`` taken as a cross-configuration mean or median of the model itself
is **not admissible** (§7.3) — use β / ρ_epi for that question.

The three decision rules in §7.4 are:

- ``"threshold"``  — P(ÊDP^med <|> θ) ≥ P*_val   (capacity / code check)
- ``"fractional"`` — P(|ÊDP^med − target| / target ≤ τ_val) ≥ P*_val
                     (experimental / reference-model benchmark)
- ``"ci_overlap"`` — HDI(ÊDP^med) overlaps an externally supplied interval
                     (benchmark with its own quantified uncertainty)

τ_val and P*_val are **per use case** and are NOT inherited from α_eq / P*.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def validation_parameter(
    mu_config: np.ndarray,
    u0: float,
    ref_idx: int,
    labels: list[str],
    *,
    ci: tuple[float, float] = (0.05, 0.95),
    subset_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    r"""Posterior draws and summary of ÊDP^med — ``uncertanty_measures.md`` §7.1.

    Parameters
    ----------
    mu_config : (S, K) array
        Posterior draws of configuration effects on the log-EDP scale.
    u0 : float
        **Externally anchored** log-scale reference for the case (§7.3).
        Passing an internally computed cross-configuration mean or median
        of ``mu_config`` is outside the spec's admissible sources.
    ref_idx : int
        Index of the reference configuration.
    labels : list[str]
        Configuration labels (length K or ``len(subset_idx)``).
    ci : tuple, default (0.05, 0.95)
        Quantile CI for the ``edp_{med,lo,hi}`` columns.
    subset_idx : (M,) int array, optional
        Restrict output to these configuration indices.

    Returns
    -------
    edp_samples : (S, M) ndarray
        Posterior draws of ÊDP^med for each retained configuration.
    summary : pd.DataFrame
        Columns: ``Config``, ``edp_{med,lo,hi}``.
    """
    mu_sub = mu_config[:, subset_idx] if subset_idx is not None else mu_config
    dmu = mu_sub - mu_config[:, [ref_idx]]
    edp = np.exp(u0 + dmu)  # (S, M)

    ci_lo, ci_hi = ci
    summary = pd.DataFrame({
        "Config": labels,
        "edp_med": np.median(edp, axis=0),
        "edp_lo": np.quantile(edp, ci_lo, axis=0),
        "edp_hi": np.quantile(edp, ci_hi, axis=0),
    })
    return edp, summary


def validation_decision(
    mu_config: np.ndarray,
    u0: float,
    ref_idx: int,
    labels: list[str],
    *,
    rule: Literal["threshold", "fractional", "ci_overlap"],
    theta: float | None = None,
    direction: Literal["below", "above"] = "below",
    target: float | None = None,
    tau_val: float | None = None,
    target_hdi: tuple[float, float] | None = None,
    p_star_val: float = 0.95,
    ci: tuple[float, float] = (0.05, 0.95),
    subset_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    r"""Per-configuration validation decision — ``uncertanty_measures.md`` §7.4.

    Computes ÊDP^med posterior draws and applies one of the three §7.4 rules.
    ``tau_val`` and ``p_star_val`` are declared per use case and are NOT
    inherited from ``α_eq`` / ``P*`` (§7.4, §7.5).

    Rules
    -----
    ``"threshold"``
        Validation passes iff P(ÊDP^med ⋛ θ) ≥ p_star_val, where the
        comparison is ``<`` when ``direction='below'`` and ``>`` when
        ``direction='above'``. Natural for capacity / code checks.

    ``"fractional"``
        Validation passes iff P(|ÊDP^med − target| / target ≤ tau_val)
        ≥ p_star_val. Natural for experimental or reference-model
        benchmarks.

    ``"ci_overlap"``
        Validation passes iff the quantile CI on ÊDP^med (set by ``ci``)
        overlaps ``target_hdi``. This rule is deterministic once the CI is
        fixed, so ``posterior_mass`` is 1.0 on pass and 0.0 on fail.

    Parameters
    ----------
    mu_config, u0, ref_idx, labels, ci, subset_idx
        See :func:`validation_parameter`.
    rule : {"threshold", "fractional", "ci_overlap"}
        Which §7.4 rule to apply.
    theta : float, optional
        Threshold (required for ``rule='threshold'``).
    direction : {"below", "above"}, default "below"
        Passing side of ``theta`` (threshold rule).
    target : float, optional
        External benchmark EDP in physical units (required for
        ``rule='fractional'``).
    tau_val : float, optional
        Fractional tolerance (required for ``rule='fractional'``).
    target_hdi : tuple[float, float], optional
        Physical-unit interval on the benchmark (required for
        ``rule='ci_overlap'``).
    p_star_val : float, default 0.95
        Posterior-mass gate. **Per use case**; not tied to the paper's
        ``cfg.decision.p_star``.

    Returns
    -------
    pd.DataFrame
        One row per configuration. Columns: ``Config``,
        ``edp_{med,lo,hi}``, ``posterior_mass``, ``decision``, ``rule``.
        ``decision`` is ``"pass"`` / ``"fail"``.
    """
    edp, summary = validation_parameter(
        mu_config, u0, ref_idx, labels, ci=ci, subset_idx=subset_idx,
    )

    if rule == "threshold":
        if theta is None:
            raise ValueError("theta required for rule='threshold'")
        if direction == "below":
            satisfied = edp < theta
        elif direction == "above":
            satisfied = edp > theta
        else:
            raise ValueError(
                f"direction must be 'below' or 'above', got {direction!r}"
            )
        mass = np.mean(satisfied, axis=0)

    elif rule == "fractional":
        if target is None or tau_val is None:
            raise ValueError(
                "target and tau_val required for rule='fractional'"
            )
        if target == 0:
            raise ValueError("target must be nonzero for fractional rule")
        satisfied = (np.abs(edp - target) / abs(target)) <= tau_val
        mass = np.mean(satisfied, axis=0)

    elif rule == "ci_overlap":
        if target_hdi is None:
            raise ValueError("target_hdi required for rule='ci_overlap'")
        lo_tgt, hi_tgt = target_hdi
        overlap = (
            (summary["edp_lo"].to_numpy() <= hi_tgt)
            & (summary["edp_hi"].to_numpy() >= lo_tgt)
        )
        mass = overlap.astype(float)

    else:
        raise ValueError(f"unknown rule {rule!r}")

    decision = np.where(mass >= p_star_val, "pass", "fail")
    summary["posterior_mass"] = mass
    summary["decision"] = decision
    summary["rule"] = rule
    return summary
