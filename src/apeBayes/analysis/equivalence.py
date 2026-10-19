"""
Epistemic equivalence analysis.

Probabilistic equivalence testing, pairwise distance/equivalence matrices,
and hierarchical clustering of configurations.

All functions accept ``sigma_denom`` (the aleatory SD per draw) rather than
a specific ``sigma_run``; the caller chooses whether that denominator is
\u03c3_src, \u03c3_GM (canonical), or \u03c3_pred. See ``uncertanty_measures.md`` \u00a74.
Paper-level defaults (\u00a75\u2013\u00a76): ``alpha = 0.4`` for the equivalence threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform


def _apply_denom(dmu: np.ndarray, sigma_denom: np.ndarray,
                 subset_idx: np.ndarray | None) -> np.ndarray:
    """Divide ``dmu`` (S, M) by ``sigma_denom`` handling 1-D or 2-D denoms."""
    if sigma_denom.ndim == 1:
        return dmu / sigma_denom[:, None]
    if sigma_denom.ndim == 2:
        denom = sigma_denom[:, subset_idx] if subset_idx is not None else sigma_denom
        if denom.shape != dmu.shape:
            raise ValueError(
                f"sigma_denom shape {denom.shape} incompatible with dmu shape {dmu.shape}"
            )
        return dmu / denom
    raise ValueError(
        f"sigma_denom must be 1-D (S,) or 2-D (S, K), got shape {sigma_denom.shape}"
    )


def _threshold_mask(dmu: np.ndarray, sigma_denom: np.ndarray,
                    alpha: float, subset_idx: np.ndarray | None) -> np.ndarray:
    """Return boolean mask |dmu| < alpha * sigma_denom."""
    if sigma_denom.ndim == 1:
        return np.abs(dmu) < alpha * sigma_denom[:, None]
    if sigma_denom.ndim == 2:
        denom = sigma_denom[:, subset_idx] if subset_idx is not None else sigma_denom
        return np.abs(dmu) < alpha * denom
    raise ValueError(
        f"sigma_denom must be 1-D (S,) or 2-D (S, K), got shape {sigma_denom.shape}"
    )


def equivalence_probability(
    mu_config: np.ndarray,
    sigma_denom: np.ndarray,
    ref_idx: int,
    labels: list[str],
    *,
    alpha: float = 0.4,
    ci: tuple[float, float] = (0.05, 0.95),
    subset_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    r"""P(|\u0394\u03bc_k| < \u03b1 \u00b7 \u03c3_denom) for each configuration vs reference.

    Parameters
    ----------
    mu_config : (S, K) array
        Configuration effects per draw.
    sigma_denom : (S,) or (S, K) array
        Aleatory denominator per draw \u2014 \u03c3_src, \u03c3_GM, or \u03c3_pred.
    ref_idx : int
        Index of the reference configuration.
    labels : list[str]
        Config labels for the subset.
    alpha : float
        Equivalence radius on the \u03b2 scale. Default 0.4 per spec \u00a75.
    subset_idx : optional
        Index array for a subset of configs.
    """
    mu_sel = mu_config[:, subset_idx] if subset_idx is not None else mu_config
    dmu = mu_sel - mu_config[:, [ref_idx]]

    beta = _apply_denom(dmu, sigma_denom, subset_idx)
    P_equiv = np.mean(_threshold_mask(dmu, sigma_denom, alpha, subset_idx), axis=0)

    ci_lo, ci_hi = ci
    return pd.DataFrame({
        "Config": labels,
        "alpha": float(alpha),
        "P_equiv": P_equiv,
        "beta_med": np.median(beta, axis=0),
        "beta_lo": np.quantile(beta, ci_lo, axis=0),
        "beta_hi": np.quantile(beta, ci_hi, axis=0),
    }).sort_values("P_equiv", ascending=False).reset_index(drop=True)


def equivalence_sweep(
    mu_config: np.ndarray,
    sigma_denom: np.ndarray,
    ref_idx: int,
    labels: list[str],
    *,
    alphas: np.ndarray | None = None,
    subset_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sweep equivalence probability over a range of \u03b1 thresholds.

    Returns a long DataFrame with columns: Config, alpha, P_equiv.
    """
    if alphas is None:
        alphas = np.linspace(0.01, 1.2, 60)

    mu_sel = mu_config[:, subset_idx] if subset_idx is not None else mu_config
    dmu = mu_sel - mu_config[:, [ref_idx]]  # (S, M)

    records = []
    for a in alphas:
        P = np.mean(_threshold_mask(dmu, sigma_denom, float(a), subset_idx), axis=0)
        for k, lbl in enumerate(labels):
            records.append({"Config": lbl, "alpha": float(a), "P_equiv": float(P[k])})

    return pd.DataFrame(records)


def epistemic_distance_matrix(
    mu_config: np.ndarray,
    sigma_denom: np.ndarray,
    *,
    alpha: float = 0.4,
    labels: list[str] | None = None,
    subset_idx: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray]:
    """NxN epistemic distance matrix.

    Computes D_ij = 1 \u2212 P(|\u03bc_i \u2212 \u03bc_j| < \u03b1\u00b7\u03c3_denom).

    ``sigma_denom`` must be a 1-D ``(S,)`` array \u2014 pairwise comparisons
    share a single station-level aleatory scale per draw, so there is no
    meaningful (S, K) broadcast for config-dependent residuals here.
    """
    if sigma_denom.ndim != 1:
        raise ValueError(
            f"epistemic_distance_matrix requires a 1-D sigma_denom, "
            f"got shape {sigma_denom.shape}"
        )
    mu_sel = mu_config[:, subset_idx] if subset_idx is not None else mu_config

    # Pairwise |\u0394\u03bc| for each draw \u2192 (S, M, M)
    dmu = mu_sel[:, :, None] - mu_sel[:, None, :]
    P = np.mean(np.abs(dmu) < alpha * sigma_denom[:, None, None], axis=0)
    D = 1.0 - P
    np.fill_diagonal(D, 0.0)

    if labels is None:
        labels = [str(i) for i in range(D.shape[0])]

    return labels, D


def epistemic_equivalence_matrix(
    mu_config: np.ndarray,
    sigma_denom: np.ndarray,
    *,
    alpha: float = 0.4,
    labels: list[str] | None = None,
    subset_idx: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray]:
    """NxN equivalence matrix: P_ij = P(|\u03bc_i \u2212 \u03bc_j| < \u03b1\u00b7\u03c3_denom)."""
    labels, D = epistemic_distance_matrix(
        mu_config, sigma_denom,
        alpha=alpha, labels=labels, subset_idx=subset_idx,
    )
    P = 1.0 - D
    np.fill_diagonal(P, 1.0)
    return labels, P


def epistemic_clusters(
    mu_config: np.ndarray,
    sigma_denom: np.ndarray,
    *,
    alpha: float = 0.4,
    labels: list[str] | None = None,
    subset_idx: np.ndarray | None = None,
    method: str = "average",
    threshold: float | None = None,
    n_clusters: int | None = None,
    p_link: float = 0.5,
) -> pd.DataFrame:
    """Hierarchical clustering on epistemic distance D = 1 - P_eq.

    Uses either a distance *threshold* or a fixed *n_clusters* count.
    If neither is given, the threshold is ``1 - p_link``: configurations
    join a cluster when their (linkage-averaged) pairwise P_eq exceeds
    ``p_link`` (default 0.5, "more likely equivalent than not").

    Before 0.4.0 the default threshold was ``1 - alpha``, which reused the
    equivalence radius alpha (a distance on the beta scale) as a
    probability, so the clusters moved with alpha for the wrong reason.

    Returns
    -------
    pd.DataFrame with columns: Config, cluster, leaf_order.
    """
    labels_out, D = epistemic_distance_matrix(
        mu_config, sigma_denom,
        alpha=alpha, labels=labels, subset_idx=subset_idx,
    )
    M = len(labels_out)
    if M <= 1:
        return pd.DataFrame({
            "Config": labels_out,
            "cluster": [1] * M,
            "leaf_order": list(range(M)),
        })

    y = squareform(D, checks=False)
    Z = linkage(y, method=method)

    if not (0.0 < p_link < 1.0):
        raise ValueError(f"p_link must be in (0, 1), got {p_link}")
    if threshold is None and n_clusters is None:
        threshold = 1.0 - p_link

    if n_clusters is not None:
        cluster_ids = fcluster(Z, t=int(n_clusters), criterion="maxclust")
    else:
        cluster_ids = fcluster(Z, t=float(threshold), criterion="distance")

    leaf_idx = leaves_list(Z)
    leaf_rank = {int(i): rank for rank, i in enumerate(leaf_idx.tolist())}

    return pd.DataFrame({
        "Config": labels_out,
        "cluster": cluster_ids.astype(int),
        "leaf_order": [leaf_rank[i] for i in range(M)],
    }).sort_values(["cluster", "leaf_order"]).reset_index(drop=True)


# ── New-record (record-marginal) contrasts ─────────────────────────────────

def new_record_contrast_var(
    *,
    config_to_case: np.ndarray,
    ref_case: int,
    sigma_run: np.ndarray,
    lambda_case: np.ndarray | None = None,
    sigma_inter: np.ndarray | None = None,
    xi_case: np.ndarray | None = None,
    sigma_u_case: np.ndarray | None = None,
    sigma_v_case: np.ndarray | None = None,
    n_runs: int = 6,
    n_stations: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Per-record discrepancy variance of config k vs the reference.

    The record terms the model does *not* share between config k (Case c)
    and the reference (Case ref) make the k-vs-ref difference vary from
    record to record. Per record, that variance is

    .. math::
        v^{rup}_k = (\lambda_c - \lambda_{ref})^2 \sigma_{run}^2
                    + [c \ne ref](\sigma_{u,c}^2 + \sigma_{u,ref}^2),\qquad
        v^{cell}_k = (\xi_c - \xi_{ref})^2 \sigma_{inter}^2
                    + [c \ne ref](\sigma_{v,c}^2 + \sigma_{v,ref}^2).

    Averaged over a new set of ``n_runs`` ruptures × ``n_stations``
    stations, the contrast Δμ_k picks up a sampling term with variance
    ``v_rup / n_runs + v_cell / (n_runs · n_stations)`` (2026-10-19
    critique, H5).

    Returns
    -------
    (var_new, var_record) : two (S, K) arrays
        ``var_new``: variance of the record-averaged contrast for a new
        record set. ``var_record``: per-record variance ``v_rup + v_cell``
        (the record-level discrepancy, without residuals).
    """
    S = sigma_run.shape[0]
    K = len(config_to_case)
    c = np.asarray(config_to_case)
    diff_c = (c != ref_case).astype(float)[None, :]
    v_rup = np.zeros((S, K))
    v_cell = np.zeros((S, K))
    if lambda_case is not None:
        v_rup += (lambda_case[:, c] - lambda_case[:, [ref_case]]) ** 2 * sigma_run[:, None] ** 2
    if sigma_inter is not None and xi_case is not None:
        v_cell += (xi_case[:, c] - xi_case[:, [ref_case]]) ** 2 * sigma_inter[:, None] ** 2
    if sigma_u_case is not None:
        v_rup += diff_c * (sigma_u_case[:, c] ** 2 + sigma_u_case[:, [ref_case]] ** 2)
    if sigma_v_case is not None:
        v_cell += diff_c * (sigma_v_case[:, c] ** 2 + sigma_v_case[:, [ref_case]] ** 2)
    var_new = v_rup / n_runs + v_cell / (n_runs * n_stations)
    return var_new, v_rup + v_cell


def record_marginal_dmu(
    dmu: np.ndarray, var_new: np.ndarray, *, seed: int = 0,
) -> np.ndarray:
    """Δμ for a new record set: ``dmu + sqrt(var_new) · z`` per draw (seeded)."""
    z = np.random.default_rng(seed).standard_normal(dmu.shape)
    return dmu + np.sqrt(var_new) * z


def decision_labels(p_eq: np.ndarray, p_star: float) -> np.ndarray:
    """``equivalent`` / ``inequivalent`` / ``undecided`` from P_eq and the P* gate."""
    return np.where(p_eq >= p_star, "equivalent",
                    np.where(p_eq <= 1.0 - p_star, "inequivalent", "undecided"))
