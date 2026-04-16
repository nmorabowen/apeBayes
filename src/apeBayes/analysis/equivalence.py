"""
Epistemic equivalence analysis.

Probabilistic equivalence testing, pairwise distance/equivalence matrices,
and hierarchical clustering of configurations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform

from ..utils import EPS


def equivalence_probability(
    mu_config: np.ndarray,
    sigma_run: np.ndarray,
    ref_idx: int,
    labels: list[str],
    *,
    alpha: float = 0.5,
    ci: tuple[float, float] = (0.05, 0.95),
    subset_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    """P(|Δμ_k| < α · σ_run) for each configuration vs reference.

    Parameters
    ----------
    mu_config : (S, K) — configuration effects.
    sigma_run : (S,) — run-level SD.
    ref_idx : int — index of reference config.
    labels : list[str] — config labels for the subset.
    alpha : float — equivalence radius in σ_run units.
    subset_idx : optional index array for a subset of configs.
    """
    mu_full = mu_config  # (S, K)
    if subset_idx is not None:
        mu_sel = mu_full[:, subset_idx]
    else:
        mu_sel = mu_full

    dmu = mu_sel - mu_full[:, [ref_idx]]
    beta = dmu / sigma_run[:, None]
    P_equiv = np.mean(np.abs(dmu) < alpha * sigma_run[:, None], axis=0)

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
    sigma_run: np.ndarray,
    ref_idx: int,
    labels: list[str],
    *,
    alphas: np.ndarray | None = None,
    subset_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sweep equivalence probability over a range of α thresholds.

    Returns a long DataFrame with columns: Config, alpha, P_equiv.
    """
    if alphas is None:
        alphas = np.linspace(0.01, 0.5, 50)

    mu_full = mu_config
    if subset_idx is not None:
        mu_sel = mu_full[:, subset_idx]
    else:
        mu_sel = mu_full
        labels = labels

    mu_ref = mu_full[:, [ref_idx]]
    dmu = mu_sel - mu_ref             # (S, M)

    records = []
    for a in alphas:
        P = np.mean(np.abs(dmu) < a * sigma_run[:, None], axis=0)
        for k, lbl in enumerate(labels):
            records.append({"Config": lbl, "alpha": float(a), "P_equiv": float(P[k])})

    return pd.DataFrame(records)


def epistemic_distance_matrix(
    mu_config: np.ndarray,
    sigma_run: np.ndarray,
    *,
    alpha: float = 0.5,
    labels: list[str] | None = None,
    subset_idx: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray]:
    """NxN epistemic distance: D_ij = 1 − P(|μ_i − μ_j| < α·σ_run).

    Parameters
    ----------
    mu_config : (S, K) — config effects per draw.
    sigma_run : (S,) — run-level SD.
    alpha : float — equivalence radius.
    labels : list[str] — config labels (length K or len(subset_idx)).
    subset_idx : optional subset indices.

    Returns
    -------
    (labels, D) where D is (M, M) symmetric, D_ii = 0.
    """
    if subset_idx is not None:
        mu_sel = mu_config[:, subset_idx]
    else:
        mu_sel = mu_config

    # Pairwise |Δμ| for each draw → (S, M, M)
    dmu = mu_sel[:, :, None] - mu_sel[:, None, :]
    P = np.mean(np.abs(dmu) < alpha * sigma_run[:, None, None], axis=0)
    D = 1.0 - P
    np.fill_diagonal(D, 0.0)

    if labels is None:
        labels = [str(i) for i in range(D.shape[0])]

    return labels, D


def epistemic_equivalence_matrix(
    mu_config: np.ndarray,
    sigma_run: np.ndarray,
    *,
    alpha: float = 0.5,
    labels: list[str] | None = None,
    subset_idx: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray]:
    """NxN equivalence: P_ij = P(|μ_i − μ_j| < α·σ_run)."""
    labels, D = epistemic_distance_matrix(
        mu_config, sigma_run,
        alpha=alpha, labels=labels, subset_idx=subset_idx,
    )
    P = 1.0 - D
    np.fill_diagonal(P, 1.0)
    return labels, P


def epistemic_clusters(
    mu_config: np.ndarray,
    sigma_run: np.ndarray,
    *,
    alpha: float = 0.5,
    labels: list[str] | None = None,
    subset_idx: np.ndarray | None = None,
    method: str = "average",
    threshold: float | None = None,
    n_clusters: int | None = None,
) -> pd.DataFrame:
    """Hierarchical clustering on epistemic distance.

    Uses either a distance *threshold* or a fixed *n_clusters* count.
    If neither is given, defaults to threshold = 1 − α.

    Returns
    -------
    pd.DataFrame with columns: Config, cluster, leaf_order.
    """
    labels_out, D = epistemic_distance_matrix(
        mu_config, sigma_run,
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

    if threshold is None and n_clusters is None:
        threshold = 1.0 - alpha

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
