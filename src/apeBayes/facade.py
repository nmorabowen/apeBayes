"""
BayesEpistemicModel — convenience facade.

Wires together data encoding, model building, sampling, posterior access,
and analysis into one object.  Notebooks call this; internals are
independently usable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from .analysis import bias, decomposition, equivalence, fitted, variance
from .config import ModelConfig
from .data import EpistemicDataset, encode_dataset
from .diagnostics.convergence import (
    diagnostics_summary,
    divergences_count,
    ess_table,
    rhat_table,
)
from .diagnostics.validation import posterior_predictive_check
from .model.comparison import FittedVariant, compare_models
from .model.flat import FlatConfigModel
from .model.hierarchical import HierarchicalConfigModel
from .model.random_slopes import RandomSlopesModel
from .model.random_slopes_interaction import RandomSlopesInteractionModel
from .model.sampling import sample_model
from .plots.style import CMAP_DIV, CMAP_SEQ, FULL_WIDTH, HALF_WIDTH
from .posterior.accessor import PosteriorAccessor

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from arviz import InferenceData

    from .model.base import ModelBuilder


def _fmt(a: float) -> str:
    """Format an α value into a stable column-label suffix (e.g. 0.4 -> '0.4')."""
    # Strip trailing zeros, keep at least one decimal for floats that look int
    s = f"{a:g}"
    return s


def _variant_tag(builder: ModelBuilder) -> str:
    """Return a short tag describing the variant class.

    Used by :func:`_detect_builder` to compare a user-supplied builder
    against the variant inferred from the posterior.
    """
    if isinstance(builder, RandomSlopesInteractionModel):
        # pylint: disable=protected-access
        return "v9" if getattr(builder, "_interaction_loading", False) else "v8"
    if isinstance(builder, RandomSlopesModel):
        return "v4-v7"
    if isinstance(builder, (HierarchicalConfigModel, FlatConfigModel)):
        return "v1-v3"
    return type(builder).__name__


def _detect_builder(post: PosteriorAccessor) -> tuple[ModelBuilder, str]:
    """Infer which ModelBuilder produced a posterior from its variable set.

    Returns
    -------
    (builder, tag)
        A default-constructed builder matching the posterior's variant and
        the variant tag string (``'v9'`` / ``'v8'`` / ``'v4-v7'`` /
        ``'v1-v3'``).

    Notes
    -----
    Probes used (see ``uncertanty_measures.md`` §7 for the σ_GM formula
    associated with each variant):

    - ``xi_case``     present → v9 RandomSlopesInteractionModel (guard:
      σ_GM raises NotImplementedError, but we still want to route there
      so the error is correct rather than silently returning σ_src).
    - ``gamma_sr``    present → v8 RandomSlopesInteractionModel.
    - ``lambda_case`` present → v4-v7 RandomSlopesModel.
    - otherwise                → v1-v3; σ_GM is σ_src either way so we
      return FlatConfigModel (HierarchicalConfigModel has identical
      σ_GM, so the dispatch is equivalent).
    """
    if post.has_var("xi_case"):
        return RandomSlopesInteractionModel(interaction_loading=True), "v9"
    if post.has_var("gamma_sr"):
        return RandomSlopesInteractionModel(), "v8"
    if post.has_var("lambda_case"):
        return RandomSlopesModel(), "v4-v7"
    return FlatConfigModel(), "v1-v3"


class BayesEpistemicModel:
    """High-level API for Bayesian epistemic uncertainty analysis.

    Usage
    -----
    >>> model = BayesEpistemicModel(df_long, cfg=my_config)
    >>> model.fit()
    >>> model.standardized_bias_table()
    >>> model.variance_budget_table()
    >>> model.equivalence_probability_table(alpha=0.4)
    """

    # ── Construction & fitting ───────────────────────────────────────────

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: ModelConfig | None = None,
        *,
        name: str | None = None,
    ) -> None:
        self.cfg = cfg or ModelConfig()
        self.name = name
        self.data: EpistemicDataset = encode_dataset(df, self.cfg)
        self._posterior: PosteriorAccessor | None = None
        self._builder: ModelBuilder | None = None
        self._sigma_gm_cache: np.ndarray | None = None

    def fit(
        self,
        *,
        builder: ModelBuilder | None = None,
        prior_predictive: bool = True,
        posterior_predictive: bool = True,
    ) -> BayesEpistemicModel:
        """Build and sample the model.

        Parameters
        ----------
        builder : ModelBuilder, optional
            Custom model builder.  Defaults to FlatConfigModel with
            settings from self.cfg.
        prior_predictive : bool
            Sample prior predictive before MCMC.
        posterior_predictive : bool
            Sample posterior predictive after MCMC.

        Returns self for chaining.
        """
        if builder is None:
            builder = FlatConfigModel()
        self._builder = builder

        pm_model = builder.build(self.data, self.cfg)
        idata = sample_model(
            pm_model,
            self.cfg.sampling,
            prior_predictive=prior_predictive,
            posterior_predictive=posterior_predictive,
        )
        self._posterior = PosteriorAccessor(idata, self.data)
        self._sigma_gm_cache = None
        return self

    @property
    def posterior(self) -> PosteriorAccessor:
        """Return the posterior accessor; raise if not yet fitted."""
        if self._posterior is None:
            raise RuntimeError("Call fit() before accessing the posterior.")
        return self._posterior

    @property
    def idata(self) -> InferenceData:
        """Return the underlying ArviZ InferenceData object."""
        return self.posterior.idata

    @property
    def is_fitted(self) -> bool:
        """Return True if the model has been fitted."""
        return self._posterior is not None

    @property
    def builder(self) -> ModelBuilder:
        """Return the ModelBuilder that produced the posterior.

        Raises
        ------
        RuntimeError
            If the model has not been fitted or loaded with a builder.
        """
        if self._builder is None:
            raise RuntimeError(
                "No ModelBuilder registered. Call fit() or load(builder=...) first."
            )
        return self._builder

    # ── Canonical aleatory denominators ──────────────────────────────────

    def sigma_src_draws(self) -> np.ndarray:
        """(S,) posterior draws of σ_src. Alias for ``posterior.sigma_src()``."""
        return self.posterior.sigma_src()

    def sigma_GM_draws(self) -> np.ndarray:
        """(S,) posterior draws of σ_GM via per-variant dispatch.

        Dispatches to the fitted model's ``sigma_GM(post)`` method per
        ``uncertanty_measures.md`` §7. Cached after the first call.
        """
        if self._sigma_gm_cache is None:
            self._sigma_gm_cache = self.builder.sigma_GM(self.posterior)
        return self._sigma_gm_cache

    def sigma_pred_draws(self, ref_idx: int | None = None) -> np.ndarray:
        """(S,) posterior draws of σ_pred = √(σ_GM² + σ_eps_eff²).

        Parameters
        ----------
        ref_idx : int | None
            Residual to fold in. When ``None``, uses the reference config
            index from the dataset (matches the spec's default).
        """
        p = self.posterior
        if ref_idx is None:
            ref_idx = p.ref_idx
        return p.sigma_pred(self.sigma_GM_draws(), ref_idx=ref_idx)

    # ── Serialization ───────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Save the fitted model (InferenceData + dataset metadata) to NetCDF.

        Parameters
        ----------
        path : str or Path
            File path (recommended extension: .nc).

        Returns
        -------
        Path to the saved file.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        path = Path(path)
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model. Call fit() first.")
        self.idata.to_netcdf(str(path))
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        df: pd.DataFrame,
        cfg: ModelConfig | None = None,
        *,
        name: str | None = None,
        builder: ModelBuilder | None = None,
    ) -> BayesEpistemicModel:
        """Reload a fitted model from a NetCDF file.

        The ``ModelBuilder`` is auto-detected from the posterior's variable
        set so σ_GM dispatch routes to the correct variant without user
        input. Probes:

        - ``xi_case``     present → v9 ``RandomSlopesInteractionModel``
        - ``gamma_sr``    present → v8 ``RandomSlopesInteractionModel``
        - ``lambda_case`` present → v4-v7 ``RandomSlopesModel``
        - otherwise                → v1-v3 ``FlatConfigModel``

        Parameters
        ----------
        path : str or Path
            Path to a .nc file saved by :meth:`save`.
        df : pd.DataFrame
            The *same* long-format DataFrame used to fit the model
            (needed to reconstruct the EpistemicDataset).
        cfg : ModelConfig, optional
            Model configuration (must match the one used for fitting).
        name : str, optional
            Human-readable model name.
        builder : ModelBuilder, optional
            Override the auto-detected variant. When provided, the class
            must match what the posterior shape implies; a mismatch raises
            ``ValueError`` rather than silently returning the wrong σ_GM.

        Returns
        -------
        BayesEpistemicModel
            Fully reconstructed fitted model.

        Raises
        ------
        ValueError
            If ``builder`` is supplied and its variant does not match the
            one detected from the posterior's variables.
        """
        import arviz as az

        path = Path(path)
        idata = az.from_netcdf(str(path))
        obj = cls(df, cfg=cfg, name=name)
        post = PosteriorAccessor(idata, obj.data)
        obj._posterior = post

        detected, detected_tag = _detect_builder(post)
        if builder is None:
            obj._builder = detected
        else:
            user_tag = _variant_tag(builder)
            if user_tag != detected_tag:
                raise ValueError(
                    f"Supplied builder is {user_tag} "
                    f"({type(builder).__name__}) but the posterior's "
                    f"variable set implies {detected_tag}. Drop the "
                    f"builder kwarg to accept the detected variant, "
                    f"or re-check the NetCDF file."
                )
            obj._builder = builder
        return obj

    # ── Analysis: bias ───────────────────────────────────────────────────

    def standardized_bias_table(
        self,
        ref: str | None = None,
        configs: list[str] | None = None,
        *,
        denominator: Literal["src", "gm", "pred"] = "gm",
    ) -> pd.DataFrame:
        """Standardised epistemic bias β relative to a reference.

        Canonical denominator is σ_GM (``denominator="gm"``); pass
        ``"src"`` for the conservative β_src and ``"pred"`` for the
        generous β_pred. See ``uncertanty_measures.md`` §4.
        """
        p = self.posterior
        ref_idx = self.data.config_label_to_idx(ref) if ref else p.ref_idx
        labels, subset_idx = self.data.subset_config_indices(configs)
        sigma_denom = self._sigma_denom_for(denominator, ref_idx)
        return bias.standardized_bias(
            p.mu_config(), sigma_denom, ref_idx, labels,
            ci=self.cfg.ci, subset_idx=subset_idx,
        )

    def bias_probability_table(
        self,
        *,
        mode: Literal["exceed_band", "within_equiv", "positive"] = "within_equiv",
        band: float | None = None,
        alpha_equiv: float | None = None,
        ref: str | None = None,
        configs: list[str] | None = None,
        denominator: Literal["src", "gm", "pred"] = "gm",
    ) -> pd.DataFrame:
        """Compute posterior probabilities for bias exceedance or equivalence.

        Defaults pull from ``self.cfg.decision``: ``alpha_equiv=alpha_eq``
        (0.4) and ``band=alpha_ladder[-1]`` (1.1). Pass explicit values
        to override.
        """
        p = self.posterior
        ref_idx = self.data.config_label_to_idx(ref) if ref else p.ref_idx
        labels, subset_idx = self.data.subset_config_indices(configs)
        dec = self.cfg.decision
        if alpha_equiv is None:
            alpha_equiv = dec.alpha_eq
        if band is None:
            band = dec.alpha_ladder[-1]
        sigma_denom = self._sigma_denom_for(denominator, ref_idx)
        return bias.bias_probability(
            p.mu_config(), sigma_denom, ref_idx, labels,
            mode=mode, band=band, alpha_equiv=alpha_equiv,
            subset_idx=subset_idx,
        )

    def _sigma_denom_for(
        self, denominator: Literal["src", "gm", "pred"], ref_idx: int,
    ) -> np.ndarray:
        """Resolve the requested aleatory denominator to an (S,) array."""
        if denominator == "src":
            return self.sigma_src_draws()
        if denominator == "gm":
            return self.sigma_GM_draws()
        if denominator == "pred":
            return self.sigma_pred_draws(ref_idx=ref_idx)
        raise ValueError(f"Unknown denominator {denominator!r}")

    # ── Analysis: headline decision report ───────────────────────────────

    def decision_report(
        self,
        *,
        ref: str | None = None,
        configs: list[str] | None = None,
        denominators: tuple[str, ...] | None = None,
        alpha_eq: float | None = None,
        alpha_ladder: tuple[float, ...] | None = None,
        p_star: float | None = None,
    ) -> pd.DataFrame:
        """Single-shot equivalence decision table per ``uncertanty_measures.md``.

        Computes β_src, β_GM, β_pred (whichever ``denominators`` selects)
        and the α-ladder P_eq table, then applies the P* gate at α_eq
        under the canonical σ_GM to label each configuration as
        ``equivalent`` / ``inequivalent`` / ``undecided``. The
        ``denominator_robust`` column flags configurations where the
        three denominators agree on the label.

        Parameters default to ``self.cfg.decision``.

        Returns
        -------
        pd.DataFrame
            One row per configuration (ordered by label). Columns:

            - ``Config``
            - ``beta_{src,gm,pred}_{med,lo,hi}`` for every requested denom
            - ``P_eq_{\u03b1}_gm`` for each α in the ladder
            - ``P_eq_{\u03b1_eq}_{src,pred}`` (sensitivity at α_eq)
            - ``decision`` (``equivalent`` | ``inequivalent`` | ``undecided``)
            - ``denominator_robust`` (bool)
        """
        dec = self.cfg.decision
        if alpha_eq is None:
            alpha_eq = dec.alpha_eq
        if alpha_ladder is None:
            alpha_ladder = dec.alpha_ladder
        if p_star is None:
            p_star = dec.p_star
        if denominators is None:
            denominators = dec.denominators

        if alpha_eq not in alpha_ladder:
            raise ValueError(
                f"alpha_eq ({alpha_eq}) must be in alpha_ladder {alpha_ladder}."
            )
        for d in denominators:
            if d not in ("src", "gm", "pred"):
                raise ValueError(f"Unknown denominator {d!r}")
        if "gm" not in denominators:
            raise ValueError(
                "decision_report requires 'gm' among denominators "
                "(it is the canonical denominator for the P* gate)."
            )

        p = self.posterior
        ref_idx = self.data.config_label_to_idx(ref) if ref else p.ref_idx
        labels, subset_idx = self.data.subset_config_indices(configs)
        mu_config = p.mu_config()
        mu_sel = mu_config[:, subset_idx] if subset_idx is not None else mu_config
        dmu = mu_sel - mu_config[:, [ref_idx]]  # (S, M)
        ci_lo, ci_hi = self.cfg.ci

        out = pd.DataFrame({"Config": labels})

        # Pre-compute each requested denominator once.
        denom_arrays: dict[str, np.ndarray] = {}
        for d in denominators:
            if d not in ("src", "gm", "pred"):
                raise ValueError(f"Unknown denominator {d!r}")
            denom_arrays[d] = self._sigma_denom_for(d, ref_idx)  # type: ignore[arg-type]

        # β summaries per denominator
        for d in denominators:
            beta = dmu / denom_arrays[d][:, None]
            out[f"beta_{d}_med"] = np.median(beta, axis=0)
            out[f"beta_{d}_lo"] = np.quantile(beta, ci_lo, axis=0)
            out[f"beta_{d}_hi"] = np.quantile(beta, ci_hi, axis=0)

        # P_eq ladder under canonical σ_GM
        gm_denom = denom_arrays["gm"]
        beta_gm = dmu / gm_denom[:, None]
        abs_beta_gm = np.abs(beta_gm)
        for a in alpha_ladder:
            out[f"P_eq_{_fmt(a)}_gm"] = np.mean(abs_beta_gm < a, axis=0)

        # P_eq at α_eq under the sensitivity denominators (if requested)
        for d in denominators:
            if d == "gm":
                continue
            beta_d = dmu / denom_arrays[d][:, None]
            out[f"P_eq_{_fmt(alpha_eq)}_{d}"] = np.mean(np.abs(beta_d) < alpha_eq, axis=0)

        # Decision column under σ_GM
        p_eq_col = out[f"P_eq_{_fmt(alpha_eq)}_gm"].to_numpy()
        decision = np.where(
            p_eq_col >= p_star, "equivalent",
            np.where(p_eq_col <= 1.0 - p_star, "inequivalent", "undecided"),
        )
        out["decision"] = decision

        # Denominator-robust = all requested denoms give the same decision
        decisions_by_denom = {}
        for d in denominators:
            col = out[f"P_eq_{_fmt(alpha_eq)}_{d}"] if d != "gm" else p_eq_col
            col_arr = col.to_numpy() if hasattr(col, "to_numpy") else col
            decisions_by_denom[d] = np.where(
                col_arr >= p_star, "equivalent",
                np.where(col_arr <= 1.0 - p_star, "inequivalent", "undecided"),
            )
        first_d = denominators[0]
        robust = np.ones(len(labels), dtype=bool)
        for d in denominators[1:]:
            robust &= decisions_by_denom[d] == decisions_by_denom[first_d]
        out["denominator_robust"] = robust

        return out

    # ── Analysis: variance ───────────────────────────────────────────────

    def variance_budget_table(
        self,
        *,
        include_interaction: bool | None = None,
    ) -> pd.DataFrame:
        """Posterior variance budget.

        Parameters
        ----------
        include_interaction : bool, optional
            If True, add a fifth "Interaction Var(γ_sr)" row (requires
            a v8+ interaction model).  If None (default), include it
            automatically when the posterior contains ``gamma_sr``.
            If False, always return the legacy 4-row budget for
            backward compatibility with v1–v7 pipelines.
        """
        p = self.posterior
        if include_interaction is None:
            include_interaction = p.has_interaction
        gamma = p.gamma_sr() if include_interaction else None
        return variance.variance_budget(
            p.mu_config(), p.delta_st(), p.sigma_run(), p.sigma_eps(),
            self.data.config_idx, ci=self.cfg.ci, nu=p.nu(),
            gamma_sr=gamma,
        )

    def variance_component_table(
        self,
        configs: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute per-configuration variance components."""
        p = self.posterior
        labels = configs or p.config_labels
        return variance.variance_components(
            p.sigma_run(), p.sigma_eps(), labels,
            ci=self.cfg.ci, nu=p.nu(),
        )

    # ── Analysis: random-slopes parameters (v4+ λ, v8+ σ_inter, v9+ ξ) ──

    def lambda_case_table(self) -> pd.DataFrame:
        """Posterior summary of per-Case run-sensitivity loadings λ."""
        p = self.posterior
        lam = p.lambda_case()
        if lam is None:
            msg = "No lambda_case in posterior. Fit with RandomSlopesModel."
            raise RuntimeError(msg)
        case_labels = p.case_labels or [f"Case{i}" for i in range(lam.shape[1])]
        ci_lo, ci_hi = self.cfg.ci
        return pd.DataFrame({
            "Case": case_labels,
            "lambda_med": np.median(lam, axis=0),
            "lambda_lo": np.quantile(lam, ci_lo, axis=0),
            "lambda_hi": np.quantile(lam, ci_hi, axis=0),
        })

    def xi_case_table(self) -> pd.DataFrame:
        """Posterior summary of per-Case interaction-sensitivity loadings ξ."""
        p = self.posterior
        xi = p.xi_case()
        if xi is None:
            msg = (
                "No xi_case in posterior. Fit with "
                "RandomSlopesInteractionModel(interaction_loading=True)."
            )
            raise RuntimeError(msg)
        case_labels = p.case_labels or [f"Case{i}" for i in range(xi.shape[1])]
        ci_lo, ci_hi = self.cfg.ci
        return pd.DataFrame({
            "Case": case_labels,
            "xi_med": np.median(xi, axis=0),
            "xi_lo": np.quantile(xi, ci_lo, axis=0),
            "xi_hi": np.quantile(xi, ci_hi, axis=0),
        })

    def sigma_inter_table(self) -> pd.DataFrame:
        """Posterior summary of the station×rupture interaction scale σ_inter."""
        p = self.posterior
        sig = p.sigma_inter()
        if sig is None:
            msg = "No sigma_inter in posterior. Fit with RandomSlopesInteractionModel."
            raise RuntimeError(msg)
        ci_lo, ci_hi = self.cfg.ci
        return pd.DataFrame({
            "component": ["sigma_inter"],
            "med": [float(np.median(sig))],
            "lo": [float(np.quantile(sig, ci_lo))],
            "hi": [float(np.quantile(sig, ci_hi))],
        })

    def sigma_rupture_table(self) -> pd.DataFrame:
        """Posterior summary of σ_run (named sigma_rupture for paper clarity)."""
        p = self.posterior
        sig = p.sigma_run()
        ci_lo, ci_hi = self.cfg.ci
        return pd.DataFrame({
            "component": ["sigma_run"],
            "med": [float(np.median(sig))],
            "lo": [float(np.quantile(sig, ci_lo))],
            "hi": [float(np.quantile(sig, ci_hi))],
        })

    # ── Analysis: decomposition ──────────────────────────────────────────

    def axiswise_decomposition_table(
        self,
        *,
        ratio_mode: Literal["var_over_sigma", "var_over_sigma2"] = "var_over_sigma",
    ) -> pd.DataFrame:
        """Decompose mu surface into tier, case, and interaction effects."""
        p = self.posterior
        f0_levels, f1_levels, grid = self.data.factor_index_grid()
        decomp = decomposition.axiswise_decomposition_draws(
            p.mu_config(), f0_levels, f1_levels, grid, p.sigma_run(),
        )
        return decomposition.axiswise_table(
            decomp, ci=self.cfg.ci, ratio_mode=ratio_mode,
            factor_names=(self.cfg.factors[0].name, self.cfg.factors[1].name),
        )

    def level_ranking_tables(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute tier- and case-level ranking tables with interaction."""
        p = self.posterior
        f0_levels, f1_levels, grid = self.data.factor_index_grid()
        decomp = decomposition.axiswise_decomposition_draws(
            p.mu_config(), f0_levels, f1_levels, grid, p.sigma_run(),
        )
        return decomposition.level_ranking_tables(
            decomp, ci=self.cfg.ci,
            factor_names=(self.cfg.factors[0].name, self.cfg.factors[1].name),
        )

    # ── Analysis: equivalence ────────────────────────────────────────────

    def equivalence_probability_table(
        self,
        *,
        alpha: float | None = None,
        ref: str | None = None,
        configs: list[str] | None = None,
        denominator: Literal["src", "gm", "pred"] = "gm",
    ) -> pd.DataFrame:
        """Compute equivalence probability at threshold ``alpha``.

        ``alpha`` defaults to ``cfg.decision.alpha_eq`` (paper default 0.4).
        ``denominator`` defaults to the canonical σ_GM.
        """
        p = self.posterior
        ref_idx = self.data.config_label_to_idx(ref) if ref else p.ref_idx
        labels, subset_idx = self.data.subset_config_indices(configs)
        if alpha is None:
            alpha = self.cfg.decision.alpha_eq
        sigma_denom = self._sigma_denom_for(denominator, ref_idx)
        return equivalence.equivalence_probability(
            p.mu_config(), sigma_denom, ref_idx, labels,
            alpha=alpha, ci=self.cfg.ci, subset_idx=subset_idx,
        )

    def equivalence_sweep_table(
        self,
        *,
        alphas: np.ndarray | None = None,
        ref: str | None = None,
        configs: list[str] | None = None,
        denominator: Literal["src", "gm", "pred"] = "gm",
    ) -> pd.DataFrame:
        """Sweep equivalence probability across a range of α values."""
        p = self.posterior
        ref_idx = self.data.config_label_to_idx(ref) if ref else p.ref_idx
        labels, subset_idx = self.data.subset_config_indices(configs)
        sigma_denom = self._sigma_denom_for(denominator, ref_idx)
        return equivalence.equivalence_sweep(
            p.mu_config(), sigma_denom, ref_idx, labels,
            alphas=alphas, subset_idx=subset_idx,
        )

    def epistemic_equivalence_matrix(
        self,
        *,
        alpha: float | None = None,
        configs: list[str] | None = None,
        denominator: Literal["src", "gm"] = "gm",
    ) -> tuple[list[str], np.ndarray]:
        """Compute the pairwise epistemic equivalence probability matrix.

        Only 1-D denominators are supported here (pairwise comparisons
        share a single station-level aleatory scale per draw), so
        ``denominator="pred"`` is not allowed.
        """
        p = self.posterior
        labels, subset_idx = self.data.subset_config_indices(configs)
        if alpha is None:
            alpha = self.cfg.decision.alpha_eq
        sigma_denom = self._sigma_denom_for(denominator, p.ref_idx)
        return equivalence.epistemic_equivalence_matrix(
            p.mu_config(), sigma_denom,
            alpha=alpha, labels=labels, subset_idx=subset_idx,
        )

    def epistemic_clusters_table(
        self,
        *,
        alpha: float | None = None,
        configs: list[str] | None = None,
        method: str = "average",
        threshold: float | None = None,
        n_clusters: int | None = None,
        denominator: Literal["src", "gm"] = "gm",
    ) -> pd.DataFrame:
        """Cluster configurations by epistemic equivalence distance."""
        p = self.posterior
        labels, subset_idx = self.data.subset_config_indices(configs)
        if alpha is None:
            alpha = self.cfg.decision.alpha_eq
        sigma_denom = self._sigma_denom_for(denominator, p.ref_idx)
        return equivalence.epistemic_clusters(
            p.mu_config(), sigma_denom,
            alpha=alpha, labels=labels, subset_idx=subset_idx,
            method=method, threshold=threshold, n_clusters=n_clusters,
        )

    # ── Analysis: fitted values ──────────────────────────────────────────

    def fitted_values(self) -> pd.DataFrame:
        """Compute posterior-mean fitted values and residuals."""
        p = self.posterior
        return fitted.posterior_mean_fitted(
            self.data.y, p.mu0(), p.mu_config(), p.delta_st(), p.b_run(),
            self.data.config_idx, self.data.station_idx, self.data.run_idx,
        )

    def r2_table(self) -> pd.DataFrame:
        """Compute posterior R-squared summary."""
        p = self.posterior
        return fitted.posterior_r2(
            self.data.y, p.mu0(), p.mu_config(), p.delta_st(), p.b_run(),
            self.data.config_idx, self.data.station_idx, self.data.run_idx,
            ci=self.cfg.ci,
        )

    def mu_hat_table(self, configs: list[str] | None = None) -> pd.DataFrame:
        """Compute configuration intercepts with posterior summaries."""
        p = self.posterior
        labels, subset_idx = self.data.subset_config_indices(configs)
        return fitted.mu_hat_table(
            p.mu0(), p.mu_config(), labels, ci=self.cfg.ci, subset_idx=subset_idx,
        )

    def delta_hat_table(self) -> pd.DataFrame:
        """Compute station random-effect posterior summaries."""
        p = self.posterior
        return fitted.delta_hat_table(p.delta_st(), p.station_labels, ci=self.cfg.ci)

    # ── Diagnostics ──────────────────────────────────────────────────────

    def diagnostics_summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """Return convergence diagnostics for all or selected parameters."""
        return diagnostics_summary(self.idata, var_names)

    def divergences_count(self) -> int:
        """Return the number of divergent transitions."""
        return divergences_count(self.idata)

    def rhat_table(self, **kwargs: Any) -> pd.DataFrame:
        """Return R-hat convergence statistics."""
        return rhat_table(self.idata, **kwargs)

    def ess_table(self, **kwargs: Any) -> pd.DataFrame:
        """Return effective sample size statistics."""
        return ess_table(self.idata, **kwargs)

    def posterior_predictive_check(self) -> pd.DataFrame:
        """Run posterior predictive check and return summary statistics."""
        p = self.posterior
        y_rep = p.y_rep()
        if y_rep is None:
            raise RuntimeError(
                "No posterior predictive samples. "
                "Refit with posterior_predictive=True."
            )
        return posterior_predictive_check(
            self.data.y, y_rep,
            config_idx=self.data.config_idx,
            config_labels=p.config_labels,
        )

    # ── Plots ───────────────────────────────────────────────────────────
    #
    # Every plot method exposes the underlying plot function's user-facing
    # kwargs explicitly. Plumbing kwargs (draw arrays, indices, column
    # names) are computed by the facade and not surfaced. No **kwargs
    # escape hatch — IDE autocomplete and the type checker see the real
    # parameter surface.

    # Diagnostics ........................................................

    def plot_rhat_bar(
        self,
        *,
        top_n: int | None = 10,
        threshold: float = 1.01,
        figsize: tuple[float, float] = (HALF_WIDTH, HALF_WIDTH),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "rhat_bar.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot R-hat convergence bar chart."""
        from .plots.diagnostics import plot_rhat_bar as _plot
        return _plot(
            self.rhat_table(),
            top_n=top_n, threshold=threshold,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_ess_bar(
        self,
        *,
        kind: str = "bulk",
        top_n: int | None = 10,
        threshold: float = 400.0,
        figsize: tuple[float, float] = (HALF_WIDTH, HALF_WIDTH),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot effective sample size bar chart."""
        from .plots.diagnostics import plot_ess_bar as _plot
        return _plot(
            self.ess_table(),
            kind=kind, top_n=top_n, threshold=threshold,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    # Bias .............................................................

    def plot_mu_triptych(
        self,
        *,
        ref: str | None = None,
        original_edp_scale: bool = False,
        annot: bool = True,
        fmt_mu: str = ".2f",
        fmt_dmu: str = ".2f",
        fmt_ratio: str = ".2f",
        cmap: str = CMAP_DIV,
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot three-panel heatmap of absolute mu, delta-mu, and median ratio."""
        ref = ref or self.data.ref_label
        mu_hat = self.mu_hat_table()
        bias_df = self.standardized_bias_table()
        from .plots.bias import plot_mu_triptych as _plot
        return _plot(
            mu_hat, bias_df,
            ref=ref, original_edp_scale=original_edp_scale,
            annot=annot, fmt_mu=fmt_mu, fmt_dmu=fmt_dmu, fmt_ratio=fmt_ratio,
            cmap=cmap, figsize=figsize, out_dir=out_dir, prefix=prefix,
        )

    def plot_standardized_bias(
        self,
        *,
        station_subplots: bool = False,
        dot_alpha: float = 0.28,
        posterior_style: str = "violin",
        figsize: tuple[float, float] | None = None,
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "standardized_bias.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot layered standardized-bias figure with density and raw data.

        Uses σ_GM (canonical denominator) for the standardised axis.
        The β draws, raw-data dots, and reference label are computed
        from the fit and injected automatically.
        """
        bias_df = self.standardized_bias_table()
        from .plots.bias import plot_standardized_bias as _plot

        d = self.data
        p = self.posterior
        mu_config = p.mu_config()                 # (S, K)
        sigma_denom = self.sigma_GM_draws()       # (S,)
        ref_idx = list(d.config_labels).index(d.ref_label)
        ref_draws = mu_config[:, ref_idx]         # (S,)
        denom_med = float(np.median(sigma_denom))

        # Posterior beta draws (for violins) — normalise by median σ_GM
        # so the violin shows Δμ uncertainty without denominator noise.
        beta_draws = (mu_config - ref_draws[:, None]) / denom_med
        beta_labels = list(d.config_labels)

        # Raw per-runkey dots and config means (for data overlay).
        n_configs = d.n_configs
        n_runs = d.n_runs
        raw_dots = np.full((n_configs, n_runs), np.nan)
        ref_by_run = np.zeros(n_runs)
        for r in range(n_runs):
            ref_mask = (d.config_idx == ref_idx) & (d.run_idx == r)
            if ref_mask.any():
                ref_by_run[r] = float(np.mean(d.y[ref_mask]))
        for k in range(n_configs):
            for r in range(n_runs):
                mask = (d.config_idx == k) & (d.run_idx == r)
                if mask.any():
                    raw_dots[k, r] = (
                        float(np.mean(d.y[mask])) - ref_by_run[r]
                    ) / denom_med
        raw_means = np.nanmean(raw_dots, axis=1)

        return _plot(
            bias_df,
            beta_draws=beta_draws, beta_labels=beta_labels,
            raw_dots=raw_dots, raw_means=raw_means, raw_labels=beta_labels,
            station_subplots=station_subplots,
            ref_label=d.ref_label,
            dot_alpha=dot_alpha, posterior_style=posterior_style,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_radar_bias_probability(
        self,
        *,
        alpha: float | None = None,
        fill_alpha: float = 0.15,
        figsize: tuple[float, float] = (HALF_WIDTH, HALF_WIDTH),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "radar_bias_probability.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot radar chart of equivalence probabilities per configuration.

        ``alpha`` defaults to ``cfg.decision.alpha_eq``.
        """
        if alpha is None:
            alpha = self.cfg.decision.alpha_eq
        equiv_df = self.equivalence_probability_table(alpha=alpha)
        from .plots.bias import plot_radar_bias_probability as _plot
        return _plot(
            equiv_df, alpha=alpha, ref=self.data.ref_label,
            fill_alpha=fill_alpha,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_bias_ridgeplot(
        self,
        *,
        station_subplots: bool = False,
        denom_name: str = "GM",
        overlap: float = 0.6,
        bw_adjust: float = 0.4,
        figsize: tuple[float, float] | None = None,
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "bias_ridgeplot.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot ridgeplot of posterior β densities (σ_GM denominator)."""
        p = self.posterior
        from .plots.bias import plot_bias_ridgeplot as _plot
        extra: dict[str, Any] = {}
        if station_subplots:
            extra = dict(
                y_obs=self.data.y,
                config_idx=self.data.config_idx,
                station_idx=self.data.station_idx,
                station_labels=list(self.data.station_labels),
            )
        return _plot(
            p.mu_config(), self.sigma_GM_draws(),
            p.config_labels, p.ref_idx,
            denom_name=denom_name, overlap=overlap, bw_adjust=bw_adjust,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
            **extra,
        )

    def plot_bias_probability(
        self,
        *,
        mode: Literal["exceed_band", "within_equiv", "positive"] = "within_equiv",
        alpha_equiv: float | None = None,
        band: float | None = None,
        ref: str | None = None,
        configs: list[str] | None = None,
        denominator: Literal["src", "gm", "pred"] = "gm",
        threshold_label: str = "",
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "bias_probability.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot bar chart of β probability under the requested mode.

        The first six kwargs are forwarded to :meth:`bias_probability_table`;
        the rest control the plot. ``prob_col='prob'`` is pinned internally
        to match the table's output column.
        """
        prob_df = self.bias_probability_table(
            mode=mode, alpha_equiv=alpha_equiv, band=band,
            ref=ref, configs=configs, denominator=denominator,
        )
        from .plots.bias import plot_bias_probability as _plot
        return _plot(
            prob_df,
            prob_col="prob", label_col="Config",
            threshold_label=threshold_label,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    # Variance ..........................................................

    def plot_variance_budget(
        self,
        *,
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "variance_budget.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot variance-budget bar chart."""
        vb = self.variance_budget_table()
        from .plots.variance import plot_variance_budget_bars as _plot
        return _plot(
            vb, figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_variance_budget_waterfall(
        self,
        *,
        figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "variance_budget_waterfall.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot variance-budget waterfall chart."""
        vb = self.variance_budget_table()
        from .plots.variance import plot_variance_budget_waterfall as _plot
        return _plot(
            vb, figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_decomposition_bars(
        self,
        *,
        figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "decomposition_bars.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot axis-wise decomposition bar chart."""
        decomp = self.axiswise_decomposition_table()
        from .plots.variance import plot_decomposition_bars as _plot
        return _plot(
            decomp, figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_sigma_stability(
        self,
        *,
        figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "sigma_stability.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot per-configuration residual-scale stability."""
        vc = self.variance_component_table()
        sigma_src_med = float(np.median(self.posterior.sigma_src()))
        from .plots.variance import plot_sigma_stability as _plot
        return _plot(
            vc, sigma_src_med=sigma_src_med,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_variance_ratio(
        self,
        *,
        figsize: tuple[float, float] = (HALF_WIDTH, 3.5),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "variance_ratio.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot variance-ratio forest across configurations."""
        p = self.posterior
        from .plots.variance import plot_variance_ratio as _plot
        return _plot(
            p.sigma_src(), p.sigma_eps(),
            p.config_labels, ci=self.cfg.ci, nu=p.nu(),
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_level_rankings(
        self,
        *,
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "level_rankings.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot tier and case level rankings."""
        tier_tbl, case_tbl, _ = self.level_ranking_tables()
        factor_names = (self.cfg.factors[0].name, self.cfg.factors[1].name)
        from .plots.variance import plot_level_rankings as _plot
        return _plot(
            tier_tbl, case_tbl, factor_names=factor_names,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_sigma_stability_triptych(
        self,
        *,
        order_by: str = "stability",
        figsize: tuple[float, float] = (FULL_WIDTH, 4.5),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "sigma_stability_triptych.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot three-panel sigma stability figure."""
        p = self.posterior
        from .plots.variance import plot_sigma_stability_triptych as _plot
        return _plot(
            p.sigma_eps(), p.sigma_src(), p.config_labels, p.ref_idx,
            ci=self.cfg.ci, nu=p.nu(),
            order_by=order_by,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    # Equivalence .......................................................

    def plot_equivalence_matrix(
        self,
        *,
        alpha: float | None = None,
        annot: bool = True,
        fmt: str = ".2f",
        cmap: str = CMAP_SEQ,
        figsize: tuple[float, float] = (HALF_WIDTH, HALF_WIDTH),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "equivalence_matrix.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot equivalence probability heatmap."""
        if alpha is None:
            alpha = self.cfg.decision.alpha_eq
        labels, P_mat = self.epistemic_equivalence_matrix(alpha=alpha)
        from .plots.equivalence import plot_equivalence_matrix as _plot
        return _plot(
            labels, P_mat, alpha=alpha,
            annot=annot, fmt=fmt, cmap=cmap,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_equivalence_matrix_with_dendrogram(
        self,
        *,
        alpha: float | None = None,
        method: str = "average",
        cluster_order: bool = True,
        annot: bool = True,
        fmt: str = ".2f",
        cmap: str = CMAP_SEQ,
        figsize: tuple[float, float] = (FULL_WIDTH, 5.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "equivalence_matrix_dendro.pdf",
    ) -> tuple[plt.Figure, np.ndarray, np.ndarray]:
        """Plot equivalence heatmap with dendrogram overlay."""
        if alpha is None:
            alpha = self.cfg.decision.alpha_eq
        labels, P_mat = self.epistemic_equivalence_matrix(alpha=alpha)
        from .plots.equivalence import (
            plot_equivalence_matrix_with_dendrogram as _plot,
        )
        return _plot(
            labels, P_mat, alpha=alpha,
            method=method, cluster_order=cluster_order,
            annot=annot, fmt=fmt, cmap=cmap,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_equivalence_dendrogram(
        self,
        *,
        alpha: float | None = None,
        method: str = "average",
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "equivalence_dendrogram.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot epistemic-distance dendrogram."""
        if alpha is None:
            alpha = self.cfg.decision.alpha_eq
        labels, P_mat = self.epistemic_equivalence_matrix(alpha=alpha)
        from .plots.equivalence import plot_equivalence_dendrogram as _plot
        return _plot(
            labels, P_mat, alpha=alpha, method=method,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_equivalence_sweep(
        self,
        *,
        figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "equivalence_sweep.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot equivalence probability sweep across alpha values."""
        sweep = self.equivalence_sweep_table()
        from .plots.equivalence import plot_equivalence_sweep as _plot
        return _plot(
            sweep, figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_equivalence_bars_plus_sweep(
        self,
        *,
        alpha: float | None = None,
        alphas: np.ndarray | None = None,
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "equivalence_bars_sweep.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot combined equivalence bars and sweep lines."""
        if alpha is None:
            alpha = self.cfg.decision.alpha_eq
        equiv = self.equivalence_probability_table(alpha=alpha)
        sweep = self.equivalence_sweep_table(alphas=alphas)
        from .plots.equivalence import (
            plot_equivalence_bars_plus_sweep as _plot,
        )
        return _plot(
            equiv, sweep, alpha=alpha,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    # Interaction (v8+, v9+) ............................................

    def plot_interaction_heatmap(
        self,
        *,
        annotate: bool = True,
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "interaction_heatmap.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot heatmap of the shared station-rupture interaction.

        Requires a v8+ interaction model in the posterior.
        """
        p = self.posterior
        if not p.has_interaction:
            raise RuntimeError(
                "Posterior has no 'gamma_sr'. Fit with "
                "RandomSlopesInteractionModel (v8+) to use this plot."
            )
        gamma = p.gamma_sr()
        assert gamma is not None
        from .plots.interaction import plot_interaction_heatmap as _plot
        return _plot(
            gamma,
            list(self.data.station_labels),
            list(self.data.run_labels),
            annotate=annotate,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_interaction_by_case(
        self,
        *,
        annotate: bool = True,
        share_scale: bool = True,
        ncols: int | None = None,
        figsize: tuple[float, float] | None = None,
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "interaction_by_case.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot per-Case panel of effective interactions.

        Works for both v8 and v9 interaction models. ``xi_case`` is
        auto-detected from the posterior when present.
        """
        p = self.posterior
        if not p.has_interaction:
            raise RuntimeError(
                "Posterior has no 'gamma_sr'. Fit with "
                "RandomSlopesInteractionModel (v8+) to use this plot."
            )
        gamma = p.gamma_sr()
        assert gamma is not None
        case_labels = p.case_labels
        if case_labels is None:
            case_labels = list(self.data.factor_levels[
                self.cfg.factors[1].name
            ])
        from .plots.interaction import plot_interaction_by_case as _plot
        return _plot(
            gamma,
            case_labels,
            list(self.data.station_labels),
            list(self.data.run_labels),
            xi_case_draws=p.xi_case(),
            annotate=annotate, share_scale=share_scale, ncols=ncols,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_interaction_forest(
        self,
        *,
        ci: tuple[float, float] | None = None,
        sort: bool = True,
        max_cells: int | None = None,
        figsize: tuple[float, float] = (HALF_WIDTH, 4.3),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "interaction_forest.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot forest plot of interaction cells with credible intervals.

        Requires a v8+ interaction model in the posterior. ``ci`` defaults
        to ``cfg.ci``.
        """
        p = self.posterior
        if not p.has_interaction:
            raise RuntimeError(
                "Posterior has no 'gamma_sr'. Fit with "
                "RandomSlopesInteractionModel (v8+) to use this plot."
            )
        gamma = p.gamma_sr()
        assert gamma is not None
        labels = p.station_run_labels
        if labels is None:
            labels = [
                f"{s}|{r}"
                for s in self.data.station_labels
                for r in self.data.run_labels
            ]
        if ci is None:
            ci = self.cfg.ci
        from .plots.interaction import plot_interaction_forest as _plot
        return _plot(
            gamma, labels, ci=ci, sort=sort, max_cells=max_cells,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    # Posterior .........................................................

    def plot_station_posteriors(
        self,
        *,
        figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "station_posteriors.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot station random-effect posterior densities."""
        p = self.posterior
        from .plots.posterior import plot_station_posteriors as _plot
        return _plot(
            p.delta_st(), p.station_labels,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_observed_vs_predicted(
        self,
        *,
        figsize: tuple[float, float] = (HALF_WIDTH, HALF_WIDTH),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "observed_vs_predicted.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot observed vs. posterior-mean predicted values."""
        p = self.posterior
        fv = self.fitted_values()
        from .plots.posterior import plot_observed_vs_predicted as _plot
        return _plot(
            self.data.y, fv["yhat_with_run"].to_numpy(),
            config_idx=self.data.config_idx,
            config_labels=p.config_labels,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_mu_density(
        self,
        *,
        configs: list[str] | None = None,
        station_idx: int | None = None,
        run_idx: int | None = None,
        original_edp_scale: bool = False,
        kind: Literal["density", "cdf"] = "density",
        normalize: bool = False,
        n_cols: int = 4,
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "mu_posterior_density.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot posterior density of configuration intercepts."""
        p = self.posterior
        from .plots.posterior import plot_mu_density as _plot
        return _plot(
            p.mu0(), p.mu_config(), p.config_labels,
            configs=configs,
            delta_st=p.delta_st() if station_idx is not None else None,
            station_idx=station_idx,
            b_run=p.b_run() if run_idx is not None else None,
            run_idx=run_idx,
            original_edp_scale=original_edp_scale,
            kind=kind, normalize=normalize, n_cols=n_cols,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_ppc(
        self,
        *,
        figsize: tuple[float, float] | None = None,
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "ppc_check.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot posterior predictive check summary."""
        ppc_df = self.posterior_predictive_check()
        from .plots.posterior import plot_ppc as _plot
        return _plot(
            ppc_df, figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_residuals(
        self,
        *,
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "residuals.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot residual diagnostics."""
        fv = self.fitted_values()
        from .plots.posterior import plot_residuals as _plot
        return _plot(
            fv, figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_raw_data(
        self,
        *,
        figsize: tuple[float, float] = (FULL_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "raw_data.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot raw EDP data by configuration and station."""
        from .plots.posterior import plot_raw_data as _plot
        return _plot(
            self.data.y,
            self.data.config_idx,
            list(self.data.config_labels),
            self.data.station_idx,
            list(self.data.station_labels),
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_ppc_density(
        self,
        *,
        n_rep_draws: int = 50,
        figsize: tuple[float, float] = (HALF_WIDTH, 3.0),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "ppc_density.pdf",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot posterior predictive density overlay."""
        p = self.posterior
        y_rep = p.y_rep()
        if y_rep is None:
            raise RuntimeError(
                "No posterior predictive samples. "
                "Refit with posterior_predictive=True."
            )
        from .plots.posterior import plot_ppc_density as _plot
        return _plot(
            self.data.y, y_rep, n_rep_draws=n_rep_draws,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_trace(
        self,
        *,
        var_names: list[str] | None = None,
        figsize: tuple[float, float] | None = None,
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "traceplot.pdf",
    ) -> plt.Figure:
        """Plot ArviZ traceplot for key parameters."""
        from .plots.posterior import plot_trace as _plot
        return _plot(
            self.idata, var_names=var_names,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_pair(
        self,
        *,
        var_names: list[str] | None = None,
        figsize: tuple[float, float] = (HALF_WIDTH, HALF_WIDTH),
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "pairplot.pdf",
    ) -> plt.Figure:
        """Plot ArviZ pair plot for key parameters."""
        from .plots.posterior import plot_pair as _plot
        return _plot(
            self.idata, var_names=var_names,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    def plot_forest_arviz(
        self,
        *,
        var_names: list[str] | None = None,
        ci: float = 0.94,
        dot_alpha: float = 0.30,
        figsize: tuple[float, float] | None = None,
        out_dir: str | Path | None = None,
        prefix: str = "",
        filename: str = "forest_arviz.pdf",
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot ArviZ forest plot with observed data overlay."""
        d = self.data
        from .plots.posterior import plot_forest_arviz as _plot
        return _plot(
            self.idata, var_names=var_names,
            observed_y=d.y,
            config_idx=d.config_idx,
            station_idx=d.station_idx,
            run_idx=d.run_idx,
            config_labels=list(d.config_labels),
            station_labels=list(d.station_labels),
            ci=ci, dot_alpha=dot_alpha,
            figsize=figsize, out_dir=out_dir, prefix=prefix, filename=filename,
        )

    # ── Model comparison ─────────────────────────────────────────────────

    def compare_variants(
        self,
        *,
        ic: str = "loo",
    ) -> tuple[pd.DataFrame, dict[str, FittedVariant]]:
        """Compare Student-t/Gaussian × hetero/homo variants.

        Returns the ArviZ comparison table and all fitted variants.
        """
        variants = {
            "StudentT_hetero": FlatConfigModel(likelihood="student_t", heteroskedastic=True),
            "Gaussian_hetero": FlatConfigModel(likelihood="gaussian", heteroskedastic=True),
            "StudentT_homo": FlatConfigModel(likelihood="student_t", heteroskedastic=False),
            "Gaussian_homo": FlatConfigModel(likelihood="gaussian", heteroskedastic=False),
        }
        return compare_models(self.data, variants, self.cfg, ic=ic)

    # ── Repr ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        name_str = f" ({self.name})" if self.name else ""
        return (
            f"BayesEpistemicModel{name_str} [{status}]\n"
            f"  configs: {self.data.n_configs}, "
            f"stations: {self.data.n_stations}, "
            f"runs: {self.data.n_runs}, "
            f"obs: {self.data.n_obs}\n"
            f"  factors: {self.cfg.factor_names}\n"
            f"  ref: {self.data.ref_label}\n"
            f"  likelihood: {self.cfg.likelihood}, "
            f"hetero: {self.cfg.heteroskedastic}"
        )
