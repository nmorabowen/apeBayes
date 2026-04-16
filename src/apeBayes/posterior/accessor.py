"""
Consistent posterior draw access.

PosteriorAccessor wraps an ArviZ InferenceData object and provides
a single, cached interface for extracting flattened posterior draws.
No more _post() vs _stack_samples_2d() confusion.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from arviz import InferenceData

from ..data import EpistemicDataset
from ..utils import flatten_posterior, student_t_var_factor, student_t_sd_factor


class PosteriorAccessor:
    """Uniform access to posterior draws from a fitted model.

    Parameters
    ----------
    idata : InferenceData
        ArviZ inference data from MCMC sampling.
    data : EpistemicDataset
        The encoded dataset used for fitting.
    """

    def __init__(self, idata: InferenceData, data: EpistemicDataset) -> None:
        self._idata = idata
        self._data = data
        self._cache: dict[str, np.ndarray] = {}

    @property
    def idata(self) -> InferenceData:
        return self._idata

    @property
    def data(self) -> EpistemicDataset:
        return self._data

    @property
    def n_samples(self) -> int:
        """Total number of posterior samples (chains × draws)."""
        mu0 = self.draws("mu0")
        return mu0.shape[0]

    # ── Core draw access ─────────────────────────────────────────────────

    def draws(self, var: str) -> np.ndarray:
        """Return flattened posterior draws for a variable.

        Parameters
        ----------
        var : str
            Variable name in ``idata.posterior``.

        Returns
        -------
        np.ndarray
            Shape (S,) for scalars, (S, K) for vectors, (S, K1, K2) etc.
            where S = n_chains × n_draws.

        Raises
        ------
        KeyError
            If the variable is not in the posterior.
        """
        if var not in self._cache:
            if var not in self._idata.posterior:
                raise KeyError(
                    f"Variable {var!r} not found in posterior. "
                    f"Available: {list(self._idata.posterior.data_vars)}"
                )
            self._cache[var] = flatten_posterior(self._idata.posterior[var])
        return self._cache[var]

    def has_var(self, var: str) -> bool:
        """Check whether a variable exists in the posterior."""
        return var in self._idata.posterior

    # ── Derived convenience draws ────────────────────────────────────────

    def mu0(self) -> np.ndarray:
        """(S,) grand intercept."""
        return self.draws("mu0").ravel()

    def mu_config(self) -> np.ndarray:
        """(S, n_configs) configuration effects."""
        return self.draws("mu_config")

    def mu_full(self) -> np.ndarray:
        """(S, n_configs) = mu0 + mu_config."""
        return self.mu0()[:, None] + self.mu_config()

    def delta_st(self) -> np.ndarray:
        """(S, n_stations) station effects."""
        return self.draws("delta_st")

    def sigma_run(self) -> np.ndarray:
        """(S,) run-level SD."""
        return self.draws("sigma_run").ravel()

    def b_run(self) -> np.ndarray:
        """(S, n_runs) run random effects."""
        return self.draws("b_run")

    # ── Random-slopes / interaction extensions (v4+, v8+, v9+) ───────────

    def lambda_case(self) -> np.ndarray | None:
        """(S, n_cases) per-Case loading on run effect, or None.

        Present in RandomSlopesModel (v4+) and descendants.
        """
        if not self.has_var("lambda_case"):
            return None
        return self.draws("lambda_case")

    def xi_case(self) -> np.ndarray | None:
        """(S, n_cases) per-Case loading on the station×rupture interaction, or None.

        Present only in v9 RandomSlopesInteractionModel with
        ``interaction_loading=True``.
        """
        if not self.has_var("xi_case"):
            return None
        return self.draws("xi_case")

    def gamma_sr(self) -> np.ndarray | None:
        """(S, n_inter) station×rupture interaction draws, or None.

        ``n_inter = n_stations * n_runs``.  Present in
        RandomSlopesInteractionModel (v8+).
        """
        if not self.has_var("gamma_sr"):
            return None
        return self.draws("gamma_sr")

    def sigma_inter(self) -> np.ndarray | None:
        """(S,) interaction scale, or None."""
        if not self.has_var("sigma_inter"):
            return None
        return self.draws("sigma_inter").ravel()

    @property
    def has_interaction(self) -> bool:
        return self.has_var("gamma_sr")

    @property
    def has_interaction_loading(self) -> bool:
        return self.has_var("xi_case")

    @property
    def case_labels(self) -> list[str] | None:
        """Case coordinate labels from posterior, or None if not present."""
        if "Case" not in self._idata.posterior.coords:
            return None
        return [str(c) for c in self._idata.posterior.coords["Case"].values]

    @property
    def station_run_labels(self) -> list[str] | None:
        """StationRun coordinate labels from posterior, or None if not present."""
        if "StationRun" not in self._idata.posterior.coords:
            return None
        return [str(c) for c in self._idata.posterior.coords["StationRun"].values]

    def nu(self) -> np.ndarray | None:
        """(S,) Student-t degrees of freedom, or None if Gaussian."""
        if not self.has_var("nu_minus2"):
            return None
        return self.draws("nu_minus2").ravel() + 2.0

    def sigma_eps(self) -> np.ndarray:
        """(S,) or (S, n_configs) residual SD.

        Returns sigma_eps_config if heteroskedastic, else sigma_eps.
        """
        if self.has_var("sigma_eps_config"):
            return self.draws("sigma_eps_config")
        return self.draws("sigma_eps").ravel()

    def sigma_eps_effective(self) -> np.ndarray:
        """Residual SD adjusted for Student-t heavy tails.

        For Student-t: sigma_eff = sigma_eps * sqrt(nu / (nu - 2)).
        For Gaussian: sigma_eff = sigma_eps.
        """
        sig = self.sigma_eps()
        nu = self.nu()
        if nu is None:
            return sig
        factor = student_t_sd_factor(nu)
        if sig.ndim == 1:
            return sig * factor
        return sig * factor[:, None]

    @property
    def is_heteroskedastic(self) -> bool:
        return self.has_var("sigma_eps_config")

    @property
    def is_student_t(self) -> bool:
        return self.has_var("nu_minus2")

    # ── Posterior predictive ─────────────────────────────────────────────

    def has_posterior_predictive(self) -> bool:
        return hasattr(self._idata, "posterior_predictive")

    def has_prior_predictive(self) -> bool:
        return hasattr(self._idata, "prior_predictive")

    def y_rep(self) -> np.ndarray | None:
        """(S, N) posterior predictive draws, or None."""
        if not self.has_posterior_predictive():
            return None
        pp = self._idata.posterior_predictive
        if "y_obs" in pp:
            return flatten_posterior(pp["y_obs"])
        return None

    # ── Labels & metadata ────────────────────────────────────────────────

    @property
    def config_labels(self) -> list[str]:
        return self._data.config_labels.tolist()

    @property
    def station_labels(self) -> list[str]:
        return self._data.station_labels.tolist()

    @property
    def run_labels(self) -> list[str]:
        return self._data.run_labels.tolist()

    @property
    def ref_idx(self) -> int:
        return self._data.ref_idx

    @property
    def ref_label(self) -> str:
        return self._data.ref_label
