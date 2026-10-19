# Changelog

## 0.4.0 — 2026-10-19 (branch `feature/review-2026-10-19`, not released)

Fixes and additions from three independent reviews of 0.3.2 (code, statistics,
model critique). **The v8 / v8.1 builders compute the same posterior as before
by default.** A pinned test checks that their PyMC graphs (free variables, order,
log-density) equal those of 0.3.2. Changes to *reported* numbers come only from
post-processing bug fixes, listed under "Changed results".

### Changed results (post-processing bug fixes)

- **R², fitted values, residual and observed-vs-predicted plots** (v4–v9). The
  linear predictor omitted `lambda_case`, `gamma_sr` and `xi_case`; it now
  includes every term of the fitted variant (`PosteriorAccessor.linear_predictor`).
  `R2_cond` and the residuals of every v4–v9 fit change.
- **Variance budget.**
  - Before: the run row was σ²_run regardless of λ, the interaction row was the
    finite-population variance of the realised γ, and the shares were
    median / sum-of-medians.
  - Now: every component is on one per-observation footing and computed per
    draw. The fixed factors use the variance of their realised levels. The
    random factors use their loaded scale: E[λ²]σ²_run, E[ξ²]σ²_inter, and the
    v10 σ²_u, σ²_v. The residual includes the Student-t factor.
  - Shares are per-draw fractions: `pct_med` / `pct_lo` / `pct_hi`, plus
    `pct_mean`, which sums to 100 and drives the bar plots.
  - `kind="finite"` gives the realised-effect version.
  - The run row is renamed `Run dispersion λ²σ²_run` and the interaction row
    `Interaction   ξ²σ²_inter`.
- **`variance_component_table(configs=[...])`** labelled rows by position. It
  now picks each config's own residual scale.
- **`plot_variance_ratio`.** The homoskedastic branch now divides by σ_src, and
  re-sorting the labels now permutes the columns with them.
- **`plot_mu_density(run_idx=...)`** adds the full record term (λ_c·b_r, plus
  γ, u and v when present) instead of a bare b_r.
- **`epistemic_clusters` default cut** is now `1 − p_link` (`p_link=0.5`). It
  was `1 − alpha`, which reused the equivalence radius as a probability.

### Sampling

- `max_treedepth` is forwarded to nutpie as `maxdepth`. Before, nutpie
  silently ran at depth 10. Set `max_treedepth=10` to reproduce a pre-0.4
  nutpie fit draw for draw.
- A nutpie runtime error is now raised. It no longer silently refits with
  PyMC. A missing nutpie still falls back to PyMC, and that is recorded in
  `idata.attrs["sampler_fallback"]`.
- The pointwise log-likelihood is stored (computed after nutpie, which does
  not produce it). `SamplingConfig.store_log_likelihood=False` turns it off.
- The sampler settings actually used are written to `idata.attrs`.

### Convergence gate

- `diagnostics.convergence_report` / `attach_convergence` record the max R-hat,
  min bulk/tail ESS, divergences and a verdict in `idata.attrs`. The verdict
  is converged when R-hat ≤ 1.01, ESS ≥ 400 and there are no divergences.
- `fit(strict=False)` warns with `ConvergenceWarning`; `strict=True` raises.
- `BayesEpistemicModel.convergence` / `.converged` give the verdict. For
  posteriors saved before 0.4 it is computed on first access.
- Every facade table carries `attrs["converged"]`. `decision_report` and the
  `MultiEDPModel.compare_*` tables carry a `converged` column.

### Model comparison

- `model.compare_fitted` / `ensure_log_likelihood` compare already-fitted
  models. For old posteriors the log-likelihood is computed from the builder.
- `BayesEpistemicModel.compare_variants(variants=...)` accepts any builders.
- `model.leave_one_group_out` / `facade.leave_one_group_out` run
  leave-one-rupture-out (or station-out) CV. Each fold is a Gaussian refit,
  and the held-out block is scored by its exact joint marginal. This is the
  critique's LORO. `model.compare_logo` ranks models with paired-fold SEs.

### Sum-to-zero option

- `RandomSlopesModel` / `RandomSlopesInteractionModel` take
  `zero_sum="last" | "zsn"`. The default `"last"` is the published,
  order-dependent scheme. `"zsn"` uses `pm.ZeroSumNormal` for `mu_config`,
  `delta_st` and `b_run`, and a 2-axis ZeroSumNormal for `gamma_sr`.

### New model: v10 `CaseRecordModel` ("vc4")

- v8 plus Case×rupture u_cr and Case×station×rupture v_csr terms, with per-Case
  SDs. It also has Case-level residual SDs, unpooled ZeroSumNormal config
  effects, and ZeroSumNormal throughout.
- σ_GM = √(σ_run² + σ_inter² + σ²_u,ref + σ²_v,ref).

### Equivalence additions

- `decision_report(record_marginal=True)`: P_eq for a new set of records. It
  adds the record-sampling term of the record effects that config k does not
  share with the reference.
- `decision_report(rope_abs=np.log(1.2))`: P_eq for a fixed absolute ROPE.
- `record_discrepancy_table()`: the per-record SD of y_k − y_ref.
- `sigma_rtr_ref_draws()` and `denominator="rtr"`: the reference
  configuration's own record-to-record SD, estimated from its data alone
  (`analysis.reference`).

### `separability_check`

- It now reports the run-pattern SDs and their ratio to the reference Case.
  This is the λ-relevant part, which correlation cannot see.
- It now reports Fisher-z intervals for every correlation.
- The documentation now says it cannot validate the rank-1 record structure.
  Use leave-one-rupture-out CV for that.

### Smaller fixes

- The builder class and its options (residual pooling and prior, `residual_tau`,
  `sigma_inter`, `sigma_lambda`, `zero_sum`, …) are saved in `idata.attrs` and
  `config.json` and restored by `load()`. A v8.1 now loads as v8.1.
- The v8.1 prior on `log_sigma_eps_bar` (formerly hard-coded N(ln 0.1, 1.5)) is
  now the constructor options `residual_log_mean` / `residual_log_sd`.
- `load()` checks the posterior's Config/Station/Run coordinates against the
  re-encoded data.
- `fit()` / `MultiEDPModel.fit_all()` without `builder=` warn
  (`FutureWarning`) that they fall back to the v1 `FlatConfigModel`.
- `encode_dataset` warns about grid configs with no observations and raises
  for an empty reference.
- The interaction-model docstring now says the published γ has a *single*
  sum-to-zero constraint, not a double one.

### Tests

- `tests/test_review_2026_10_19.py`: regression tests for every fix, run on
  point-mass posteriors (no MCMC).
- `tests/test_sampling.py` (marker `sampling`, about 2 min): v8 and v10 fits
  on crossed synthetic data. It checks recovery, R², the budget, labels, the
  gate, the log-likelihood, LORO and save/load.
- `tests/test_canonical_slow.py` (marker `slow`, opt-in with `-m slow`): v8
  reproduces the canonical Paper 1 σ_GM for PFA mid.
