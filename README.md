# apeBayes

Bayesian hierarchical quantification of epistemic uncertainty in computational mechanics.

## Model versions

The model lineage grows in the direction of "how much of the station × rupture
structure is modeled explicitly, and how is the residual scale handled."

- **v1-v3** — `FlatConfigModel` / `HierarchicalConfigModel`. Fixed or
  hierarchically-shrunk per-configuration effects on top of a station effect;
  no explicit rupture (run) structure.
- **v4-v7** — `RandomSlopesModel`. Adds a shared random rupture effect
  `b_run`, rescaled per Case level by `lambda_case` (the "random slopes").
  Station × rupture interaction is not modeled explicitly — any such
  structure is absorbed into the (optionally heteroskedastic) residual
  scale.
- **v8** — `RandomSlopesInteractionModel`. Paper 1's model. Adds an explicit
  station × rupture interaction term `gamma_sr` with its own shared scale
  `sigma_inter`, so that interaction variance no longer has to be absorbed
  into the residual.
- **v8.1** — `RandomSlopesInteractionModel(residual_pooling="partial")`.
  The same model as v8, with a partially pooled prior on the per-configuration
  residual scales (`sigma_eps_config`) in place of v8's independent
  `HalfNormal` per configuration. It was introduced because the free,
  independent scales can collapse toward zero when the shared random
  effects (`gamma_sr`, `b_run`) absorb one configuration's noise, which
  starves that configuration's residual of the signal needed to estimate its
  own scale. The partially pooled prior lets configurations borrow strength
  from each other instead. The pooling strength is set by the prior on
  `tau_sigma_eps` (HalfNormal(0.5)) and is provisional: on the San Ramon
  data it left the drift EDPs unchanged but shrank the residual scales of
  the floor-acceleration EDPs toward a common value, moving `sigma_GM` by
  about a quarter where heteroskedasticity is real. Check `sigma_GM`
  against the free-scale fit before adopting it for an EDP. The spread
  prior itself is configurable via `residual_tau` / `residual_tau_dist`
  on the constructor (default HalfNormal(0.5), which is v8.1 as released).
- **v9** — `RandomSlopesInteractionModel(interaction_loading=True)`.
  Experimental. Adds a per-Case loading `xi_case` on the station × rupture
  interaction `gamma_sr`, analogous to how `lambda_case` loads the rupture
  effect `b_run`.
- **v10 ("vc4")** — `CaseRecordModel` (0.4.0). Proposed by the 2026-10-19
  model critique. v8's record block is rank 1: one rupture pattern scaled by
  `lambda_case`, plus one `gamma_sr` shared by every configuration.
  Leave-one-rupture-out CV on the Paper 1 data rejected it by 56–75 nats.
  v10 is v8 plus two nested terms:
  - a Case × rupture effect `u_case_run`, and
  - a Case × station × rupture effect `v_case_sr`,
  each with a per-Case SD (`sigma_u_case`, `sigma_v_case`). It also uses
  Case-level residual SDs (`sigma_eps_case`), unpooled config effects, and
  `ZeroSumNormal` for every sum-to-zero. v8 is v10 with σ_u = σ_v = 0 and
  per-config residuals. Here
  σ_GM = √(σ_run² + σ_inter² + σ²_u,ref + σ²_v,ref).
  Whether the papers adopt it is the author's decision; the v8 / v8.1
  builders and their published numbers are unchanged.
- **v11 ("ctS")** — `CaseRecordModel(case_station=True, tier_station=True)`
  (0.5.0). Proposed by the engstruct-2026 diagnosis (2026-10-25) and adopted
  for both San Ramón papers (2026-10-26 / 2026-10-28). Each Case follows the
  spectrum at its own period, so stations that differ in spectral shape give
  every Case its own station pattern. v10 has no term for it, and on that
  data its chains go bimodal (Student-t residual with ν ≈ 2). v11 is v10
  plus:
  - a Case × station effect `case_station` (SD `sigma_case_station`), and
  - a Tier × station effect `tier_station` (SD `sigma_tier_station`),
  both zero-sum over stations. The offsets Δμ_k and σ_GM do not contain
  them; with a balanced design μ_k is still the equal-weight station
  average. Either term can be used alone ("v10+cS", "v10+tS"); with both off
  the graph is v10's. `sigma_GM(post, case=...)` gives any Case's
  record-to-record SD from the same fit (λ_c² σ²_run + σ²_inter + σ²_u,c +
  σ²_v,c), e.g. to read the offsets against another reference.

All random-slopes builders accept `zero_sum="last"` (default, the published
scheme) or `zero_sum="zsn"`. With `"last"`, the last level is minus the sum
of the others, so its prior variance is (n−1)× larger and results depend on
level order. `"zsn"` uses `pm.ZeroSumNormal` and puts a two-axis zero sum on
`gamma_sr`.

### Checks to run on any fit (0.4.0)

- **Convergence.** `model.convergence` / `model.converged`. `fit(strict=True)`
  raises unless max R-hat ≤ 1.01, min bulk/tail ESS ≥ 400 and there are no
  divergences. Every exported table carries the flag.
- **Record structure.** `model.leave_one_group_out(group="run")` for each
  candidate builder, then `apeBayes.model.compare_logo`. PSIS-LOO
  (`compare_fitted`) cannot test the record structure, because it leaves out
  single observations that share their rupture with the rest.
- **What the decision generalises to.** `decision_report(record_marginal=True,
  rope_abs=np.log(1.2))` adds two things:
  - the decision for a new set of records, not just these 18; and
  - a ROPE that is not estimated from the same six ruptures.
- **The denominator.** `denominator="rtr"` / `sigma_rtr_ref_draws()` use the
  reference configuration's own record-to-record SD. σ_GM is the record
  variance shared across configurations; it is not 4D's own.
  `record_discrepancy_table()` gives the per-record SD of y_k − y_ref, since
  median equivalence is not record-level interchangeability.

### `separability_check`: a pre-fit screen, not a test of the model

`RandomSlopesModel` and `RandomSlopesInteractionModel` (v4-v9) assume that a
rupture's effect is *shared* across Case levels: each Case rescales the same
run pattern through `lambda_case`. `separability_check` looks at this in the
raw long-format frame before any sampling. It centres the EDP within each
(configuration, station) group, averages the residual by (Case level, run),
and returns the Case-by-Case Pearson correlation matrix of those run
patterns. Its attributes are:

- `min_pairwise_corr`
- `corr_ci`: Fisher-z intervals
- `run_pattern_sd`
- `sd_ratio_to_ref`: each Case's pattern SD over the reference Case's, a
  raw analogue of λ

What it cannot do:

- **It cannot detect λ.** Correlation ignores scale; read `sd_ratio_to_ref`
  for that.
- **It has little power.** With 6 ruptures the 95 % interval for r = 0.47
  is (−0.55, 0.93).
- **Passing it does not validate the rank-1 structure.** On the San Ramon
  PFA-roof data the minimum correlation was 0.47, passing the 0.4 rule of
  thumb, yet leave-one-rupture-out CV rejected v8 by 62 nats.

A low minimum correlation (below about 0.4) is a warning that the shared
pattern fails. A high one proves nothing; test the structure with
`leave_one_group_out`.

See `CHANGELOG.md` for the 0.4.0 fixes, including which reported numbers they
change.
