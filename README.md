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

### `separability_check`: a pre-fit diagnostic

`RandomSlopesModel` and `RandomSlopesInteractionModel` (v4-v9) all assume
that a rupture's effect is *shared* across Case levels — each Case merely
rescales the same underlying run pattern through `lambda_case`. That
assumption is only checkable after looking at the data: `separability_check`
computes it directly from the raw long-format frame, before any sampling.
It centers the EDP within each (configuration, station) group, averages the
residual by (Case level, run), and returns the Case-by-Case Pearson
correlation matrix of those run patterns, together with
`result.attrs["min_pairwise_corr"]` and `result.attrs["all_near_one"]`. A low
minimum pairwise correlation (rule of thumb: below about 0.4) means the
common-rupture-effect assumption is violated and `lambda_case` will not be
identified; correlations all near one mean the Case levels differ by little
more than a constant offset, leaving nothing case-specific for `lambda_case`
to fit.
