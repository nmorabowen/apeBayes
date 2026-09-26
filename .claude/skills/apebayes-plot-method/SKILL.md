---
name: apebayes-plot-method
description: >
  Read before adding or changing a plot in apeBayes: a function in src/apeBayes/plots/
  or a BayesEpistemicModel.plot_* method in src/apeBayes/facade.py. A checklist of the
  plot-layer conventions (stamp before save, fs() font sizes, explicit keyword arguments,
  labels that match the data, smoke tests), each pointing at the commit that explains it.
  It covers changing the library, not using its plots in an analysis notebook.
---

# apeBayes: adding or changing a plot

Read this before adding or changing a plot function (`src/apeBayes/plots/*.py`) or its
`BayesEpistemicModel.plot_*` wrapper in `src/apeBayes/facade.py`. Plot work is the
repo's most common change: 20 of its 48 non-merge commits touch `plots/`.

Each item names the commit whose message explains it; run `git show <sha>`. This repo's
commit messages are its lessons archive, so the reasons live there, not here.

## The plot function (`plots/<topic>.py`)

- [ ] Keep it pure. It takes arrays or frames already computed by `analysis/` (never a
      model or a posterior) and returns a `Figure` or `(Figure, Axes | ndarray)`.
- [ ] Take `figsize`, `out_dir=None`, `prefix=""` and `filename="<name>.pdf"`, and finish
      with `savefig(fig, out_dir, filename, prefix=prefix)` from `plots.helpers`. It does
      nothing when `out_dir` is None.
- [ ] Set every font size with `fs(delta)` from `plots.style`, never with a number. That
      covers `fontsize`, `title_fontsize`, `labelsize`, `leaf_font_size` and
      `annot_kws={"size": ...}`. A number ignores `apply_style(context=...)`:
      `git show 62190cf` (the delta table is in `fs`'s docstring). Seven number-valued
      sizes survive from before that commit; don't copy them.
- [ ] Take colours and widths from `plots.style` (`NAVY`, `CASE_COLORS`, `STATION_COLORS`,
      `VARIANCE_COLORS`, `CMAP_DIV`, `FULL_WIDTH`, `HALF_WIDTH`, ...) and
      `helpers.case_color`, not raw hex or the default cycle: `git show 9751f7f`.
- [ ] If rows or spokes are configurations, add an `order: Literal["tier", "case",
      "input"] = "tier"` knob and sort with `plots.helpers.order_config_labels(..., by=order)`:
      `git show 3d7e09b`. `utils.order_config_labels` is a different function (`major=`,
      `sep=`) used by `data.py`.
- [ ] Make the label match the data. The TeX for a β denominator comes from
      `_denom_math(denom_name)` in `plots/bias.py`, and the draws are picked by the same
      `denom_name`. A σ_src label once sat over σ_GM draws: `git show c5d30f0`. The
      canonical denominator is σ_GM; `\sigma_{\mathrm{run}}` labels survived the rename
      twice (`git show 18dae46`, `git show 3d7e09b`).
- [ ] A visual summary shows the same posterior as the table beside it. Violins divide by
      per-draw σ like the CI does. 18dae46 reversed 8a04afa's median normalisation for
      exactly this reason; don't reintroduce it.

## The facade method (`BayesEpistemicModel.plot_<name>`)

- [ ] Use explicit keyword arguments, with no `**kw` pass-through. A split `**kw` once
      misrouted typos silently, and the plot's default `prob_col` didn't match the
      table's column: `git show 6786940`.
- [ ] Type string options as `Literal[...]` and document every choice in the numpy-style
      Parameters block: `git show c5d30f0`.
- [ ] Take decision defaults (α_eq, the α ladder, P*) from `self.cfg.decision`, not
      literals: `git show 4b65528`.
- [ ] Call the plot function with `out_dir=None`, then
      `return self._stamp_and_save(result, out_dir=out_dir, prefix=prefix, filename=filename)`.
      The model-name stamp must land before the PDF is written: `git show 2be0817`.
      `plot_mu_triptych` is the one documented exception (see `_stamp`).
- [ ] Pass column names from the table the facade just built, not from the plot
      function's defaults (`git show 6786940`).

## Tests, then finish

- [ ] Add a smoke test in `tests/test_facade/`, modelled on
      `test_rho_and_validation_plots.py`: `matplotlib.use("Agg")`, a model built from
      `synthetic_long_df` + `default_config` with `stub_posterior_v8` (no sampling),
      assertions on ticks, labels or artists, then `plt.close(fig)`.
- [ ] If the on-disk file matters, assert on the written file, like
      `test_stamp_survives_on_disk_save` in `tests/test_facade/test_name_stamp.py`.
- [ ] Run `ruff check src tests`, `mypy src` and `pytest` against this checkout's `src/`
      (AGENTS.md, "Install and test": pytest can pick up another installed apeBayes).
