# apeBayes: working rules for agents

apeBayes is a PyMC library for Bayesian hierarchical quantification of epistemic
uncertainty (the San Ramon SSI study, Papers 1 and 2). GitHub `nmorabowen/apeBayes`,
default branch `main`. The public API is `__all__` in `src/apeBayes/__init__.py`; the
model lineage (v1 to v9, v8.1) is the README's "Model versions". Don't restate either here.

## Task guides: read the matching one before starting

| Doing this | Read first |
|---|---|
| Adding or changing a plot (`src/apeBayes/plots/`, `BayesEpistemicModel.plot_*`) | [`.claude/skills/apebayes-plot-method/SKILL.md`](.claude/skills/apebayes-plot-method/SKILL.md) |

This repo has no lessons archive. Its commit messages are long and say why, so
they are the archive: guide items point at `git show <sha>`. Write yours the same way.

## Install and test

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash; .venv/bin on Linux/macOS
pip install -e ".[dev]"          # add ",fast" for the optional nutpie sampler
python -c "import apeBayes; print(apeBayes.__file__)"   # must point into THIS checkout's src/
```

There is no CI. Run all three gates before every commit:

```bash
ruff check src tests
mypy src
pytest                           # 192 tests, 10-20 s; none runs the sampler
```

Traps that cost time here:

1. **pytest can test someone else's apeBayes.** The src layout has no `pythonpath`
   setting, so tests import whatever `apeBayes` is installed. On the author's Windows
   machine the default `python` holds an editable 0.2.0 install from a second, stale
   clone (`~/Github/apeBayes`, frozen at 3819a35): `pytest` there fails collecting
   `tests/test_analysis/test_bias.py`, or silently tests old code. Check `__file__`
   as above, or run with `PYTHONPATH=src`.
2. **pyarrow is a runtime dependency** (bundles write `data.parquet`). Without it the
   four bundle round-trip tests in `tests/test_facade/test_bundle_roundtrip.py` fail.
3. **mypy needs `pandas-stubs`** (in the dev extra). Without it `mypy src` reports 20
   `import-untyped` errors that are not real.
4. **`ruff check --fix` deletes unused function-local imports (F401).** In f60272d it
   turned six import smoke tests in `tests/test_imports.py` into `pass`. Read every
   `--fix` diff under `tests/`; keep a deliberate import with `# noqa: F401`.
5. **Without CI, the gates regress.** "ruff clean, mypy clean" was set up in 3819a35
   and regressed twice (fixed in 81712d8 and f60272d). 2be0817 was committed with
   `import apeBayes` failing (fixed in d6a18ef).
6. **pytest never samples.** It checks model graphs at build time and the analysis and
   plot layers on stub posteriors (`tests/conftest.py`). Loader and model changes were
   checked by hand: load the six Paper 1 bundles and reproduce their β values (see
   f60272d, 60909f1). Say in the commit whether you did.
7. **Check `git status` before `git add -A`.** Editor temp files (`<file>.tmp.<pid>.<ms>`)
   were committed twice (eb57b04, 874efba). `.gitignore` now drops them.

## Layout

```
src/apeBayes/
  facade.py      BayesEpistemicModel: fit, save/load, *_table(), plot_*(), decision_report()
  multi_edp.py   MultiEDPModel / EDPSpec: one BayesEpistemicModel per EDP
  config.py      ModelConfig, PriorConfig, SamplingConfig, DecisionConfig, FactorSpec
  data.py        encode_dataset -> EpistemicDataset (long-format frame -> index arrays)
  model/         ModelBuilder subclasses (v1-v9), sample_model, compare_models
  posterior/     PosteriorAccessor: named access to posterior draws
  analysis/      pure numpy/pandas: bias, equivalence, decomposition, variance, validation
  diagnostics/   convergence; separability_check (pre-fit)
  plots/         pure plot functions; style.py (palette, apply_style, fs), helpers.py (savefig)
tests/           mirrors src/ (test_analysis/, test_facade/, ...); fixtures in conftest.py
```

`analysis/` and `plots/` take arrays and frames and return tables and figures.
`facade.py` wires posterior to analysis to plots and owns the defaults (from `cfg.decision`).

## Contracts other code depends on

- **The `.apebayes` bundle** (see the "Bundle format" block in `facade.py`): a directory
  `<name>.apebayes/` with `idata.nc`, `config.json` and `data.parquet`, or the same three
  members in `<name>.apebayes.zip`; a plain `.nc` is legacy and needs `df=` at load.
  Changing `config.json`'s shape means bumping `_BUNDLE_SCHEMA_VERSION`; an unknown version
  raises on purpose. Saved Paper 1 and Paper 2 fits are reloaded, never refitted: the
  4b65528 rewrite dropped the bundle format and left 31 saved bundles unloadable until
  60909f1.
- **PyMC variable names are persisted.** Bundles on disk, and scripts in the separate
  epistemic-uncertainty research repo, read the posterior by name (`xarray.open_dataset`
  on `idata.nc`, `group="posterior"`: `mu_config`, `sigma_run`, `sigma_inter`, `gamma_sr`,
  `lambda_case`, ...). Rename on the Python side only: `PriorConfig.sigma_src` kept the
  PyMC name `sigma_run` (356b174). `idata.nc` is NetCDF4/HDF5; a reader without netCDF4
  can open it with `engine="h5netcdf"`.
- **`load()` infers the model variant from those names** (`_detect_builder`: `xi_case`
  is v9, `gamma_sr` v8, `lambda_case` v4-v7, otherwise v1-v3). The wrong builder silently
  gives σ_src where σ_GM was meant (93756f5). A variant with new variables updates
  `_detect_builder`, `_variant_tag` and its `sigma_GM()` (eed89f0).
- **New options on an existing model default to the released behaviour, unchanged**
  (`residual_pooling="none"` in 838cf7c). Published numbers come from those defaults; a
  prior default change shifts every β (8e3d5d2, reverted in 852f19d).
- **Specs outside this repo.** Docstrings cite `uncertanty_measures.md` (§4 to §7: σ_GM,
  the α ladder, P*), `library_integration_plan.md` and `reference_paper_figure_standards.md`.
  They live in the author's paper folder, not in git. Ask before acting on a § reference
  you can't read.

## Releases, branches, commits

- The version lives in two places, `pyproject.toml` and `__version__` in
  `src/apeBayes/__init__.py`; bump both. A model change also gets a README "Model
  versions" entry. Releases are tagged `vX.Y.Z`.
- Until April 2026 work was committed on `main` directly. The v8.1 and `residual_tau`
  work (0.3.1, 0.3.2) went through `feature/<slug>` branches merged into `main` with
  merge commits. Base PRs on `main`.
- Commit subject in the imperative. The body says why and ends with what you verified
  (ruff, mypy, pytest counts, bundles reloaded).
