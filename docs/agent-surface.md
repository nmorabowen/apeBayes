# Agent surface for apeBayes: AGENTS.md and one task guide

Revision 1. Not yet adversarially reviewed.

Status: built on branch `claude/agent-surface`, cut from `main` @ `53abef8` (v0.3.2); draft PR #1.
Method: the agent-surface playbook, first piloted in the Ladruno OpenSees fork (WP-115). The
method carries over between repos; the rules don't. Everything below comes from this repo's own history.

## Problem

The repo had no `CLAUDE.md`/`AGENTS.md`, no docs folder and no CI. An agent arriving here
cannot tell:

- which apeBayes its `pytest` imports. The src layout has no `pythonpath`, and on the author's
  machine the default Python holds an editable 0.2.0 install from a second, stale clone.
- that the three gates (ruff, mypy, pytest) are manual. They have regressed twice.
- which names and formats other code depends on: the `.apebayes` bundle, the persisted PyMC
  variable names, and the variant detection in `load()`.

## Phase 0 baseline (2026-09-25)

**Archive:** none in the repo: no gotchas doc, ledger, CHANGELOG, ADRs or docs. The 48
non-merge commit messages are long and explain why; they are the de facto archive. User
memory: one entry mentions apeBayes (project `gitAPE-Epistemic-Uncertanty`,
`canva-connector-workflow.md`: reading a bundle's `idata.nc` with `engine="h5netcdf"`).

**Repeats in the commit history** (a mistake fixed, then made again):

| Lesson | First fixed | Repeated in | Recorded before the repeat? |
|---|---|---|---|
| ruff and mypy clean | 3819a35 (2026-04-16) sets both up | regressions fixed in 81712d8 (04-17) and f60272d (09-10) | pyproject config and the 3819a35 message |
| Editor temp files `*.py.tmp.<pid>.<ms>` | eb57b04 (04-16) removes one | 874efba (04-17) removes another | no, subject line only |
| Font sizes as numbers | 13336bc (04-16) removes them from the forest plot | 40d105f (04-16) and f50b6b6 (04-19) add new ones; 62190cf (04-20) migrates 38 to `fs()` | 13336bc message (forest plot only) |
| Stale `σ_run` labels after the σ_src/σ_GM rename | f3e11e0 (04-17) | 18dae46 (04-18), 3d7e09b (04-21) | f3e11e0 message |

That is 4 lessons and 7 repeat events. None of them is a repeat of a lesson kept in an archive,
because there is no archive.

**Gate: fail (no archive).** Per the playbook, only Phase 1 applies here. Phase 2 is added
because the history shows one clear recurring kind of work: 20 of the 48 non-merge commits touch
`src/apeBayes/plots/`.

**Phase 6 baseline:** 7 repeat events in 48 commits (2026-04-15 to 2026-09-12). After about 10
work packages, count the fix commits and review findings that match an item already in
`AGENTS.md` or the guide. If that number does not drop, stop adding guides.

## Shape

1. **`AGENTS.md` (new).** It covers the install, the three gates and seven traps, the layout,
   the cross-repo contracts, and the release/branch/commit habits read from the history.
   `CLAUDE.md` is the one line `@AGENTS.md`.
   *Accept:* every command it gives was run; every sha and repo path it cites resolves
   (checked by a script, see Results).
2. **One guide: `.claude/skills/apebayes-plot-method/SKILL.md`**, at most 100 lines. Each
   item points at the commit whose message explains it, and none copies the explanation.
   *Accept:* 73 lines; all 9 cited shas resolve; each item was checked against the current code.
3. **`.gitignore`.** Add `.claude/*` + `!.claude/skills/`: agent worktrees live in
   `.claude/worktrees/`, and the main checkout showed `?? .claude/` while this one existed.
   Add `*.tmp.[0-9]*` for the editor leftovers committed twice.
   *Accept:* `git check-ignore` matches both historical temp-file names; the guide is not
   ignored; `git ls-files -i -c --exclude-standard` is empty.
4. **Moved from memory:** the bundle-reading note (the `idata.nc` format and
   `engine="h5netcdf"`) now sits in AGENTS.md under "Contracts". The memory file is unchanged.

## Rejected approaches

- **A quirk lint (Phase 3).** The gate failed, so no lint ships. For the record, the best
  candidate is number-valued font sizes in `plots/` and `facade.py`. It is mechanical (an AST
  keyword scan) and has an incident: an AST scan finds 45 number-valued font sizes at 62190cf~1.
  But the fix commit would not pass. The same scan finds 7 at 62190cf: six the migration didn't
  reach (`leaf_font_size`, `labelsize`, an `annot_kws` size, the stamp's `fontsize=8`), plus
  `title_fontsize=7`, which the fix commit added itself. The rule would be red on day one until
  those are fixed.
- **A lint for stale `σ_run` labels.** `sigma_run` is also the persisted PyMC name and an
  `analysis/` parameter name, so a pattern would be mostly noise.
- **A lint for stamp-before-save.** An AST survey of all 35 `plot_*` wrappers found every one
  calling the inner plot with `out_dir=None`, except the documented `plot_mu_triptych`. There
  is no live instance and no repeat.
- **A second guide for model options** (838cf7c, ce2b54a, eed89f0, 93756f5). There were 4
  commits, too few to call a recurring kind of work. The contracts such a guide would carry
  (persisted names, `_detect_builder`, defaults unchanged, the version in two places) are in AGENTS.md.
- **Adding CI.** It would stop the ruff/mypy regressions, but it is a new gate, not part of the
  agent surface. It is left as an open question for the owner.
- **Fixing what the survey found.** Changes to production code and tests are out of scope for
  this PR; see the next section.

## Findings for the owner (not fixed here)

- **`tests/test_imports.py` is hollow.** `ruff --fix` (F401) in f60272d replaced the imports in
  7 smoke tests with `pass`. The pre-f60272d file (`git show 23fdabb:tests/test_imports.py`)
  still passes against HEAD (9 passed), so it can be restored with `# noqa: F401`.
- **Seven font sizes are numbers, not `fs()`,** so they don't follow `apply_style(context=...)`:
  `facade.py:1449` (`fontsize=8`, the name stamp), `plots/bias.py:76` (`annot_kws` size 7),
  `plots/equivalence.py:77,164` (`leaf_font_size=7`), `plots/interaction.py:108,235`
  (`labelsize=7`), `plots/posterior.py:383` (`title_fontsize=7`).
- **The comment block above `_stamp` in `facade.py` is stale.** It still describes a
  bottom-right `"name: ..."` watermark; since 2be0817 the stamp is the bare name, top-centre.
- **This is the machine environment, not the repo.** The default Python's editable apeBayes
  0.2.0 points at `~/Github/apeBayes`, a clone frozen at 3819a35. Both `pyarrow` and
  `pandas-stubs` are missing.

## Results (2026-09-25, in the worktree, default Python 3.11.9)

| Check | Result |
|---|---|
| `python -m pytest` (no `PYTHONPATH`) | imports `~/Github/apeBayes/src` (0.2.0); collection error in `test_bias.py` |
| `PYTHONPATH=src python -m pytest` | 188 passed, 4 failed (bundle round-trip: `pyarrow` missing); 11.7 s |
| `ruff check src tests` | All checks passed |
| `PYTHONPATH=src python -m mypy src` | 20 errors, all `import-untyped` for pandas (`pandas-stubs` missing) |
| `git archive 2be0817 src` then `import apeBayes` | `ImportError: cannot import name 'validation'` (confirms the AGENTS.md trap) |
| Cited shas and paths (scratchpad script) | AGENTS.md: 15 shas and 7 paths resolve; guide: 9 shas resolve and 3 of its 4 paths exist (the fourth is the glob `plots/*.py`) |
| Reading a Paper 1 bundle's `idata.nc` with xarray 2026.4.0 | the default engine works (h5netcdf 1.8.1 is picked, netCDF4 is absent); `engine="h5netcdf"` also works |

Not verified: `pip install -e ".[dev]"` in a fresh venv. Nothing was installed on this machine.

## Open questions

- Add a CI workflow running ruff, mypy and pytest on push? It would have caught 81712d8,
  f60272d and 2be0817.
- Fix the seven number-valued font sizes, and then ship the font-size rule as a lint?
- Restore `tests/test_imports.py`?
