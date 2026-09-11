# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important guidelines

- Do not `os.path` but use `from pathlib import Path` instead.

## What this is

NMMA is a Python library for nuclear-physics and multi-messenger (gravitational-wave + electromagnetic)
Bayesian inference of binary neutron star / compact binary mergers. It is built on top of `bilby`/`bilby_pipe`
for sampling and provides its own likelihoods, priors, and lightcurve/EOS/population models. Companion paper:
Pang et al. 2023, *Nature Communications* 14, 8352.

## Commands

Install (editable, with dev/test extras):
```bash
pip install -e ".[dev]"
# other extras: grb, neuralnet, production, sampler, doc — combine as needed, e.g. ".[grb,neuralnet,dev]"
```

Run the full test suite:
```bash
pytest tests/*/*.py
```

Run a single test file / test case (prefer this over the full suite while iterating):
```bash
pytest tests/em/model_test.py
pytest tests/em/model_test.py::TestLightCurveModelContainer::test_init_sets_expected_attributes
```

Coverage (as run in CI):
```bash
python -m coverage run --source nmma -m pytest tests/*/*.py
```

Lint (flake8 config lives in `.flake8`; CI only hard-fails on syntax errors/undefined names, the rest is
`--exit-zero`):
```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude docs
```

Formatting is enforced via pre-commit (black + flake8 + basic hygiene hooks):
```bash
pre-commit run --all-files
```

Some tests are gated on the `NMMA_FIESTA_SURROGATES` environment variable pointing at a local checkout of the
`nuclear-multimessenger-astronomy/fiesta-surrogates` HuggingFace repo (CI downloads it via `hf download`); those
tests `unittest.SkipTest` themselves when it isn't set or `fiesta` isn't installed, so this is safe to leave
unset locally.

## Test layout (mid-migration)

Tests are being moved from the historical `nmma/tests/` (being removed) to a top-level `tests/` tree that
mirrors the `nmma/<subpackage>/` layout, e.g. `tests/em/model_test.py` tests `nmma/em/model.py`. When adding
tests for a module, put them under `tests/<subpackage>/`, not back under `nmma/tests/`. CI invokes
`pytest tests/*/*.py`, i.e. it only discovers files one directory below `tests/` — keep new test files inside a
subpackage directory, not directly in `tests/`.

Tests are `unittest.TestCase`-based (not bare pytest functions/fixtures). Favor building objects via
`object.__new__(SomeClass)` plus hand-set attributes when a class's `__init__` does expensive/networked work
(e.g. downloading SVD models from GitLab) — see `TestSVDLightCurveModel` in `tests/em/model_test.py` for the
pattern — rather than mocking the constructor.

## Architecture

The package is organized by physical domain, not by layer; each domain subpackage typically has its own
`*_parsing.py` (CLI args), a likelihood module, and (for `em`) the model zoo itself:

- `nmma/core/` — shared infrastructure used by every other subpackage: `parsing.py` (base argparse/configargparse
  setup shared by all CLI entry points, `nmma_base_parsing`/`parsing_and_logging`), `base.py` (`bilby_sampling`,
  `multi_analysis_loop` — the actual sampler-invocation loop), `mpi_setup.py` (`pbilby_sampling`, parallel-bilby/MPI
  path), `conversion.py` (parameter conversions, e.g. redshift↔distance), `constants.py`, `gitlab.py` (on-demand
  download of pretrained SVD lightcurve models from the `nmma-models` GitLab repo), `utils.py`.
- `nmma/em/` — electromagnetic (lightcurve) side: `model.py` is the model zoo (all `LightCurveModelContainer`
  subclasses: SVD-interpolated models, fiesta neural-surrogate models, simple analytic kilonova/bolometric/GRB/
  supernova/shock-cooling/host-galaxy models, and `CombinedLightCurveModelContainer` for summing several), plus
  `em_likelihood.py`, `prior.py`, `lightcurve_generation.py`/`lightcurve_handling.py`, `systematics.py`,
  `training.py` (SVD model training/benchmarking), `io.py`, `cluster_handling.py` (SLURM helpers).
- `nmma/gw/` — gravitational-wave likelihood/parsing/inputs; thin domain layer over bilby's own GW inference.
- `nmma/eos/` — equation-of-state generation (`eos_gen.py`), TOV solving (`tov.py`), EOS likelihood/processing.
- `nmma/population/` — population-level likelihood for hierarchical/population inference.
- `nmma/joint/` — combines GW + EM (+ EOS/population) into one joint analysis: `joint_likelihood.py`
  (`MultiMessengerLikelihood`), `main.py` (the `nmma-analysis` entry point, forks stdout/stderr on non-zero MPI
  rank before importing further), `generation.py` (`nmma-generation`, builds the data dump consumed by `main.py`),
  `injection_handling.py`.
- `nmma/mlmodel/` — neural-net components (ResNet/embedding/normalizing-flow models) used by fiesta-style
  surrogate lightcurve models; includes checked-in `.pth` weight files.
- `nmma/post_processing/` — resampling (`gwem-resampling`), Hubble constant estimation, NS characteristics,
  maximum-mass constraints, plotting.

Most domain subpackages follow the same request flow: a `*_parsing.py` builds an `argparse`/`configargparse`
parser via `nmma_base_parsing` → the resulting namespace drives model/prior/likelihood construction in that
subpackage → `core.base.bilby_sampling` (or `core.mpi_setup.pbilby_sampling` under MPI) actually runs the sampler.
CLI entry points are declared in `pyproject.toml` under `[project.scripts]` — that's the authoritative map from
command name (e.g. `lightcurve-analysis`, `nmma-analysis`, `svdmodel-download`) to the Python function it calls.

Pretrained/large model assets are not vendored in the repo: SVD lightcurve models are pulled from GitLab on
first use (`core/gitlab.py`), and fiesta neural-surrogate models come from a separate `fiesta-surrogates`
HuggingFace repo. Prior definitions for each model live as flat `.prior` files under `priors/` at the repo root.

## Repository conventions

- Python 3.12–3.13 only (see `requires-python` in `pyproject.toml`).
- Contribution flow is fork → branch → PR against `main`; PRs must have passing tests (see
  `doc/contributing.md`). Style is enforced by black + flake8 via pre-commit, not by hand.
