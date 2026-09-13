# NMMA test suite

Notes on how these tests are built, what they assume about the environment,
and what they found. Written alongside the suites for `joint`, `mlmodel`,
`population` and `post_processing`.

## Running them

```bash
python -m pytest tests/                      # everything
python -m pytest tests/joint -q              # one subpackage
python -m pytest tests/eos/tov_test.py       # one module
python -m pytest tests/eos/tov_test.py::TestClass::test_name
```

Test modules are named `<module>_test.py` and mirror the package layout, so a
change to `nmma/joint/generation.py` belongs in
`tests/joint/generation_test.py`. Plain `pytest tests/` collects everything;
there is no glob to remember.

Black and flake8 are clean across `joint`, `mlmodel`, `population` and
`post_processing`. Eight older modules under `core` and `em` are not yet
black-formatted, so `black --check tests/` fails on those alone.

Note that the top-level `CLAUDE.md` still documents an older layout
(`nmma/tests/test_<subpackage>/` collected with `nmma/tests/*/*.py`). That is
stale and worth correcting.

## Layout

| Subpackage | Modules | Tests |
| --- | --- | --- |
| `core` | 8 | 491 |
| `eos` | 5 | 300 |
| `joint` | 6 | 375 |
| `post_processing` | 7 | 278 |
| `mlmodel` | 5 | 260 |
| `em` | 1 | 93 |
| `population` | 1 | 52 |

## Conventions

Everything is `unittest`, matching what was already here before these suites
were added.

**Test names are sentences.** `test_a_violated_constraint_floors_the_likelihood`
rather than `test_constraint_2`. A failure report should read as a statement
about the code, so the name carries the claim and the assertion checks it.

**Docstrings explain why, not what.** A class docstring says what the thing
under test is for in the physics or the pipeline, because that is the part a
reader cannot recover from the code. The assertions already say what happens.

**Mixins carry shared setup.** `InjectionCreatorMixin`, `ResamplerMixin`,
`PostmergerMixin` and friends build the fixtures once and let each test class
use them. Temporary directories go through `tempfile.mkdtemp` with a
`tearDown` or `addCleanup` that removes them.

**Real objects wherever they are cheap.** These are scientific pipelines, and
a test that mocks the physics away mostly tests the mock. So:

- The injection creator is built through the actual command line parser and a
  small on-disk equation-of-state table, and the conversion chain runs for
  real end to end.
- The machine-learning tests build real networks. A small one-dimensional
  ResNet on a 121-point light curve runs in milliseconds, so shapes,
  initialisation, freezing and gradient flow are all checked against actual
  tensors.
- The population tests use the real `scipy` distributions.
- Two tests load the weight files shipped in the package, to confirm the
  released architecture still matches them.

**Stubs only where the real thing is expensive or unavailable.** Signal
injection into a 2048-second segment, light curve generation, and the
per-messenger setup helpers are stubbed. The two `pymultinest` mixins are
paired with a plain `object` base instead of the solver, which exercises the
prior and the likelihood without needing the MultiNest shared library. Stubs
are small named classes rather than bare `MagicMock` where the test asserts
on behaviour, because a `MagicMock` answers every question and so proves
little.

**Optional dependencies skip, they do not error.** `torch` is the `neuralnet`
extra, so every class in `tests/mlmodel` carries a `needs_torch` guard and the
imports sit in a `try`/`except ImportError`. With torch absent all 260 skip
cleanly; a class that subclasses `nn.Module` at module level needs a
conditional base for this to hold. The same pattern covers `ligo.lw` in
`tests/joint`.

### Two traps worth knowing

**`bilby.core.likelihood.JointLikelihood` deepcopies its sub-likelihoods.**
Assertions have to read `likelihood.likelihoods[i]`, never the instances
passed to the constructor.

**Several `pandas` columns are integer-typed.** Assigning a Python `bool` into
an `int64` column raises `LossySetitemError`, so tests that poke a
`tests_passed` column write `0` and `1`.

## What the tests found

Pinned defects each have a test that asserts the current behaviour, named so
the defect is visible, with a comment saying what the fix should change.
Whoever fixes one flips the assertion.

### Breaks a documented workflow today

- **`post_processing/resampling.py`** asks `arviz` for a credible interval
  with `hdi_prob=`, renamed to `prob=` in arviz 1.0. `pyproject.toml`
  requires `arviz>=1.2`, so this raises `TypeError` on every call, taking out
  both the `gwem-Hubble-estimate` and `combine-EOS` console scripts.
- **`post_processing/marginalisation.py`** asks `em.em_parsing` for
  `lc_marginalisation_parser`, which lives in `post_processing.parser`. The
  routine raises on its first statement, so the marginalisation workflow
  cannot run at all.
- **`post_processing/ns_characteristics.py`** reads `args.EOSPath` where the
  parser defines `--EOSpath`, giving `args.EOSpath`. `combine-EOS` dies before
  loading a single posterior. Past that, the output is written with
  `sep="\s+"`, which is a valid separator for reading but not for writing.
- **`post_processing/plotting_routines.py`**: `plot_multi_corner` reads four
  attribute names the corner plot parser never sets (`prior`, `verbose`,
  `bestfit_json`) and unpacks a whole mapping into two names
  (`plot_keys, plot_labels = mapping.items()`), which only works for a
  mapping of exactly two parameters.
- **`population/pop_likelihood.py`**: the joint parser defaults
  `--population-model` to `uniform`, which matches neither implemented model,
  so a population run started without an explicit model builds an unusable
  likelihood. An unknown name is also accepted silently and only fails later
  inside the sampler.
- **`joint/joint_likelihood.py:147`** dereferences a `True` placeholder as if
  it were an `EoSConverter` whenever there is no gravitational-wave
  messenger, so an electromagnetic-plus-equation-of-state run cannot be set
  up.
- **`joint/injection_handling.py:35`** copies `prior_dict` onto `prior_file`
  but leaves `prior_dict` set, so `bilby_pipe` receives both.
  `--prior-dict` is unusable for injections either way: a mapping string
  fails as a missing prior file, a path fails in the ini-dict reader. A real
  mapping still works.

### Silently wrong results

- **`post_processing/hubble_estimates.py`**: `generate_logprob` updates its
  accumulator in place and appends the same array object each pass, so every
  row holds the final value. The trend over accumulated events is flattened
  to its endpoint, which is the entire point of the routine. Appending a copy
  fixes it.
- **`population/pop_likelihood.py:15`** passes the intended maximum mass of
  2.0 as `scipy`'s `scale`, which is a width rather than an upper bound. The
  flat model's support becomes 1.1 to 3.1 solar masses, so implausibly heavy
  neutron stars get finite probability and the density is 0.5 instead of
  1.11. The peak model sets its truncation in standard deviations and is
  correct.
- **`post_processing/marginalisation.py`** fills the secondary spin from the
  `m2` column rather than `a2`, so every light curve is built with a spin
  equal to a neutron-star mass.
- **`mlmodel/dataprocessing.py:244`** strips merge suffixes with `rstrip`,
  which removes any trailing underscore or `y` rather than the two-character
  suffix. A band named `ztfy` would become `ztf`. The three current ZTF bands
  happen to survive.
- **`mlmodel/inference.py:8`** seeds the posterior from a set literal, so
  column order follows the per-process hash seed. Values stay correctly
  keyed, but the recorded parameter-key order, and so the corner plot axis
  order, varies run to run.
- **`post_processing/plotting_routines.py`**: `resampling_corner_plot` passes
  its output directory as `corner_plot`'s fourth positional argument, which
  is the figure to draw onto, not the save path. The figure is never written.
- **`joint/generation.py:51`** removes from the list it is iterating, so
  consecutive unneeded argument groups survive into the written config.
- **`joint/multi_parsing.py:108`** shares one destination between an optional
  positional and the `--data-dump` flag, so argparse overwrites the flag with
  `None`. The flag only takes effect when a positional is also present.
- **`mlmodel/dataprocessing.py:146`** builds the leading pad up to one step
  before the first observation, but `numpy.arange` already excludes its stop
  value, so one row is lost and the join is a double step. Only the time
  column is affected; the network is fed the magnitudes.

### Dead code paths and unreachable guards

- **`post_processing/resampling.py`** guards the compactness against
  `ZeroDivisionError`, but the radius comes from `numpy.interp` and numpy
  division by a zero float yields infinity with a warning rather than
  raising. A secondary the equation of state cannot support is scored instead
  of rejected. The check has to test the radius directly.
- **`post_processing/marginalisation.py`**: `get_all_gw_quantities` computes
  the effective spin from `a1` and `a2` several lines before the loop that
  defaults them to zero, so the spinless template format the reader falls
  back to is rejected. The ejecta nuisance parameter is also only assigned in
  two of the four mass orderings, so a binary with both components above the
  maximum mass hits an unbound local.
- **`mlmodel/dataprocessing.py:174`**: `pad_all_dfs` calls the padding
  function with one argument where two are required, so it raises for any
  input. Nothing in the package calls it, which is why it went unnoticed.
- **`post_processing/resampling.py`**: `find_spread_from_resampling` calls
  the resampling method twice per weighting and throws the first result away,
  doubling the cost of every trend.
- **`joint/joint_parsing.py`** sets a description for the injection parser,
  then composes the joint likelihood parser last, which overwrites it. The
  injection script's help shows the wrong description.

### Limitations rather than bugs

- **`mlmodel/resnet.py:403`**: dilated downsampling only works when each
  layer holds a single block, because the basic block refuses any dilation
  above one. The bottleneck network has no such limit. Both sides are pinned.
- **`population/pop_likelihood.py`** computes the pairing term as the
  logarithm of the mass ratio raised to the exponent rather than the exponent
  times the logarithm, so the power underflows before the logarithm is taken.
  Only exponents above about a thousand reach this.
- **`post_processing/hubble_estimates.py`** normalises with
  `scipy.special.logsumexp` but only imports `scipy.stats`. It works because
  `scipy.stats` pulls `scipy.special` in as a side effect.
- **`post_processing/plotting_routines.py`** builds axis titles from the
  significant figures of the credible interval, which a fixed parameter does
  not have, so a posterior holding a delta-function parameter overflows.
- Both training loops in `mlmodel` only flush their running loss every tenth
  batch, so a short epoch reports zero rather than its real loss.
- **`post_processing/hubble_estimates.py`** writes its trend file to the
  working directory, ignoring `--outdir`, unlike every other script.

## Environment notes

- **torch.** Not installed when the `mlmodel` suite was written, though
  `CLAUDE.md` and CI both install the `neuralnet` extra. Installing
  `torch 2.9.1+cpu`, `torchvision 0.24.1+cpu` and `nflows` fixed it. The
  first attempt pulled a CUDA build whose bundled `triton` segfaulted on any
  optimizer construction, which looked like a code bug for a while and was
  not. The CPU wheel does not depend on `triton`. If you want the CUDA build,
  reinstall torch and triton together as a matched pair.
- **pymultinest** needs a separately built shared library and is not
  importable here. Both post-processing samplers defer that import into their
  entry points, so the mixins are testable without it.
- **ligo.lw** is not installed, so one legacy injection-format test skips.
- **arviz** is 1.3.0, which is what surfaced the credible-interval break.
- Plotting tests force the `Agg` backend and restore global matplotlib
  settings in `tearDown`, because importing the plotting helpers turns LaTeX
  rendering on and a developer machine may have no LaTeX.

## Writing more

A few things worth repeating:

- Check the real return signature before asserting on it. Several plotting
  helpers return `(figure, limits)` rather than a figure.
- Avoid comparing two independently randomised runs. Seed the generator, or
  set a zero learning rate so the quantity under test holds still.
- Watch for global random state. `reweight_to_flat_mass_prior` resamples, so
  tests about counting bypass it; `estimate_observable_trend` shuffles with
  the global `random` module rather than its seeded generator, so its
  orderings are not reproducible from `args.seed`.
- `pymultinest` both raises and calls `sys.exit` when MPI is missing, so a
  guard around it needs `BaseException`.
- Two-sided p-values can land exactly on a threshold. Mock the percentile
  rather than relying on where a random draw falls.
