import inspect
import logging
import os
import pickle
import signal
import sys
import traceback
from datetime import timedelta
from functools import wraps
from io import BufferedWriter
from pathlib import Path
from time import sleep, time
from typing import Any

import dynesty
import numpy as np
from bilby.core.sampler import base_sampler as bs
from bilby.core.sampler.dynesty import (
    Dynesty as BilbyDynesty,
    dynesty_stats_plot as stats_plot,
)
from bilby.core.utils import random as bilby_random
from dynesty.plotting import runplot, traceplot
from matplotlib import pyplot as plt
from numpy.random import PCG64, Generator, SeedSequence
from pandas import DataFrame
from schwimmbad import MPIPool, MultiPool, SerialPool

from .conversion import label_mapping
from .parsing import NLIVE_KWARGS, PARALLEL_SAMPLERS
from .utils import logger, read_bestfit_from_posterior, rejection_sample

# signals on which the master checkpoints and exits, and which pool processes ignore
TERMINATION_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)


def process_mpi_kwargs(args) -> tuple[dict, dict]:
    """Collect the settings of the driver of a parallel run for
    ``args.sampler`` ("dynesty" or "pocomc", see
    ``nmma.core.parallel_setup.pbilby_sampling``) from the CLI arguments.

    Parameters
    ----------
    args: argparse.Namespace
        Must provide ``sampler``; ``sampling_seed``, ``plot``, the run
        control (the keys of ``NMMASampler.run_control_defaults``) and
        ``sampler_kwargs`` fall back to defaults.

    Returns
    -------
    run_control_kwargs: dict
        ``sampling_seed``, ``plot`` and the run control, where None (also
        for arguments missing in ``args``) stands for the default of the
        sampler.
    sampler_kwargs: dict
        A copy of ``args.sampler_kwargs``, which the driver resolves against
        the defaults of bilby's (or pocomc_bilby's) sampler.
    """
    if args.sampler not in PARALLEL_SAMPLERS:
        raise ValueError(f"Sampler not available in MPI setup: {args.sampler}")

    run_control_kwargs = dict(
        sampling_seed=getattr(args, "sampling_seed", 42),
        plot=getattr(args, "plot", True),
    )
    run_control_kwargs |= {
        key: getattr(args, key, None) for key in NMMASampler.run_control_defaults
    }
    sampler_kwargs = dict(getattr(args, "sampler_kwargs", None) or {})
    return run_control_kwargs, sampler_kwargs


def bilby_init_kwargs(args, likelihood, injection_parameters) -> dict:
    """Create a dict of always-set bilby kwargs"""
    return dict(
        outdir=args.outdir,
        label=args.label,
        injection_parameters=injection_parameters,
        meta_data=dict(likelihood=likelihood.meta_data),
        soft_init=True,
        use_ratio=True,
    )


def time_storage(func):
    """Decorator that logs how long the wrapped call took, via
    ``logger.info``."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        duration = timedelta(seconds=time() - start)
        logger.info(f"{func.__name__} took {duration} ")
        return result

    return wrapper


class NMMASampler(bs.Sampler):
    """Sampler-independent part of parallel runs' master process:
    checkpointing, iteration/time limits, interruption, saving
    The other processes of the pool only evaluate plain likelihood and prior

    To be subclassed with ``external_init_keys``, ``external_run_keys`` and
    ``run_control_defaults["n_check_point"]``, and implement
    ``start_sampler``, ``_run_sampler``, ``write_current_state``,
    ``current_weighted_samples`` and ``fill_sampler_result``.

    The settings are kept apart by where they go:

    - ``bilby_init_kwargs``: the arguments of bilby's ``Sampler.__init__``;
    - ``self.kwargs``: bilby's resolution of ``sampler_kwargs`` against
      ``default_kwargs`` (bilby's defaults, updated by
      ``nmma_default_kwargs``), from which ``external_init_kwargs`` and
      ``external_run_kwargs`` are taken for the external sampler (dynesty's
      or pocomc's own classes); settings only the bilby wrapper or nmma
      knows are read from ``self.kwargs`` where they are used;
    - ``run_control_kwargs``: nmma's run control, never passed to bilby.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed CLI arguments
    prior: bilby.core.prior.PriorDict
    likelihood: bilby.core.likelihood.Likelihood
    injection_parameters: dict, optional
    sampler_kwargs: dict, optional
        The settings of the sampler. Keys in ``nmma_enforced_kwargs`` are
        ignored with a warning, keys unknown to the sampler raise a ValueError,
        and a seed (see ``seed_kwargs``) overrides ``sampling_seed``.
    sampling_seed: int, default 42
        Seeds all random number generators of the run (see
        ``seed_random_states``), unless ``sampler_kwargs`` sets a seed.
    plot: bool, default False
        Whether to plot the result.
    meta_data: dict, optional
        Extra metadata to store alongside the result.
    **run_control_kwargs
        ``check_point_delta_t``, ``n_check_point``, ``max_its``,
        ``max_run_time`` and ``checkpoint_plot``; None stands for the value
        in ``run_control_defaults``.
    """

    # settings that nmma sets itself, so that --sampler-kwargs cannot override
    nmma_enforced_kwargs = frozenset(
        {
            "outdir",
            "label",
            "plot",
            "use_ratio",
            "soft_init",
            "skip_import_verification",
            "injection_parameters",
            "meta_data",
            # "result_class",
            # "likelihood_benchmark",
            # "exit_code",
            "resume",
            "check_point",
            "check_point_plot",
            "check_point_delta_t",
            "n_check_point",
            "pool",
            *bs.Sampler.npool_equiv_kwargs,
        }
    )
    # defaults that deviate from bilby's, and settings that only nmma knows
    nmma_default_kwargs = {}
    # the run control used where the CLI leaves it None
    run_control_defaults = dict(
        check_point_delta_t=1800,
        n_check_point=2000,
        max_its=1e10,
        max_run_time=1e10,
        checkpoint_plot=False,
    )

    def __init__(
        self,
        args,
        prior,
        likelihood,
        injection_parameters=None,
        sampler_kwargs: dict | None = None,
        sampling_seed: int = 42,
        plot: bool = False,
        meta_data: dict | None = None,
        **run_control_kwargs,
    ):
        args.plot = plot
        self.args = args
        self.interrupted = False
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        for signum in TERMINATION_SIGNALS:
            signal.signal(signum, self.handle_termination_signal)

        sampler_kwargs = self.add_cli_sampler_kwargs(dict(sampler_kwargs or {}))
        sampling_seed = self.resolve_sampling_seed(sampler_kwargs, sampling_seed)
        super().__init__(
            likelihood,
            prior,
            plot=plot,
            skip_import_verification=False,
            **bilby_init_kwargs(args, likelihood, injection_parameters),
            **self.checked_sampler_kwargs(sampler_kwargs),
        )

        self.resume_file = Path(self.outdir) / f"{self.label}_checkpoint_resume.pickle"
        self.samples_file = Path(self.outdir) / f"{self.label}_samples.parquet"

        # All generators are seeded here and saved across restarts, which makes
        # runs fully deterministic, as is important for reproducibility
        self.seed_random_states(sampling_seed)

        self.meta_data = dict(meta_data or {})
        self.sampling_time = 0.0
        self.set_run_control(run_control_kwargs)
        self._split_sampler_kwargs()

    def handle_termination_signal(self, signum: int, frame: Any) -> None:
        """Flag the run as interrupted, so that ``checkpoint_or_stop``
        checkpoints and exits at the next iteration boundary, where the
        sampler state is consistent. A second signal exits immediately,
        falling back to the last periodic checkpoint."""
        if self.interrupted:
            logger.warning("Received second termination signal, exiting now.")
            sys.exit(self.exit_code)
        logger.info(
            f"Received {signal.Signals(signum).name}. Checkpointing and exiting "
            "gracefully after the current iteration."
        )
        self.interrupted = True

    @property
    def default_kwargs(self) -> dict:
        """bilby's defaults of the sampler, updated by
        ``nmma_default_kwargs``."""
        return super().default_kwargs | self.nmma_default_kwargs

    @property
    def external_init_keys(self) -> set[str]:
        """The keys of ``self.kwargs`` that initialise the external
        sampler; defined by subclasses."""
        raise NotImplementedError

    @property
    def external_run_keys(self) -> set[str]:
        """The keys of ``self.kwargs`` that run the external sampler;
        defined by subclasses."""
        raise NotImplementedError

    def add_cli_sampler_kwargs(self, sampler_kwargs: dict) -> dict:
        """``sampler_kwargs`` with the sampler settings that have their own
        CLI argument; none by default."""
        return sampler_kwargs

    @property
    def seed_kwargs(self) -> list[str]:
        """The keys by which ``sampler_kwargs`` can set the sampling seed:
        bilby's equivalents of it and the sampler's own key."""
        return [*self.sampling_seed_equiv_kwargs, self.sampling_seed_key]

    def resolve_sampling_seed(self, sampler_kwargs: dict, sampling_seed: int) -> int:
        """The seed in ``sampler_kwargs`` (the first of ``seed_kwargs``
        present), else ``sampling_seed``."""
        seeds = {
            key: sampler_kwargs[key]
            for key in self.seed_kwargs
            if key in sampler_kwargs
        }
        if not seeds:
            return sampling_seed
        key, seed = next(iter(seeds.items()))
        if len(set(seeds.values())) > 1:
            logger.warning(
                f"Conflicting seeds {seeds} in --sampler-kwargs, using {key}."
            )
        logger.info(
            f"Using {key}={seed} from --sampler-kwargs as sampling seed instead "
            f"of {sampling_seed}."
        )
        return seed

    def checked_sampler_kwargs(self, sampler_kwargs: dict) -> dict:
        """``sampler_kwargs`` without the keys in ``nmma_enforced_kwargs``
        (ignored with a warning) and in ``seed_kwargs`` (resolved by
        ``resolve_sampling_seed``)."""
        enforced = self.nmma_enforced_kwargs & set(sampler_kwargs)
        if enforced:
            logger.warning(f"Ignoring {sorted(enforced)} in parallel runs.")
        dropped = enforced | set(self.seed_kwargs)
        return {k: v for k, v in sampler_kwargs.items() if k not in dropped}

    def _verify_kwargs_against_default_kwargs(self):
        """bilby's check of ``self.kwargs``, but raising a ValueError for
        unknown keys, which bilby would only drop with a warning."""
        unknown = set(self.kwargs) - set(self.default_kwargs)
        if unknown:
            raise ValueError(
                f"Unknown settings {sorted(unknown)} in --sampler-kwargs for "
                f"{self.sampler_name}."
            )
        super()._verify_kwargs_against_default_kwargs()

    def set_run_control(self, run_control_kwargs: dict) -> None:
        """Set ``run_control_kwargs`` from ``run_control_defaults``, updated
        by the values in ``run_control_kwargs`` that are not None, and each
        of them as an attribute (overriding those set by bilby's samplers)."""
        given = {k: v for k, v in run_control_kwargs.items() if v is not None}
        self.run_control_kwargs = self.run_control_defaults | given
        self.n_check_point = self.run_control_kwargs["n_check_point"]
        self.check_point_delta_t = self.run_control_kwargs["check_point_delta_t"]
        self.max_its = self.run_control_kwargs["max_its"]
        self.max_run_time = self.run_control_kwargs["max_run_time"]
        self.checkpoint_plot = self.run_control_kwargs["checkpoint_plot"]

    def _split_sampler_kwargs(self) -> None:
        """Set ``external_init_kwargs`` and ``external_run_kwargs``: the
        settings in ``self.kwargs`` under ``external_init_keys`` and
        ``external_run_keys``, without those in ``nmma_enforced_kwargs``,
        which ``start_sampler`` passes itself."""
        self.external_init_kwargs = self.select_kwargs(self.external_init_keys)
        self.external_run_kwargs = self.select_kwargs(self.external_run_keys)

    def select_kwargs(self, keys: set[str]) -> dict:
        """The settings in ``self.kwargs`` under ``keys``, except those in
        ``nmma_enforced_kwargs``."""
        return {
            key: value
            for key, value in self.kwargs.items()
            if key in keys and key not in self.nmma_enforced_kwargs
        }

    def seed_random_states(self, sampling_seed: int) -> None:
        """Seed all random number generators the master draws from:
        ``self.rstate`` from ``seed_sequence`` itself, and bilby's and
        numpy's global generators from independent children of it.

        bilby's generator estimates the normalisation of constrained priors
        (and draws pocomc's prior samples). Only the master is seeded, since
        workers draw from per-task generators (see
        ``Dynesty.get_initial_points_from_prior``) if at all.
        """
        self.sampling_seed = sampling_seed
        seed_sequence = SeedSequence(sampling_seed)
        self.rstate = Generator(PCG64(seed_sequence))
        bilby_seed, numpy_seed = seed_sequence.spawn(2)
        bilby_random.seed(bilby_seed)
        np.random.seed(numpy_seed.generate_state(1))
        logger.debug(f"Setting random state = {self.rstate} (seed={sampling_seed})")

    def random_states(self) -> dict:
        """The states of all generators seeded by ``seed_random_states``,
        to be checkpointed."""
        return dict(
            rstate=self.rstate,
            numpy=np.random.get_state(),
            bilby=bilby_random.rng.bit_generator.state,
        )

    def restore_random_states(self, states: dict) -> None:
        """Restore generator states saved via ``random_states``."""
        self.rstate = states["rstate"]
        np.random.set_state(states["numpy"])
        bilby_random.rng.bit_generator.state = states["bilby"]

    def run_sampler(self):
        """Drive the sampler with periodic checkpointing: checkpoint every
        ``check_point_delta_t`` seconds or ``n_check_point`` iterations, and
        stop early (checkpointing first) if ``max_its``/``max_run_time`` is
        reached.

        Returns
        -------
        The sampler-specific output for ``format_result``, or None if stopped
        early via max_its/max_run_time.
        """
        logger.info(
            f"Beginning sampling with checkpoints every {self.check_point_delta_t}s "
            f" or {self.n_check_point} iterations until max {self.max_its} "
            f"iterations or max run time {timedelta(seconds=self.max_run_time)}."
        )
        self.run_start_time = self.last_checkpoint_time = time()
        self.last_checkpoint_it = 0
        self.completed = self._run_sampler()

    def stdout_sampling_log(self, step_info: str) -> None:
        """Overwrite the current terminal line with ``step_info``."""
        sys.stdout.write(f"\033[K {step_info}\r")
        sys.stdout.flush()

    def checkpoint_or_stop(self, it: int) -> bool:
        """Apply the run control at an iteration boundary.

        Checkpoints and exits if the run was interrupted, checkpoints and
        returns True if ``max_its``/``max_run_time`` is reached, and
        checkpoints if the checkpoint cadence is due.

        Parameters
        ----------
        it: int
            Iterations completed in this session.

        Returns
        -------
        bool
            Whether sampling should stop.
        """
        run_time = time() - self.run_start_time
        if self.interrupted:
            self.checkpoint_plot = False
            self.checkpointing()
            logger.info("Received termination signal. Checkpointing and exiting.")
            sys.exit(self.exit_code)

        elif it >= self.max_its or run_time > self.max_run_time:
            self.checkpointing()
            logger.info(
                f" Stopping: {it}/{self.max_its} iterations completed after "
                f"{timedelta(seconds=run_time)}/{timedelta(seconds=self.max_run_time)}s"
            )
            return True

        elif (
            time() - self.last_checkpoint_time >= self.check_point_delta_t
            or it - self.last_checkpoint_it >= self.n_check_point
        ):
            self.last_checkpoint_it = it
            self.checkpointing()
        return False

    def write_update(self, with_plot=True) -> None:
        """Add the time since the last checkpoint to ``sampling_time``."""
        now = time()
        self.sampling_time += now - self.last_checkpoint_time
        self.last_checkpoint_time = now
        self.write_current_state()
        if with_plot:
            self.plot_current_state()

    def checkpointing(self) -> None:
        """Write the sample dump and sampler state to disk (optionally
        plotting), and log ``message`` if given."""
        self.write_sample_dump(*self.current_weighted_samples())
        self.write_update(self.checkpoint_plot)

    def plot_current_state(self) -> None:
        """Checkpoint plots; none by default."""

    @time_storage
    def write_sample_dump(self, samples: np.ndarray, weights: np.ndarray) -> None:
        """Rejection sample the current weighted samples into
        ``self.samples_file``.

        Draws from its own generator, not ``self.rstate``:
        Otherwise its results would depend on the checkpoint cadence.
        """
        rstate = Generator(PCG64(self.sampling_seed))
        samples, keep = rejection_sample(samples, weights, rstate)

        logger.info(f"Writing {np.sum(keep)} current samples to {self.samples_file}")
        df = DataFrame(samples, columns=self._search_parameter_keys)
        df.to_parquet(self.samples_file, index=False)

    def dump_checkpoint(self, obj) -> None:
        """Atomically pickle ``obj`` to ``self.resume_file``, via a temp
        file, so an interruption never leaves a broken resume file."""
        print("")
        cp_time = self.sampling_time
        if self.resume_file.is_file():
            cp_time = time() - self.resume_file.stat().st_mtime
        logger.info(f"Write new checkpoint after {timedelta(seconds=cp_time)}")

        temp_filename = self.resume_file.with_suffix(".temp")
        with open(temp_filename, "wb") as file:
            with BufferedWriter(file) as buffer:
                pickle.dump(obj, buffer, protocol=pickle.HIGHEST_PROTOCOL)
        temp_filename.rename(self.resume_file)
        logger.info(f"Written checkpoint file {self.resume_file}")

    def storable_metadata(self):
        """Assemble this run's metadata for the bilby Result.

        Returns
        -------
        dict
        """
        meta_data = self.meta_data
        meta_data["args"] = vars(self.args).copy()  # convert to dict for storing
        meta_data["likelihood"] = self.likelihood.meta_data
        meta_data["external_init_kwargs"] = self.external_init_kwargs
        meta_data["external_run_kwargs"] = self.external_run_kwargs
        meta_data = self.floatify_dict(meta_data)
        return meta_data

    def floatify_dict(self, d):
        """Recursively convert numpy floating values in ``d`` (mutated
        and returned) to plain Python floats, e.g. so it's
        JSON/YAML-serializable. Leaves other value types (ints,
        strings, lists) untouched."""
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = self.floatify_dict(v)
            elif isinstance(v, np.floating):
                d[k] = float(v)
        return d

    def posterior_from_weights(
        self,
        samples: np.ndarray,
        weights: np.ndarray,
        log_likelihoods: np.ndarray,
        rejection_sample_posterior: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Turn weighted samples into equally weighted posterior samples.

        Parameters
        ----------
        samples: numpy.ndarray
            Weighted samples, one row per sample.
        weights: numpy.ndarray
            Their (not necessarily normalised) weights.
        log_likelihoods: numpy.ndarray
            Their log-likelihoods.
        rejection_sample_posterior: bool, default True
            Rejection sample (unique samples, fewer of them) instead of
            resampling with replacement (as many samples as given).

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The posterior samples and their log-likelihoods.
        """
        samples = np.asarray(samples)
        if rejection_sample_posterior:
            _, keep = rejection_sample(samples, weights, self.rstate)
            indices = np.flatnonzero(keep)
            logger.info(f"Rejection sampled to obtain {len(indices)} posterior samples")
        else:
            indices = dynesty.utils.resample_equal(
                np.arange(len(weights)), weights / np.sum(weights), self.rstate
            )
            logger.info("Resampling weighted samples to posterior samples in place.")
        return samples[indices], np.asarray(log_likelihoods)[indices]

    def format_result(
        self,
        result_format=None,
        rejection_sample_posterior=None,
    ):
        """
        Packs the variables from the run into a bilby result object

        Parameters
        ----------
        sampler_result
            The sampler-specific output of ``run_sampler``.
        result_format: str
            The format to save the result
        rejection_sample_posterior: bool
            Whether to generate the posterior samples by rejection sampling the
            weighted samples or resampling with replacement
        """
        if not self.completed:
            return None
        samples, weights, log_likelihoods = self.fill_sampler_result()
        result = self.result
        # all weighted samples, in the format of bilby's dynesty
        result.nested_samples = DataFrame(samples, columns=self._search_parameter_keys)
        result.nested_samples["weights"] = weights
        result.nested_samples["log_likelihood"] = log_likelihoods
        result.sampling_time = timedelta(seconds=self.sampling_time)
        logger.info(f"Sampling time = {result.sampling_time}")
        result.meta_data = self.storable_metadata()
        logger.info(f"Number of lnl calls = {result.num_likelihood_evaluations}")
        result.log_noise_evidence = self.likelihood.noise_log_likelihood()
        result.log_evidence = result.log_noise_evidence + result.log_bayes_factor
        if rejection_sample_posterior is None:
            rejection_sample_posterior = self.args.rejection_sample_posterior
        result.samples, result.log_likelihood_evaluations = self.posterior_from_weights(
            samples, weights, log_likelihoods, rejection_sample_posterior
        )

        result.samples_to_posterior(priors=result.priors)
        result.posterior = self.likelihood.posterior_conversion(result.posterior)
        result.save_posterior_samples()

        extra_keys = set(result.posterior.columns) - set(self._search_parameter_keys)
        posterior_keys = self._search_parameter_keys.copy()
        posterior_keys.extend(extra_keys)
        result.meta_data["posterior_keys"] = posterior_keys
        posterior_labels = result.parameter_labels_with_unit.copy()
        posterior_labels.extend([label_mapping.get(k, k) for k in extra_keys])
        result.meta_data["posterior_labels"] = posterior_labels
        if self.samples_file.is_file():
            self.samples_file.unlink()  # remove temp file after succesful run

        if result_format is None:
            result_format = self.args.result_format
        logger.info(
            f"Saving result to {self.outdir}/{self.label}_result.{result_format}"
        )
        result.save_to_file(extension=result_format)

        if self.plot:
            self.plot_result(result)
        logger.info("Finished formatting result.")
        return result

    def plot_result(self, result) -> None:
        """Corner plot of the posterior and the likelihood's final
        diagnostics; failures are logged, not raised."""
        logger.info("Creating corner plot of posterior samples.")
        try:
            injection_parameters = (
                {
                    k: v
                    for k, v in self.injection_parameters.items()
                    if k in self._search_parameter_keys
                }
                if self.injection_parameters
                else None
            )
            result.plot_corner(parameters=injection_parameters, priors=True, dpi=200)
        except Exception as e:
            logger.warning(f"Failed to create corner plot: {e}")
        try:
            logger.info("Creating diagnostic plots.")
            bestfit_params = read_bestfit_from_posterior(
                result.posterior, "max_posterior"
            )
            self.likelihood.final_diagnostics(bestfit_params, self.args, result)
        except Exception as e:
            logger.warning(
                f"Failed to create diagnostic plots: {e} \n{traceback.format_exc()}"
            )


class Dynesty(NMMASampler, BilbyDynesty):
    """The driver of dynesty on the master process: initializes or resumes a
    dynesty NestedSampler, runs it with periodic checkpointing, and formats
    the final bilby Result.

    The settings of the sampler are those of bilby's ``Dynesty``, with its
    defaults, except for ``nmma_default_kwargs``.
    """

    # queue_size (or any of bilby's aliases of it) is a setting here: it
    # changes the sampling, so it must not follow the pool size implicitly
    nmma_enforced_kwargs = (
        NMMASampler.nmma_enforced_kwargs - set(bs.Sampler.npool_equiv_kwargs)
    ) | {
        # "nestcheck",
        # "print_method",
        "rejection_sample_posterior",
        "live_points",
        "use_pool",
        # "logl_args",
        # "logl_kwargs",
        # "ptform_args",
        # "ptform_kwargs",
        "periodic",
        "reflective",
        "add_live",
        "print_progress",
        "print_func",
        "checkpoint_file",
        "checkpoint_every",
        "rstate",
    }
    nmma_default_kwargs = dict(sample="acceptance-walk")

    @property
    def external_init_keys(self) -> set[str]:
        """The settings of dynesty's ``NestedSampler`` in bilby's ``Dynesty``."""
        return set(self._dynesty_init_kwargs)

    @property
    def external_run_keys(self) -> set[str]:
        """The settings of dynesty's ``Sampler.sample``, which runs the
        iterations here (bilby's ``Dynesty`` takes those of ``run_nested``)."""
        return set(inspect.signature(dynesty.sampler.Sampler.sample).parameters)

    def add_cli_sampler_kwargs(self, sampler_kwargs: dict) -> dict:
        """``sampler_kwargs`` with ``nlive`` from the CLI (``args.nlive``),
        unless they set it or one of its aliases themselves."""
        nlive = getattr(self.args, "nlive", None)
        if nlive is not None and not NLIVE_KWARGS & set(sampler_kwargs):
            sampler_kwargs["nlive"] = nlive
        return sampler_kwargs

    @time_storage
    def start_sampler(self, pool):
        """
        Start from a saved state or initialise.

        The required information to reconstruct the state of the run is read from a
        pickle file.

        Parameters
        ----------
        pool: a Schwimmbad-pool object

        Sets ``self.sampler`` (a resumed instance if a resume file exists
        and was read successfully, else a freshly-created one) and
        ``self.sampling_time`` in place; nothing is returned.
        """

        if self.resume_file.is_file():
            logger.info(f"Reading resume file {self.resume_file}")
            with open(self.resume_file, "rb") as file:
                sampler = pickle.load(file)
                if sampler.added_live:
                    sampler._remove_live_points()

                # reset pool, but keep the queued proposals and queue_size: the
                # former are finished evaluations, and redrawing them or
                # changing the latter would break reproducibility
                sampler.pool = pool
                sampler.mapper = pool.map if sampler.queue_size > 1 else map

            sampler.prior_transform = pooled_prior_transform
            sampler.loglikelihood = dynesty.utils.LogLikelihood(
                pooled_log_likelihood, sampler.ndim
            )

            self.sampling_time = sampler.sampling_time
            self.restore_random_states(sampler.random_states)
            self.sampler = sampler
            requested = self.external_init_kwargs["queue_size"]
            if requested not in (None, sampler.queue_size):
                logger.warning(
                    f"Changed queue_size from {sampler.queue_size} to {requested}. "
                    "This will most likely break reproducibility!"
                )
        else:
            logger.info(
                f"Resume file {self.resume_file} does not exist. "
                f"Initializing sampling points with pool size={pool.size}"
            )
            live_points = self.get_initial_points_from_prior(
                pool, pooled_initial_point_from_prior
            )
            self.external_init_kwargs["queue_size"] = (
                self.external_init_kwargs["queue_size"] or pool.size
            )
            logger.info(f"Initialize NestedSampler with {self.external_init_kwargs}")
            init_kwargs = self.sampler_init_kwargs | dict(
                pool=pool,
                live_points=live_points,
                rstate=self.rstate,
                queue_size=self.external_init_kwargs["queue_size"],
            )
            self.sampler = dynesty.NestedSampler(
                pooled_log_likelihood,
                pooled_prior_transform,
                ndim=self.ndim,
                **init_kwargs,
            )
            self.sampling_time = 0.0

        if self.sampler.queue_size < pool.size:
            logger.warning(
                f"queue_size={self.sampler.queue_size} is smaller than the "
                f"{pool.size} pool workers, so some of them idle."
            )
        self.dlogz = self.external_run_kwargs["dlogz"]
        logger.info(f"Run criteria: {self.external_run_kwargs}")

        logger.info(
            f"Starting sampling for job {self.label}, with pool size={pool.size}"
        )

    def _run_sampler(self) -> bool:
        """Run the dynesty sampling loop

        Returns
        -------
        bool
            Whether sampling completed, i.e. was not stopped early via
            max_its/max_run_time.
        """
        for it, res in enumerate(self.sampler.sample(**self.external_run_kwargs)):
            step_info = self.get_step_info_str(res, it)
            self.stdout_sampling_log(step_info)
            if self.checkpoint_or_stop(it):
                return False

        # Adding the final set of live points.
        for it_final, res in enumerate(self.sampler.add_live_points()):
            pass
        return True

    @time_storage
    def get_initial_points_from_prior(self, pool, find_live):
        """
        Generates a set of initial points, drawn from the prior

        Parameters
        ----------
        pool: schwimmbad.MPIPool
            Schwimmbad pool for MPI parallelisation

        Returns
        -------
        (numpy.ndarray, numpy.ndarray, numpy.ndarray)
            Returns a tuple (unit, theta, logl) where
            unit: point in the unit cube
            theta: scaled value to prior space
            logl: log(likelihood)

        """
        # Create a new rstate for each point, otherwise each task will generate
        # the same random number, and the rstate on master will not be incremented.
        # The argument to self.rstate.integers() is a very large integer.
        # These rstates aren't used after this map, but each time they are created,
        # a different (but deterministic) seed is used.
        seed_gen = SeedSequence(self.rstate.integers(9223372036854775807))
        rstates = [Generator(PCG64(n)) for n in seed_gen.spawn(self.nlive)]
        units, thetas, logls = map(np.array, zip(*pool.map(find_live, rstates)))
        return units, thetas, logls

    @staticmethod
    def initial_point_from_prior(
        sampler: bs.Sampler, rstate: Generator
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Draw an initial point from the prior of ``sampler`` (the bilby
        Sampler of any process of the pool), subject to constraints applied
        both to the prior and the likelihood.

        We remove any points where the likelihood or prior is infinite or NaN.

        The `log_likelihood` often converts infinite values to large
        finite values so we catch those.
        """
        while True:
            unit = rstate.random(sampler.ndim)
            theta = sampler.prior_transform(unit)

            if Dynesty._is_bad_value(sampler.log_prior(theta)):
                continue
            logl = sampler.log_likelihood(theta)
            if Dynesty._is_bad_value(logl):
                continue

            return unit, theta, logl

    @staticmethod
    def _is_bad_value(val) -> bool:
        return not np.isfinite(val) or abs(val) >= np.finfo(float).max

    def get_step_info_str(self, *args):
        """Generate a log string for the current sampling step.
        Logs will look like:
        #:282|eff(%):26.406|logl*:-inf<-160.2<inf|logz:-165.5+/-0.1|dlogz:1038.1>0.1

        Adapted from dynesty
        https://github.com/joshspeagle/dynesty/blob/bb1c5d5f9504c9c3bbeffeeba28ce28806b42273/py/dynesty/utils.py#L349

        Parameters
        ----------
        kwargs: dict
            keyword arguments passed to dynesty.utils.get_print_fn_args

        Returns
        -------
        str
            Formatted log string for the current sampling step.
        """
        niter, _, mid_str, _ = dynesty.utils.get_print_fn_args(
            *args,
            dlogz=self.dlogz,
            ncall=self.sampler.ncall,
        )
        custom_str = [f"#: {niter:d}"] + mid_str
        custom_str = "|".join(custom_str).replace(" ", "")
        return custom_str

    def current_weighted_samples(self) -> tuple[np.ndarray, np.ndarray]:
        """The dead points so far and their importance weights."""
        data = self.sampler.saved_run.D
        weights = np.exp(np.asarray(data["logwt"]) - data["logz"][-1])
        return np.asarray(data["v"]), weights

    @time_storage
    def write_current_state(self):
        """Pickle ``self.sampler`` (with ``self.sampling_time`` attached)
        to ``self.resume_file``, via a temp file for an atomic write.
        Temporarily strips the pool/loglikelihood/prior_transform
        references first, so they aren't (expensively, and often
        unpicklably) pickled along with it."""
        # avoid expensive pickling of easily rebuilt objects
        pool = self.sampler.pool
        mapper = self.sampler.mapper
        logl_func = self.sampler.loglikelihood
        prior_func = self.sampler.prior_transform
        try:
            # Temporarily remove to accelerate pickling
            self.sampler.pool = None
            self.sampler.loglikelihood = None
            self.sampler.prior_transform = None
            self.sampler.mapper = map

            self.sampler.sampling_time = self.sampling_time
            # self.rstate is the sampler's own rstate, so pickling both together
            # keeps them one object
            self.sampler.random_states = self.random_states()
            self.dump_checkpoint(self.sampler)
        except:
            logger.warning("Cannot write checkpoint file!")
        finally:
            self.sampler.pool = pool
            self.sampler.mapper = mapper
            self.sampler.loglikelihood = logl_func
            self.sampler.prior_transform = prior_func

    @time_storage
    def plot_current_state(self):
        """Save trace/run/stats checkpoint plots to
        ``{outdir}/{label}_checkpoint_{trace,run,stats}.png``. Each plot
        failing is caught and logged individually, without aborting the
        others."""
        # labels = [label.replace("_", " ") for label in search_parameter_keys]
        for name, func, obj in zip(
            ["trace", "run", "stats"],
            [traceplot, runplot, stats_plot],
            [self.sampler.results, self.sampler.results, self.sampler],
        ):
            try:
                fig, _ = func(obj)
                fig.tight_layout()
                fig.savefig(f"{self.outdir}/{self.label}_checkpoint_{name}.png")

            except Exception as e:
                logger.warning(e)
                logger.warning(f"Failed to create dynesty {name} plot at checkpoint")
            finally:
                plt.close("all")

    def fill_sampler_result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Set the evidence and call count on ``self.result``.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray, numpy.ndarray)
            The nested samples, their importance weights and log-likelihoods.
        """
        dyn_result = self.sampler.results
        self.result.num_likelihood_evaluations = np.sum(dyn_result.ncall)
        self.result.log_bayes_factor = dyn_result.logz[-1]
        self.result.log_evidence_err = dyn_result.logzerr[-1]
        weights = np.exp(dyn_result["logwt"] - self.result.log_bayes_factor)
        return dyn_result.samples, weights, dyn_result.logl


class IdlingMPIPool(MPIPool):
    """``schwimmbad.MPIPool`` whose idle workers sleep instead of
    busy-polling.

    A waiting worker spins on ``Iprobe`` for ``spin_time`` seconds, so the
    back-to-back tasks of one ``map`` are picked up without delay, and then
    only polls every ``sleep_time`` seconds. This frees the cores of idle
    workers, e.g. for pocomc's flow training on the master.
    """

    spin_time = 0.01
    sleep_time = 0.001

    def wait(self, callback=None) -> None:
        """Serve tasks from the master until it sends None (see
        ``schwimmbad.MPIPool.wait``), idling via ``wait_for_task``."""
        if self.is_master():
            return

        from mpi4py import MPI

        status = MPI.Status()
        while True:
            self.wait_for_task(MPI.ANY_TAG)
            task = self.comm.recv(source=self.master, tag=MPI.ANY_TAG, status=status)
            if task is None:
                break

            func, arg = task
            self.comm.ssend(func(arg), self.master, status.tag)

        if callback is not None:
            callback()

    def wait_for_task(self, tag: int) -> None:
        """Block until a message from the master is available."""
        start = time()
        while not self.comm.Iprobe(source=self.master, tag=tag):
            if time() - start > self.spin_time:
                sleep(self.sleep_time)


def run_sampler(pool, likelihood, prior, args, injection_parameters, meta_data):
    global worker
    run_control_kwargs, sampler_kwargs = process_mpi_kwargs(args)
    if args.sampler == "dynesty":
        sampler_class = Dynesty
    elif args.sampler == "pocomc":
        # imported here, so that only the master ever loads torch
        from .pocomc import PocoMC as sampler_class

    worker = sampler_class(
        args,
        prior,
        likelihood,
        injection_parameters,
        sampler_kwargs=sampler_kwargs,
        meta_data=meta_data,
        **run_control_kwargs,
    )
    if worker.cached_result is not None:
        logger.info(f"Using the cached result of {worker.label}.")
        return worker.cached_result
    worker.meta_data["pool"] = dict(kind=type(pool).__name__, size=pool.size)
    worker.start_sampler(pool)
    worker.run_sampler()
    return worker.format_result()


def build_pool_worker(args, prior, likelihood, injection_parameters) -> bs.Sampler:
    """The plain bilby Sampler on which a process of the pool (an MPI rank
    other than 0, or a ``MultiPool`` process) evaluates the likelihood and prior,
    with the driver's ``bilby_init_kwargs``.

    The process ignores termination signals: the master handles them, and
    needs the pool until it has checkpointed.
    """
    for signum in TERMINATION_SIGNALS:
        signal.signal(signum, signal.SIG_IGN)
    return bs.Sampler(
        likelihood,
        prior,
        skip_import_verification=True,
        **bilby_init_kwargs(args, likelihood, injection_parameters),
    )


def pbilby_sampling(
    likelihood,
    prior,
    args,
    injection_parameters,
    meta_data={},
    **kwargs,
):
    """Run dynesty or pocomc (``args.sampler``) in parallel, along one of
    three paths that share ``run_sampler``: under MPI (``run_under_mpi``),
    with ``args.cpus`` local processes (``run_with_processes``; None: all
    CPUs), or, for a single CPU, serially (``run_serially``).

    Parameters
    ----------
    likelihood, prior
        As elsewhere in NMMA.
    args: argparse.Namespace
        CLI args; also updated in place from ``kwargs``.
    injection_parameters: dict, optional
    meta_data: dict, optional
    **kwargs
        Merged onto ``args`` (kwargs take priority), so CLI args can be
        overridden at the call site.

    Returns
    -------
    bilby.core.result.Result
        The formatted result (on the master only).
    """
    # kwargs > args in priority, so that command line arguments can override the config
    args.__dict__.update(kwargs)
    run_args = (likelihood, prior, args, injection_parameters, meta_data)
    try:
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() > 1:
            return run_under_mpi(*run_args)
    except ImportError:
        pass
    cpus = getattr(args, "cpus", None) or os.cpu_count()
    if cpus > 1:
        return run_with_processes(*run_args, cpus=cpus)
    return run_serially(*run_args)


def run_under_mpi(likelihood, prior, args, injection_parameters, meta_data):
    """The MPI path. The ranks other than 0 build their pool worker.
      Rank 0 runs the sampler.
    A failure on rank 0, or its exit with a non-zero code after an
    interruption, aborts all ranks, so that mpirun reports it.
    """
    from mpi4py import MPI

    global worker
    if MPI.COMM_WORLD.Get_rank() != 0:
        # each rank evaluates the likelihood it built itself, so that the
        # emulators keep their compiled state across evaluations
        worker = build_pool_worker(args, prior, likelihood, injection_parameters)

    pool_class = IdlingMPIPool if args.sampler == "pocomc" else MPIPool
    try:
        with pool_class() as pool:
            return run_sampler(
                pool, likelihood, prior, args, injection_parameters, meta_data
            )
    except SystemExit as exit_info:
        # rank 0 exits with a code after an interruption, others with 0
        if exit_info.code:
            MPI.COMM_WORLD.Abort(exit_info.code)
        raise
    except Exception:
        logger.exception("Sampling failed on the master.")
        MPI.COMM_WORLD.Abort(1)


def run_with_processes(
    likelihood, prior, args, injection_parameters, meta_data, cpus: int
):
    """The path with ``cpus`` local processes: a ``MultiPool`` forks them
    before the driver exists. Errors propagate; leaving the pool's context ends
    its processes.

    Raises
    ------
    RuntimeError
        If JAX's backend is already initialised in this process, since the
        forked processes would deadlock (run under mpirun instead).
    """
    if "jax" in sys.modules:
        raise RuntimeError(
            "JAX is already initialised in this process (e.g. by a fiesta "
            "model), so forked pool processes would deadlock. Run under mpirun "
            "instead of with --cpus > 1."
        )
    worker_args = (args, prior, likelihood, injection_parameters)

    def init_pool_worker(*worker_args) -> None:
        """Initializer of each process of a ``MultiPool``: silence its logging
        and build its pool worker from ``worker_args`` (see
        ``build_pool_worker``), as the MPI ranks other than 0 do in
        ``run_under_mpi``."""
        for name in ("nmma", "bilby"):
            logging.getLogger(name).setLevel(logging.CRITICAL)
        global worker
        worker = build_pool_worker(*worker_args)

    with MultiPool(
        processes=cpus, initializer=init_pool_worker, initargs=worker_args
    ) as pool:
        return run_sampler(
            pool, likelihood, prior, args, injection_parameters, meta_data
        )


def run_serially(likelihood, prior, args, injection_parameters, meta_data):
    """The path without further processes: the driver evaluates everything
    itself, in a ``SerialPool``."""
    with SerialPool() as pool:
        # SerialPool reports 0 workers, but dynesty's queue needs at least 1
        pool.size = 1
        return run_sampler(
            pool, likelihood, prior, args, injection_parameters, meta_data
        )


# Pool functions. These are read in the global scope by each process of the pool
def pooled_initial_point_from_prior(rstate):
    """``Dynesty.initial_point_from_prior`` on the module-level ``worker``
    (set by ``run_sampler``, ``run_under_mpi`` or ``init_pool_worker``), for
    dispatch via the schwimmbad pool."""
    return Dynesty.initial_point_from_prior(worker, rstate)


def pooled_log_likelihood(v_array):
    """``worker.log_likelihood(v_array)``, for dispatch via the
    schwimmbad pool."""
    return worker.log_likelihood(v_array)


def pooled_prior_transform(u_array):
    """``worker.prior_transform(u_array)``, for dispatch via the
    schwimmbad pool."""
    return worker.prior_transform(u_array)
