import pickle

import numpy as np
import pocomc
import torch
from pocomc_bilby.plugin import PocoMC as BilbyPocoMC
from pocomc_bilby.prior import PriorWrapper

from .parallel_setup import NMMASampler, pooled_log_likelihood, time_storage
from .utils import logger


class NMMAPocoMCSampler(pocomc.Sampler):
    """``pocomc.Sampler`` that calls ``iteration_hook`` at every iteration
    boundary (the only point at which its state is consistent).

    Parameters
    ----------
    prior, likelihood, **kwargs
        See ``pocomc.Sampler``. ``likelihood`` must be picklable by
        reference (a module-level function) when using a pool.
    iteration_hook: callable
        Called without arguments before every iteration after warm-up.
    """

    def __init__(
        self,
        prior,
        likelihood,
        iteration_hook,
        **kwargs,
    ):
        super().__init__(prior, likelihood, **kwargs)
        # hand the pool the bare likelihood: pocomc's FunctionWrapper would make
        # every worker import pocomc, and with it torch
        self.log_likelihood = likelihood
        self.iteration_hook = iteration_hook

    def _not_termination(self, current_particles: dict) -> bool:
        """pocomc's termination check, calling ``iteration_hook`` before
        every further iteration."""
        not_terminated = super()._not_termination(current_particles)
        if not_terminated:
            self.iteration_hook()
        return not_terminated


class PocoMC(NMMASampler, BilbyPocoMC):
    """The driver of pocomc on the master process: initializes or resumes an
    ``NMMAPocoMCSampler``, runs it with periodic checkpointing at its
    iteration boundaries, and formats the final bilby Result.

    The settings of the sampler are those of pocomc_bilby's ``PocoMC``,
    except for ``nmma_default_kwargs``.
    """

    nmma_enforced_kwargs = NMMASampler.nmma_enforced_kwargs | {
        "save_every",
        "resume_state_path",
        "track_sampling_time",
    }
    nmma_default_kwargs = dict(
        evaluate_constraints_in_prior=False,
        balance_particles=False,
    )
    run_control_defaults = NMMASampler.run_control_defaults | dict(n_check_point=3)
    # settings among pocomc_bilby's init_kwargs that only the wrapper knows
    wrapper_init_keys = {"evaluate_constraints_in_prior"}

    @property
    def external_init_keys(self) -> set[str]:
        """The settings of ``pocomc.Sampler`` in pocomc_bilby's ``PocoMC``."""
        return set(self.init_kwargs) - self.wrapper_init_keys

    @property
    def external_run_keys(self) -> set[str]:
        """The settings of ``pocomc.Sampler.run`` in pocomc_bilby's
        ``PocoMC``."""
        return set(self.run_kwargs)

    def _verify_kwargs_against_default_kwargs(self):
        """Fill in a missing particle number as pocomc would, so that
        pocomc_bilby can compare them and both can be balanced, before the
        checks of ``NMMASampler`` and pocomc_bilby."""
        if self.kwargs.get("n_active") is None:
            self.kwargs["n_active"] = self.kwargs["n_effective"] // 2
        elif self.kwargs.get("n_effective") is None:
            self.kwargs["n_effective"] = 2 * self.kwargs["n_active"]
        if self.kwargs["evaluate_constraints_in_prior"]:
            logger.warning(
                "Evaluating the prior constraints in pocomc's prior is only "
                "reasonable if the likelihood checks no constraints itself."
            )
        super()._verify_kwargs_against_default_kwargs()

    def seed_random_states(self, sampling_seed: int) -> None:
        """``NMMASampler.seed_random_states``, plus torch's generator (used in
        the flow training) from a further child of ``seed_sequence``."""
        super().seed_random_states(sampling_seed)
        torch.manual_seed(self.rstate.integers(98765432123456789))

    def random_states(self) -> dict:
        """``NMMASampler.random_states``, plus torch's."""
        return super().random_states() | dict(torch=torch.get_rng_state())

    def restore_random_states(self, states: dict) -> None:
        """``NMMASampler.restore_random_states``, plus torch's."""
        super().restore_random_states(states)
        torch.set_rng_state(states["torch"])

    def balance_to_pool(self, n_workers: int) -> None:
        """Balance "n_active"/"n_effective" in ``external_init_kwargs`` to
        ``n_workers`` if ``balance_particles`` is set (see
        ``balance_particle_numbers``)."""
        n_active = self.external_init_kwargs["n_active"]
        n_effective = self.external_init_kwargs["n_effective"]
        if not self.kwargs["balance_particles"]:
            if n_active % n_workers:
                logger.warning(
                    f"n_active={n_active} is not a multiple of the {n_workers} "
                    "pool workers, so some of them idle in every round of "
                    "likelihood evaluations (set balance_particles: true in "
                    "--sampler-kwargs to adjust n_active and n_effective)."
                )
            return

        n_workers = max(n_workers, 1)
        bal_n_active = max(round(n_active / n_workers), 1) * n_workers
        bal_n_eff = round(n_effective * bal_n_active / n_active)
        if (bal_n_active, bal_n_eff) != (n_active, n_effective):
            logger.info(
                f"Balancing pocomc to {n_workers} pool workers: n_active "
                f"{n_active} -> {bal_n_active}, n_effective {n_effective} -> "
                f"{bal_n_eff}."
            )
        self.external_init_kwargs.update(n_active=bal_n_active, n_effective=bal_n_eff)

    @time_storage
    def start_sampler(self, pool) -> None:
        """Build the sampler for ``pool`` and resume it from
        ``self.resume_file`` if that exists.

        Parameters
        ----------
        pool: a Schwimmbad-pool object
        """
        self.balance_to_pool(pool.size)
        self.sampler = NMMAPocoMCSampler(
            prior=PriorWrapper(
                self.priors,
                self._search_parameter_keys,
                self.kwargs["evaluate_constraints_in_prior"],
            ),
            likelihood=pooled_log_likelihood,
            iteration_hook=self.handle_iteration,
            pool=pool,
            n_dim=self.ndim,
            periodic=self._get_pocomc_boundaries("periodic"),
            reflective=self._get_pocomc_boundaries("reflective"),
            output_dir=self.outdir,
            output_label=self.label,
            **self.external_init_kwargs,
        )
        if self.resume_file.is_file():
            self.read_current_state()
        else:
            logger.info(f"Resume file {self.resume_file} does not exist.")
        self.first_iteration = self.sampler.t
        logger.info(
            f"Starting sampling for job {self.label}, with pool size={pool.size}, "
            f"n_active={self.sampler.n_active} and "
            f"n_effective={self.sampler.n_effective}"
        )

    def handle_iteration(self) -> None:
        """The sampler's ``iteration_hook``: apply the run control (see
        ``checkpoint_or_stop``), raising ``StopSampling`` to stop early."""
        it = self.sampler.t - self.first_iteration
        self.stdout_sampling_log(self.get_step_info_str())
        if self.checkpoint_or_stop(it):
            raise StopSampling

    def get_step_info_str(self) -> str:
        """Log string for the current iteration, e.g.
        #:12|beta:0.0213|ESS:512|logz:-23.418|acc:0.412|steps:6|calls:9216
        """
        particles = self.sampler.current_particles
        return (
            f"#:{self.sampler.t}|beta:{particles['beta']:.3g}"
            f"|ESS:{particles['ess']:.0f}|logz:{particles['logz']:.3f}"
            f"|acc:{particles['accept']:.3f}|steps:{particles['steps']}"
            f"|calls:{particles['calls']}"
        )

    def _run_sampler(self) -> bool:
        """Run pocomc.

        Returns
        -------
        bool
            Whether sampling completed, i.e. was not stopped early via
            max_its/max_run_time.
        """
        try:
            self.sampler.run(**self.external_run_kwargs)
        except StopSampling:
            return False
        return True

    def current_weighted_samples(self) -> tuple[np.ndarray, np.ndarray]:
        """All particles so far, importance weighted to the posterior."""
        samples, weights, _, _ = self.sampler.posterior()
        return samples, weights

    @time_storage
    def write_current_state(self) -> None:
        """Write the sampler state to ``self.resume_file``, without the
        attributes ``start_sampler`` rebuilds,
        but with the sampling time, all random states, and the search
        parameters and their bounds to check the resume against."""
        sampler_state = {
            key: value
            for key, value in self.sampler.__dict__.items()
            if key
            not in {
                "pbar",
                "pool",
                "distribute",
                "prior",
                "log_prior",
                "sample_prior",
                "bounds",
                "log_likelihood",
                "iteration_hook",
            }
        }
        checkpoint = dict(
            sampler_state=sampler_state,
            sampling_time=self.sampling_time,
            random_states=self.random_states(),
            prior_normalizations=self.priors._cached_normalizations,
            search_parameter_keys=list(self._search_parameter_keys),
            bounds=self.sampler.bounds,
        )
        try:
            self.dump_checkpoint(checkpoint)
        except Exception as e:
            logger.warning(f"Cannot write checkpoint file! {e}")

    def read_current_state(self) -> None:
        """Restore the state written by ``write_current_state`` onto the
        freshly built ``self.sampler``.

        Raises
        ------
        ValueError
            If the checkpoint was written for different search parameters
            or prior bounds.
        """
        logger.info(f"Reading resume file {self.resume_file}")
        with open(self.resume_file, "rb") as file:
            checkpoint = pickle.load(file)
        self.check_resume_compatibility(checkpoint)

        self.sampler.__dict__.update(checkpoint["sampler_state"])
        self.sampling_time = checkpoint["sampling_time"]
        self.restore_random_states(checkpoint["random_states"])
        self.priors._cached_normalizations.update(checkpoint["prior_normalizations"])

        resumed = dict(
            n_active=self.sampler.n_active, n_effective=self.sampler.n_effective
        )
        if resumed != {k: self.external_init_kwargs[k] for k in resumed}:
            logger.info(f"Resumed run keeps its particle numbers {resumed}.")
        self.external_init_kwargs.update(resumed)

    def check_resume_compatibility(self, checkpoint: dict) -> None:
        """Raise a ValueError unless ``checkpoint`` was written for the
        current search parameters and prior bounds."""
        same_keys = checkpoint["search_parameter_keys"] == list(
            self._search_parameter_keys
        )
        if not (
            same_keys and np.array_equal(checkpoint["bounds"], self.sampler.bounds)
        ):
            raise ValueError(
                f"{self.resume_file} was written for a different prior, so the "
                "run cannot be resumed. Delete it to start over."
            )

    def fill_sampler_result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Set the evidence and call count on ``self.result``.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray, numpy.ndarray)
            The particles, their importance weights and log-likelihoods.
        """
        # the log-priors are pocomc's, without the constraints of bilby's
        # PriorDict.ln_prob, so bilby recomputes them for the posterior
        samples, weights, log_likelihoods, _ = self.sampler.posterior()
        log_bayes_factor, log_bayes_factor_err = self.sampler.evidence()
        self.result.num_likelihood_evaluations = self.sampler.calls
        self.result.log_bayes_factor = log_bayes_factor
        self.result.log_evidence_err = (
            np.nan if log_bayes_factor_err is None else log_bayes_factor_err
        )
        return samples, weights, log_likelihoods


class StopSampling(Exception):
    """Raised from the iteration hook to stop pocomc early."""
