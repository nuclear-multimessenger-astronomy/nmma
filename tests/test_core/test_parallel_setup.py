import json
import shutil
import signal
import subprocess
import sys
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bilby.core.likelihood import Likelihood
from bilby.core.prior import PriorDict, Uniform
from bilby.core.result import Result, read_in_result
from bilby.core.sampler.dynesty import Dynesty as BilbyDynesty
from bilby.core.utils import random as bilby_random

from nmma.core.parallel_setup import (
    Dynesty,
    NMMASampler,
    build_pool_worker,
    pbilby_sampling,
    process_mpi_kwargs,
    time_storage,
)

N_MPI_RANKS = 3
# local processes, as many as the pool workers under mpirun
N_LOCAL_PROCESSES = N_MPI_RANKS - 1
RUN_TIMEOUT = 600
needs_mpirun = pytest.mark.skipif(shutil.which("mpirun") is None, reason="needs mpirun")


class BoundedGaussianLikelihood(Likelihood):
    """Gaussian log-likelihood in x, returning ``outside_value`` instead for
    x < ``cut`` (like a likelihood constraint), so that the rejection of bad
    values has something real to act on."""

    meta_data = {}

    def __init__(
        self, mean: float, sigma: float, cut: float, outside_value: float = -np.inf
    ):
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.cut = cut
        self.outside_value = outside_value

    def log_likelihood(self, parameters=None) -> float:
        if parameters["x"] < self.cut:
            return self.outside_value
        return -0.5 * ((parameters["x"] - self.mean) / self.sigma) ** 2

    def noise_log_likelihood(self) -> float:
        return 0.0

    def posterior_conversion(self, posterior):
        return posterior


def assert_succeeded(completed: subprocess.CompletedProcess, err_tail=2000) -> None:
    """The run in ``completed`` exited with 0; shows its stderr otherwise."""
    assert completed.returncode == 0, completed.stderr[-err_tail:]


class TestTimeStorage:
    def setup_method(self) -> None:
        self.summands = (2, 3)

        @time_storage
        def add(a: int, b: int) -> int:
            return a + b

        self.add = add

    def test_keeps_name_and_result(self) -> None:
        assert self.add.__name__ == "add"
        assert self.add(*self.summands) == sum(self.summands)


class TestBuildPoolWorker:
    """The pool worker, against a small real prior and likelihood, which the
    tests of the samplers share."""

    seed = 1
    parameter_names = ["x", "phi", "psi"]
    x_range = (-1.0, 1.0)
    # BoundedGaussianLikelihood: Gaussian in x, and outside_value below x_cut
    x_mean = 0.3
    x_sigma = 0.1
    x_cut = -0.5
    label = "worker_test"
    outdir = Path(__file__).with_name("pool_worker_outdir")
    termination_signals = (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)

    @classmethod
    def setup_class(cls) -> None:
        # one Namespace per test class, built from the class's own outdir,
        # which starts empty and is shared by the tests of the class
        cls.args = Namespace(outdir=str(cls.outdir), label=cls.label)
        shutil.rmtree(cls.outdir, ignore_errors=True)

    @classmethod
    def teardown_class(cls) -> None:
        shutil.rmtree(cls.outdir, ignore_errors=True)

    def setup_method(self) -> None:
        # handlers installed by a test must not outlive it
        self.original_handlers = {
            s: signal.getsignal(s) for s in self.termination_signals
        }
        self.prior = self.build_prior(self.x_range)
        self.likelihood_kwargs = dict(
            mean=self.x_mean, sigma=self.x_sigma, cut=self.x_cut
        )
        self.likelihood = BoundedGaussianLikelihood(**self.likelihood_kwargs)
        self.worker = build_pool_worker(
            self.args, self.prior, self.likelihood, injection_parameters=None
        )
        self.rstate = np.random.default_rng(self.seed)

    def teardown_method(self) -> None:
        for signum, handler in self.original_handlers.items():
            signal.signal(signum, handler)

    def build_prior(self, x_range: tuple[float, float]) -> PriorDict:
        """Uniform prior over ``parameter_names``: x in ``x_range``, with a
        periodic phi and a reflective psi."""
        x, phi, psi = self.parameter_names
        return PriorDict(
            {
                x: Uniform(*x_range, name=x),
                phi: Uniform(0.0, 2 * np.pi, name=phi, boundary="periodic"),
                psi: Uniform(0.0, np.pi, name=psi, boundary="reflective"),
            }
        )

    def test_setup(self) -> None:
        assert self.worker.outdir == str(self.outdir)
        assert self.worker.label == self.label
        assert self.worker._search_parameter_keys == self.parameter_names

    def test_meta_data_holds_likelihood_meta_data(self) -> None:
        # compared by bilby with the meta data of a cached result
        assert self.worker.meta_data == dict(likelihood=self.likelihood.meta_data)

    def test_termination_signal_handlers(self) -> None:
        for signum in self.termination_signals:
            assert signal.getsignal(signum) == signal.SIG_IGN


class NMMASamplerTestContainer(TestBuildPoolWorker):
    """Tests of NMMASampler behaviour, run through its concrete subclasses.

    The runs in subprocesses are described by ``self.runs`` and started on
    first request by ``completed_run``; each runs once per test class, and the
    tests read the shared result and checkpoint files.
    """

    sampler_class = None
    bilby_class = None
    sampler_args = {}
    max_its_before_stop = 2
    # the defaults in which the sampler deliberately deviates from bilby's class
    deviations_from_bilby = {}
    # settings that bilby sets to a bound method of the sampler instance
    instance_bound_settings = ()
    # sampler settings that make the result independent of the pool size
    pool_independent_settings = {}
    # the pool of the sampler under mpirun
    mpi_pool_kind = "MPIPool"

    @classmethod
    def setup_class(cls):
        super().setup_class()
        if shutil.which("mpirun") is not None:
            # pools of the three paths of pbilby_sampling, as stored in the result
            cls.mpi_pool = dict(kind=cls.mpi_pool_kind, size=N_MPI_RANKS - 1)

        cls.n_draws = 3
        # the runs of self.runs that completed, by name, shared by all tests
        cls.completed_runs = {}
        cls.local_pool = dict(kind="MultiPool", size=N_LOCAL_PROCESSES)
        cls.serial_pool = dict(kind="SerialPool", size=1)
        cls.args.__dict__.update(
            sampler=cls.sampler_class.__name__.lower(),
            sampling_seed=cls.seed,
            plot=False,
            check_point_delta_t=1e9,
            n_check_point=None,
            max_its=1e10,
            max_run_time=1e10,
            checkpoint_plot=False,
            result_format="json",
            rejection_sample_posterior=True,
            **cls.sampler_args,
        )

    def setup_method(self) -> None:
        super().setup_method()
        self.sampler = self.build_sampler()
        # the runs, by name: ``run`` is one of the three paths, ``starts_from``
        # a run whose files it continues (its checkpoint is copied unless both
        # share the label), and the rest are overrides of the run's arguments;
        # the label defaults to the name
        self.runs = dict(
            mpi=dict(run=self.run_under_mpi),
            mpi_rerun=dict(run=self.run_under_mpi, starts_from="mpi", label="mpi"),
            stopped=dict(run=self.run_under_mpi, max_its=self.max_its_before_stop),
            # resumed with more processes than the stopped run had workers
            resumed=dict(
                run=self.run_with_processes,
                starts_from="stopped",
                cpus=N_LOCAL_PROCESSES + 1,
            ),
            local=dict(run=self.run_with_processes),
            serial=dict(
                run=self.run_serially,
                sampler_kwargs=self.args.sampler_kwargs
                | self.pool_independent_settings,
            ),
        )

    def build_sampler(
        self, sampler_overrides: dict | None = None, **arg_overrides
    ) -> NMMASampler:
        """The sampler as ``run_sampler`` builds it on the master, from
        ``self.args`` overridden by ``arg_overrides``, and its
        ``sampler_kwargs`` updated by ``sampler_overrides``."""
        args = Namespace(**(vars(self.args) | arg_overrides))
        args.sampler_kwargs = args.sampler_kwargs | (sampler_overrides or {})
        run_control_kwargs, sampler_kwargs = process_mpi_kwargs(args)
        return self.sampler_class(
            args,
            self.prior,
            self.likelihood,
            None,
            sampler_kwargs=sampler_kwargs,
            **run_control_kwargs,
        )

    def completed_run(self, name: str) -> subprocess.CompletedProcess:
        """The run ``name`` of ``self.runs``, started on its first request
        (after the run it starts from) and shared by all tests of the class."""
        if name not in self.completed_runs:
            arg_overrides = dict(self.runs[name])
            run = arg_overrides.pop("run")
            starts_from = arg_overrides.pop("starts_from", None)
            label = arg_overrides.setdefault("label", name)
            if starts_from is not None:
                assert_succeeded(self.completed_run(starts_from))
                source_label = self.runs[starts_from].get("label", starts_from)
                if source_label != label:
                    shutil.copy(
                        self.build_sampler(label=source_label).resume_file,
                        self.build_sampler(label=label).resume_file,
                    )
            self.completed_runs[name] = run(**arg_overrides)
        return self.completed_runs[name]

    def run_under_mpi(
        self, prior: PriorDict | None = None, **arg_overrides
    ) -> subprocess.CompletedProcess:
        config = self.run_config(prior, arg_overrides)
        return subprocess.run(
            ["mpirun", "-n", str(N_MPI_RANKS), sys.executable, __file__, config],
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT,
        )

    def run_with_processes(
        self, cpus: int = N_LOCAL_PROCESSES, **arg_overrides
    ) -> subprocess.CompletedProcess:
        config = self.run_config(None, arg_overrides | dict(cpus=cpus))
        return subprocess.run(
            [sys.executable, __file__, config],
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT,
        )

    def run_serially(self, **arg_overrides) -> subprocess.CompletedProcess:
        """As ``run_with_processes``, but with a single CPU."""
        return self.run_with_processes(cpus=1, **arg_overrides)

    def run_config(self, prior: PriorDict | None, arg_overrides: dict) -> str:
        prior = prior or self.prior
        label = "run"
        prior.to_json(outdir=self.outdir, label=label)
        return json.dumps(
            dict(
                args=vars(self.args) | arg_overrides,
                prior_file=str(self.outdir / f"{label}_prior.json"),
                likelihood_kwargs=self.likelihood_kwargs,
            )
        )

    def read_result(self, label: str) -> Result:
        return read_in_result(self.result_file(label))

    def result_file(self, label: str) -> Path:
        return self.outdir / f"{label}_result.json"

    def assert_same_results(self, *labels: str) -> None:
        """The saved results of the runs ``labels`` are identical."""
        first, *others = [self.read_result(label) for label in labels]
        for other in others:
            assert other.log_evidence == first.log_evidence
            assert other.posterior.equals(first.posterior)

    def draw_from_random_states(self, sampler: NMMASampler) -> list:
        """One draw from each generator in ``sampler.random_states()``."""
        return [
            sampler.rstate.random(),
            np.random.random(),
            bilby_random.rng.random(),
        ]

    # in-process behaviour

    def test_init(self) -> None:
        assert self.outdir.is_dir()
        assert not self.sampler.interrupted
        self.sampler.handle_termination_signal(signal.SIGTERM, None)
        assert self.sampler.interrupted

        with pytest.raises(SystemExit) as exit_info:
            self.sampler.handle_termination_signal(signal.SIGTERM, None)
        assert exit_info.value.code == self.sampler.exit_code

    def test_termination_signal_handlers(self) -> None:
        # the sampler, built after the pool worker, replaces the worker's
        # SIG_IGN with its own handler
        for signum in self.termination_signals:
            assert signal.getsignal(signum) == self.sampler.handle_termination_signal

    def test_n_check_point(self) -> None:
        custom_n_check_point = 7
        default = self.sampler_class.run_control_defaults["n_check_point"]
        assert self.sampler.n_check_point == default
        sampler = self.build_sampler(n_check_point=custom_n_check_point)
        assert sampler.n_check_point == custom_n_check_point
        assert sampler.run_control_kwargs["n_check_point"] == custom_n_check_point

    def test_seeding(self) -> None:
        # reproducible
        draws = self.sampler.rstate.random(self.n_draws)
        assert np.array_equal(self.build_sampler().rstate.random(self.n_draws), draws)

        seed_setting = {self.sampler.sampling_seed_key: self.seed + 1}
        sampler = self.build_sampler(sampler_overrides=seed_setting)
        assert sampler.sampling_seed == self.seed + 1

    def test_restore_random_states(self) -> None:
        # deep-copied, as written to and read from a checkpoint
        states = deepcopy(self.sampler.random_states())
        draws = self.draw_from_random_states(self.sampler)
        restored = self.build_sampler()
        restored.restore_random_states(states)
        assert self.draw_from_random_states(restored) == draws

    def test_default_sampler_kwargs_match_bilby(self) -> None:
        sampler = self.build_sampler(sampler_kwargs={})
        _, sampler_kwargs = process_mpi_kwargs(sampler.args)
        sampler_kwargs = sampler.add_cli_sampler_kwargs(sampler_kwargs)
        bilby_sampler = self.bilby_class(
            self.likelihood,
            self.prior,
            outdir=str(self.outdir),
            label=self.label,
            soft_init=True,
            **sampler_kwargs,
        )
        expected = bilby_sampler.kwargs | self.deviations_from_bilby
        for key in self.instance_bound_settings:
            expected[key] = sampler.kwargs[key]
        assert sampler.kwargs == expected

    def test_unknown_sampler_kwargs_raise(self) -> None:
        unknown_sampler_setting = "not_a_sampler_setting"
        unknown_setting = {unknown_sampler_setting: True}
        with pytest.raises(ValueError, match=unknown_sampler_setting):
            self.build_sampler(sampler_overrides=unknown_setting)

    def test_enforced_sampler_kwargs_are_ignored(self) -> None:
        custom_check_point_delta_t = 5.0
        enforced_setting = dict(check_point_delta_t=custom_check_point_delta_t)
        sampler = self.build_sampler(sampler_overrides=enforced_setting)
        assert sampler.check_point_delta_t == self.args.check_point_delta_t
        assert "check_point_delta_t" not in sampler.kwargs

    def test_posterior_from_weights(self) -> None:
        weighted_samples = np.repeat(np.arange(6.0)[:, None], 3, axis=1)
        weights = np.array([0.0, 1.0, 0.0, 3.0, 2.0, 0.0])
        sample_log_likelihoods = -np.arange(6.0)
        weighted = weights > 0
        for rejection_sample_posterior in [True, False]:
            samples, log_likelihoods = self.sampler.posterior_from_weights(
                weighted_samples,
                weights,
                sample_log_likelihoods,
                rejection_sample_posterior,
            )
            rows = samples[:, 0].astype(int)
            assert weighted[rows].all(), "zero-weight samples must not survive"
            assert np.array_equal(log_likelihoods, sample_log_likelihoods[rows])
            if not rejection_sample_posterior:
                assert len(samples) == len(weights)

    def test_write_sample_dump_is_independent_of_rstate(self) -> None:
        dummy_samples = np.repeat(np.arange(6.0)[:, None], 3, axis=1)
        dummy_weights = np.array([0.0, 1.0, 0.0, 3.0, 2.0, 0.0])
        draws = self.build_sampler().rstate.random(self.n_draws)
        self.sampler.write_sample_dump(dummy_samples, dummy_weights)
        first = pd.read_parquet(self.sampler.samples_file)
        self.sampler.write_sample_dump(dummy_samples, dummy_weights)

        assert list(first.columns) == self.parameter_names
        assert first.equals(pd.read_parquet(self.sampler.samples_file))
        assert np.array_equal(self.sampler.rstate.random(self.n_draws), draws)

    def test_storable_metadata(self) -> None:
        meta_data = self.sampler.storable_metadata()
        assert meta_data["args"]["label"] == self.label
        assert meta_data["external_run_kwargs"] == self.sampler.external_run_kwargs
        assert meta_data["external_init_kwargs"] is self.sampler.external_init_kwargs

    def test_floatify_dict(self) -> None:
        nested_float = np.float64(1.5)
        floats = self.sampler.floatify_dict(dict(outer=dict(inner=nested_float)))
        assert type(floats["outer"]["inner"]) is float

    def test_external_kwargs_are_split_from_kwargs(self) -> None:
        for external_kwargs, keys in [
            (self.sampler.external_init_kwargs, self.sampler.external_init_keys),
            (self.sampler.external_run_kwargs, self.sampler.external_run_keys),
        ]:
            assert set(external_kwargs) <= keys
            assert not set(external_kwargs) & self.sampler.nmma_enforced_kwargs
            for key, value in external_kwargs.items():
                assert value is self.sampler.kwargs[key]

    # runs under mpirun

    @needs_mpirun
    def test_run_under_mpi_completes(self) -> None:
        assert_succeeded(self.completed_run("mpi"))
        result = self.read_result("mpi")

        # loose sanity bound for the small runs here, not a convergence test
        log_evidence_tolerance = 0.5
        analytic_log_evidence = np.log(
            np.sqrt(2 * np.pi) * self.x_sigma / np.ptp(self.x_range)
        )
        log_evidence_error = result.log_evidence - analytic_log_evidence
        assert abs(log_evidence_error) < log_evidence_tolerance
        assert (result.posterior["x"] >= self.x_cut).all()
        assert {"weights", "log_likelihood"} <= set(result.nested_samples.columns)
        assert result.meta_data["pool"] == self.mpi_pool
        assert not self.build_sampler(label="mpi").samples_file.is_file()

    @needs_mpirun
    def test_run_under_mpi_stops_at_max_its(self) -> None:
        assert_succeeded(self.completed_run("stopped"))
        stopped = self.build_sampler(label="stopped")
        assert not self.result_file("stopped").is_file()
        assert stopped.resume_file.is_file()
        assert stopped.samples_file.is_file()

    @needs_mpirun
    def test_rerun_of_completed_run_uses_cached_result(self) -> None:
        assert_succeeded(self.completed_run("mpi"))
        first = self.read_result("mpi")
        completed = self.completed_run("mpi_rerun")
        assert_succeeded(completed)
        assert "Using the cached result of mpi" in completed.stderr
        assert self.read_result("mpi").posterior.equals(first.posterior)

    # runs with local processes, compared with the other paths

    @needs_mpirun
    def test_run_with_processes_matches_run_under_mpi(self) -> None:
        for name in ["mpi", "local"]:
            assert_succeeded(self.completed_run(name))
        self.assert_same_results("mpi", "local")
        assert self.read_result("local").meta_data["pool"] == self.local_pool

    def test_run_serially_matches_run_with_processes(self) -> None:
        for name in ["serial", "local"]:
            assert_succeeded(self.completed_run(name))
        self.assert_same_results("serial", "local")
        assert self.read_result("serial").meta_data["pool"] == self.serial_pool

    @needs_mpirun
    def test_resumed_run_matches_uninterrupted_run(self) -> None:
        # stopped under mpirun and resumed with more local processes, which
        # must keep the sampling of the stopped run (e.g. dynesty's queue_size)
        for name in ["mpi", "resumed"]:
            assert_succeeded(self.completed_run(name))
        self.assert_same_results("mpi", "resumed")


class TestDynesty(NMMASamplerTestContainer):
    sampler_class = Dynesty
    bilby_class = BilbyDynesty
    outdir = Path(__file__).with_name("dynesty_outdir")
    instance_bound_settings = ("print_func",)
    deviations_from_bilby = dict(sample="acceptance-walk")
    nlive = 50
    dlogz = 0.5
    maxmcmc = 500
    sampler_args = dict(nlive=nlive, sampler_kwargs=dict(dlogz=dlogz, maxmcmc=maxmcmc))
    max_its_before_stop = 4
    # dynesty's proposals depend on its queue, by default the pool size
    pool_independent_settings = dict(queue_size=N_LOCAL_PROCESSES)

    def assert_initial_points_valid(self, worker) -> None:
        for _ in range(10):
            unit, theta, logl = Dynesty.initial_point_from_prior(worker, self.rstate)
            assert np.isfinite(logl), f"bad value {logl} at {theta}"
            assert theta[0] >= self.x_cut
            assert ((unit >= 0.0) & (unit <= 1.0)).all()

    def test_boundaries_reach_dynesty_via_bilby(self) -> None:
        bilby_init_kwargs = self.sampler.sampler_init_kwargs
        assert bilby_init_kwargs["periodic"] == [self.parameter_names.index("phi")]
        assert bilby_init_kwargs["reflective"] == [self.parameter_names.index("psi")]

    def test_external_run_kwargs(self) -> None:
        run_kwargs = self.sampler.external_run_kwargs
        # settings of dynesty's Sampler.sample that are not enforced
        run_keys = {"maxiter", "maxcall", "dlogz", "logl_max", "save_bounds"}
        assert set(run_kwargs) <= run_keys
        assert run_kwargs["dlogz"] == self.dlogz

    def test_aliasing(self):
        alias = dict(npool=2 * N_LOCAL_PROCESSES, npoints=2 * self.nlive)
        sampler = self.build_sampler(sampler_overrides=alias)
        queue_size = sampler.external_init_kwargs["queue_size"]
        assert queue_size == alias["npool"]
        assert sampler.nlive == alias["npoints"]

    def test_initial_point_from_prior_rejects_neg_inf(self) -> None:
        self.assert_initial_points_valid(self.worker)

    def test_initial_point_from_prior_rejects_nan(self) -> None:
        nan_likelihood = BoundedGaussianLikelihood(
            **self.likelihood_kwargs, outside_value=np.nan
        )
        nan_worker = build_pool_worker(
            self.args, self.prior, nan_likelihood, injection_parameters=None
        )
        self.assert_initial_points_valid(nan_worker)


@needs_mpirun
class TestIdlingMPIPool:
    def test_map(self) -> None:
        # just do a very silly test of the mapping
        tasks = [-1, -2, -3, -4, -5]
        script = (
            "from nmma.core.parallel_setup import IdlingMPIPool\n"
            "with IdlingMPIPool() as pool:\n"
            f"    print(pool.map(abs, {tasks}))\n"
        )

        completed = subprocess.run(
            ["mpirun", "-n", str(N_MPI_RANKS), sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT,
        )
        assert_succeeded(completed)
        assert completed.stdout.strip() == str([abs(t) for t in tasks])


if __name__ == "__main__":
    # executed by every rank of NMMASamplerTestContainer.run_under_mpi, or by the
    # master of NMMASamplerTestContainer.run_with_processes
    config = json.loads(sys.argv[1])
    pbilby_sampling(
        BoundedGaussianLikelihood(**config["likelihood_kwargs"]),
        PriorDict.from_json(config["prior_file"]),
        Namespace(**config["args"]),
        None,
    )
