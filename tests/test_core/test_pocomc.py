from pathlib import Path

import numpy as np
import pytest
import torch
from bilby.core.prior import PriorDict, Uniform
from pocomc_bilby.plugin import PocoMC as BilbyPocoMC
from pocomc_bilby.prior import PriorWrapper
from test_parallel_setup import (
    NMMASamplerTestContainer,
    needs_mpirun,
)

from nmma.core.pocomc import NMMAPocoMCSampler, PocoMC, StopSampling


class TestNMMAPocoMCSampler:
    seed = 5
    n_total = 256
    n_effective = 64
    n_active = 32
    pytorch_threads = 1
    parameter_names = ["a", "b", "c"]
    parameter_range = (-1.0, 1.0)

    def setup_method(self) -> None:
        # pocomc sets the torch threads of the whole process to pytorch_threads
        self.threads_before = torch.get_num_threads()
        self.iterations_in_hook = []
        prior = PriorDict(
            {
                name: Uniform(*self.parameter_range, name=name)
                for name in self.parameter_names
            }
        )
        self.sampler = NMMAPocoMCSampler(
            prior=PriorWrapper(prior, self.parameter_names, False),
            likelihood=self.dummy_log_likelihood,
            iteration_hook=self.record_iteration,
            n_effective=self.n_effective,
            n_active=self.n_active,
            pytorch_threads=self.pytorch_threads,
            random_state=self.seed,
        )

    def teardown_method(self) -> None:
        torch.set_num_threads(self.threads_before)

    @staticmethod
    def dummy_log_likelihood(theta) -> float:
        return -0.5 * np.sum(((np.asarray(theta) - 0.3) / 0.1) ** 2)

    def record_iteration(self) -> None:
        self.iterations_in_hook.append(self.sampler.t)

    def stop_sampling(self) -> None:
        raise StopSampling

    def test_log_likelihood_notwrapped(self) -> None:
        assert self.sampler.log_likelihood is self.dummy_log_likelihood

    def test_run_calls_hook_at_every_iteration_boundary(self) -> None:
        self.sampler.run(n_total=self.n_total, progress=False)
        first, *_ = self.iterations_in_hook
        expected = list(range(first, self.sampler.t))
        assert self.iterations_in_hook == expected

    def test_hook_stops_run(self) -> None:
        self.sampler.iteration_hook = self.stop_sampling
        with pytest.raises(StopSampling):
            self.sampler.run(n_total=self.n_total, progress=False)


class TestPocoMC(NMMASamplerTestContainer):
    sampler_class = PocoMC
    bilby_class = BilbyPocoMC
    outdir = Path(__file__).with_name("pocomc_outdir")
    mpi_pool_kind = "IdlingMPIPool"
    deviations_from_bilby = dict(
        evaluate_constraints_in_prior=False,
        balance_particles=False,
    )
    n_effective = 64
    n_active = 32
    n_total = 256
    sampler_args = dict(
        sampler_kwargs=dict(
            n_effective=n_effective,
            n_active=n_active,
            n_total=n_total,
            n_evidence=512,
            progress=False,
        ),
    )

    def setup_method(self) -> None:
        super().setup_method()
        self.n_workers = 12
        # (n_active, n_effective) -> balanced to n_workers
        self.balanced_particle_numbers = {
            (30, 64): (24, 51),
            (256, 512): (252, 504),
            (5, 10): (12, 24),
            (24, 48): (24, 48),
        }
        self.changed_x_range = (-2.0, 2.0)
        self.runs["changed_prior"] = dict(
            run=self.run_under_mpi,
            starts_from="stopped",
            prior=self.build_prior(self.changed_x_range),
        )
        # pocomc_bilby's settings that the sampler controls itself
        self.enforced_run_settings = ("save_every", "resume_state_path")
        # settings of pocomc.Sampler that the sampler passes itself
        self.call_site_init_keys = (
            "n_dim",
            "periodic",
            "reflective",
            "output_dir",
            "output_label",
        )

    def particle_numbers(self, sampler: PocoMC) -> tuple[int, int]:
        kwargs = sampler.external_init_kwargs
        return kwargs["n_active"], kwargs["n_effective"]

    def build_with_particle_numbers(
        self, n_active: int | None, n_effective: int | None, **sampler_overrides
    ) -> PocoMC:
        particle_numbers = dict(n_active=n_active, n_effective=n_effective)
        return self.build_sampler(
            sampler_overrides=particle_numbers | sampler_overrides
        )

    def test_external_init_kwargs(self) -> None:
        init_kwargs = self.sampler.external_init_kwargs
        assert self.particle_numbers(self.sampler) == (self.n_active, self.n_effective)
        assert init_kwargs["random_state"] is None
        for key in [*self.deviations_from_bilby, *self.call_site_init_keys]:
            assert key not in init_kwargs

    def test_boundaries(self) -> None:
        periodic = self.sampler._get_pocomc_boundaries("periodic")
        reflective = self.sampler._get_pocomc_boundaries("reflective")
        assert periodic == [self.parameter_names.index("phi")]
        assert reflective == [self.parameter_names.index("psi")]

    def test_external_run_kwargs(self) -> None:
        run_kwargs = self.sampler.external_run_kwargs
        assert run_kwargs["n_total"] == self.n_total
        for key in self.enforced_run_settings:
            assert key not in run_kwargs

    def test_init_sampler_kwargs_derives_missing_particle_number(self) -> None:
        derived_n_active = self.build_with_particle_numbers(None, self.n_effective)
        assert self.particle_numbers(derived_n_active) == (
            self.n_effective // 2,
            self.n_effective,
        )
        derived_n_effective = self.build_with_particle_numbers(self.n_active, None)
        assert self.particle_numbers(derived_n_effective) == (
            self.n_active,
            2 * self.n_active,
        )

    def test_restore_random_states_includes_torch(self) -> None:
        states = self.sampler.random_states()
        draws = torch.rand(self.n_draws)
        self.build_sampler().restore_random_states(states)
        assert torch.equal(torch.rand(self.n_draws), draws)

    def test_balance_to_pool_keeps_particle_numbers_by_default(self) -> None:
        for given in self.balanced_particle_numbers:
            sampler = self.build_with_particle_numbers(*given)
            sampler.balance_to_pool(self.n_workers)
            assert self.particle_numbers(sampler) == given

    def test_balance_to_pool_balances_particle_numbers(self) -> None:
        for given, balanced in self.balanced_particle_numbers.items():
            sampler = self.build_with_particle_numbers(*given, balance_particles=True)
            sampler.balance_to_pool(self.n_workers)
            assert self.particle_numbers(sampler) == balanced, given

    @needs_mpirun
    def test_resume_rejects_changed_prior(self) -> None:
        completed = self.completed_run("changed_prior")
        assert "different prior" in completed.stderr
        assert not self.result_file("changed_prior").is_file()
        # the failure must also reach the scheduler, via mpirun's exit code
        assert completed.returncode != 0
