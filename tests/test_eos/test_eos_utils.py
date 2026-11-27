from argparse import Namespace

import numpy as np
import pytest
from helpers import EOS_MACRO_DIR, N_EOS_MACRO, eos_macro_tables

from nmma.core.conversion import DisjointMassGrid, MassGrid
from nmma.eos.eos_processing import TabulatedEoSGenerator
from nmma.eos.eos_utils import TOVDataset


def padded_eos_tables() -> tuple[np.ndarray, ...]:
    """(masses, radii, lambdas) of the test EOSs, each of shape
    (N_EOS_MACRO, longest table), NaN-padded behind the shorter tables."""
    tables = eos_macro_tables(log_lambda=False)
    length = max(table.shape[1] for table in tables)
    radii, masses, lambdas = (
        np.array(
            [
                np.pad(t[col], (0, length - len(t[col])), constant_values=np.nan)
                for t in tables
            ]
        )
        for col in range(3)
    )
    return masses, radii, lambdas


class TestTOVDataset:
    """The solver curves of the tabulated test EOSs. They differ in length, so the
    shorter ones are NaN-padded, and every table ends at its TOV mass."""

    ref_masses = np.array([1.4, 1.8, 2.2])
    float_precision = 1e-12
    # masses between table nodes, for linear interpolation in sqrt(M_tov - m)
    between_nodes_masses = np.array([1.4, 1.8])

    def setup_method(self) -> None:
        self.masses, self.radii, self.lambdas = padded_eos_tables()
        self.tov_masses = np.nanmax(self.masses, axis=1)
        self.params = np.arange(N_EOS_MACRO, dtype=float)[:, None]
        self.dataset = TOVDataset(self.params, self.masses, self.radii, self.lambdas)
        self.grid = MassGrid(interpolation="quadratic")

    def test_len_and_subset(self) -> None:
        assert len(self.dataset) == N_EOS_MACRO
        subset = self.dataset.subset([1, 4])
        assert len(subset) == 2
        assert subset.params == pytest.approx(self.params[[1, 4]])
        assert subset.mass[1] == pytest.approx(self.masses[4], nan_ok=True)

    def test_stable_branch_drops_padding(self) -> None:
        for i in range(N_EOS_MACRO):
            masses, radii, lambdas = self.dataset.stable_branch(i)
            valid = np.isfinite(self.masses[i])
            assert masses == pytest.approx(self.masses[i][valid])
            assert radii == pytest.approx(self.radii[i][valid])
            assert lambdas == pytest.approx(self.lambdas[i][valid])

    def test_stable_branch_cuts_at_the_maximum_mass(self) -> None:
        "An unstable branch behind the maximum mass is dropped"
        n = np.isfinite(self.masses[0]).sum()
        masses, radii, lambdas = (
            np.append(a[0, :n], a[0, n - 2])[None]
            for a in (self.masses, self.radii, self.lambdas)
        )
        dataset = TOVDataset(self.params[:1], masses, radii, lambdas)
        branch_masses, _, _ = dataset.stable_branch(0)
        assert branch_masses == pytest.approx(self.masses[0, :n])

    def test_stable_branch_rejects_non_ascending_masses(self) -> None:
        masses = self.masses.copy()
        masses[0, 1] = masses[0, 0]
        dataset = TOVDataset(self.params, masses, self.radii, self.lambdas)
        assert dataset.stable_branch(0) is None
        assert dataset.stable_branch(1) is not None

    def test_reference_values_reproduce_the_table_nodes(self) -> None:
        for i in range(N_EOS_MACRO):
            valid = np.isfinite(self.masses[i])
            node_masses = self.masses[i][valid]
            radii, ln_lambdas = self.dataset.subset([i]).reference_values(node_masses)
            assert radii[0] == pytest.approx(self.radii[i][valid])
            assert ln_lambdas[0] == pytest.approx(np.log(self.lambdas[i][valid]))

    def test_reference_values_are_nan_outside_the_stable_branch(self) -> None:
        radii, ln_lambdas = self.dataset.reference_values(self.ref_masses)
        min_masses = np.nanmin(self.masses, axis=1)
        below = self.dataset.reference_values([0.5 * min_masses.min()])
        assert np.all(np.isnan(below[0])) and np.all(np.isnan(below[1]))
        above_tov = self.ref_masses[None, :] > self.tov_masses[:, None]
        assert np.isnan(radii).tolist() == above_tov.tolist()
        assert np.isnan(ln_lambdas).tolist() == above_tov.tolist()

    def test_prepare_targets_selects_by_tov_mass(self) -> None:
        m_tov_range = (2.2, 3.0)
        good, targets = self.dataset.prepare_targets(self.grid, m_tov_range)
        in_range = (self.tov_masses >= m_tov_range[0]) & (
            self.tov_masses <= m_tov_range[1]
        )
        assert 0 < len(good) < N_EOS_MACRO
        assert good.tolist() == np.flatnonzero(in_range).tolist()
        assert self.dataset.rejected["M_tov outside range"] == N_EOS_MACRO - len(good)
        assert targets[:, -1] == pytest.approx(self.tov_masses[good])

    def test_prepare_targets_rejects_curves_starting_above_m_min(self) -> None:
        grid = MassGrid(m_min=0.5 * np.nanmin(self.masses))
        good, targets = self.dataset.prepare_targets(grid)
        assert len(good) == 0 and len(targets) == 0
        assert (
            self.dataset.rejected[f"curve starts above m_min={grid.m_min}"]
            == N_EOS_MACRO
        )

    def test_prepare_targets_resamples_onto_the_grid(self) -> None:
        "The targets are the reference values at the grid masses of each EOS"
        good, targets = self.dataset.prepare_targets(self.grid)
        assert good.tolist() == list(range(N_EOS_MACRO))
        n = self.grid.n_points
        for i in good:
            grid_masses = self.grid.masses(self.tov_masses[i])
            radii, ln_lambdas = self.dataset.subset([i]).reference_values(grid_masses)
            assert targets[i, :n] == pytest.approx(radii[0])
            assert targets[i, n : 2 * n] == pytest.approx(ln_lambdas[0])
            assert targets[i, -1] == self.tov_masses[i]

    def test_targets_match_the_tabulated_generator(self) -> None:
        generator = TabulatedEoSGenerator(Namespace(eos_data=EOS_MACRO_DIR), self.grid)
        _, targets = self.dataset.prepare_targets(self.grid)
        n = self.grid.n_points
        curves = generator.resampled_curves
        assert targets[:, :n] == pytest.approx(curves[:, 0])
        assert targets[:, n : 2 * n] == pytest.approx(curves[:, 1])
        assert targets[:, -1] == pytest.approx(generator.m_tov)

    def test_vs_generator(self) -> None:
        """The tabulated generator reproduces its raw tables exactly at the masses of
        its grid that are shared by all EOSs, i.e. below the split value of a
        disjoint grid, and has their TOV masses"""
        grid = DisjointMassGrid(
            n_low=11, n_high=10, split_value=2.0, interpolation="quadratic"
        )
        generator = TabulatedEoSGenerator(Namespace(eos_data=EOS_MACRO_DIR), grid)
        shared_masses = grid.masses(self.tov_masses[0])[: grid.n_low]
        dataset = generator.reference_dataset()

        characteristics = dataset.test_vs_generator(generator, shared_masses)
        assert characteristics["m_tov_pred"] == pytest.approx(self.tov_masses)
        assert characteristics["m_tov"] == pytest.approx(self.tov_masses)
        assert characteristics["summary"]["m_tov"]["max_abs"] == pytest.approx(
            0.0, abs=self.float_precision
        )
        for mass in shared_masses:
            for obs in ("rad", "lam"):
                abs_rel = characteristics["abs_rel"][str(mass)][obs]
                assert abs_rel == pytest.approx(0.0, abs=self.float_precision)
