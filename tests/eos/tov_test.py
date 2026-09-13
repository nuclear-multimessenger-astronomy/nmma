import lal
import numpy as np
import pytest
import scipy.constants

from nmma.eos import tov
from nmma.eos.eos_gen import EOS_with_CSE


def polytropic_low_density_eos(K=10.0, gamma=2.0, n_max=0.3, num=200):
    """A smooth analytic stand-in for a tabulated crust-plus-outer-core EOS.

    Units follow the EOS module's convention: number density in fm^-3,
    pressure and energy density in MeV fm^-3.
    """
    n = np.logspace(np.log10(1e-3), np.log10(n_max), num)
    energy_density = n * 939.565 + K * np.power(n, gamma)
    pressure = K * (gamma - 1.0) * np.power(n, gamma)
    return dict(n=n, p=pressure, e=energy_density)


class AnalyticEoS:
    """Minimal EOS interface exercised by tov_ode.

    Energy density and pressure are simple power laws in the pseudo
    enthalpy, which keeps the log-log slope that tov_ode asks for
    constant and known.
    """

    def __init__(self, e_scale=500.0, p_scale=50.0, e_index=1.5, p_index=2.5):
        self.e_scale = e_scale
        self.p_scale = p_scale
        self.e_index = e_index
        self.p_index = p_index

    def energy_density_from_pseudo_enthalpy(self, h):
        return self.e_scale * np.power(h, self.e_index)

    def pressure_from_pseudo_enthalpy(self, h):
        return self.p_scale * np.power(h, self.p_index)

    def log_dedp_from_log_pressure(self, log_p):
        return self.e_index / self.p_index


class TestUnitConversions:
    """The module converts from particle-physics units (MeV fm^-3) to
    geometric units, which is the only place those factors are defined."""

    def test_particle_to_si(self):
        assert tov.particle_to_SI == scipy.constants.e * 1e51

    def test_si_to_geometric(self):
        expected = scipy.constants.G / np.power(scipy.constants.c, 4.0)
        assert tov.SI_to_geometric == expected

    def test_particle_to_geometric_is_the_composition_of_the_two(self):
        assert (
            tov.particle_to_geometric == tov.particle_to_SI * tov.SI_to_geometric
        )

    def test_geometric_conversion_has_the_expected_order_of_magnitude(self):
        # 1 MeV fm^-3 is about 1.3e-12 m^-2 in geometric units.
        assert np.log10(tov.particle_to_geometric) == pytest.approx(-11.88, abs=1.5 * 10**-2)


class TestCalcK2:
    """calc_k2 turns the metric perturbation at the surface into the
    dimensionless tidal Love number, so it may only depend on the
    compactness M/R and on y = R b / H."""

    def test_depends_only_on_y_and_not_on_the_scale_of_h_and_b(self):
        reference = tov.calc_k2(1.0, 0.2, 1.0, 2.0)
        for factor in [0.5, 3.0, 100.0]:
            assert tov.calc_k2(1.0, 0.2, factor * 1.0, factor * 2.0) == pytest.approx(reference)

    def test_depends_only_on_the_compactness_and_not_on_the_radius(self):
        reference = tov.calc_k2(1.0, 0.2, 1.0, 2.0)
        # R and M doubled keeps C, and b/H is rescaled to keep y
        assert tov.calc_k2(2.0, 0.4, 4.0, 4.0) == pytest.approx(reference)

    def test_is_positive_over_the_neutron_star_range_of_compactness(self):
        for compactness in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            k2 = tov.calc_k2(1.0, compactness, 1.0, 2.0)
            assert k2 > 0.0, f"C={compactness}"
            assert k2 < 1.0, f"C={compactness}"

    def test_accepts_array_input(self):
        compactness = np.array([0.1, 0.2, 0.3])
        k2 = tov.calc_k2(1.0, compactness, 1.0, 2.0)
        assert k2.shape == compactness.shape
        assert np.all(np.isfinite(k2))

    def test_matches_the_scalar_result_elementwise(self):
        compactness = np.array([0.1, 0.25])
        k2 = tov.calc_k2(1.0, compactness, 1.0, 2.0)
        assert k2[0] == pytest.approx(tov.calc_k2(1.0, 0.1, 1.0, 2.0))
        assert k2[1] == pytest.approx(tov.calc_k2(1.0, 0.25, 1.0, 2.0))


class TestTovOde:
    """The ODE is integrated inwards in pseudo enthalpy, so the radius,
    the enclosed mass and the perturbation all decrease as h grows."""

    def setup_method(self):
        self.eos = AnalyticEoS()
        self.h = 0.2
        self.state = [8.0e3, 1.5e3, 6.4e7, 1.6e4]

    def test_returns_four_derivatives(self):
        dydt = tov.tov_ode(self.h, self.state, self.eos)
        assert len(dydt) == 4
        assert np.all(np.isfinite(dydt))

    def test_radius_and_mass_decrease_with_increasing_pseudo_enthalpy(self):
        drdh, dmdh, _, _ = tov.tov_ode(self.h, self.state, self.eos)
        assert drdh < 0.0
        assert dmdh < 0.0

    def test_mass_derivative_follows_the_continuity_equation(self):
        r = self.state[0]
        drdh, dmdh, _, _ = tov.tov_ode(self.h, self.state, self.eos)
        energy_density = (
            self.eos.energy_density_from_pseudo_enthalpy(self.h)
            * tov.particle_to_geometric
        )
        assert dmdh / (4.0 * np.pi * r * r * energy_density * drdh) == pytest.approx(1.0)

    def test_metric_perturbation_derivative_is_b_times_drdh(self):
        b = self.state[3]
        drdh, _, dHdh, _ = tov.tov_ode(self.h, self.state, self.eos)
        assert dHdh == pytest.approx(b * drdh)

    def test_a_real_eos_object_can_be_integrated_at_its_centre(self):
        eos = EOS_with_CSE(polytropic_low_density_eos(), n_lim=1.0, N_seg=3)
        hc = eos.pseudo_enthalpy_from_pressure(30.0)
        state = [1.0e3, 1.0e2, 1.0e6, 2.0e3]
        dydt = tov.tov_ode(hc, state, eos)
        assert np.all(np.isfinite(dydt))


class TestTOVSolver:
    """End-to-end solve of one stellar model on a synthetic but physically
    reasonable EOS. Masses come back in geometric metres, so lal.MRSUN_SI
    and 1e3 convert to solar masses and kilometres."""

    @classmethod
    def setup_class(cls):
        cls.eos = EOS_with_CSE(
            polytropic_low_density_eos(), n_connect=0.16, n_lim=1.0, N_seg=3, seed=42
        )

    def solve(self, central_pressure):
        mass, radius, k2 = tov.TOVSolver(self.eos, central_pressure)
        return mass / lal.MRSUN_SI, radius / 1e3, k2, mass / radius

    def test_returns_a_neutron_star_of_plausible_mass_and_radius(self):
        mass, radius, k2, compactness = self.solve(30.0)
        assert 1.0 < mass < 2.0, f"mass {mass}"
        assert 8.0 < radius < 16.0, f"radius {radius}"
        assert 0.0 < k2 < 0.5, f"k2 {k2}"
        assert compactness < 0.5

    def test_mass_and_radius_grow_along_the_stable_branch(self):
        masses, radii = [], []
        for central_pressure in [10.0, 20.0, 30.0, 50.0]:
            mass, radius, _, _ = self.solve(central_pressure)
            masses.append(mass)
            radii.append(radius)
        assert np.all(np.diff(masses) > 0.0), str(masses)
        assert np.all(np.diff(radii) > 0.0), str(radii)

    def test_love_number_decreases_as_the_star_becomes_more_compact(self):
        k2s = [self.solve(p)[2] for p in [10.0, 30.0, 50.0, 80.0]]
        assert np.all(np.diff(k2s) < 0.0), str(k2s)

    def test_tidal_deformability_is_in_the_range_probed_by_gw170817(self):
        _, _, k2, compactness = self.solve(50.0)
        lambda_tidal = 2.0 / 3.0 * k2 * np.power(compactness, -5.0)
        assert 10.0 < lambda_tidal < 5000.0, f"lambda {lambda_tidal}"

    def test_the_solution_is_deterministic(self):
        first = tov.TOVSolver(self.eos, 30.0)
        second = tov.TOVSolver(self.eos, 30.0)
        np.testing.assert_allclose(first, second)
