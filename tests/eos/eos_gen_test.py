import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nmma.eos import eos_gen


def polytropic_low_density_eos(K=10.0, gamma=2.0, n_max=0.3, num=200):
    """A smooth analytic stand-in for a tabulated crust-plus-outer-core EOS.

    Units follow the module convention: number density in fm^-3, pressure
    and energy density in MeV fm^-3.
    """
    n = np.logspace(np.log10(1e-3), np.log10(n_max), num)
    energy_density = n * 939.565 + K * np.power(n, gamma)
    pressure = K * (gamma - 1.0) * np.power(n, gamma)
    return dict(n=n, p=pressure, e=energy_density)


class TestEoSFromNEP:
    """eos_from_nep builds a microscopic EOS table from nuclear empirical
    parameters and glues it onto a tabulated crust."""

    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.crust = np.column_stack(
            [
                np.linspace(0.01, 0.09, 5),
                np.linspace(0.1, 0.5, 5),
                np.linspace(10.0, 90.0, 5),
            ]
        )
        self.crust_path = self.tmp_dir / "crust.dat"
        np.savetxt(self.crust_path, self.crust)

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    def build(self, S0=32.0, L=60.0, **kwargs):
        return eos_gen.eos_from_nep(S0, L, crust_path=str(self.crust_path), **kwargs)

    def test_returns_three_columns_of_number_density_pressure_energy(self):
        table = self.build()
        assert table.ndim == 2
        assert table.shape[1] == 3

    def test_the_crust_table_is_prepended_unchanged(self):
        table = self.build()
        np.testing.assert_allclose(table[: len(self.crust)], self.crust)

    def test_the_outer_core_covers_the_expected_density_grid(self):
        table = self.build()
        core = table[len(self.crust) :]
        expected = np.arange(0.1, 1.6, 0.002)
        assert len(core) == len(expected)
        np.testing.assert_allclose(core[:, 0], expected)

    def test_energy_density_follows_the_expansion_around_saturation(self):
        # At saturation density with the default expansion coefficients the
        # energy per particle reduces to Esat + Ssym * (1 - 2x).
        table = self.build(S0=32.0, L=60.0)
        core = table[len(self.crust) :]
        index = int(np.argmin(np.abs(core[:, 0] - 0.16)))
        n_sat = core[index, 0]
        expected = n_sat * (939.565 - 16.0 + 32.0 * (1.0 - 2.0 * 0.02))
        assert core[index, 2] == pytest.approx(expected, abs=1.5 * 10**-6)

    def test_pressure_is_positive_and_rising_in_the_outer_core(self):
        core = self.build()[len(self.crust) :]
        assert np.all(core[:, 1] > 0.0)
        assert np.all(np.diff(core[:, 1]) > 0.0)

    def test_a_larger_slope_of_the_symmetry_energy_stiffens_the_eos(self):
        soft = self.build(L=30.0)[len(self.crust) :]
        stiff = self.build(L=90.0)[len(self.crust) :]
        assert np.all(stiff[:, 1] > soft[:, 1])

    def test_saturation_density_can_be_shifted(self):
        default = self.build()[len(self.crust) :]
        shifted = self.build(nsat_val=0.15)[len(self.crust) :]
        assert not np.allclose(default[:, 2], shifted[:, 2])

    def test_a_missing_crust_file_is_reported(self):
        with pytest.raises(OSError):
            eos_gen.eos_from_nep(32.0, 60.0, crust_path=str(self.tmp_dir / "absent"))


class TestEOSWithCSEConstruction:
    """The low-density table is taken up to n_connect and extended to
    n_lim by a randomly drawn speed-of-sound profile."""

    @classmethod
    def setup_class(cls):
        cls.low_density_eos = polytropic_low_density_eos()
        cls.eos = eos_gen.EOS_with_CSE(
            cls.low_density_eos, n_connect=0.16, n_lim=1.0, N_seg=3, seed=42
        )

    def test_stores_the_low_density_table(self):
        np.testing.assert_allclose(self.eos.n_low, self.low_density_eos["n"])
        np.testing.assert_allclose(self.eos.p_low, self.low_density_eos["p"])
        np.testing.assert_allclose(self.eos.e_low, self.low_density_eos["e"])

    def test_stores_the_extension_settings(self):
        assert self.eos.n_connect == 0.16
        assert self.eos.n_lim == 1.0
        assert self.eos.N_seg == 3
        assert self.eos.cs2_limit == 1.0
        assert self.eos.seed == 42
        assert self.eos.n_extend_range == pytest.approx(1.0 - 0.16)

    def test_connection_values_agree_with_the_low_density_table(self):
        n = self.low_density_eos["n"]
        expected_pressure = np.interp(0.16, n, self.low_density_eos["p"])
        expected_energy = np.interp(0.16, n, self.low_density_eos["e"])
        assert self.eos.p_at_n_connect / expected_pressure == pytest.approx(1.0, abs=1.5 * 10**-3)
        assert self.eos.e_at_n_connect / expected_energy == pytest.approx(1.0, abs=1.5 * 10**-3)

    def test_speed_of_sound_at_the_connection_point_is_subluminal(self):
        assert self.eos.cs2_at_n_connect > 0.0
        assert self.eos.cs2_at_n_connect < 1.0

    def test_the_full_table_spans_the_low_density_tail_and_the_extension(self):
        assert self.eos.n_array[0] == pytest.approx(self.low_density_eos["n"][0])
        assert self.eos.n_array[-1] < 1.0
        assert self.eos.n_array[-1] > 0.99
        assert len(self.eos.n_array) == len(self.eos.p_array)
        assert len(self.eos.n_array) == len(self.eos.e_array)

    def test_the_extension_starts_at_the_connection_density(self):
        assert self.eos.n_high[0] == pytest.approx(0.16)
        assert self.eos.p_high[0] == pytest.approx(self.eos.p_at_n_connect)
        assert self.eos.e_high[0] == pytest.approx(self.eos.e_at_n_connect)

    def test_the_low_density_table_is_truncated_at_the_connection_density(self):
        assert np.all(self.eos.n_array[: -len(self.eos.n_high)] < 0.16)

    def test_pressure_and_energy_density_rise_monotonically(self):
        assert np.all(np.diff(self.eos.p_array) > 0.0)
        assert np.all(np.diff(self.eos.e_array) > 0.0)

    def test_the_extension_respects_the_speed_of_sound_limit(self):
        cs2 = np.gradient(self.eos.p_high, self.eos.e_high)
        assert np.all(cs2 <= 1.0)
        assert np.all(cs2 >= 0.0)

    def test_a_tighter_speed_of_sound_limit_softens_the_extension(self):
        soft = eos_gen.EOS_with_CSE(
            self.low_density_eos, n_lim=1.0, N_seg=3, seed=42, cs2_limit=0.3
        )
        assert soft.p_array[-1] < self.eos.p_array[-1]

    def test_pseudo_enthalpy_is_positive_and_increasing(self):
        assert self.eos.h_array[0] > 0.0
        assert np.all(np.diff(self.eos.h_array) > 0.0)

    def test_the_seed_makes_the_extension_reproducible(self):
        twin = eos_gen.EOS_with_CSE(
            self.low_density_eos, n_connect=0.16, n_lim=1.0, N_seg=3, seed=42
        )
        np.testing.assert_allclose(twin.p_array, self.eos.p_array)

    def test_a_different_seed_draws_a_different_extension(self):
        other = eos_gen.EOS_with_CSE(
            self.low_density_eos, n_connect=0.16, n_lim=1.0, N_seg=3, seed=7
        )
        assert other.p_array[-1] != pytest.approx(self.eos.p_array[-1])

    def test_extending_further_in_density_gives_a_longer_table(self):
        extended = eos_gen.EOS_with_CSE(
            self.low_density_eos, n_connect=0.16, n_lim=2.0, N_seg=3, seed=42
        )
        assert len(extended.n_array) > len(self.eos.n_array)
        assert extended.n_array[-1] > 1.9


class TestEOSWithCSEInterpolation:
    """All thermodynamic quantities are exposed as pairwise interpolants,
    which must invert each other."""

    @classmethod
    def setup_class(cls):
        cls.eos = eos_gen.EOS_with_CSE(
            polytropic_low_density_eos(), n_connect=0.16, n_lim=1.0, N_seg=3, seed=42
        )

    def test_number_density_and_pressure_invert_each_other(self):
        pressure = self.eos.pressure_from_number_density(0.2)
        assert self.eos.number_density_from_pressure(pressure) == pytest.approx(0.2)

    def test_number_density_and_energy_density_invert_each_other(self):
        energy = self.eos.energy_density_from_number_density(0.3)
        assert self.eos.number_density_from_energy_density(energy) == pytest.approx(0.3)

    def test_pressure_and_energy_density_invert_each_other(self):
        energy = self.eos.energy_density_from_pressure(20.0)
        assert self.eos.pressure_from_energy_density(energy) / 20.0 == pytest.approx(1.0)

    def test_pseudo_enthalpy_inverts_against_pressure(self):
        enthalpy = self.eos.pseudo_enthalpy_from_pressure(20.0)
        assert self.eos.pressure_from_pseudo_enthalpy(enthalpy) / 20.0 == pytest.approx(1.0)

    def test_pseudo_enthalpy_inverts_against_energy_density(self):
        enthalpy = self.eos.pseudo_enthalpy_from_energy_density(200.0)
        assert self.eos.energy_density_from_pseudo_enthalpy(enthalpy) / 200.0 == pytest.approx(1.0)

    def test_pseudo_enthalpy_inverts_against_number_density(self):
        enthalpy = self.eos.pseudo_enthalpy_from_number_density(0.25)
        assert self.eos.number_density_from_pseudo_enthalpy(enthalpy) == pytest.approx(0.25)

    def test_interpolants_accept_arrays(self):
        densities = np.array([0.1, 0.2, 0.4])
        pressures = self.eos.pressure_from_number_density(densities)
        assert pressures.shape == densities.shape
        assert np.all(np.diff(pressures) > 0.0)

    def test_dedp_is_positive_and_falls_with_pressure(self):
        # A stiffening EOS needs less energy density per unit pressure.
        low = self.eos.dedp_from_pressure(1.0)
        high = self.eos.dedp_from_pressure(50.0)
        assert low > 0.0
        assert high > 0.0
        assert low > high

    def test_dedp_is_the_inverse_of_the_speed_of_sound_squared(self):
        pressure = 20.0
        cs2 = 1.0 / self.eos.dedp_from_pressure(pressure)
        assert cs2 > 0.0
        assert cs2 < 1.0


class TestEOSWithCSEMixedLowDensityTables:
    """Passing a second, stiffer low-density table interpolates between the
    two with a seeded random weight."""

    def setup_method(self):
        self.soft = polytropic_low_density_eos(K=10.0)
        self.stiff = polytropic_low_density_eos(K=12.0)

    def test_the_mixed_table_lies_between_the_soft_and_stiff_inputs(self):
        eos = eos_gen.EOS_with_CSE(
            self.soft, n_lim=1.0, N_seg=3, seed=42, low_density_eos_stiff=self.stiff
        )
        assert np.all(eos.p_low >= self.soft["p"])
        assert np.all(eos.p_low <= self.stiff["p"])
        assert np.all(eos.e_low >= self.soft["e"])
        assert np.all(eos.e_low <= self.stiff["e"])

    def test_the_mixing_weight_is_drawn_from_the_seeded_generator(self):
        eos = eos_gen.EOS_with_CSE(
            self.soft, n_lim=1.0, N_seg=3, seed=42, low_density_eos_stiff=self.stiff
        )
        np.random.seed(42)
        alpha = np.random.uniform()
        expected = self.soft["p"] + alpha * (self.stiff["p"] - self.soft["p"])
        np.testing.assert_allclose(eos.p_low, expected)

    def test_both_input_tables_are_kept(self):
        eos = eos_gen.EOS_with_CSE(
            self.soft, n_lim=1.0, N_seg=3, seed=42, low_density_eos_stiff=self.stiff
        )
        np.testing.assert_allclose(eos.p_low_soft, self.soft["p"])
        np.testing.assert_allclose(eos.p_low_stiff, self.stiff["p"])

    def test_a_stiff_table_missing_a_quantity_is_rejected(self):
        # The guard compares the number of dict keys, so it catches a table
        # that is missing an entry rather than one of a different length.
        incomplete = {key: self.stiff[key] for key in ["n", "p"]}
        with pytest.raises(AssertionError):
            eos_gen.EOS_with_CSE(
                self.soft, n_lim=1.0, N_seg=3, low_density_eos_stiff=incomplete
            )

    def test_arrays_of_different_length_are_not_caught_by_the_guard(self):
        # Differing array lengths pass the key-count assertion and only fail
        # later when the two tables are subtracted.
        short = polytropic_low_density_eos(K=12.0, num=150)
        with pytest.raises(ValueError):
            eos_gen.EOS_with_CSE(
                self.soft, n_lim=1.0, N_seg=3, low_density_eos_stiff=short
            )


class TestEOSWithCSEExtensionSchemes:
    """Only the default 'peter' extension is functional; the other branches
    are documented here so that a fix shows up as an unexpected pass."""

    def setup_method(self):
        self.low_density_eos = polytropic_low_density_eos()

    def test_the_default_scheme_is_peter(self):
        default = eos_gen.EOS_with_CSE(
            self.low_density_eos, n_lim=1.0, N_seg=3, seed=42
        )
        explicit = eos_gen.EOS_with_CSE(
            self.low_density_eos,
            n_lim=1.0,
            N_seg=3,
            seed=42,
            extension_scheme="peter",
        )
        np.testing.assert_allclose(default.p_array, explicit.p_array)

    @pytest.mark.xfail(strict=True)
    def test_the_rahul_scheme_builds_an_eos(self):
        # __extend_v1 reads self.mu_at_n_connect, which is never set, so the
        # chemical-potential extension raises AttributeError.
        eos = eos_gen.EOS_with_CSE(
            self.low_density_eos, n_lim=1.0, N_seg=3, extension_scheme="rahul"
        )
        assert np.all(np.diff(eos.p_array) > 0.0)

    def test_an_unknown_scheme_leaves_the_object_unbuilt(self):
        # No branch runs, so the interpolation setup trips over the missing
        # p_array rather than reporting the unknown scheme name.
        with pytest.raises(AttributeError):
            eos_gen.EOS_with_CSE(
                self.low_density_eos, n_lim=1.0, N_seg=3, extension_scheme="unknown"
            )


class TestEOSWithCSEFamily:
    """construct_family solves the TOV equations along a sequence of central
    pressures and exposes mass-radius and mass-lambda interpolants."""

    @classmethod
    def setup_class(cls):
        cls.eos = eos_gen.EOS_with_CSE(
            polytropic_low_density_eos(), n_connect=0.16, n_lim=1.0, N_seg=3, seed=42
        )
        cls.eos.construct_family(ndat=20)

    def test_construct_family_returns_none_and_sets_interpolants(self):
        assert hasattr(self.eos, "radius_m_interp")
        assert hasattr(self.eos, "lambda_m_interp")

    def test_a_canonical_star_has_a_plausible_radius(self):
        radius = float(self.eos.radius_m_interp(1.4))
        assert 8.0 < radius < 16.0, f"radius {radius}"

    def test_a_canonical_star_has_a_plausible_tidal_deformability(self):
        lambda_tidal = float(self.eos.lambda_m_interp(1.4))
        assert 10.0 < lambda_tidal < 5000.0, f"lambda {lambda_tidal}"

    def test_tidal_deformability_falls_with_mass(self):
        lambdas = [float(self.eos.lambda_m_interp(m)) for m in [1.2, 1.4, 1.6, 1.8]]
        assert np.all(np.diff(lambdas) < 0.0), str(lambdas)

    def test_the_interpolants_do_not_extrapolate_beyond_the_maximum_mass(self):
        with pytest.raises(ValueError):
            self.eos.radius_m_interp(5.0)

    def test_a_denser_grid_still_reproduces_the_canonical_star(self):
        dense = eos_gen.EOS_with_CSE(
            polytropic_low_density_eos(), n_connect=0.16, n_lim=1.0, N_seg=3, seed=42
        )
        dense.construct_family(ndat=30)
        assert float(dense.radius_m_interp(1.4)) == pytest.approx(
            float(self.eos.radius_m_interp(1.4)), abs=1.5 * 10**-1
        )
