import numpy as np
import pytest
from astropy import constants as const
from astropy import cosmology
from astropy import units as u
from bilby.gw import cosmology as bilby_cosmo

from nmma.core import constants


class TestFundamentalConstants:
    """The module's stated contract is that it defers to astropy.constants,
    so each constant is checked against the astropy value it claims to be
    rather than against a hard-coded number."""

    def test_solar_mass_and_speed_of_light_in_cgs(self):
        assert constants.msun_cgs == const.M_sun.cgs.value
        assert constants.c_cgs == const.c.cgs.value

    def test_speed_of_light_units_are_consistent(self):
        assert constants.c_SI == const.c.si.value
        assert constants.c_kms == pytest.approx(constants.c_SI / 1000.0)
        assert constants.c_cgs == pytest.approx(constants.c_SI * 100.0, abs=1.0)

    def test_planck_and_boltzmann_constants(self):
        assert constants.h == const.h.cgs.value
        assert constants.kb == const.k_B.cgs.value

    def test_gravitational_constant_in_neutron_star_units(self):
        expected = const.G.to(u.km**3 / u.solMass / u.second**2).value
        assert constants.G_in_ns_units == expected

    def test_seconds_a_day(self):
        assert constants.seconds_a_day == 86400


class TestDistanceConstants:
    def test_megaparsec_in_cm(self):
        assert constants.Mpc == pytest.approx(const.pc.cgs.value * 1e6)

    def test_reference_distance_is_ten_parsec(self):
        assert constants.D == pytest.approx(10 * const.pc.cgs.value)
        assert constants.Mpc / constants.D == pytest.approx(1e5)


class TestDerivedConstants:
    def test_radiation_constant_follows_from_stefan_boltzmann(self):
        assert constants.sigSB == const.sigma_sb.cgs.value
        assert constants.arad == pytest.approx(4 * constants.sigSB / constants.c_cgs)

    def test_electron_volt_per_planck_constant(self):
        assert constants.eV_per_h_SI == pytest.approx(const.e.si.value / const.h.si.value)

    def test_geometrised_solar_mass_in_km(self):
        # The comment in the module quotes this value explicitly.
        assert constants.geom_msun_km == pytest.approx(1.476625038050125, abs=1.5 * 10**(-9))

    def test_solar_mass_energy_in_ergs(self):
        expected = (const.M_sun * const.c**2).cgs.value
        assert constants.msun_to_ergs / expected == pytest.approx(1.0)

    def test_mev_per_fm3_conversion_matches_documented_value(self):
        # The module comment states 1 MeV/fm**3 is 8.9653E-7 Msun/km**3.
        assert constants.MeV_per_fm3_to_Msun_per_km3 == pytest.approx(8.9653e-7, abs=1.5 * 10**(-11))

    def test_proton_mass_in_solar_masses(self):
        assert float(constants.particle_mass) == pytest.approx(const.m_p.value / const.M_sun.value)


class TestPulsarTimingConstants:
    def test_geometrised_solar_mass_in_seconds(self):
        expected = (const.M_sun * const.G / const.c**3).value
        assert constants.msun_s == expected
        # roughly 4.93 microseconds, the standard value quoted in the literature
        assert constants.msun_s * 1e6 == pytest.approx(4.9254909, abs=1.5 * 10**(-5))

    def test_microsecond_variant_is_scaled_consistently(self):
        assert constants.msun_mus == pytest.approx(constants.msun_s * 1e6)

    def test_einstein_factor(self):
        assert constants.einstein_factor == pytest.approx(constants.msun_s ** (2 / 3))


class TestCosmology:
    """set_cosmology/get_cosmology wrap bilby's global cosmology, so the
    module-level default has to be restored after every test to keep the
    rest of the suite independent of test ordering."""

    def setup_method(self):
        self.original = constants.get_cosmology()

    def teardown_method(self):
        constants.set_cosmology(self.original)

    def test_default_cosmology_is_planck18(self):
        assert constants.default_cosmology is cosmology.Planck18
        assert constants.get_cosmology().name == "Planck18"

    def test_set_cosmology_with_none_restores_the_default(self):
        constants.set_cosmology("Planck15")
        assert constants.get_cosmology().name == "Planck15"
        constants.set_cosmology(None)
        assert constants.get_cosmology().name == "Planck18"

    def test_set_cosmology_accepts_a_name(self):
        returned = constants.set_cosmology("Planck15")
        assert returned.name == "Planck15"
        assert constants.get_cosmology().name == "Planck15"

    def test_set_cosmology_accepts_a_cosmology_object(self):
        returned = constants.set_cosmology(cosmology.WMAP9)
        assert returned.name == "WMAP9"

    def test_set_cosmology_also_updates_bilby(self):
        # NMMA and bilby must not disagree about the cosmology, since the
        # GW side of a joint run converts distances through bilby's global.
        constants.set_cosmology("Planck15")
        assert bilby_cosmo.get_cosmology().name == "Planck15"

    def test_set_cosmology_returns_the_same_object_as_get_cosmology(self):
        returned = constants.set_cosmology("Planck15")
        assert returned is constants.get_cosmology()

    def test_cosmology_is_usable_for_distance_conversion(self):
        distance = constants.get_cosmology().luminosity_distance(0.01).value
        assert np.isfinite(distance)
        assert distance > 0.0

