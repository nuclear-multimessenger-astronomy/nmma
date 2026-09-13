import numpy as np
import pandas as pd
import pytest
from astropy import cosmology as astropy_cosmology

from nmma.core import constants
from nmma.core import conversion
from nmma.core.constants import geom_msun_km, msun_s


class TestValToScalar:
    def test_a_scalar_is_passed_through(self):
        assert conversion.val_to_scalar(3.0) == 3.0

    def test_a_single_element_array_becomes_a_python_scalar(self):
        value = conversion.val_to_scalar(np.array([3.0]))
        assert value == 3.0
        assert isinstance(value, float)

    def test_a_multi_element_array_is_left_as_an_array(self):
        np.testing.assert_allclose(
            conversion.val_to_scalar(np.array([1.0, 2.0])), [1.0, 2.0]
        )

    def test_a_list_of_one_becomes_a_scalar(self):
        assert conversion.val_to_scalar([2.5]) == 2.5


class TestDistanceConversions:
    def setup_method(self):
        self.original_cosmology = constants.get_cosmology()

    def teardown_method(self):
        constants.set_cosmology(self.original_cosmology)

    def test_distance_modulus_matches_its_definition(self):
        # mag_app - mag_abs = 5 log10(d / 10 pc), with d in Mpc
        assert conversion.distance_modulus_nmma(40.0) == pytest.approx(5.0 * np.log10(40e6 / 10.0))

    def test_distance_modulus_at_ten_parsec_is_zero(self):
        assert conversion.distance_modulus_nmma(1e-5) == pytest.approx(0.0)

    def test_luminosity_distance_to_redshift_inverts_the_cosmology(self):
        cosmology = constants.get_cosmology()
        distance = cosmology.luminosity_distance(0.05).value
        assert conversion.luminosity_distance_to_redshift(distance) == pytest.approx(0.05, abs=1.5 * 10**(-6))

    def test_luminosity_distance_to_redshift_accepts_a_pandas_series(self):
        distances = pd.Series([40.0, 100.0])
        redshifts = conversion.luminosity_distance_to_redshift(distances)
        assert len(redshifts) == 2
        assert redshifts[0] < redshifts[1]

    def test_many_distances_go_through_the_interpolated_grid(self):
        # more than 50 entries switches to the interpolation branch; it has
        # to agree with the exact inversion to within interpolation error
        distances = np.linspace(40.0, 400.0, 60)
        interpolated = conversion.luminosity_distance_to_redshift(distances)
        exact = np.array(
            [conversion.luminosity_distance_to_redshift(float(d)) for d in distances]
        )
        np.testing.assert_allclose(interpolated, exact, rtol=1e-4)

    def test_get_cosmo_grids_spans_the_requested_range(self):
        cosmology = constants.get_cosmology()
        dist_grid, z_grid = conversion.get_cosmo_grids(40.0, 400.0, cosmology)
        assert len(dist_grid) == 50
        assert dist_grid[0] == pytest.approx(40.0, abs=1.5 * 10**(-3))
        assert dist_grid[-1] == pytest.approx(400.0, abs=1.5 * 10**(-3))
        assert np.all(np.diff(z_grid) > 0)

    def test_get_redshift_prefers_an_explicit_redshift(self):
        assert conversion.get_redshift({"redshift": 0.1, "luminosity_distance": 40.0}) == 0.1

    def test_get_redshift_falls_back_to_the_luminosity_distance(self):
        assert conversion.get_redshift(
            {"luminosity_distance": 40.0}
        ) == pytest.approx(conversion.luminosity_distance_to_redshift(40.0))

    def test_get_redshift_without_distance_information_is_zero(self):
        redshift = conversion.get_redshift({"mass_1": np.array([1.4, 1.3])})
        np.testing.assert_allclose(redshift, [0.0, 0.0])


class TestCosmologyToDistance:
    def setup_method(self):
        self.original_cosmology = constants.get_cosmology()

    def teardown_method(self):
        constants.set_cosmology(self.original_cosmology)

    def test_redshift_is_derived_from_the_distance(self):
        parameters = conversion.cosmology_to_distance(
            {"Hubble_constant": 70.0, "luminosity_distance": 40.0}
        )
        assert "redshift" in parameters
        assert parameters["redshift"] > 0.0

    def test_distance_is_derived_from_the_redshift(self):
        parameters = conversion.cosmology_to_distance(
            {"Hubble_constant": 70.0, "redshift": 0.01}
        )
        assert parameters["luminosity_distance"] == pytest.approx(43.1546, abs=1.5 * 10**(-3))

    def test_a_larger_hubble_constant_gives_a_smaller_distance(self):
        low = conversion.cosmology_to_distance(
            {"Hubble_constant": 60.0, "redshift": 0.01}
        )["luminosity_distance"]
        high = conversion.cosmology_to_distance(
            {"Hubble_constant": 80.0, "redshift": 0.01}
        )["luminosity_distance"]
        assert low > high

    def test_omega_matter_is_honoured(self):
        first = conversion.cosmology_to_distance(
            {"Hubble_constant": 70.0, "Omega_matter": 0.2, "redshift": 0.5}
        )["luminosity_distance"]
        second = conversion.cosmology_to_distance(
            {"Hubble_constant": 70.0, "Omega_matter": 0.4, "redshift": 0.5}
        )["luminosity_distance"]
        assert first != pytest.approx(second)

    def test_neither_redshift_nor_distance_raises(self):
        with pytest.raises(KeyError):
            conversion.cosmology_to_distance({"Hubble_constant": 70.0})

    def test_array_valued_hubble_constant_takes_the_per_sample_branch(self):
        # cosmology.clone raises a ValueError for an array-valued H0, which
        # sends the conversion down the one-cosmology-per-sample path
        parameters = conversion.cosmology_to_distance(
            {
                "Hubble_constant": np.array([60.0, 80.0]),
                "redshift": np.array([0.01, 0.01]),
            }
        )
        distances = parameters["luminosity_distance"]
        assert len(distances) == 2
        assert distances[0] > distances[1]

    def test_array_valued_hubble_constant_with_distances_gives_redshifts(self):
        parameters = conversion.cosmology_to_distance(
            {
                "Hubble_constant": np.array([60.0, 80.0]),
                "luminosity_distance": np.array([40.0, 40.0]),
            }
        )
        redshifts = parameters["redshift"]
        assert len(redshifts) == 2
        assert redshifts[0] < redshifts[1]


class TestSourceFrameMasses:
    def test_source_masses_are_redshifted_detector_masses(self):
        parameters = conversion.source_frame_masses(
            {"mass_1": 1.5, "mass_2": 1.3, "redshift": 0.1}
        )
        assert float(parameters["mass_1_source"]) == pytest.approx(1.5 / 1.1)
        assert float(parameters["mass_2_source"]) == pytest.approx(1.3 / 1.1)

    def test_redshift_is_computed_from_the_distance_when_absent(self):
        parameters = conversion.source_frame_masses(
            {"mass_1": 1.5, "mass_2": 1.3, "luminosity_distance": 100.0}
        )
        assert "redshift" in parameters
        assert float(parameters["mass_1_source"]) < 1.5

    def test_existing_source_masses_are_not_overwritten(self):
        parameters = conversion.source_frame_masses(
            {"mass_1": 1.5, "mass_2": 1.3, "redshift": 0.1, "mass_1_source": 99.0}
        )
        assert parameters["mass_1_source"] == 99.0

    def test_derived_mass_parameters_are_added(self):
        parameters = conversion.source_frame_masses(
            {"mass_1": 1.5, "mass_2": 1.3, "redshift": 0.0}
        )
        for key in ["chirp_mass", "total_mass", "mass_ratio", "symmetric_mass_ratio"]:
            assert key in parameters


class TestObservationAngleConversion:
    def test_theta_jn_is_converted_to_degrees(self):
        parameters = conversion.observation_angle_conversion({"theta_jn": np.pi / 4})
        assert parameters["KNtheta"] == pytest.approx(45.0)
        assert parameters["inclination_EM"] == pytest.approx(np.pi / 4)

    def test_angles_above_ninety_degrees_are_folded_back(self):
        # the kilonova is symmetric about the orbital plane, so theta and
        # pi - theta are the same viewing geometry
        parameters = conversion.observation_angle_conversion({"theta_jn": 0.75 * np.pi})
        assert parameters["KNtheta"] == pytest.approx(45.0)

    def test_cos_theta_jn_is_accepted(self):
        parameters = conversion.observation_angle_conversion({"cos_theta_jn": 0.0})
        assert parameters["KNtheta"] == pytest.approx(90.0)

    def test_inclination_em_takes_precedence(self):
        parameters = conversion.observation_angle_conversion(
            {"theta_jn": np.pi / 4, "inclination_EM": np.pi / 6}
        )
        assert parameters["KNtheta"] == pytest.approx(30.0)

    def test_kntheta_is_converted_back_to_radians(self):
        parameters = conversion.observation_angle_conversion({"KNtheta": 60.0})
        assert parameters["inclination_EM"] == pytest.approx(np.pi / 3)

    def test_no_angle_information_gives_an_on_axis_view(self):
        parameters = conversion.observation_angle_conversion({"mass_1": 1.4})
        assert parameters["KNtheta"] == pytest.approx(0.0)
        assert parameters["inclination_EM"] == pytest.approx(0.0)


class TestMassConversions:
    def test_source_frame_helpers_add_component_masses(self):
        for func in [conversion.bbh_source_frame, conversion.bns_source_frame]:
            parameters = func(
                {"chirp_mass": 1.2, "mass_ratio": 0.9, "luminosity_distance": 40.0}
            )
            assert "mass_1_source" in parameters, func.__name__
            assert "mass_2_source" in parameters, func.__name__

    def test_bns_source_frame_keeps_tidal_parameters(self):
        parameters = conversion.bns_source_frame(
            {
                "chirp_mass": 1.2,
                "mass_ratio": 0.9,
                "luminosity_distance": 40.0,
                "lambda_1": 400.0,
                "lambda_2": 600.0,
            }
        )
        assert "lambda_1" in parameters

    def test_mass_ratio_to_eta_peaks_at_equal_masses(self):
        assert conversion.mass_ratio_to_eta(1.0) == pytest.approx(0.25)
        assert conversion.mass_ratio_to_eta(0.5) < 0.25

    def test_component_masses_to_mass_quantities(self):
        chirp_mass, eta, mass_ratio = conversion.component_masses_to_mass_quantities(
            2.0, 2.0
        )
        assert eta == pytest.approx(0.25)
        assert mass_ratio == pytest.approx(1.0)
        assert chirp_mass == pytest.approx(4.0 * 0.25**0.6)

    def test_chirp_mass_and_eta_round_trip_through_component_masses(self):
        mass_1, mass_2 = conversion.chirp_mass_and_eta_to_component_masses(1.2, 0.24)
        chirp_mass, eta, _ = conversion.component_masses_to_mass_quantities(mass_1, mass_2)
        assert chirp_mass == pytest.approx(1.2)
        assert eta == pytest.approx(0.24)

    def test_chirp_mass_and_eta_to_component_masses_orders_the_masses(self):
        mass_1, mass_2 = conversion.chirp_mass_and_eta_to_component_masses(1.2, 0.24)
        assert mass_1 >= mass_2

    def test_effective_tidal_deformabilities_for_equal_masses(self):
        # for lambda_1 == lambda_2 and q == 1, lambda_tilde reduces to lambda
        # and the asymmetric combination vanishes
        lambda_tilde, dlambda_tilde = (
            conversion.tidal_deformabilities_and_mass_ratio_to_eff_tidal_deformabilities(
                500.0, 500.0, 1.0
            )
        )
        assert lambda_tilde == pytest.approx(500.0)
        assert dlambda_tilde == pytest.approx(0.0)

    def test_effective_tidal_deformability_grows_with_the_deformabilities(self):
        small, _ = (
            conversion.tidal_deformabilities_and_mass_ratio_to_eff_tidal_deformabilities(
                100.0, 100.0, 0.9
            )
        )
        large, _ = (
            conversion.tidal_deformabilities_and_mass_ratio_to_eff_tidal_deformabilities(
                800.0, 800.0, 0.9
            )
        )
        assert large > small

    def test_reweight_to_flat_mass_prior_thins_the_dataframe(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "chirp_mass": rng.uniform(1.0, 2.0, 1000),
                "mass_ratio": rng.uniform(0.5, 1.0, 1000),
            }
        )
        reweighted = conversion.reweight_to_flat_mass_prior(df)
        assert len(reweighted) == 300
        assert set(reweighted.columns) == set(df.columns)

    def test_convert_mtot_mni_fills_linear_masses(self):
        parameters = conversion.convert_mtot_mni(
            {"log10_mni": -1.0, "log10_mtot": 0.0, "log10_mrp": -2.0, "xmix": 0.5}
        )
        assert parameters["mni"] == pytest.approx(0.1)
        assert parameters["mtot"] == pytest.approx(1.0)
        assert parameters["mrp"] == pytest.approx(0.01)
        assert parameters["mni_c"] == pytest.approx(0.1)
        assert parameters["mrp_c"] == pytest.approx(0.5 * (1.0 - 0.1) - 0.01)

    def test_convert_mtot_mni_keeps_explicit_linear_masses(self):
        parameters = conversion.convert_mtot_mni(
            {"mni": 0.2, "mtot": 1.0, "mrp": 0.01, "xmix": 0.5}
        )
        assert parameters["mni_c"] == pytest.approx(0.2)


class TestPulsarTimingConversions:
    def test_binary_mass_function_definition(self):
        assert conversion.binary_mass_function(1.4, 1.1, 0.9) == pytest.approx((1.1 * 0.9) ** 3 / (1.4 + 1.1) ** 2)

    def test_mass_function_inverts_back_to_sin_i(self):
        mass_function = conversion.binary_mass_function(1.4, 1.1, 0.9)
        assert conversion.mass_parameters_to_sini(1.4 + 1.1, mass_function, 1.1) == pytest.approx(0.9)

    def test_shapiro_delay_is_edge_on_maximal(self):
        edge_on = conversion.shapiro_delay(1.1, 1.0)
        inclined = conversion.shapiro_delay(1.1, 0.5)
        assert edge_on > inclined

    def test_shapiro_delay_scales_with_the_companion_mass(self):
        assert conversion.shapiro_delay(2.0, 1.0) / conversion.shapiro_delay(1.0, 1.0) == pytest.approx(2.0)

    def test_shapiro_delay_units_are_microseconds(self):
        # range = msun_s * m_comp, expressed in microseconds
        assert conversion.shapiro_delay(1.0, 1.0) == pytest.approx(msun_s * 1e6)

    def test_einstein_delay_orbital_factor_vanishes_for_a_circular_orbit(self):
        assert conversion.einstein_delay_orbital_factor(1e5, 0.0) == pytest.approx(0.0)

    def test_einstein_delay_grows_with_eccentricity(self):
        assert conversion.einstein_delay(1.4, 1.1, 1e5, 0.5) > conversion.einstein_delay(1.4, 1.1, 1e5, 0.1)

    def test_einstein_delay_composes_its_two_factors(self):
        factor = conversion.einstein_delay_orbital_factor(1e5, 0.3)
        assert conversion.einstein_delay(1.4, 1.1, 1e5, 0.3) == pytest.approx(conversion.simplified_einstein_delay(1.4, 1.1, factor))


class TestEOSConversions:
    def setup_method(self):
        # a monotonic stand-in mass-radius-lambda sequence; the TOV point is
        # the maximum of the mass column
        self.masses = np.array([1.0, 1.4, 1.6, 2.0, 1.9])
        self.radii = np.array([12.0, 12.4, 12.5, 12.0, 11.0])
        self.lambdas = np.array([2000.0, 500.0, 300.0, 50.0, 40.0])

    def test_tov_point_is_the_maximum_mass(self):
        tov_mass, tov_radius, r14, r16 = conversion.EOS_to_ns_parameters(
            self.radii, self.masses, self.lambdas
        )
        assert tov_mass == pytest.approx(2.0)
        assert tov_radius == pytest.approx(12.0)

    def test_canonical_radii_are_interpolated_at_1_4_and_1_6(self):
        _, _, r14, r16 = conversion.EOS_to_ns_parameters(
            self.radii, self.masses, self.lambdas
        )
        assert r14 == pytest.approx(12.4)
        assert r16 == pytest.approx(12.5)

    def test_system_parameters_are_interpolated_for_both_components(self):
        masses = np.array([1.0, 1.4, 1.6, 2.0])
        radii = np.array([12.0, 12.4, 12.5, 12.0])
        lambdas = np.array([2000.0, 500.0, 300.0, 50.0])
        lambda_1, lambda_2, radius_1, radius_2 = conversion.EOS_to_system_parameters(
            radii, masses, lambdas, 1.6, 1.4
        )
        assert lambda_1 == pytest.approx(300.0)
        assert lambda_2 == pytest.approx(500.0)
        assert radius_1 == pytest.approx(12.5)
        assert radius_2 == pytest.approx(12.4)

    def test_a_mass_outside_the_tabulated_range_gives_zero_radius(self):
        # the 0 radius is the sentinel the ejecta fitting uses to decide a
        # component is not a neutron star under this equation of state
        masses = np.array([1.0, 1.4, 1.6, 2.0])
        radii = np.array([12.0, 12.4, 12.5, 12.0])
        lambdas = np.array([2000.0, 500.0, 300.0, 50.0])
        lambda_1, _, radius_1, _ = conversion.EOS_to_system_parameters(
            radii, masses, lambdas, 2.5, 1.4
        )
        assert radius_1 == 0.0
        assert lambda_1 == 0.0

    def test_lambda_to_compactness_decreases_with_deformability(self):
        assert conversion.lambda_to_compactness(100.0) > conversion.lambda_to_compactness(1000.0)

    def test_lambda_to_compactness_is_in_a_physical_range(self):
        compactness = conversion.lambda_to_compactness(400.0)
        assert 0.1 < compactness < 0.25

    def test_mass_and_compactness_to_radius_inverts_the_compactness(self):
        radius = conversion.mass_and_compactness_to_radius(1.4, 0.16)
        assert radius == pytest.approx(1.4 / 0.16 * geom_msun_km)

    def test_a_black_hole_compactness_gives_zero_radius(self):
        assert conversion.mass_and_compactness_to_radius(1.4, 0.6) == 0.0

    def test_mass_and_compactness_to_radius_is_vectorised(self):
        radii = conversion.mass_and_compactness_to_radius(
            np.array([1.4, 1.4]), np.array([0.16, 0.6])
        )
        assert radii[0] > 0.0
        assert radii[1] == 0.0

    def test_radii_from_qur_adds_radii_and_the_canonical_radius(self):
        parameters = conversion.radii_from_qur(
            {
                "mass_1_source": 1.4,
                "mass_2_source": 1.3,
                "lambda_1": 400.0,
                "lambda_2": 600.0,
            }
        )
        assert parameters["radius_1"] > 8.0
        assert parameters["radius_1"] < 20.0
        assert parameters["R_16"] > 0.0

    def test_radii_from_qur_gives_the_softer_star_the_smaller_radius(self):
        parameters = conversion.radii_from_qur(
            {
                "mass_1_source": 1.4,
                "mass_2_source": 1.4,
                "lambda_1": 200.0,
                "lambda_2": 800.0,
            }
        )
        assert parameters["radius_1"] < parameters["radius_2"]


class TestGRBJetConversions:
    def test_gaussian_jet_isotropic_equivalent_exceeds_the_true_energy(self):
        e_iso = conversion.gaussian_jet_energy_to_central_isotropic_energy_equivalent(
            1e50, 0.1, 4.0
        )
        assert e_iso > 1e50

    def test_gaussian_jet_energy_scales_linearly(self):
        first = conversion.gaussian_jet_energy_to_central_isotropic_energy_equivalent(
            1e50, 0.1, 4.0
        )
        second = conversion.gaussian_jet_energy_to_central_isotropic_energy_equivalent(
            2e50, 0.1, 4.0
        )
        assert second / first == pytest.approx(2.0)

    def test_gaussian_jet_result_is_real(self):
        # the expression is evaluated with complex error functions whose
        # imaginary part must cancel
        e_iso = conversion.gaussian_jet_energy_to_central_isotropic_energy_equivalent(
            1e50, 0.1, 4.0
        )
        assert isinstance(float(e_iso), float)
        assert np.isfinite(e_iso)

    def test_a_narrower_core_concentrates_more_energy_on_axis(self):
        narrow = conversion.gaussian_jet_energy_to_central_isotropic_energy_equivalent(
            1e50, 0.05, 4.0
        )
        wide = conversion.gaussian_jet_energy_to_central_isotropic_energy_equivalent(
            1e50, 0.2, 4.0
        )
        assert narrow > wide

    def test_powerlaw_jet_isotropic_equivalent_exceeds_the_true_energy(self):
        e_iso = conversion.powerlaw_jet_energy_to_central_isotropic_energy_equivalent(
            1e50, 0.1, 4.0, 2.0
        )
        assert e_iso > 1e50

    def test_powerlaw_jet_energy_scales_linearly(self):
        first = conversion.powerlaw_jet_energy_to_central_isotropic_energy_equivalent(
            1e50, 0.1, 4.0, 2.0
        )
        second = conversion.powerlaw_jet_energy_to_central_isotropic_energy_equivalent(
            3e50, 0.1, 4.0, 2.0
        )
        assert second / first == pytest.approx(3.0)

    def test_a_steeper_powerlaw_tail_concentrates_more_energy_on_axis(self):
        steep = conversion.powerlaw_jet_energy_to_central_isotropic_energy_equivalent(
            1e50, 0.1, 4.0, 6.0
        )
        shallow = conversion.powerlaw_jet_energy_to_central_isotropic_energy_equivalent(
            1e50, 0.1, 4.0, 1.0
        )
        assert steep > shallow


class TestEjectaFittingBase:
    def test_the_base_class_produces_no_ejecta(self):
        parameters = conversion.EjectaFitting()({"mass_1": 1.4})
        for key in conversion.EjectaFitting.mass_fitting_keys:
            assert parameters[key] == -np.inf

    def test_explicitly_sampled_ejecta_parameters_are_preferred(self):
        parameters = conversion.EjectaFitting()({"mass_1": 1.4, "log10_mej": -2.0})
        assert parameters["log10_mej"] == -2.0
        assert parameters["log10_mej_dyn"] == -np.inf

    def test_the_input_dictionary_is_updated_in_place(self):
        parameters = {"mass_1": 1.4}
        returned = conversion.EjectaFitting()(parameters)
        assert returned is parameters


class TestNSBHEjectaFitting:
    def setup_method(self):
        self.fitter = conversion.NSBHEjectaFitting()

    def test_isco_of_a_non_spinning_black_hole_is_six_masses(self):
        assert self.fitter.chibh2risco(0.0) == pytest.approx(6.0)

    def test_isco_shrinks_for_prograde_spin(self):
        assert self.fitter.chibh2risco(0.9) < self.fitter.chibh2risco(0.0)

    def test_isco_grows_for_retrograde_spin(self):
        assert self.fitter.chibh2risco(-0.9) > self.fitter.chibh2risco(0.0)

    def test_extremal_spin_gives_the_expected_isco_limits(self):
        assert self.fitter.chibh2risco(1.0) == pytest.approx(1.0, abs=1.5 * 10**(-6))
        assert self.fitter.chibh2risco(-1.0) == pytest.approx(9.0, abs=1.5 * 10**(-6))

    def test_baryon_mass_exceeds_the_gravitational_mass(self):
        assert self.fitter.baryon_mass_NS(1.4, 0.16) > 1.4

    def test_baryon_mass_correction_grows_with_compactness(self):
        assert self.fitter.baryon_mass_NS(1.4, 0.20) > self.fitter.baryon_mass_NS(1.4, 0.10)

    def test_remnant_disk_and_dynamic_masses_are_non_negative(self):
        for chi_bh in [-0.5, 0.0, 0.9]:
            assert self.fitter.remnant_disk_mass_fitting(6.0, 1.4, 0.16, chi_bh) >= 0.0
            assert self.fitter.dynamic_mass_fitting(6.0, 1.4, 0.16, chi_bh) >= 0.0

    def test_a_rapidly_spinning_black_hole_disrupts_the_star_more(self):
        # a smaller ISCO lets more material stay outside the horizon
        assert self.fitter.remnant_disk_mass_fitting(6.0, 1.4, 0.16, 0.9) > self.fitter.remnant_disk_mass_fitting(6.0, 1.4, 0.16, 0.0)

    def test_nsbh_conversion_returns_four_ejecta_quantities(self):
        parameters = dict(
            mass_1_source=np.array([6.0]),
            mass_2_source=np.array([1.4]),
            radius_2=np.array([12.0]),
            chi_1=np.array([0.9]),
            alpha=np.array([0.0]),
            ratio_zeta=np.array([0.5]),
        )
        result = self.fitter.nsbh_parameter_conversion(parameters)
        assert result.shape == (4, 1)

    def test_nsbh_conversion_never_produces_a_grb_energy(self):
        # the fourth slot, log10_E0, is left at -inf for NSBH systems
        parameters = dict(
            mass_1_source=np.array([6.0]),
            mass_2_source=np.array([1.4]),
            radius_2=np.array([12.0]),
            chi_1=np.array([0.9]),
            alpha=np.array([0.0]),
            ratio_zeta=np.array([0.5]),
        )
        assert self.fitter.nsbh_parameter_conversion(parameters)[3][0] == -np.inf

    def test_spin_is_built_from_the_tilt_when_chi_1_is_absent(self):
        base = dict(
            mass_1_source=np.array([6.0]),
            mass_2_source=np.array([1.4]),
            radius_2=np.array([12.0]),
            alpha=np.array([0.0]),
            ratio_zeta=np.array([0.5]),
        )
        from_components = self.fitter.nsbh_parameter_conversion(
            dict(base, a_1=np.array([0.9]), tilt_1=np.array([0.0]))
        )
        from_chi = self.fitter.nsbh_parameter_conversion(dict(base, chi_1=np.array([0.9])))
        np.testing.assert_allclose(from_components, from_chi)

    def test_an_aligned_spin_reduces_to_the_spin_magnitude(self):
        base = dict(
            mass_1_source=np.array([6.0]),
            mass_2_source=np.array([1.4]),
            radius_2=np.array([12.0]),
            alpha=np.array([0.0]),
            ratio_zeta=np.array([0.5]),
        )
        # tilt_1 = pi/2 leaves no aligned spin component at all
        in_plane = self.fitter.nsbh_parameter_conversion(
            dict(base, a_1=np.array([0.9]), tilt_1=np.array([np.pi / 2]))
        )
        non_spinning = self.fitter.nsbh_parameter_conversion(
            dict(base, chi_1=np.array([0.0]))
        )
        np.testing.assert_allclose(in_plane, non_spinning, atol=1e-12)

    def test_cos_tilt_1_alone_currently_raises(self):
        # The spin fallback reads
        #   converted_parameters.get("cos_tilt_1", np.cos(...["tilt_1"]))
        # and Python evaluates a dict.get default eagerly, so tilt_1 is
        # required even when cos_tilt_1 is present and the default is never
        # used. A run that samples in a_1/cos_tilt_1 without also carrying
        # tilt_1 therefore fails here. This test documents that known gap
        # rather than asserting it is correct.
        parameters = dict(
            mass_1_source=np.array([6.0]),
            mass_2_source=np.array([1.4]),
            radius_2=np.array([12.0]),
            alpha=np.array([0.0]),
            ratio_zeta=np.array([0.5]),
            a_1=np.array([0.9]),
            cos_tilt_1=np.array([1.0]),
        )
        with pytest.raises(KeyError):
            self.fitter.nsbh_parameter_conversion(parameters)

    def test_a_non_disrupting_system_gives_no_ejecta(self):
        # a heavy, non-spinning black hole swallows the star whole
        parameters = dict(
            mass_1_source=np.array([30.0]),
            mass_2_source=np.array([1.4]),
            radius_2=np.array([12.0]),
            chi_1=np.array([0.0]),
            alpha=np.array([0.0]),
            ratio_zeta=np.array([0.5]),
        )
        result = self.fitter.nsbh_parameter_conversion(parameters)
        assert result[1][0] == -np.inf


class TestBNSEjectaFitting:
    def setup_method(self):
        self.fitter = conversion.BNSEjectaFitting()
        self.parameters = dict(
            mass_1_source=np.array([1.4]),
            mass_2_source=np.array([1.3]),
            radius_1=np.array([13.0]),
            radius_2=np.array([13.1]),
            alpha=np.array([0.04]),
            ratio_zeta=np.array([0.5]),
            TOV_mass=np.array([2.0854]),
            R_16=np.array([12.0 * geom_msun_km]),
        )

    def test_disk_mass_is_floored_at_ten_to_the_minus_three(self):
        # a total mass far above the threshold mass gives prompt collapse
        log10_mdisk = self.fitter.log10_disk_mass_fitting(10.0, 1.0, 2.0, 12.0)
        assert log10_mdisk == pytest.approx(-3.0)

    def test_a_lighter_binary_leaves_a_more_massive_disk(self):
        light = self.fitter.log10_disk_mass_fitting(2.6, 0.9, 2.0, 12.0)
        heavy = self.fitter.log10_disk_mass_fitting(3.2, 0.9, 2.0, 12.0)
        assert light > heavy

    def test_dynamic_mass_fittings_are_non_negative(self):
        assert self.fitter.dynamic_mass_fitting_KrFo(1.4, 1.3, 0.16, 0.15) >= 0.0
        assert self.fitter.dynamic_mass_fitting_KrFo(1.9, 1.0, 0.25, 0.10) >= 0.0

    def test_dynamic_mass_fittings_agree_within_an_order_of_magnitude(self):
        krfo = self.fitter.dynamic_mass_fitting_KrFo(1.4, 1.3, 0.16, 0.15)
        codimame = 10 ** self.fitter.log10_dynamic_mass_fitting_CoDiMaMe(
            1.4, 1.3, 0.16, 0.15
        )
        assert abs(np.log10(krfo / codimame)) < 1.0

    def test_dynamic_ejecta_velocity_is_a_sensible_fraction_of_light_speed(self):
        velocity = self.fitter.dynamic_vel_fitting_Radice2018(1.4, 1.3, 0.16, 0.15)
        assert 0.05 < velocity < 0.5

    def test_prompt_collapse_dynamic_mass_is_positive(self):
        mdyn = self.fitter.dynamic_mass_fitting_prompt_collapse(1.4, 1.3, 400.0, 600.0)
        assert mdyn > 0.0

    def test_prompt_collapse_dynamic_mass_grows_with_deformability(self):
        assert self.fitter.dynamic_mass_fitting_prompt_collapse(
            1.4, 1.3, 800.0, 1000.0
        ) > self.fitter.dynamic_mass_fitting_prompt_collapse(1.4, 1.3, 200.0, 300.0)

    def test_prompt_collapse_velocity_is_a_sensible_fraction_of_light_speed(self):
        velocity = self.fitter.dynamic_vel_fitting_prompt_collapse(1.4, 1.3, 0.16, 0.15)
        assert 0.05 < velocity < 0.5

    def test_prompt_collapse_disk_mass_is_capped(self):
        log10_mdisk = self.fitter.log10_disk_mass_fitting_prompt_collapse(
            1.4, 1.3, 2000.0, 2000.0
        )
        assert log10_mdisk <= -1.0

    def test_black_hole_spin_fitting_is_a_physical_spin(self):
        chi_bh = self.fitter.chiBH_fitting(1.4, 1.3, 400.0, 600.0)
        assert 0.0 < chi_bh < 1.0

    def test_ejecta_conversion_returns_finite_masses_for_a_plausible_binary(self):
        log10_mej_dyn, log10_mej_wind, log10_mej_total, log10_mdisk = (
            self.fitter.bns_ejecta_conversion(self.parameters)
        )
        assert np.isfinite(log10_mej_dyn[0])
        assert np.isfinite(log10_mej_wind[0])
        assert np.isfinite(log10_mej_total[0])
        assert np.isfinite(log10_mdisk)

    def test_the_total_ejecta_mass_is_the_sum_of_its_components(self):
        log10_mej_dyn, log10_mej_wind, log10_mej_total, _ = (
            self.fitter.bns_ejecta_conversion(self.parameters)
        )
        assert log10_mej_total[0] == pytest.approx(np.log10(10**log10_mej_dyn[0] + 10**log10_mej_wind[0]))

    def test_the_wind_ejecta_scale_with_the_disk_conversion_efficiency(self):
        low = self.fitter.bns_ejecta_conversion(
            dict(self.parameters, ratio_zeta=np.array([0.1]))
        )[1][0]
        high = self.fitter.bns_ejecta_conversion(
            dict(self.parameters, ratio_zeta=np.array([0.5]))
        )[1][0]
        assert high - low == pytest.approx(np.log10(0.5 / 0.1))

    def test_bns_ejecta_conversion_rejects_non_ns_component(self):
        """Regression test carried over from the retired
        nmma/tests/test_core/conversion.py.

        radius_1/radius_2 are 0 (not a real Schwarzschild radius) for a mass
        outside the equation of state's tabulated range, so compactness =
        mass*geom_msun_km/radius is inf for that component.
        dynamic_mass_fitting_KrFo/log10_disk_mass_fitting clip negative fit
        values via np.maximum(0, .), which silently turned that inf into a
        finite mdyn_fit=0.0 before np.isfinite() downstream had any chance to
        notice; log10(0 + alpha) then came out as an ordinary finite number
        for a system that isn't actually a BNS under this equation of state.
        With mass_1_source=2.5 (above ALF2's ~2.09 Msun TOV mass) and
        radius_1=0, this used to return a finite log10_mej_dyn instead of -inf.
        """
        parameters = dict(
            mass_1_source=np.array([2.5]),
            mass_2_source=np.array([1.3]),
            radius_1=np.array([0.0]),  # outside ALF2's mass range -> not a NS
            radius_2=np.array([13.1]),
            alpha=np.array([0.04]),
            ratio_zeta=np.array([0.5]),
            TOV_mass=np.array([2.0854]),
            R_16=np.array([12.0]),
        )
        log10_mej_dyn, log10_mej_wind, log10_mej_total, _ = (
            self.fitter.bns_ejecta_conversion(parameters)
        )
        assert not np.isfinite(log10_mej_dyn[0])
        assert not np.isfinite(log10_mej_wind[0])
        assert not np.isfinite(log10_mej_total[0])

    def test_grb_energy_defaults_to_a_top_hat_jet(self):
        log10_e_iso = self.fitter.grb_energy_conversion(
            self.parameters, np.array([-1.0])
        )
        assert np.all(np.isfinite(log10_e_iso))

    def test_grb_energy_uses_the_gaussian_jet_when_a_wing_is_given(self):
        parameters = dict(self.parameters, thetaCore=0.1, thetaWing=0.4)
        log10_e_iso = self.fitter.grb_energy_conversion(parameters, np.array([-1.0]))
        assert np.isfinite(log10_e_iso)

    def test_grb_energy_uses_the_powerlaw_jet_when_b_is_given(self):
        gaussian = self.fitter.grb_energy_conversion(
            dict(self.parameters, thetaCore=0.1, alphaWing=4.0), np.array([-1.0])
        )
        powerlaw = self.fitter.grb_energy_conversion(
            dict(self.parameters, thetaCore=0.1, alphaWing=4.0, b=2.0), np.array([-1.0])
        )
        assert float(gaussian) != pytest.approx(float(powerlaw))

    def test_grb_energy_scales_with_the_disk_mass(self):
        low = self.fitter.grb_energy_conversion(self.parameters, np.array([-2.0]))
        high = self.fitter.grb_energy_conversion(self.parameters, np.array([-1.0]))
        np.testing.assert_allclose(high - low, 1.0)

    def test_a_larger_wind_fraction_leaves_less_energy_for_the_jet(self):
        low_zeta = self.fitter.grb_energy_conversion(
            dict(self.parameters, ratio_zeta=np.array([0.1])), np.array([-1.0])
        )
        high_zeta = self.fitter.grb_energy_conversion(
            dict(self.parameters, ratio_zeta=np.array([0.9])), np.array([-1.0])
        )
        assert low_zeta > high_zeta

    def test_an_explicit_jet_energy_is_not_overwritten(self):
        parameters = dict(self.parameters, log10_E0=np.array([50.0]))
        result = self.fitter.bns_parameter_conversion(parameters)
        assert float(result[3][0]) == pytest.approx(50.0)

    def test_non_finite_results_are_normalised_to_minus_infinity(self):
        parameters = dict(
            self.parameters,
            radius_1=np.array([0.0]),
            radius_2=np.array([0.0]),
        )
        result = self.fitter.bns_parameter_conversion(parameters)
        assert np.all(result[:3] == -np.inf)

    def test_the_fitter_is_callable_and_fills_the_parameter_dictionary(self):
        parameters = {k: v for k, v in self.parameters.items()}
        returned = self.fitter(parameters)
        for key in conversion.EjectaFitting.mass_fitting_keys:
            assert key in returned


class TestKilonovaEjectaFitting:
    def setup_method(self):
        self.fitter = conversion.KilonovaEjectaFitting()

    def test_a_scalar_bns_is_routed_to_the_bns_fitting(self):
        parameters = dict(
            mass_1_source=1.4,
            mass_2_source=1.3,
            radius_1=13.0,
            radius_2=13.1,
            alpha=0.04,
            ratio_zeta=0.5,
            TOV_mass=2.0854,
            R_16=12.0 * geom_msun_km,
        )
        expected = conversion.BNSEjectaFitting().bns_parameter_conversion(dict(parameters))
        np.testing.assert_allclose(
            self.fitter.ejecta_parameter_conversion(parameters), expected
        )

    def test_a_scalar_nsbh_is_routed_to_the_nsbh_fitting(self):
        parameters = dict(
            mass_1_source=6.0,
            mass_2_source=1.4,
            radius_1=0.0,  # the heavier object is a black hole
            radius_2=12.0,
            chi_1=0.9,
            alpha=0.0,
            ratio_zeta=0.5,
        )
        expected = conversion.NSBHEjectaFitting().nsbh_parameter_conversion(
            dict(parameters)
        )
        np.testing.assert_allclose(
            self.fitter.ejecta_parameter_conversion(parameters), expected
        )

    def test_a_scalar_binary_black_hole_produces_no_ejecta(self):
        result = self.fitter.ejecta_parameter_conversion(
            dict(radius_1=0.0, radius_2=0.0)
        )
        np.testing.assert_allclose(result, np.full(4, -np.inf))

    def test_kn_ejecta_fitting_requires_both_components_to_be_ns(self):
        """Regression test carried over from the retired
        nmma/tests/test_core/conversion.py.

        The routing used to send a row to bns_parameter_conversion whenever
        radius_1>0 alone, without also checking radius_2>0. mass_1 >= mass_2
        by convention, so radius_1>0 usually implies radius_2>0 too (a lighter
        mass is inside the equation of state's mass range whenever a heavier
        one is), but not always: mass_2 can fall below the tabulated minimum
        while mass_1 is a valid neutron-star mass, and that row was still
        wrongly treated as a BNS, silently publishing a finite ejecta mass
        instead of -inf.
        """
        # two rows, to force numpy's vectorized np.where routing path (the
        # if/elif scalar path only ever runs for single-injection calls,
        # which the real pipeline, always operating on a whole dataframe,
        # never does; a length-1 array can be truth-tested directly by numpy
        # without raising, so it wouldn't actually exercise the routing bug)
        parameters = dict(
            mass_1_source=np.array([1.8, 1.4]),
            mass_2_source=np.array([1.3, 1.3]),
            # row 0: radius_1 > 0 alone would wrongly route to BNS
            radius_1=np.array([13.0, 13.2]),
            # row 0: mass_2 isn't a real NS; row 1: a genuine BNS
            radius_2=np.array([0.0, 13.1]),
            alpha=np.array([0.04, 0.04]),
            ratio_zeta=np.array([0.5, 0.5]),
            TOV_mass=np.array([2.0854, 2.0854]),
            R_16=np.array([12.0, 12.0]),
            # np.where evaluates both the bns_ and nsbh_parameter_conversion
            # branches eagerly for every row (only the result is masked
            # afterwards), so nsbh_parameter_conversion's inputs must be
            # valid for all rows too, even those that end up routed to BNS.
            chi_1=np.array([0.0, 0.0]),
        )
        log10_mej_dyn, log10_mej_wind, log10_mej_total, _ = (
            self.fitter.ejecta_parameter_conversion(parameters)
        )
        assert not np.isfinite(log10_mej_dyn[0])
        assert not np.isfinite(log10_mej_wind[0])
        assert not np.isfinite(log10_mej_total[0])
        assert np.isfinite(log10_mej_dyn[1])

    def test_a_vectorised_binary_black_hole_row_produces_no_ejecta(self):
        parameters = dict(
            mass_1_source=np.array([10.0, 1.4]),
            mass_2_source=np.array([8.0, 1.3]),
            radius_1=np.array([0.0, 13.2]),
            radius_2=np.array([0.0, 13.1]),
            alpha=np.array([0.04, 0.04]),
            ratio_zeta=np.array([0.5, 0.5]),
            TOV_mass=np.array([2.0854, 2.0854]),
            R_16=np.array([12.0, 12.0]),
            chi_1=np.array([0.0, 0.0]),
        )
        result = self.fitter.ejecta_parameter_conversion(parameters)
        assert np.all(result[:, 0] == -np.inf)
        assert np.isfinite(result[0, 1])


class TestMultimessengerConversion:
    def setup_method(self):
        self.original_cosmology = constants.get_cosmology()

    def teardown_method(self):
        constants.set_cosmology(self.original_cosmology)

    def test_conversions_are_applied_in_order(self):
        calls = []

        def first(parameters):
            calls.append("first")
            parameters["a"] = 1
            return parameters

        def second(parameters):
            calls.append("second")
            parameters["b"] = parameters["a"] + 1
            return parameters

        converter = conversion.MultimessengerConversion(first, second)
        result = converter.core_conversion({})
        assert calls == ["first", "second"]
        assert result["b"] == 2

    def test_no_conversions_leaves_the_parameters_untouched(self):
        converter = conversion.MultimessengerConversion()
        assert converter.core_conversion({"a": 1}) == {"a": 1}

    def test_identity_conversion_returns_its_input(self):
        parameters = {"a": 1}
        assert conversion.MultimessengerConversion().identity_conversion(parameters) is parameters

    def test_single_element_arrays_are_flattened_to_scalars(self):
        converter = conversion.MultimessengerConversion()
        result = converter.convert_to_multimessenger_parameters(
            {"mass_1": np.array([1.4])}
        )
        assert isinstance(result["mass_1"], float)

    def test_added_keys_are_reported_when_requested(self):
        converter = conversion.MultimessengerConversion(conversion.bbh_source_frame)
        result, added_keys = converter.convert_to_multimessenger_parameters(
            {"chirp_mass": 1.2, "mass_ratio": 0.9, "luminosity_distance": 40.0},
            add_new_keys=True,
        )
        assert "mass_1_source" in added_keys
        assert "chirp_mass" not in added_keys
        assert "mass_1_source" in result

    def test_from_args_is_not_implemented_yet(self):
        with pytest.raises(NotImplementedError):
            conversion.MultimessengerConversion.from_args(None)

    def test_from_dict_builds_only_the_requested_conversions(self):
        converter = conversion.MultimessengerConversion.from_dict({"ejecta": True})
        assert len(converter._conversions) == 1
        assert isinstance(converter._conversions[0], conversion.KilonovaEjectaFitting)

    def test_from_dict_is_empty_for_an_empty_instruction(self):
        assert len(conversion.MultimessengerConversion.from_dict({})._conversions) == 0

    def test_from_dict_sets_the_cosmology_and_adds_the_distance_conversion(self):
        converter = conversion.MultimessengerConversion.from_dict({"cosmo": "Planck15"})
        assert constants.get_cosmology().name == "Planck15"
        assert converter._conversions[0] is conversion.cosmology_to_distance

    def test_from_dict_orders_cosmology_gw_eos_ejecta_em_and_custom(self):
        gw, eos, em, custom = (lambda p: p for _ in range(4))
        converter = conversion.MultimessengerConversion.from_dict(
            {
                "custom": custom,
                "em": em,
                "ejecta": True,
                "eos": eos,
                "gw": gw,
                "cosmo": None,
            }
        )
        conversions = converter._conversions
        assert conversions[0] is conversion.cosmology_to_distance
        assert conversions[1] is gw
        assert conversions[2] is eos
        assert isinstance(conversions[3], conversion.KilonovaEjectaFitting)
        assert conversions[4] is em
        assert conversions[5] is custom

    def test_basic_cbc_wires_up_the_standard_chain(self):
        eos_conversion, em_conversion = (lambda p: p for _ in range(2))
        converter = conversion.MultimessengerConversion.basic_cbc(
            eos_conversion, em_conversion
        )
        conversions = converter._conversions
        assert conversions[0] is conversion.bbh_source_frame
        assert conversions[1] is eos_conversion
        assert isinstance(conversions[2], conversion.KilonovaEjectaFitting)
        assert conversions[3] is em_conversion

    def test_a_full_chain_turns_binary_parameters_into_ejecta_parameters(self):
        # the point of the conversion layer: an EM model in a joint run sees
        # quantities the GW likelihood never mentions
        converter = conversion.MultimessengerConversion.basic_cbc(
            conversion.radii_from_qur, conversion.observation_angle_conversion
        )
        result = converter.convert_to_multimessenger_parameters(
            {
                "chirp_mass": 1.188,
                "mass_ratio": 0.9,
                "luminosity_distance": 40.0,
                "lambda_1": 400.0,
                "lambda_2": 600.0,
                "theta_jn": 0.5,
                "alpha": 0.04,
                "ratio_zeta": 0.5,
                "TOV_mass": 2.0854,
            }
        )
        assert "log10_mej" in result
        assert "KNtheta" in result
        assert np.isfinite(result["log10_mej"])


class TestLabelMapping:
    def test_every_label_is_latex(self):
        for key, label in conversion.label_mapping.items():
            assert label.startswith("$"), key
            assert label.endswith("$"), key

    def test_the_ejecta_parameters_produced_by_the_fitting_all_have_labels(self):
        for key in conversion.EjectaFitting.mass_fitting_keys:
            assert key in conversion.label_mapping

    def test_core_sampling_parameters_have_labels(self):
        for key in [
            "luminosity_distance",
            "chirp_mass",
            "mass_ratio",
            "redshift",
            "Hubble_constant",
            "KNtheta",
        ]:
            assert key in conversion.label_mapping

