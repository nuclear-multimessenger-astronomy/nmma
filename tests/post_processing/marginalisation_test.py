import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from nmma.post_processing import marginalisation


def template_row(with_spins=True, with_angles=True):
    """The columns a search-template file provides, in the dictionary form
    the quantity helper accepts."""
    data = {
        "m1": np.array([1.6, 1.7]),
        "m2": np.array([1.4, 1.3]),
        "dist": np.array([40.0, 45.0]),
    }
    if with_spins:
        data["a1"] = np.array([0.02, 0.03])
        data["a2"] = np.array([0.01, 0.02])
    if with_angles:
        data["theta_jn"] = np.array([0.3, 0.4])
        data["tilt1"] = np.array([0.0, 0.0])
        data["tilt2"] = np.array([0.0, 0.0])
    return data


class TestGetAllGWQuantitiesFromMasses(unittest.TestCase):
    """A search template gives component masses, and the derived mass
    quantities the light curve model needs are filled in from them."""

    def test_the_mass_quantities_are_derived_from_the_components(self):
        data = marginalisation.get_all_gw_quantities(template_row())
        for key in ["mchirp", "eta", "q"]:
            self.assertIn(key, data, msg=key)

    def test_the_chirp_mass_sits_below_both_components(self):
        # For comparable masses the chirp mass is about 0.87 of the mean, so
        # it falls under the lighter star rather than between the two.
        data = marginalisation.get_all_gw_quantities(template_row())
        self.assertTrue(np.all(data["mchirp"] < data["m2"]))
        self.assertTrue(np.all(data["mchirp"] > 0.8 * data["m2"]))

    def test_the_symmetric_mass_ratio_stays_below_a_quarter(self):
        data = marginalisation.get_all_gw_quantities(template_row())
        self.assertTrue(np.all(data["eta"] <= 0.25))

    def test_every_template_is_weighted_equally(self):
        data = marginalisation.get_all_gw_quantities(template_row())
        self.assertAlmostEqual(float(np.atleast_1d(data["weight"])[0]), 0.5)

    def test_the_effective_spin_is_the_mass_weighted_average(self):
        data = marginalisation.get_all_gw_quantities(template_row())
        expected = (1.6 * 0.02 + 1.4 * 0.01) / (1.6 + 1.4)
        self.assertAlmostEqual(data["chi_eff"][0], expected)

    def test_the_aligned_spin_columns_are_preferred_when_present(self):
        data = template_row()
        data["spin1z"] = np.array([0.05, 0.06])
        data["spin2z"] = np.array([0.04, 0.05])
        result = marginalisation.get_all_gw_quantities(data)
        np.testing.assert_allclose(result["a1"], [0.05, 0.06])

    def test_missing_orientation_angles_default_to_zero(self):
        data = marginalisation.get_all_gw_quantities(template_row(with_angles=False))
        for key in ["theta_jn", "tilt1", "tilt2"]:
            self.assertEqual(data[key], 0.0, msg=key)


class TestGetAllGWQuantitiesWithoutSpins(unittest.TestCase):
    """A template file written without spin columns is one of the two formats
    the script explicitly reads."""

    def test_a_template_without_spins_cannot_be_processed(self):
        # The effective spin is computed from a1 and a2 several lines before
        # the loop that defaults them to zero, so the spinless format the
        # reader falls back to is rejected here. Moving the defaulting loop
        # above the effective-spin calculation is the fix.
        with self.assertRaises(KeyError) as caught:
            marginalisation.get_all_gw_quantities(template_row(with_spins=False))
        self.assertIn("a1", str(caught.exception))

    def test_the_defaulting_loop_does_cover_the_spins(self):
        # The intent is clearly there; it just runs too late.
        source = __import__("pathlib").Path(marginalisation.__file__).read_text()
        self.assertIn('for key in ["a1", "a2", "theta_jn", "tilt1", "tilt2"]', source)

    def test_supplying_the_spins_explicitly_works_around_it(self):
        data = template_row(with_spins=False)
        data["a1"] = np.zeros(2)
        data["a2"] = np.zeros(2)
        result = marginalisation.get_all_gw_quantities(data)
        np.testing.assert_allclose(result["chi_eff"], [0.0, 0.0])


class TestGetAllGWQuantitiesFromChirpMass(unittest.TestCase):
    """A posterior file gives a chirp mass and mass ratio instead, and the
    component masses are reconstructed from them."""

    def data(self):
        return {
            "mc": np.array([1.2, 1.21]),
            "q": np.array([0.9, 0.85]),
            "a1": np.array([0.02, 0.03]),
            "a2": np.array([0.01, 0.02]),
            "theta_jn": np.array([0.3, 0.4]),
            "tilt1": np.array([0.0, 0.0]),
            "tilt2": np.array([0.0, 0.0]),
            "dist": np.array([40.0, 45.0]),
        }

    def test_the_component_masses_are_reconstructed(self):
        result = marginalisation.get_all_gw_quantities(self.data())
        self.assertIn("m1", result)
        self.assertIn("m2", result)

    def test_the_primary_is_the_heavier_component(self):
        result = marginalisation.get_all_gw_quantities(self.data())
        self.assertTrue(np.all(result["m1"] >= result["m2"]))

    def test_the_chirp_mass_is_carried_over_under_its_full_name(self):
        result = marginalisation.get_all_gw_quantities(self.data())
        np.testing.assert_allclose(result["mchirp"], [1.2, 1.21])

    def test_the_symmetric_mass_ratio_is_derived_from_the_mass_ratio(self):
        result = marginalisation.get_all_gw_quantities(self.data())
        expected = 0.9 / (1 + 0.9) ** 2
        self.assertAlmostEqual(result["eta"][0], expected)

    def test_the_reconstructed_masses_reproduce_the_chirp_mass(self):
        result = marginalisation.get_all_gw_quantities(self.data())
        m1, m2 = result["m1"], result["m2"]
        reconstructed = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        np.testing.assert_allclose(reconstructed, result["mchirp"], rtol=1e-6)


class TestMarginalisedLightcurveExpectation(unittest.TestCase):
    """The routine that draws equation-of-state and template pairs and builds
    one light curve per draw. Only its guard rails are reachable without a
    full set of surrogate weights and sky localisation inputs."""

    def test_the_routine_looks_for_its_parser_in_the_wrong_module(self):
        # It asks the electromagnetic parsing module for
        # lc_marginalisation_parser, but that function lives in the
        # post-processing parser module. The routine therefore raises on its
        # very first statement, so the marginalisation workflow cannot run at
        # all. Importing the parser from the right module is the fix.
        from nmma.em import em_parsing
        from nmma.post_processing import parser as pp_parser

        self.assertFalse(hasattr(em_parsing, "lc_marginalisation_parser"))
        self.assertTrue(hasattr(pp_parser, "lc_marginalisation_parser"))
        with self.assertRaises(AttributeError) as caught:
            marginalisation.marginalised_lightcurve_expectation_from_gw_samples()
        self.assertIn("lc_marginalisation_parser", str(caught.exception))

    def test_no_input_format_at_all_would_exit_once_the_parser_is_found(self):
        args = MagicMock()
        args.generation_seed = 42
        args.template_file = None
        args.hdf5_file = None
        args.coinc_file = None
        with patch.object(
            marginalisation.emp,
            "lc_marginalisation_parser",
            create=True,
        ):
            with patch.object(
                marginalisation.emp, "parsing_and_logging", return_value=args
            ):
                with patch.object(
                    marginalisation.utils, "set_filters", return_value=["ztfg"]
                ):
                    with patch.object(
                        marginalisation.model, "create_light_curve_model_from_args"
                    ):
                        with patch.object(
                            marginalisation.conv.MultimessengerConversion, "basic_cbc"
                        ):
                            with patch.object(marginalisation, "EoSConverter"):
                                with patch.object(
                                    marginalisation,
                                    "load_tabulated_macro_eos_set_to_dict",
                                    return_value=({}, np.ones(2) / 2, 2),
                                ):
                                    with self.assertRaises(SystemExit):
                                        marginalisation.marginalised_lightcurve_expectation_from_gw_samples()

    def test_the_secondary_spin_is_filled_from_the_secondary_mass(self):
        # The parameter dictionary sets the secondary spin from the m2
        # column rather than the a2 column, so every light curve is built
        # with a spin equal to a neutron-star mass. That is far outside any
        # physical spin range.
        source = __import__("pathlib").Path(marginalisation.__file__).read_text()
        self.assertIn('"a_2": data_out["m2"][idy]', source)
        self.assertIn('"a_1": data_out["a1"][idy]', source)

    def test_a_binary_with_both_components_above_the_maximum_mass_is_unhandled(self):
        # The ejecta nuisance parameter is only assigned in two of the four
        # mass orderings. When both components exceed the maximum mass, which
        # a search template can easily produce, neither branch runs and the
        # parameter is referenced before assignment.
        source = __import__("pathlib").Path(marginalisation.__file__).read_text()
        self.assertIn("if (m1 < mMax) and (m2 < mMax):", source)
        self.assertIn("elif (m1 > mMax) and (m2 < mMax):", source)
        self.assertNotIn("else:\n            alpha", source)

    def test_the_error_scale_is_zeroed_so_the_curves_are_noise_free(self):
        # The spread being measured is the one from the binary parameters and
        # the equation of state, not from photometric noise.
        source = __import__("pathlib").Path(marginalisation.__file__).read_text()
        self.assertIn("args.mag_error_scale = 0", source)

    def test_the_default_filter_set_is_used_when_none_are_requested(self):
        source = __import__("pathlib").Path(marginalisation.__file__).read_text()
        self.assertIn('filters = "u,g,r,i,z,y,J,H,K"', source)


if __name__ == "__main__":
    unittest.main()
