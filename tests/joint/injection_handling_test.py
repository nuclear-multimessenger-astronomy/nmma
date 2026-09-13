import json
import operator
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from nmma.core.parsing import nmma_base_parsing
from nmma.joint import injection_handling
from nmma.joint.injection_handling import NMMAInjectionCreator
from nmma.joint.joint_parsing import injection_parsing

# The ejecta conversion is evaluated for both the BNS and the NSBH fitting
# whenever it is handed a whole table, so an injection prior has to supply
# the parameters both formulae need: component masses, a distance, the two
# ejecta nuisance parameters and the aligned spins.
INJECTION_PRIOR = """\
mass_1 = Uniform(minimum=1.2, maximum=2.0, name='mass_1')
mass_2 = Uniform(minimum=1.2, maximum=2.0, name='mass_2')
luminosity_distance = Uniform(minimum=10, maximum=100, name='luminosity_distance')
alpha = Uniform(minimum=0.0, maximum=0.2, name='alpha')
ratio_zeta = Uniform(minimum=0.1, maximum=0.5, name='ratio_zeta')
chi_1 = Uniform(minimum=-0.05, maximum=0.05, name='chi_1')
chi_2 = Uniform(minimum=-0.05, maximum=0.05, name='chi_2')
"""


def write_eos_file(path, maximum_mass=2.2):
    """A monotonic radius-mass-lambda table, in the column order the EOS
    machinery reads."""
    masses = np.linspace(0.5, maximum_mass, 40)
    radii = 12.0 - 0.3 * masses
    lambdas = 1000.0 * np.exp(-2.0 * masses)
    np.savetxt(path, np.column_stack([radii, masses, lambdas]))


class InjectionCreatorMixin:
    """Builds the creator through the real parser, so the tests stay honest
    about which command line actually reaches the class."""

    n_injection = 5

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.prior_file = self.tmp_dir / "injection.prior"
        self.prior_file.write_text(INJECTION_PRIOR)
        self.eos_file = self.tmp_dir / "eos.dat"
        write_eos_file(self.eos_file)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def make_args(self, *extra, prior_file=None):
        return nmma_base_parsing(
            injection_parsing,
            [
                "--prior-file",
                str(prior_file or self.prior_file),
                "--outdir",
                str(self.tmp_dir),
                "--injection-file",
                "injections",
                "--n-injection",
                str(self.n_injection),
                "--generation-seed",
                "42",
            ]
            + list(extra),
        )

    def make_creator(self, *extra, **kwargs):
        return NMMAInjectionCreator(self.make_args(*extra, **kwargs))

    def simple_creator(self, *extra):
        """A creator with every test and conversion switched off, for the
        parts that only deal with drawing and bookkeeping."""
        return self.make_creator("--simple-setup", *extra)

    def eos_creator(self, *extra):
        return self.make_creator("--eos-file", str(self.eos_file), *extra)


class TestBNSDistribution(unittest.TestCase):
    """The population test weights each draw by its mass ratio, folded so
    that it never exceeds one."""

    def test_a_lighter_secondary_gives_the_mass_ratio(self):
        self.assertAlmostEqual(injection_handling.BNS_distribution(2.0, 1.0), 0.5)

    def test_an_equal_mass_binary_gets_the_highest_weight(self):
        self.assertAlmostEqual(injection_handling.BNS_distribution(1.5, 1.5), 1.0)

    def test_a_heavier_secondary_is_folded_back_below_one(self):
        self.assertAlmostEqual(injection_handling.BNS_distribution(1.0, 2.0), 0.5)

    def test_the_weight_never_exceeds_one(self):
        masses = np.linspace(1.0, 3.0, 20)
        weights = injection_handling.BNS_distribution(masses, masses[::-1])
        self.assertTrue(np.all(weights <= 1.0))

    def test_it_works_elementwise_over_arrays(self):
        weights = injection_handling.BNS_distribution(
            np.array([2.0, 1.0]), np.array([1.0, 2.0])
        )
        np.testing.assert_allclose(weights, [0.5, 0.5])


class TestInitialisation(InjectionCreatorMixin, unittest.TestCase):
    def test_a_simple_setup_skips_every_check(self):
        creator = self.simple_creator()
        self.assertFalse(creator.include_checks)

    def test_a_simple_setup_builds_no_test_routines(self):
        creator = self.simple_creator()
        self.assertFalse(hasattr(creator, "test_routines"))

    def test_a_full_setup_enables_the_checks(self):
        self.assertTrue(self.eos_creator().include_checks)

    def test_the_priors_are_read_from_the_prior_file(self):
        creator = self.simple_creator()
        for parameter in ["mass_1", "mass_2", "luminosity_distance"]:
            self.assertIn(parameter, creator.priors, msg=parameter)

    def test_the_requested_number_of_injections_is_kept(self):
        self.assertEqual(self.simple_creator().n_injection, self.n_injection)

    def test_the_output_file_lands_in_the_output_directory(self):
        creator = self.simple_creator()
        self.assertEqual(Path(creator.filename).parent, self.tmp_dir)
        self.assertEqual(Path(creator.filename).name, "injections.json")

    def test_a_csv_request_is_written_in_the_dat_format(self):
        # bilby_pipe's writer only knows dat and json, so a csv request is
        # served by the tab-separated writer.
        creator = self.simple_creator("--extension", "csv")
        self.assertEqual(creator.extension, "dat")

    def test_a_json_request_keeps_its_extension(self):
        self.assertEqual(self.simple_creator("--extension", "json").extension, "json")

    def test_a_prior_dictionary_is_used_when_given_as_a_mapping(self):
        args = self.make_args("--simple-setup")
        args.prior_file = None
        args.prior_dict = {
            "mass_1": "Uniform(1.2, 2.0, 'mass_1')",
            "mass_2": "Uniform(1.2, 2.0, 'mass_2')",
        }
        creator = NMMAInjectionCreator(args)
        self.assertIn("mass_1", creator.priors)

    def test_a_prior_dictionary_given_as_a_string_cannot_be_used(self):
        # The string branch copies prior_dict onto prior_file but leaves
        # prior_dict set, so bilby_pipe is handed both. A dictionary string
        # fails as a missing prior file, and a path fails in the ini-dict
        # reader, which makes --prior-dict unusable for injections. Clearing
        # prior_dict once it has been copied would fix both.
        args = self.make_args("--simple-setup")
        args.prior_dict = "{mass_1: Uniform(1.2, 2.0, 'mass_1')}"
        with self.assertRaises(FileNotFoundError):
            NMMAInjectionCreator(args)

    def test_a_prior_file_path_passed_as_a_prior_dictionary_also_fails(self):
        args = self.make_args("--simple-setup")
        args.prior_dict = str(self.prior_file)
        with self.assertRaises(Exception):
            NMMAInjectionCreator(args)

    def test_keyword_arguments_are_set_as_attributes(self):
        creator = NMMAInjectionCreator(
            self.make_args("--simple-setup"), label="my_label"
        )
        self.assertEqual(creator.label, "my_label")

    def test_the_random_generator_is_seeded_from_the_generation_seed(self):
        first = self.simple_creator().rng.random(3)
        second = self.simple_creator().rng.random(3)
        np.testing.assert_array_equal(first, second)

    def test_the_redraw_limit_is_taken_from_the_arguments(self):
        creator = self.eos_creator("--max-redraws", "4")
        self.assertEqual(creator.max_redraws, 4)

    def test_a_binary_type_without_an_eos_file_is_refused(self):
        # The filter has to evaluate the ejecta formula against an EOS, so
        # asking for one without the other cannot be honoured.
        with self.assertRaises(ValueError):
            self.make_creator("--binary-type", "BNS")

    def test_a_binary_type_with_an_eos_file_is_accepted(self):
        creator = self.eos_creator("--binary-type", "BNS")
        self.assertEqual(creator.binary_type_filter, "BNS")

    def test_no_binary_type_leaves_the_filter_off(self):
        self.assertIsNone(self.eos_creator().binary_type_filter)

    def test_no_external_injection_file_leaves_the_path_unset(self):
        self.assertIsNone(self.simple_creator().gw_injection_file)

    def test_an_external_injection_file_is_kept_as_a_path(self):
        creator = self.simple_creator("--gw-injection-file", "legacy.xml")
        self.assertEqual(creator.gw_injection_file, Path("legacy.xml"))

    def test_the_reference_frequency_is_carried_over(self):
        creator = self.simple_creator("--reference-frequency", "50")
        self.assertEqual(creator.reference_frequency, 50.0)


class TestSetupTestRoutines(InjectionCreatorMixin, unittest.TestCase):
    """Which routines run, and which conversions they need, is decided by
    the --tests specification."""

    def routine_names(self, creator):
        return [routine.__name__ for routine in creator.test_routines]

    def test_no_tests_means_no_routines(self):
        self.assertEqual(self.routine_names(self.eos_creator()), [])

    def test_an_ejecta_test_adds_the_ejecta_routine_and_conversion(self):
        creator = self.eos_creator("--tests", "ejecta")
        self.assertEqual(self.routine_names(creator), ["test_ejecta"])
        self.assertTrue(creator.conv_instructions["ejecta"])

    def test_a_population_test_adds_only_its_routine(self):
        creator = self.eos_creator("--tests", "population")
        self.assertEqual(self.routine_names(creator), ["test_population"])

    def test_several_tests_are_all_registered(self):
        creator = self.eos_creator("--tests", "ejecta", "population")
        self.assertEqual(
            sorted(self.routine_names(creator)), ["test_ejecta", "test_population"]
        )

    def test_an_snr_test_records_the_comparison_and_the_threshold(self):
        creator = self.eos_creator("--tests", "snr>12")
        self.assertEqual(self.routine_names(creator), ["test_snr"])
        self.assertIs(creator.snr_op, operator.gt)
        self.assertEqual(creator.snr_threshold, 12.0)

    def test_an_snr_test_sets_up_the_interferometers(self):
        creator = self.eos_creator("--tests", "snr>12")
        self.assertTrue(len(creator.ifos) > 0)

    def test_a_peak_magnitude_test_records_the_comparison_and_the_reference(self):
        with patch.object(injection_handling, "create_injection_model") as create_model:
            create_model.return_value = MagicMock(filters=["ztfg"])
            with patch.object(injection_handling.utils, "create_detection_limit"):
                creator = self.eos_creator(
                    "--tests", "peak_magnitude<22", "--em-model", "Bu2019lm"
                )
        self.assertEqual(self.routine_names(creator), ["test_detectability"])
        self.assertIs(creator.mag_op, operator.lt)
        self.assertEqual(creator.ref_mag, 22.0)

    def test_the_eos_conversion_is_always_registered(self):
        self.assertIn("eos", self.eos_creator().conv_instructions)

    def test_the_source_frame_conversion_is_registered_even_without_an_snr_test(self):
        # The EOS conversion needs source-frame masses, which only the
        # gravitational-wave step produces, so it is added regardless.
        creator = self.eos_creator("--tests", "ejecta")
        self.assertIs(
            creator.conv_instructions["gw"], injection_handling.bbh_source_frame
        )

    def test_an_snr_test_supplies_the_same_source_frame_conversion(self):
        creator = self.eos_creator("--tests", "snr>12")
        self.assertIs(
            creator.conv_instructions["gw"], injection_handling.bbh_source_frame
        )

    def test_a_hubble_constant_prior_adds_the_cosmology_conversion(self):
        prior_file = self.tmp_dir / "hubble.prior"
        prior_file.write_text(
            INJECTION_PRIOR
            + "Hubble_constant = Uniform(minimum=60, maximum=80, name='Hubble_constant')\n"
        )
        creator = self.make_creator(
            "--eos-file", str(self.eos_file), prior_file=prior_file
        )
        self.assertIn("cosmo", creator.conv_instructions)

    def test_without_a_hubble_prior_no_cosmology_conversion_is_added(self):
        self.assertNotIn("cosmo", self.eos_creator().conv_instructions)


class TestSetupPostProcessing(InjectionCreatorMixin, unittest.TestCase):
    def names(self, creator):
        return [step.__name__ for step in creator.postprocessing]

    def test_nothing_requested_leaves_a_single_no_op_step(self):
        creator = self.eos_creator()
        self.assertEqual(self.names(creator), ["dummy_postprocess"])

    def test_the_no_op_step_returns_the_table_unchanged(self):
        creator = self.eos_creator()
        frame = pd.DataFrame({"mass_1": [1.4]})
        self.assertIs(creator.postprocessing[0](frame), frame)

    def test_an_ejecta_step_is_registered(self):
        creator = self.eos_creator("--post-processing", "ejecta")
        self.assertEqual(self.names(creator), ["compute_ejecta"])

    def test_an_snr_step_is_registered_and_sets_up_the_interferometers(self):
        creator = self.eos_creator("--post-processing", "snr")
        self.assertEqual(self.names(creator), ["add_snrs"])
        self.assertTrue(len(creator.ifos) > 0)

    def test_an_snr_step_is_not_repeated_when_the_snr_is_already_tested(self):
        # The test routine already computes the SNR, so adding it again
        # would double the work.
        creator = self.eos_creator("--tests", "snr>12", "--post-processing", "snr")
        self.assertNotIn("add_snrs", self.names(creator))

    def test_a_lightcurve_step_is_registered(self):
        with patch.object(injection_handling, "create_injection_model") as create_model:
            create_model.return_value = MagicMock(filters=["ztfg"])
            with patch.object(injection_handling.utils, "create_detection_limit"):
                creator = self.eos_creator(
                    "--post-processing", "lightcurve", "--em-model", "Bu2019lm"
                )
        self.assertEqual(self.names(creator), ["prepare_lightcurves"])

    def test_several_steps_are_all_registered(self):
        creator = self.eos_creator("--post-processing", "snr", "ejecta")
        self.assertEqual(sorted(self.names(creator)), ["add_snrs", "compute_ejecta"])


class TestAdjustedPriorDraw(InjectionCreatorMixin, unittest.TestCase):
    """Every draw is put into the convention the rest of the code assumes:
    the primary is the heavier object."""

    def test_the_primary_is_never_lighter_than_the_secondary(self):
        creator = self.simple_creator()
        creator.columns_to_remove = None
        frame = creator.adjusted_prior_draw()
        self.assertTrue((frame["mass_1"] >= frame["mass_2"]).all())

    def test_the_masses_are_swapped_rather_than_resampled(self):
        creator = self.simple_creator()
        creator.columns_to_remove = None
        with patch.object(creator, "get_injection_dataframe") as draw:
            draw.return_value = pd.DataFrame(
                {"mass_1": [1.0, 2.0], "mass_2": [2.0, 1.0]}
            )
            frame = creator.adjusted_prior_draw()
        np.testing.assert_allclose(frame["mass_1"], [2.0, 2.0])
        np.testing.assert_allclose(frame["mass_2"], [1.0, 1.0])

    def test_a_draw_without_masses_is_left_alone(self):
        creator = self.simple_creator()
        creator.columns_to_remove = None
        with patch.object(creator, "get_injection_dataframe") as draw:
            draw.return_value = pd.DataFrame({"luminosity_distance": [40.0]})
            frame = creator.adjusted_prior_draw()
        self.assertEqual(frame.columns.tolist(), ["luminosity_distance"])

    def test_columns_already_supplied_elsewhere_are_dropped(self):
        creator = self.simple_creator()
        creator.columns_to_remove = ["luminosity_distance"]
        frame = creator.adjusted_prior_draw()
        self.assertNotIn("luminosity_distance", frame.columns)
        self.assertIn("mass_1", frame.columns)

    def test_the_right_number_of_samples_is_drawn(self):
        creator = self.simple_creator()
        creator.columns_to_remove = None
        self.assertEqual(len(creator.adjusted_prior_draw()), self.n_injection)


class TestGeneratePrelimDataframe(InjectionCreatorMixin, unittest.TestCase):
    def test_every_prior_parameter_is_drawn(self):
        frame = self.simple_creator().generate_prelim_dataframe()
        for parameter in ["mass_1", "mass_2", "luminosity_distance", "alpha"]:
            self.assertIn(parameter, frame.columns, msg=parameter)

    def test_each_row_is_numbered_for_later_reference(self):
        frame = self.simple_creator().generate_prelim_dataframe()
        self.assertIn("simulation_id", frame.columns)
        self.assertEqual(frame["simulation_id"].tolist(), list(range(self.n_injection)))

    def test_the_requested_number_of_rows_is_produced(self):
        frame = self.simple_creator().generate_prelim_dataframe()
        self.assertEqual(len(frame), self.n_injection)

    def test_the_columns_coming_from_the_prior_are_recorded(self):
        creator = self.simple_creator()
        creator.generate_prelim_dataframe()
        self.assertIn("mass_1", creator.use_prior_columns)

    def test_nothing_is_marked_for_removal_without_an_external_file(self):
        creator = self.simple_creator()
        creator.generate_prelim_dataframe()
        self.assertEqual(creator.columns_to_remove, [])

    def test_an_external_file_decides_how_many_injections_are_made(self):
        # The external file fixes the set of systems, so its length wins
        # over the requested number.
        creator = self.simple_creator()
        external = pd.DataFrame({"mass_1": [1.4, 1.5], "mass_2": [1.3, 1.2]})
        with patch.object(
            creator, "handle_incomplete_injection_file", return_value=external
        ):
            frame = creator.generate_prelim_dataframe()
        self.assertEqual(creator.n_injection, 2)
        self.assertEqual(len(frame), 2)

    def test_parameters_from_an_external_file_are_not_resampled(self):
        creator = self.simple_creator()
        external = pd.DataFrame({"mass_1": [1.9, 1.8], "mass_2": [1.3, 1.2]})
        with patch.object(
            creator, "handle_incomplete_injection_file", return_value=external
        ):
            frame = creator.generate_prelim_dataframe()
        np.testing.assert_allclose(frame["mass_1"], [1.9, 1.8])
        self.assertEqual(sorted(creator.columns_to_remove), ["mass_1", "mass_2"])

    def test_the_remaining_parameters_are_still_drawn_from_the_prior(self):
        creator = self.simple_creator()
        external = pd.DataFrame({"mass_1": [1.9, 1.8], "mass_2": [1.3, 1.2]})
        with patch.object(
            creator, "handle_incomplete_injection_file", return_value=external
        ):
            frame = creator.generate_prelim_dataframe()
        self.assertIn("luminosity_distance", frame.columns)
        self.assertFalse(frame["luminosity_distance"].isna().any())

    def test_an_existing_simulation_id_is_kept(self):
        creator = self.simple_creator()
        external = pd.DataFrame({"simulation_id": [7, 9], "mass_1": [1.9, 1.8]})
        with patch.object(
            creator, "handle_incomplete_injection_file", return_value=external
        ):
            frame = creator.generate_prelim_dataframe()
        self.assertEqual(frame["simulation_id"].tolist(), [7, 9])


class TestHandleIncompleteInjectionFile(InjectionCreatorMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.creator = self.simple_creator()

    def test_no_file_gives_an_empty_table(self):
        frame = self.creator.handle_incomplete_injection_file(None)
        self.assertTrue(frame.empty)

    def test_an_unknown_format_is_refused(self):
        with self.assertRaises(ValueError):
            self.creator.handle_incomplete_injection_file(Path("injections.txt"))

    def test_a_json_injection_file_is_read(self):
        path = self.tmp_dir / "legacy.json"
        frame = pd.DataFrame({"mass_1": [1.4], "mass_2": [1.3]})
        with patch.object(
            injection_handling, "read_injection_file", return_value=frame
        ) as read:
            result = self.creator.handle_incomplete_injection_file(path)
        read.assert_called_once_with(path)
        self.assertIs(result, frame)

    def test_a_legacy_table_goes_through_the_conversion_reader(self):
        path = self.tmp_dir / "legacy.xml"
        self.creator.trigger_time = 0.0
        with patch.object(self.creator, "file_to_dataframe") as reader:
            self.creator.handle_incomplete_injection_file(path)
        reader.assert_called_once_with(
            path, self.creator.reference_frequency, trigger_time=0.0
        )

    def test_every_accepted_suffix_is_dispatched(self):
        self.creator.trigger_time = 0.0
        with patch.object(self.creator, "file_to_dataframe"):
            with patch.object(injection_handling, "read_injection_file"):
                for suffix in [".json", ".xml", ".dat"]:
                    self.creator.handle_incomplete_injection_file(
                        Path(f"legacy{suffix}")
                    )


class TestTestWrap(InjectionCreatorMixin, unittest.TestCase):
    """Each candidate table is converted, checked against the constraint
    priors and then handed to every test routine."""

    def test_the_conversion_adds_the_derived_parameters(self):
        creator = self.eos_creator("--tests", "ejecta")
        frame = creator.test_wrap(creator.generate_prelim_dataframe())
        for parameter in ["mass_1_source", "radius_1", "log10_mej_dyn", "TOV_mass"]:
            self.assertIn(parameter, frame.columns, msg=parameter)

    def test_a_pass_or_fail_column_is_added(self):
        creator = self.eos_creator("--tests", "ejecta")
        frame = creator.test_wrap(creator.generate_prelim_dataframe())
        self.assertIn("tests_passed", frame.columns)
        self.assertEqual(len(frame["tests_passed"]), self.n_injection)

    def test_the_constraints_are_evaluated_per_column_not_over_the_table(self):
        # Passing the DataFrame itself makes bilby fall back to accepting
        # every row, so the columns are handed over as a plain dictionary.
        creator = self.eos_creator()
        frame = creator.generate_prelim_dataframe()
        with patch.object(
            creator.priors, "evaluate_constraints", return_value=np.ones(len(frame))
        ) as evaluate:
            creator.test_wrap(frame)
        passed = evaluate.call_args.args[0]
        self.assertIsInstance(passed, dict)
        self.assertIn("mass_1", passed)

    def test_every_routine_is_run_on_the_converted_table(self):
        creator = self.eos_creator()
        first, second = MagicMock(), MagicMock()
        creator.test_routines = [first, second]
        creator.test_wrap(creator.generate_prelim_dataframe())
        first.assert_called_once()
        second.assert_called_once()
        self.assertIn("tests_passed", first.call_args.args[0].columns)

    def test_the_original_table_is_not_modified(self):
        creator = self.eos_creator("--tests", "ejecta")
        frame = creator.generate_prelim_dataframe()
        original = frame.columns.tolist()
        creator.test_wrap(frame)
        self.assertEqual(frame.columns.tolist(), original)


class TestTestRoutines(InjectionCreatorMixin, unittest.TestCase):
    """Each routine narrows an existing pass-or-fail column in place."""

    def setUp(self):
        super().setUp()
        self.creator = self.eos_creator()

    def frame(self, **columns):
        columns.setdefault("tests_passed", [True] * len(next(iter(columns.values()))))
        return pd.DataFrame(columns)

    def test_finite_ejecta_masses_pass_the_ejecta_test(self):
        frame = self.frame(log10_mej_dyn=[-2.0, -3.0], log10_mej_wind=[-2.0, -2.5])
        self.creator.test_ejecta(frame)
        self.assertTrue(frame["tests_passed"].all())

    def test_a_non_finite_dynamical_ejecta_mass_fails(self):
        frame = self.frame(log10_mej_dyn=[-2.0, -np.inf], log10_mej_wind=[-2.0, -2.5])
        self.creator.test_ejecta(frame)
        self.assertEqual(frame["tests_passed"].tolist(), [True, False])

    def test_a_non_finite_wind_ejecta_mass_fails(self):
        frame = self.frame(log10_mej_dyn=[-2.0], log10_mej_wind=[np.nan])
        self.creator.test_ejecta(frame)
        self.assertFalse(frame["tests_passed"].iloc[0])

    def test_a_row_that_already_failed_stays_failed(self):
        frame = pd.DataFrame(
            {
                "log10_mej_dyn": [-2.0],
                "log10_mej_wind": [-2.0],
                "tests_passed": [False],
            }
        )
        self.creator.test_ejecta(frame)
        self.assertFalse(frame["tests_passed"].iloc[0])

    def test_a_secondary_below_one_solar_mass_fails_the_population_test(self):
        frame = self.frame(
            mass_1=[1.8, 1.8], mass_2=[1.6, 0.5], mass_2_source=[1.6, 0.5]
        )
        with patch.object(
            injection_handling, "rejection_sample", return_value=(None, np.ones(2))
        ):
            self.creator.test_population(frame)
        self.assertEqual(frame["tests_passed"].tolist(), [True, False])

    def test_the_population_test_rejection_samples_on_the_mass_ratio(self):
        frame = self.frame(mass_1=[1.8], mass_2=[1.6], mass_2_source=[1.6])
        with patch.object(
            injection_handling, "rejection_sample", return_value=(None, np.array([0]))
        ) as sampler:
            self.creator.test_population(frame)
        self.assertFalse(frame["tests_passed"].iloc[0])
        self.assertIs(sampler.call_args.args[2], self.creator.rng)

    def test_the_snr_test_compares_the_computed_snr_with_the_threshold(self):
        self.creator.snr_op, self.creator.snr_threshold = operator.gt, 12.0
        frame = self.frame(mass_1=[1.4, 1.4])

        def add_snrs(df):
            df["snr"] = [20.0, 5.0]
            return df

        with patch.object(self.creator, "add_snrs", side_effect=add_snrs):
            self.creator.test_snr(frame)
        self.assertEqual(frame["tests_passed"].tolist(), [True, False])

    def test_the_detectability_test_accepts_a_lightcurve_reaching_the_limit(self):
        self.creator.mag_op, self.creator.ref_mag = operator.lt, 22.0
        self.creator.lc_model = MagicMock()
        self.creator.lc_model.gen_detector_lc.return_value = (
            np.array([1.0]),
            {"ztfg": np.array([20.0])},
        )
        frame = self.frame(mass_1=[1.4])
        self.creator.test_detectability(frame)
        self.assertTrue(frame["tests_passed"].iloc[0])

    def test_the_detectability_test_rejects_a_lightcurve_that_stays_faint(self):
        self.creator.mag_op, self.creator.ref_mag = operator.lt, 22.0
        self.creator.lc_model = MagicMock()
        self.creator.lc_model.gen_detector_lc.return_value = (
            np.array([1.0]),
            {"ztfg": np.array([25.0])},
        )
        frame = self.frame(mass_1=[1.4])
        self.creator.test_detectability(frame)
        self.assertFalse(frame["tests_passed"].iloc[0])

    def test_one_bright_filter_is_enough_to_be_detectable(self):
        self.creator.mag_op, self.creator.ref_mag = operator.lt, 22.0
        self.creator.lc_model = MagicMock()
        self.creator.lc_model.gen_detector_lc.return_value = (
            np.array([1.0]),
            {"ztfg": np.array([25.0]), "ztfr": np.array([19.0])},
        )
        frame = self.frame(mass_1=[1.4])
        self.creator.test_detectability(frame)
        self.assertTrue(frame["tests_passed"].iloc[0])


class TestRefillFailedTests(InjectionCreatorMixin, unittest.TestCase):
    """Failed draws are replaced from fresh prior samples and retested,
    until everything passes or the redraw budget runs out."""

    def setUp(self):
        super().setUp()
        self.creator = self.simple_creator()
        self.creator.use_prior_columns = ["mass_1"]
        self.creator.max_redraws = 3
        self.creator.test_routines = []

    def frame(self, masses, passed):
        return pd.DataFrame({"mass_1": masses, "tests_passed": passed})

    def always_passes(self, df):
        df = df.copy()
        df["tests_passed"] = True
        return df

    def always_fails(self, df):
        df = df.copy()
        df["tests_passed"] = False
        return df

    def test_a_table_that_already_passes_is_returned_unchanged(self):
        frame = self.frame([1.4, 1.5], [True, True])
        with patch.object(self.creator, "adjusted_prior_draw") as draw:
            draw.return_value = pd.DataFrame({"mass_1": [9.0]})
            result = self.creator.refill_failed_tests(frame)
        np.testing.assert_allclose(result["mass_1"], [1.4, 1.5])

    def test_a_failed_row_is_replaced_by_a_fresh_draw(self):
        frame = self.frame([1.4, 1.5], [True, False])
        with patch.object(self.creator, "adjusted_prior_draw") as draw:
            draw.return_value = pd.DataFrame({"mass_1": [9.0, 9.1]})
            with patch.object(
                self.creator, "test_wrap", side_effect=self.always_passes
            ):
                result = self.creator.refill_failed_tests(frame)
        np.testing.assert_allclose(result["mass_1"], [1.4, 9.0])

    def test_a_row_that_passed_first_time_is_never_redrawn(self):
        frame = self.frame([1.4, 1.5], [False, True])
        with patch.object(self.creator, "adjusted_prior_draw") as draw:
            draw.return_value = pd.DataFrame({"mass_1": [9.0, 9.1]})
            with patch.object(
                self.creator, "test_wrap", side_effect=self.always_passes
            ):
                result = self.creator.refill_failed_tests(frame)
        self.assertEqual(result["mass_1"].iloc[1], 1.5)

    def test_the_retest_result_is_written_back_for_the_redrawn_rows(self):
        # Every column the retest recomputes has to land back on the row,
        # not just the pass-or-fail flag.
        frame = pd.DataFrame(
            {"mass_1": [1.4, 1.5], "derived": [0.0, 0.0], "tests_passed": [True, False]}
        )

        def retest(df):
            df = df.copy()
            df["derived"] = 5.0
            df["tests_passed"] = True
            return df

        with patch.object(self.creator, "adjusted_prior_draw") as draw:
            draw.return_value = pd.DataFrame({"mass_1": [9.0, 9.1]})
            with patch.object(self.creator, "test_wrap", side_effect=retest):
                result = self.creator.refill_failed_tests(frame)
        self.assertEqual(result["derived"].tolist(), [0.0, 5.0])

    def test_more_draws_are_taken_when_the_reserve_runs_out(self):
        frame = self.frame([1.4, 1.5, 1.6], [False, False, False])
        with patch.object(
            self.creator,
            "adjusted_prior_draw",
            side_effect=lambda: pd.DataFrame({"mass_1": [9.0, 9.1]}),
        ) as draw:
            with patch.object(
                self.creator, "test_wrap", side_effect=self.always_passes
            ):
                self.creator.refill_failed_tests(frame)
        self.assertGreater(draw.call_count, 1)

    def test_exhausting_the_redraw_budget_is_an_error(self):
        frame = self.frame([1.4], [False])
        with patch.object(self.creator, "adjusted_prior_draw") as draw:
            draw.return_value = pd.DataFrame({"mass_1": [9.0] * 10})
            with patch.object(self.creator, "test_wrap", side_effect=self.always_fails):
                with self.assertRaises(ValueError):
                    self.creator.refill_failed_tests(frame)

    def test_the_error_names_the_redraw_limit_as_the_thing_to_raise(self):
        frame = self.frame([1.4], [False])
        with patch.object(self.creator, "adjusted_prior_draw") as draw:
            draw.return_value = pd.DataFrame({"mass_1": [9.0] * 10})
            with patch.object(self.creator, "test_wrap", side_effect=self.always_fails):
                with self.assertRaises(ValueError) as caught:
                    self.creator.refill_failed_tests(frame)
        self.assertIn("max_redraws", str(caught.exception))


class TestTestingAndPostprocessing(InjectionCreatorMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.creator = self.eos_creator("--tests", "ejecta")

    def test_the_bookkeeping_column_is_not_written_to_the_injection_file(self):
        frame = self.creator.testing_and_postprocessing(
            self.creator.generate_prelim_dataframe()
        )
        self.assertNotIn("tests_passed", frame.columns)

    def test_the_derived_parameters_are_present_in_the_result(self):
        frame = self.creator.testing_and_postprocessing(
            self.creator.generate_prelim_dataframe()
        )
        for parameter in ["mass_1_source", "lambda_1", "log10_mej_dyn"]:
            self.assertIn(parameter, frame.columns, msg=parameter)

    def test_only_the_sampled_parameters_are_kept_when_asked(self):
        creator = self.eos_creator("--tests", "ejecta", "--original-parameters")
        frame = creator.testing_and_postprocessing(creator.generate_prelim_dataframe())
        self.assertNotIn("tests_passed", frame.columns)
        self.assertIn("mass_1", frame.columns)

    def test_every_postprocessing_step_is_applied(self):
        step = MagicMock()
        self.creator.postprocessing = [step]
        self.creator.testing_and_postprocessing(
            self.creator.generate_prelim_dataframe()
        )
        step.assert_called_once()

    def test_a_redraw_still_leaves_the_derived_columns_populated(self):
        # Rows that passed on the first draw keep the columns the
        # conversion added, rather than losing them to the original table.
        frame = self.creator.generate_prelim_dataframe()
        tested = self.creator.test_wrap(frame)
        tested.loc[0, "tests_passed"] = 0
        with patch.object(self.creator, "test_wrap", return_value=tested):
            with patch.object(
                self.creator, "refill_failed_tests", side_effect=lambda df: df
            ):
                result = self.creator.testing_and_postprocessing(frame)
        self.assertIn("mass_1_source", result.columns)
        self.assertFalse(result["mass_1_source"].isna().any())

    def test_the_binary_type_filter_runs_when_requested(self):
        creator = self.eos_creator("--binary-type", "BNS")
        with patch.object(
            creator, "filter_by_binary_type", side_effect=lambda df: df
        ) as filter_step:
            creator.testing_and_postprocessing(creator.generate_prelim_dataframe())
        filter_step.assert_called_once()

    def test_the_binary_type_filter_is_skipped_when_not_requested(self):
        with patch.object(self.creator, "filter_by_binary_type") as filter_step:
            self.creator.testing_and_postprocessing(
                self.creator.generate_prelim_dataframe()
            )
        filter_step.assert_not_called()


class TestFilterByBinaryType(InjectionCreatorMixin, unittest.TestCase):
    """A one-shot filter that applies one binary type's ejecta formula to
    every row and drops those the chosen EOS cannot support."""

    def setUp(self):
        super().setUp()
        self.creator = self.eos_creator("--binary-type", "BNS")

    def converted_frame(self):
        return self.creator.param_conversion.core_conversion(
            self.creator.generate_prelim_dataframe()
        )

    def test_an_unknown_binary_type_is_refused(self):
        self.creator.binary_type_filter = "BBH"
        with self.assertRaises(ValueError):
            self.creator.filter_by_binary_type(pd.DataFrame({"mass_1": [1.4]}))

    def test_the_ejecta_masses_are_computed_for_every_row(self):
        frame = self.creator.filter_by_binary_type(self.converted_frame())
        self.assertIn("log10_mej_dyn", frame.columns)
        self.assertIn("log10_mej_wind", frame.columns)

    def test_a_consistent_binary_keeps_all_of_its_injections(self):
        frame = self.converted_frame()
        self.assertEqual(len(self.creator.filter_by_binary_type(frame)), len(frame))

    def test_rows_with_a_non_finite_ejecta_mass_are_dropped(self):
        stub = MagicMock()
        stub.mass_fitting_keys = ["log10_mej_dyn", "log10_mej_wind"]
        stub.ejecta_parameter_conversion.return_value = [
            np.array([-2.0, -np.inf]),
            np.array([-2.0, -2.0]),
        ]
        with patch.object(injection_handling, "BNSEjectaFitting", return_value=stub):
            frame = self.creator.filter_by_binary_type(
                pd.DataFrame({"mass_1": [1.8, 1.9]})
            )
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame["mass_1"].tolist(), [1.8])

    def test_the_surviving_rows_are_renumbered(self):
        stub = MagicMock()
        stub.mass_fitting_keys = ["log10_mej_dyn", "log10_mej_wind"]
        stub.ejecta_parameter_conversion.return_value = [
            np.array([-np.inf, -2.0]),
            np.array([-2.0, -2.0]),
        ]
        with patch.object(injection_handling, "BNSEjectaFitting", return_value=stub):
            frame = self.creator.filter_by_binary_type(
                pd.DataFrame({"mass_1": [1.8, 1.9]})
            )
        self.assertEqual(frame.index.tolist(), [0])

    def test_an_already_sampled_ejecta_mass_is_overwritten(self):
        # The flag is an explicit request to recompute the ejecta from this
        # EOS, so a prior that samples those keys must not win.
        stub = MagicMock()
        stub.mass_fitting_keys = ["log10_mej_dyn", "log10_mej_wind"]
        stub.ejecta_parameter_conversion.return_value = [
            np.array([-2.0]),
            np.array([-2.5]),
        ]
        frame = pd.DataFrame(
            {"mass_1": [1.8], "log10_mej_dyn": [-9.0], "log10_mej_wind": [-9.0]}
        )
        with patch.object(injection_handling, "BNSEjectaFitting", return_value=stub):
            result = self.creator.filter_by_binary_type(frame)
        self.assertEqual(result["log10_mej_dyn"].iloc[0], -2.0)
        self.assertEqual(result["log10_mej_wind"].iloc[0], -2.5)

    def test_the_neutron_star_black_hole_formula_can_be_chosen(self):
        self.creator.binary_type_filter = "NSBH"
        stub = MagicMock()
        stub.mass_fitting_keys = ["log10_mej_dyn", "log10_mej_wind"]
        stub.ejecta_parameter_conversion.return_value = [
            np.array([-2.0]),
            np.array([-2.0]),
        ]
        with patch.object(injection_handling, "NSBHEjectaFitting", return_value=stub):
            frame = self.creator.filter_by_binary_type(pd.DataFrame({"mass_1": [1.8]}))
        self.assertEqual(len(frame), 1)


class TestGenerateInjectionFile(InjectionCreatorMixin, unittest.TestCase):
    def test_the_injection_file_is_written(self):
        creator = self.eos_creator("--tests", "ejecta")
        creator.generate_injection_file()
        self.assertTrue(Path(creator.filename).is_file())

    def test_the_file_holds_the_requested_number_of_injections(self):
        creator = self.eos_creator("--tests", "ejecta")
        creator.generate_injection_file()
        content = json.loads(Path(creator.filename).read_text())["injections"][
            "content"
        ]
        self.assertEqual(len(content["mass_1"]), self.n_injection)

    def test_the_derived_parameters_reach_the_file(self):
        creator = self.eos_creator("--tests", "ejecta")
        creator.generate_injection_file()
        content = json.loads(Path(creator.filename).read_text())["injections"][
            "content"
        ]
        for parameter in ["mass_1_source", "lambda_1", "log10_mej_dyn"]:
            self.assertIn(parameter, content, msg=parameter)

    def test_the_bilby_random_generator_is_seeded_before_drawing(self):
        # The prior draws come from bilby's internal generator, so the
        # generation seed only takes effect if it is seeded here.
        creator = self.simple_creator()
        with patch("bilby.core.utils.random.seed") as seed:
            creator.generate_injection_file()
        seed.assert_called_once_with(creator.generation_seed)

    def test_the_same_seed_gives_the_same_injections(self):
        first = self.simple_creator()
        first.generate_injection_file()
        first_content = Path(first.filename).read_text()
        second = self.simple_creator()
        second.generate_injection_file()
        self.assertEqual(first_content, Path(second.filename).read_text())

    def test_a_different_seed_gives_different_injections(self):
        first = self.simple_creator()
        first.generate_injection_file()
        first_content = Path(first.filename).read_text()
        second = self.simple_creator("--generation-seed", "1234")
        second.generate_injection_file()
        self.assertNotEqual(first_content, Path(second.filename).read_text())

    def test_a_simple_setup_skips_the_tests_entirely(self):
        creator = self.simple_creator()
        with patch.object(creator, "testing_and_postprocessing") as tests:
            creator.generate_injection_file()
        tests.assert_not_called()

    def test_a_full_setup_runs_the_tests(self):
        creator = self.eos_creator()
        with patch.object(
            creator, "testing_and_postprocessing", side_effect=lambda df: df
        ) as tests:
            creator.generate_injection_file()
        tests.assert_called_once()


class TestComputeEjecta(InjectionCreatorMixin, unittest.TestCase):
    def test_the_ejecta_parameters_are_added_to_the_table(self):
        creator = self.eos_creator()
        frame = creator.param_conversion.core_conversion(
            creator.generate_prelim_dataframe()
        )
        result = creator.compute_ejecta(frame)
        for parameter in ["log10_mej_dyn", "log10_mej_wind", "log10_mej"]:
            self.assertIn(parameter, result.columns, msg=parameter)

    def test_the_fitting_object_is_instantiated_before_it_is_called(self):
        # The fitting class takes no constructor arguments; only its call
        # accepts the table.
        creator = self.eos_creator()
        frame = pd.DataFrame({"mass_1": [1.8]})
        with patch.object(injection_handling, "KilonovaEjectaFitting") as fitting:
            creator.compute_ejecta(frame)
        fitting.assert_called_once_with()
        fitting.return_value.assert_called_once_with(frame)


class TestInitialiseIfos(unittest.TestCase):
    """The SNR test needs a detector network and a waveform generator, built
    from the requested detectors."""

    def creator(self, detectors, waveform_arguments=None):
        creator = object.__new__(NMMAInjectionCreator)
        creator.initialise_ifos(
            Namespace(
                gw_detectors=detectors,
                waveform_arguments=waveform_arguments or {},
            )
        )
        return creator

    def test_a_comma_separated_string_is_split_into_detectors(self):
        creator = self.creator("H1,L1")
        self.assertEqual([ifo.name for ifo in creator.ifos], ["H1", "L1"])

    def test_a_list_of_detectors_is_used_directly(self):
        creator = self.creator(["H1", "V1"])
        self.assertEqual([ifo.name for ifo in creator.ifos], ["H1", "V1"])

    def test_the_einstein_telescope_expands_into_its_three_arms(self):
        creator = self.creator(["ET"])
        self.assertEqual([ifo.name for ifo in creator.ifos], ["ET1", "ET2", "ET3"])

    def test_the_einstein_telescope_is_appended_after_the_other_detectors(self):
        creator = self.creator(["ET", "CE"])
        self.assertEqual(
            [ifo.name for ifo in creator.ifos], ["CE", "ET1", "ET2", "ET3"]
        )

    def test_the_lowest_usable_frequency_is_the_most_restrictive_one(self):
        creator = self.creator("H1,L1")
        self.assertEqual(
            creator.f_min, max(ifo.minimum_frequency for ifo in creator.ifos)
        )

    def test_the_sampling_frequency_satisfies_nyquist_for_the_network(self):
        creator = self.creator("H1,L1")
        highest = min(ifo.maximum_frequency for ifo in creator.ifos)
        self.assertEqual(creator.sampling_frequency, 2 * highest)

    def test_the_segment_is_long_enough_for_an_early_warning_signal(self):
        self.assertEqual(self.creator("H1,L1").duration, 2048.0)

    def test_a_tidal_waveform_is_used_by_default(self):
        creator = self.creator("H1,L1")
        self.assertEqual(
            creator.waveform_gen.waveform_arguments["waveform_approximant"],
            "IMRPhenomXAS_NRTidalv3",
        )

    def test_the_waveform_arguments_can_be_overridden(self):
        creator = self.creator(
            "H1,L1", {"waveform_approximant": "IMRPhenomPv2_NRTidal"}
        )
        self.assertEqual(
            creator.waveform_gen.waveform_arguments["waveform_approximant"],
            "IMRPhenomPv2_NRTidal",
        )

    def test_the_frequency_limits_follow_the_network(self):
        creator = self.creator("H1,L1")
        arguments = creator.waveform_gen.waveform_arguments
        self.assertEqual(arguments["minimum_frequency"], creator.f_min)


class TestFileToDataframe(InjectionCreatorMixin, unittest.TestCase):
    """The legacy reader converts an external table into NMMA's parameter
    names, and needs the ligo.lw library to do it."""

    def setUp(self):
        super().setUp()
        self.creator = self.simple_creator()

    def ligo_lw_available(self):
        try:
            import ligo.lw  # noqa: F401
        except ImportError:
            return False
        return True

    def test_the_missing_library_is_reported_with_how_to_install_it(self):
        if self.ligo_lw_available():
            self.skipTest("ligo.lw is installed, so the guard cannot fire")
        with self.assertRaises(ImportError) as caught:
            self.creator.file_to_dataframe(Path("legacy.xml"), 20.0)
        self.assertIn("python-ligo-lw", str(caught.exception))

    def test_the_dependency_is_checked_before_the_format_is_looked_at(self):
        # The import guard runs first, so an unsupported suffix is only
        # reported once the library is present.
        if self.ligo_lw_available():
            self.skipTest("ligo.lw is installed, so the guard cannot fire")
        with self.assertRaises(ImportError):
            self.creator.file_to_dataframe(Path("legacy.txt"), 20.0)

    def test_an_unsupported_format_is_refused(self):
        if not self.ligo_lw_available():
            self.skipTest("ligo.lw is not installed")
        with self.assertRaises(ValueError):
            self.creator.file_to_dataframe(Path("legacy.txt"), 20.0)


class TestMultiRunSetup(unittest.TestCase):
    """The slurm helper writes one directory per injection, each with its
    own prior and an analysis script filled in from a template."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.analysis_file = self.tmp_dir / "analysis.sh"
        self.analysis_file.write_text(
            "lightcurve-analysis --prior PRIOR --outdir OUTDIR "
            "--injection-file INJOUT --injection-num INJNUM\n"
        )
        self.args = Namespace(
            outdir=str(self.tmp_dir), analysis_file=str(self.analysis_file)
        )
        self.creator = MagicMock()
        self.creator.generate_prelim_dataframe.return_value = pd.DataFrame(
            {"mass_1": [1.4, 1.5]}
        )
        patch.object(
            injection_handling, "parsing_and_logging", return_value=self.args
        ).start()
        patch.object(
            injection_handling, "NMMAInjectionCreator", return_value=self.creator
        ).start()
        self.addCleanup(patch.stopall)
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def script(self, index):
        return (self.tmp_dir / str(index) / "inference.sh").read_text()

    def test_one_directory_is_made_per_injection(self):
        injection_handling.multi_run_setup()
        self.assertTrue((self.tmp_dir / "0").is_dir())
        self.assertTrue((self.tmp_dir / "1").is_dir())

    def test_each_directory_gets_its_own_analysis_script(self):
        injection_handling.multi_run_setup()
        self.assertTrue((self.tmp_dir / "0" / "inference.sh").is_file())

    def test_the_prior_placeholder_points_at_the_written_prior(self):
        injection_handling.multi_run_setup()
        self.assertIn(str(self.tmp_dir / "0" / "injection.prior"), self.script(0))

    def test_the_output_placeholder_points_at_the_run_directory(self):
        injection_handling.multi_run_setup()
        self.assertIn(f"--outdir {self.tmp_dir / '0'}", self.script(0))

    def test_the_lightcurve_placeholder_points_into_the_run_directory(self):
        injection_handling.multi_run_setup()
        self.assertIn(str(self.tmp_dir / "0" / "lc.csv"), self.script(0))

    def test_the_injection_number_is_the_row_index(self):
        injection_handling.multi_run_setup()
        self.assertIn("--injection-num 1", self.script(1))

    def test_no_placeholder_is_left_behind(self):
        injection_handling.multi_run_setup()
        for placeholder in ["PRIOR", "OUTDIR", "INJOUT", "INJNUM"]:
            self.assertNotIn(placeholder, self.script(0), msg=placeholder)

    def test_the_prior_is_written_into_each_directory(self):
        injection_handling.multi_run_setup()
        self.assertEqual(self.creator.priors.to_file.call_count, 2)


class TestGenerateInjectionEntryPoint(unittest.TestCase):
    def test_the_arguments_are_parsed_when_none_are_given(self):
        with patch.object(
            injection_handling, "nmma_base_parsing", return_value=Namespace()
        ) as parsing:
            with patch.object(injection_handling, "NMMAInjectionCreator"):
                injection_handling.generate_injection()
        parsing.assert_called_once_with(injection_handling.injection_parsing)

    def test_given_arguments_are_used_without_reparsing(self):
        args = Namespace(label="run")
        with patch.object(injection_handling, "nmma_base_parsing") as parsing:
            with patch.object(
                injection_handling, "NMMAInjectionCreator"
            ) as creator_class:
                injection_handling.generate_injection(args)
        parsing.assert_not_called()
        creator_class.assert_called_once_with(args)

    def test_the_injection_file_is_generated(self):
        with patch.object(injection_handling, "NMMAInjectionCreator") as creator_class:
            injection_handling.generate_injection(Namespace())
        creator_class.return_value.generate_injection_file.assert_called_once_with()

    def test_the_module_main_forwards_to_the_generator(self):
        args = Namespace(label="run")
        with patch.object(injection_handling, "generate_injection") as generate:
            injection_handling.main(args)
        generate.assert_called_once_with(args)


if __name__ == "__main__":
    unittest.main()
