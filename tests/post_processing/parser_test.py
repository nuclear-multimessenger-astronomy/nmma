import argparse
import shutil
import tempfile
import unittest
from pathlib import Path

from nmma.core.parsing import nmma_base_parsing
from nmma.post_processing import parser as pp_parser


class ParserMixin:
    """Each parsing function takes a bare parser and returns it with the
    arguments of one post-processing script attached."""

    def setUp(self):
        self.parser = argparse.ArgumentParser()

    def group_titles(self, parser):
        return [group.title for group in self.parser._action_groups]


class TestJointPostprocessParser(ParserMixin, unittest.TestCase):
    """The settings every trend script shares: where to write, how wide a
    credible interval to quote and how many realisations to average over."""

    def setUp(self):
        super().setUp()
        self.parser = pp_parser.joint_postprocess_parser(self.parser)

    def test_the_output_directory_is_required(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args([])

    def test_the_output_directory_is_parsed(self):
        self.assertEqual(self.parser.parse_args(["--outdir", "out"]).outdir, "out")

    def test_the_credible_interval_defaults_to_ninety_five_percent(self):
        self.assertAlmostEqual(
            self.parser.parse_args(["--outdir", "out"]).cred_interval, 0.95
        )

    def test_the_credible_interval_is_a_float(self):
        args = self.parser.parse_args(["--outdir", "out", "--cred-interval", "0.68"])
        self.assertAlmostEqual(args.cred_interval, 0.68)

    def test_the_random_seed_defaults_to_forty_two(self):
        self.assertEqual(self.parser.parse_args(["--outdir", "out"]).seed, 42)

    def test_the_detections_file_has_two_aliases(self):
        for flag in ["-d", "--detections-file", "--detectable"]:
            args = self.parser.parse_args(["--outdir", "out", flag, "det.dat"])
            self.assertEqual(args.detections_file, "det.dat", msg=flag)

    def test_the_event_count_is_an_optional_integer(self):
        args = self.parser.parse_args(["--outdir", "out"])
        self.assertIsNone(args.Nevent)
        args = self.parser.parse_args(["--outdir", "out", "--Nevent", "50"])
        self.assertEqual(args.Nevent, 50)

    def test_the_reordering_and_posterior_counts_have_defaults(self):
        args = self.parser.parse_args(["--outdir", "out"])
        self.assertEqual(args.N_reordering, 100)
        self.assertEqual(args.N_posterior_samples, 10000)

    def test_the_reordering_and_posterior_counts_are_integers(self):
        args = self.parser.parse_args(
            ["--outdir", "out", "--N-reordering", "5", "--N-posterior-samples", "7"]
        )
        self.assertEqual(args.N_reordering, 5)
        self.assertEqual(args.N_posterior_samples, 7)


class TestR14Parser(ParserMixin, unittest.TestCase):
    """The radius-trend script, reached through the combine-EOS console
    script."""

    required = [
        "--outdir",
        "out",
        "--label",
        "run",
        "--gwR14trend",
        "gw.dat",
        "--GWEMsamples",
        "samples",
        "--Neos",
        "5000",
        "--EOSpath",
        "eos",
    ]

    def setUp(self):
        super().setUp()
        self.parser = pp_parser.R14_parser(self.parser)

    def test_the_shared_trend_arguments_are_composed_in(self):
        args = self.parser.parse_args(self.required)
        self.assertAlmostEqual(args.cred_interval, 0.95)
        self.assertEqual(args.N_reordering, 100)

    def test_every_required_argument_is_enforced(self):
        for index in range(0, len(self.required), 2):
            trimmed = self.required[:index] + self.required[index + 2 :]
            with self.assertRaises(SystemExit, msg=self.required[index]):
                self.parser.parse_args(trimmed)

    def test_the_true_radius_has_a_default(self):
        self.assertAlmostEqual(self.parser.parse_args(self.required).R14_true, 11.55)

    def test_the_equation_of_state_count_is_an_integer(self):
        self.assertEqual(self.parser.parse_args(self.required).Neos, 5000)

    def test_the_equation_of_state_prior_is_optional(self):
        self.assertIsNone(self.parser.parse_args(self.required).EOS_prior)

    def test_the_detection_probability_is_optional(self):
        self.assertIsNone(self.parser.parse_args(self.required).pdet)

    def test_the_eos_directory_destination_is_lower_case(self):
        # The trend script reads this value as args.EOSPath, with a capital
        # P, which the parser never sets. Either the flag or the reader has
        # to change for combine-EOS to run at all.
        args = self.parser.parse_args(self.required)
        self.assertTrue(hasattr(args, "EOSpath"))
        self.assertFalse(hasattr(args, "EOSPath"))

    def test_the_description_names_the_quantity_being_trended(self):
        self.assertIn("R14", self.parser.description)


class TestHubbleParser(ParserMixin, unittest.TestCase):
    """The Hubble-constant trend script, reached through the
    gwem-Hubble-estimate console script."""

    required = [
        "--outdir",
        "out",
        "--output-label",
        "run",
        "--GWsamples",
        "gw",
        "--EMsamples",
        "em",
        "--injection",
        "inj.json",
        "--inject-Hubble",
        "70",
    ]

    def setUp(self):
        super().setUp()
        self.parser = pp_parser.Hubble_parser(self.parser)

    def test_the_shared_trend_arguments_are_composed_in(self):
        args = self.parser.parse_args(self.required)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.N_posterior_samples, 10000)

    def test_every_required_argument_is_enforced(self):
        for index in range(0, len(self.required), 2):
            trimmed = self.required[:index] + self.required[index + 2 :]
            with self.assertRaises(SystemExit, msg=self.required[index]):
                self.parser.parse_args(trimmed)

    def test_the_injected_hubble_constant_is_a_float(self):
        self.assertAlmostEqual(
            self.parser.parse_args(self.required).inject_Hubble, 70.0
        )

    def test_the_prior_sample_count_has_a_default(self):
        self.assertEqual(self.parser.parse_args(self.required).N_prior_samples, 10000)

    def test_the_p_value_threshold_is_off_unless_given(self):
        # Without a threshold no injection is discarded for a badly
        # recovered gravitational-wave posterior.
        self.assertIsNone(self.parser.parse_args(self.required).p_value_threshold)

    def test_the_p_value_threshold_is_a_float(self):
        args = self.parser.parse_args(self.required + ["--p-value-threshold", "0.05"])
        self.assertAlmostEqual(args.p_value_threshold, 0.05)

    def test_the_description_names_both_messengers(self):
        self.assertIn("GW", self.parser.description)
        self.assertIn("EM", self.parser.description)


class TestResamplingParser(ParserMixin, unittest.TestCase):
    """The gravitational-wave and electromagnetic resampling script, reached
    through the gwem-resampling console script."""

    required = [
        "--outdir",
        "out",
        "--GWsamples",
        "gw.dat",
        "--EMsamples",
        "em.dat",
        "--EOSpath",
        "eos",
        "--Neos",
        "5000",
        "--GWprior",
        "gw.prior",
        "--EMprior",
        "em.prior",
    ]

    def setUp(self):
        super().setUp()
        self.parser = pp_parser.resampling_parser(self.parser)

    def test_every_required_argument_is_enforced(self):
        for index in range(0, len(self.required), 2):
            trimmed = self.required[:index] + self.required[index + 2 :]
            with self.assertRaises(SystemExit, msg=self.required[index]):
                self.parser.parse_args(trimmed)

    def test_the_live_point_count_has_a_default(self):
        self.assertEqual(self.parser.parse_args(self.required).nlive, 1024)

    def test_the_two_ejecta_are_treated_separately_by_default(self):
        self.assertFalse(self.parser.parse_args(self.required).total_ejecta_mass)

    def test_the_ejecta_can_be_combined_into_a_total(self):
        args = self.parser.parse_args(self.required + ["--total-ejecta-mass"])
        self.assertTrue(args.total_ejecta_mass)

    def test_a_binary_neutron_star_is_assumed_by_default(self):
        self.assertFalse(self.parser.parse_args(self.required).withNSBH)

    def test_a_neutron_star_black_hole_source_can_be_requested(self):
        self.assertTrue(self.parser.parse_args(self.required + ["--withNSBH"]).withNSBH)

    def test_it_does_not_inherit_the_shared_trend_arguments(self):
        # This script infers one event rather than a trend over many, so it
        # has no reordering or credible-interval settings.
        args = self.parser.parse_args(self.required)
        self.assertFalse(hasattr(args, "N_reordering"))
        self.assertFalse(hasattr(args, "cred_interval"))


class TestMaximumMassParser(ParserMixin, unittest.TestCase):
    """The post-merger maximum-mass constraint script."""

    required = [
        "--outdir",
        "out",
        "--joint-posterior",
        "post.dat",
        "--prior",
        "p.prior",
        "--eos-path-macro",
        "macro",
        "--eos-path-micro",
        "micro",
    ]

    def setUp(self):
        super().setUp()
        self.parser = pp_parser.maximum_mass_parser(self.parser)

    def test_every_required_argument_is_enforced(self):
        for index in range(0, len(self.required), 2):
            trimmed = self.required[:index] + self.required[index + 2 :]
            with self.assertRaises(SystemExit, msg=self.required[index]):
                self.parser.parse_args(trimmed)

    def test_both_equation_of_state_directories_are_parsed(self):
        args = self.parser.parse_args(self.required)
        self.assertEqual(args.eos_path_macro, "macro")
        self.assertEqual(args.eos_path_micro, "micro")

    def test_the_kepler_limit_is_not_assumed_by_default(self):
        self.assertFalse(self.parser.parse_args(self.required).use_M_Kepler)

    def test_the_kepler_limit_can_be_requested(self):
        args = self.parser.parse_args(self.required + ["--use-M-Kepler"])
        self.assertTrue(args.use_M_Kepler)

    def test_the_live_point_count_has_a_default(self):
        self.assertEqual(self.parser.parse_args(self.required).nlive, 1024)

    def test_the_equation_of_state_count_is_not_an_argument(self):
        # It is counted from the macroscopic directory instead.
        self.assertFalse(hasattr(self.parser.parse_args(self.required), "Neos"))


class TestCornerPlotParser(ParserMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.parser = pp_parser.corner_plot_parser(self.parser)

    def test_at_least_one_posterior_file_is_required(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args([])

    def test_several_posterior_files_are_collected(self):
        args = self.parser.parse_args(["-f", "first.csv", "second.csv"])
        self.assertEqual(args.posterior_files, ["first.csv", "second.csv"])

    def test_several_legend_labels_are_collected(self):
        args = self.parser.parse_args(["-f", "a.csv", "-l", "$first$", "$second$"])
        self.assertEqual(args.label_name, ["$first$", "$second$"])

    def test_the_truth_can_come_from_an_injection_file(self):
        args = self.parser.parse_args(["-f", "a.csv", "-i", "inj.json", "-n", "3"])
        self.assertEqual(args.injection_json, "inj.json")
        self.assertEqual(args.injection_num, 3)

    def test_the_truth_can_come_from_a_best_fit_file_instead(self):
        args = self.parser.parse_args(["-f", "a.csv", "--bestfit-params", "bf.json"])
        self.assertEqual(args.bestfit_params, "bf.json")

    def test_the_two_truth_sources_are_not_mutually_exclusive_in_the_parser(self):
        # The help text says to use one or the other, but the parser accepts
        # both and leaves the choice to the plotting routine.
        args = self.parser.parse_args(
            ["-f", "a.csv", "-i", "inj.json", "--bestfit-params", "bf.json"]
        )
        self.assertIsNotNone(args.injection_json)
        self.assertIsNotNone(args.bestfit_params)

    def test_the_corner_keywords_default_to_an_empty_mapping_string(self):
        self.assertEqual(self.parser.parse_args(["-f", "a.csv"]).kwargs, "{}")

    def test_the_corner_keywords_are_kept_as_a_string_for_later_evaluation(self):
        args = self.parser.parse_args(
            ["-f", "a.csv", "--kwargs", "{'plot_datapoints': False}"]
        )
        self.assertEqual(args.kwargs, "{'plot_datapoints': False}")

    def test_the_prior_file_and_output_name_are_optional(self):
        args = self.parser.parse_args(["-f", "a.csv"])
        self.assertIsNone(args.prior_filename)
        self.assertIsNone(args.output)


class TestLightcurveMarginalisationParser(ParserMixin, unittest.TestCase):
    """The marginalisation script composes the electromagnetic parsing
    groups, because it generates light curves as it goes."""

    def setUp(self):
        super().setUp()
        self.parser = pp_parser.lc_marginalisation_parser(self.parser)

    def test_the_electromagnetic_groups_are_composed_in(self):
        titles = [group.title for group in self.parser._action_groups]
        self.assertIn("EM model arguments", titles)
        self.assertIn("EM analysis time arguments", titles)

    def test_the_three_input_formats_are_all_optional(self):
        # The script picks whichever of the three was supplied and exits if
        # none were.
        args = self.parser.parse_args([])
        self.assertIsNone(args.template_file)
        self.assertIsNone(args.hdf5_file)
        self.assertIsNone(args.coinc_file)

    def test_each_input_format_is_parsed(self):
        args = self.parser.parse_args(
            ["--template-file", "t.dat", "--hdf5-file", "h.h5", "--coinc-file", "c.xml"]
        )
        self.assertEqual(args.template_file, "t.dat")
        self.assertEqual(args.hdf5_file, "h.h5")
        self.assertEqual(args.coinc_file, "c.xml")

    def test_the_trigger_time_defaults_to_the_first_binary_neutron_star(self):
        self.assertEqual(self.parser.parse_args([]).gps, 1187008882)

    def test_the_marginalisation_count_defaults_to_one_hundred(self):
        self.assertEqual(self.parser.parse_args([]).Nmarg, 100)

    def test_the_equation_of_state_directory_has_an_alias(self):
        for flag in ["--eos-data", "--eos-dir"]:
            args = self.parser.parse_args([flag, "eos"])
            self.assertEqual(args.eos_data, "eos", msg=flag)

    def test_the_equation_of_state_weights_have_two_aliases(self):
        for flag in ["-e", "--eos-weights", "--gw170817-eos"]:
            args = self.parser.parse_args([flag, "w.dat"])
            self.assertEqual(args.eos_weights, "w.dat", msg=flag)

    def test_the_generation_seed_defaults_to_forty_two(self):
        self.assertEqual(self.parser.parse_args([]).generation_seed, 42)

    def test_the_skymap_is_optional_and_has_a_short_alias(self):
        self.assertIsNone(self.parser.parse_args([]).skymap)
        self.assertEqual(self.parser.parse_args(["-s", "sky.fits"]).skymap, "sky.fits")


class TestParsersThroughTheBaseParser(unittest.TestCase):
    """The console scripts reach these parsers through nmma_base_parsing,
    which adds config-file support."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_the_resampling_arguments_can_come_from_a_config_file(self):
        config = self.tmp_dir / "resampling.yaml"
        config.write_text(
            "outdir: out\nGWsamples: gw.dat\nEMsamples: em.dat\n"
            "EOSpath: eos\nNeos: 100\nGWprior: gw.prior\nEMprior: em.prior\n"
        )
        args = nmma_base_parsing(pp_parser.resampling_parser, [str(config)])
        self.assertEqual(args.Neos, 100)
        self.assertEqual(args.EOSpath, "eos")

    def test_the_command_line_overrides_the_config_file(self):
        config = self.tmp_dir / "resampling.yaml"
        config.write_text(
            "outdir: out\nGWsamples: gw.dat\nEMsamples: em.dat\n"
            "EOSpath: eos\nNeos: 100\nGWprior: gw.prior\nEMprior: em.prior\n"
        )
        args = nmma_base_parsing(
            pp_parser.resampling_parser, [str(config), "--Neos", "7"]
        )
        self.assertEqual(args.Neos, 7)

    def test_a_missing_required_argument_is_still_enforced(self):
        config = self.tmp_dir / "partial.yaml"
        config.write_text("outdir: out\n")
        with self.assertRaises(SystemExit):
            nmma_base_parsing(pp_parser.resampling_parser, [str(config)])


if __name__ == "__main__":
    unittest.main()
