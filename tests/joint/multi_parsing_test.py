import argparse
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import configargparse

from nmma.joint import multi_parsing


class ParserMixin:
    def setUp(self):
        self.parser = argparse.ArgumentParser()

    def group_titles(self, parser):
        return [group.title for group in parser._action_groups]


class TestEMSettingsParsing(ParserMixin, unittest.TestCase):
    """The observed light curve is the one EM input the joint pipeline needs
    beyond what the shared EM parsing already provides."""

    def setUp(self):
        super().setUp()
        self.parser = multi_parsing.em_settings_parsing(self.parser)

    def test_the_group_is_attached(self):
        self.assertIn("EM analysis input arguments", self.group_titles(self.parser))

    def test_the_light_curve_path_is_parsed(self):
        args = self.parser.parse_args(["--light-curve-data", "lc.dat"])
        self.assertEqual(args.light_curve_data, "lc.dat")

    def test_the_light_curve_path_is_optional(self):
        self.assertIsNone(self.parser.parse_args([]).light_curve_data)

    def test_the_group_is_populated_through_the_configargparse_alias(self):
        # The function calls group.add() rather than group.add_argument();
        # that short alias exists only because importing configargparse
        # patches it onto argparse's group class, so a plain argparse
        # installation would not be enough.
        group = argparse.ArgumentParser().add_argument_group("probe")
        self.assertEqual(group.add, group.add_argument)


class TestMultiDynestyParsing(ParserMixin, unittest.TestCase):
    """Settings for the parallelised dynesty run, several of which carry a
    --dynesty-prefixed alias so a bilby_pipe config keeps working."""

    def setUp(self):
        super().setUp()
        self.parser = multi_parsing.multi_dynesty_parsing(self.parser)

    def test_the_group_is_attached(self):
        self.assertIn("Setting for the Dynesty Sampler", self.group_titles(self.parser))

    def test_the_sampler_defaults_to_dynesty(self):
        self.assertEqual(self.parser.parse_args([]).sampler, "dynesty")

    def test_the_bounding_method_defaults_to_live_points(self):
        self.assertEqual(self.parser.parse_args([]).bound, "live")

    def test_the_bounding_method_has_a_dynesty_prefixed_alias(self):
        self.assertEqual(self.parser.parse_args(["--bound", "multi"]).bound, "multi")
        self.assertEqual(
            self.parser.parse_args(["--dynesty-bound", "multi"]).bound, "multi"
        )

    def test_the_sampling_method_defaults_to_an_acceptance_walk(self):
        self.assertEqual(self.parser.parse_args([]).sample, "acceptance-walk")

    def test_the_sampling_method_has_a_dynesty_prefixed_alias(self):
        self.assertEqual(self.parser.parse_args(["--sample", "rwalk"]).sample, "rwalk")
        self.assertEqual(
            self.parser.parse_args(["--dynesty-sample", "rwalk"]).sample, "rwalk"
        )

    def test_saving_bounds_is_off_by_default_because_resume_files_grow(self):
        self.assertFalse(self.parser.parse_args([]).save_bounds)
        self.assertTrue(self.parser.parse_args(["--save-bounds"]).save_bounds)

    def test_the_checkpoint_interval_is_an_integer_number_of_steps(self):
        self.assertEqual(self.parser.parse_args([]).n_check_point, 2000)
        args = self.parser.parse_args(["--n-check-point", "500"])
        self.assertEqual(args.n_check_point, 500)
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--n-check-point", "often"])


class TestMiscSettings(ParserMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.parser = multi_parsing.add_misc_settings(self.parser)

    def test_the_group_is_attached(self):
        self.assertIn("Misc. Settings", self.group_titles(self.parser))

    def test_clean_is_a_flag_with_a_short_alias(self):
        self.assertFalse(self.parser.parse_args([]).clean)
        self.assertTrue(self.parser.parse_args(["-c"]).clean)
        self.assertTrue(self.parser.parse_args(["--clean"]).clean)

    def test_plotting_is_off_by_default(self):
        self.assertFalse(self.parser.parse_args([]).plot)
        self.assertTrue(self.parser.parse_args(["--plot"]).plot)


class TestRunParsing(ParserMixin, unittest.TestCase):
    """The analysis stage is pointed at the data dump written by the
    generation stage, either positionally or by flag."""

    def setUp(self):
        super().setUp()
        self.parser = multi_parsing.run_parsing(self.parser)

    def test_the_group_is_attached(self):
        self.assertIn("Setting for the Main run", self.group_titles(self.parser))

    def test_the_data_dump_can_be_given_positionally(self):
        args = self.parser.parse_args(["run_data_dump.pickle"])
        self.assertEqual(args.data_dump, "run_data_dump.pickle")

    def test_the_positional_and_the_flag_share_one_destination(self):
        dests = [
            action.dest for action in self.parser._actions if action.dest == "data_dump"
        ]
        self.assertEqual(len(dests), 2)

    def test_the_data_dump_flag_on_its_own_is_lost(self):
        # Sharing one destination between an optional positional
        # (nargs="?") and a flag means argparse fills the positional with
        # its None default *after* the flag has been stored, so
        # "--data-dump path" alone silently resolves to no data dump.
        args = self.parser.parse_args(["--data-dump", "run_data_dump.pickle"])
        self.assertIsNone(args.data_dump)

    def test_the_flag_wins_only_when_a_positional_is_also_present(self):
        args = self.parser.parse_args(
            ["placeholder", "--data-dump", "run_data_dump.pickle"]
        )
        self.assertEqual(args.data_dump, "run_data_dump.pickle")

    def test_the_data_dump_is_optional(self):
        self.assertIsNone(self.parser.parse_args([]).data_dump)

    def test_the_output_directory_and_label_override_the_dump(self):
        args = self.parser.parse_args(["--outdir", "here", "--label", "rerun"])
        self.assertEqual(args.outdir, "here")
        self.assertEqual(args.label, "rerun")

    def test_the_override_arguments_are_unset_by_default(self):
        args = self.parser.parse_args([])
        self.assertIsNone(args.outdir)
        self.assertIsNone(args.label)

    def test_the_result_format_defaults_to_hdf5(self):
        self.assertEqual(self.parser.parse_args([]).result_format, "hdf5")
        args = self.parser.parse_args(["--result-format", "json"])
        self.assertEqual(args.result_format, "json")


class TestRemoveArgumentFromParser(unittest.TestCase):
    """bilby_pipe's parser carries arguments NMMA either replaces or does
    not honour, and they are removed by flag name, not destination."""

    def setUp(self):
        self.parser = argparse.ArgumentParser(conflict_handler="resolve")
        self.parser.add_argument("--keep-me", default="kept")
        self.parser.add_argument("--remove-me", default="removed")

    def dests(self):
        return {action.dest for action in self.parser._actions}

    def test_an_argument_is_removed(self):
        multi_parsing.remove_argument_from_parser(self.parser, "remove-me")
        self.assertNotIn("remove_me", self.dests())

    def test_the_other_arguments_are_untouched(self):
        multi_parsing.remove_argument_from_parser(self.parser, "remove-me")
        self.assertIn("keep_me", self.dests())
        self.assertEqual(self.parser.parse_args([]).keep_me, "kept")

    def test_a_removed_argument_is_no_longer_accepted(self):
        multi_parsing.remove_argument_from_parser(self.parser, "remove-me")
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--remove-me", "value"])

    def test_the_dashed_flag_name_is_matched_against_the_underscored_dest(self):
        self.parser.add_argument("--two-words", default=1)
        multi_parsing.remove_argument_from_parser(self.parser, "two-words")
        self.assertNotIn("two_words", self.dests())

    def test_removing_an_absent_argument_is_not_an_error(self):
        multi_parsing.remove_argument_from_parser(self.parser, "never-existed")
        self.assertIn("keep_me", self.dests())

    def test_a_removal_that_cannot_be_resolved_is_logged_not_raised(self):
        # Stripping bilby_pipe's arguments must never abort parser
        # construction, so a failed resolution only warns.
        with patch.object(
            self.parser, "_handle_conflict_resolve", side_effect=ValueError("nope")
        ):
            with patch.object(multi_parsing.logger, "warning") as warning:
                multi_parsing.remove_argument_from_parser(self.parser, "remove-me")
        warning.assert_called_once()
        self.assertIn("remove-me", warning.call_args.args[0])


class TestReducedBilbyPipeParser(unittest.TestCase):
    """NMMA reuses bilby_pipe's parser but strips the scheduler, plotting
    and sampler arguments it provides itself or cannot honour."""

    @classmethod
    def setUpClass(cls):
        cls.parser = multi_parsing._create_reduced_bilby_pipe_parser()
        cls.dests = {action.dest for action in cls.parser._actions}

    def test_the_scheduler_arguments_are_gone(self):
        for dest in ["scheduler", "scheduler_args", "accounting", "request_memory"]:
            self.assertNotIn(dest, self.dests, msg=dest)

    def test_the_sampler_arguments_are_gone_because_nmma_adds_its_own(self):
        for dest in ["sampler", "sampling_seed", "sampler_kwargs"]:
            self.assertNotIn(dest, self.dests, msg=dest)

    def test_the_plotting_arguments_are_gone(self):
        for dest in ["plot_corner", "plot_skymap", "plot_waveform", "plot_format"]:
            self.assertNotIn(dest, self.dests, msg=dest)

    def test_the_local_running_arguments_are_gone(self):
        for dest in ["local", "local_generation", "local_plot", "osg", "email"]:
            self.assertNotIn(dest, self.dests, msg=dest)

    def test_the_data_and_detector_arguments_are_kept(self):
        for dest in ["detectors", "duration", "prior_file", "trigger_time"]:
            self.assertIn(dest, self.dests, msg=dest)

    def test_the_ini_positional_is_inherited_and_still_required(self):
        # bilby_pipe marks ini as its config-file argument and keeps it
        # required even at top_level=False, which is why every
        # parse_generation_args call has to fill that slot.
        ini = next(action for action in self.parser._actions if action.dest == "ini")
        self.assertEqual(ini.option_strings, [])
        self.assertTrue(ini.required)


class TestCreateGenerationParser(unittest.TestCase):
    """The generation parser is the bilby_pipe parser plus every NMMA
    messenger group, because generation has to build all of the data."""

    @classmethod
    def setUpClass(cls):
        cls.parser = multi_parsing.create_nmma_generation_parser()
        cls.titles = [group.title for group in cls.parser._action_groups]

    def test_the_bilby_pipe_groups_are_inherited(self):
        for title in ["Detector arguments", "Waveform arguments", "Prior arguments"]:
            self.assertIn(title, self.titles, msg=title)

    def test_every_messenger_group_is_present(self):
        for title in [
            "EM analysis input arguments",
            "GW input arguments",
            "EOS input arguments",
            "Tabulated EOS input arguments",
        ]:
            self.assertIn(title, self.titles, msg=title)

    def test_the_dynesty_groups_are_present_because_the_sampler_is_all(self):
        self.assertIn("Dynesty Settings", self.titles)
        self.assertIn("Setting for the Dynesty Sampler", self.titles)

    def test_the_main_run_group_is_absent_because_generation_writes_the_dump(self):
        self.assertNotIn("Setting for the Main run", self.titles)

    def test_conflicting_arguments_resolve_rather_than_raise(self):
        self.assertEqual(self.parser.conflict_handler, "resolve")

    def test_the_version_action_reports_both_nmma_and_bilby(self):
        action = next(
            action for action in self.parser._actions if action.dest == "version"
        )
        self.assertIn("bilby=", action.version)


class TestParseGenerationArgs(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_the_parser_is_returned_alongside_the_arguments(self):
        args, parser = multi_parsing.parse_generation_args([""])
        self.assertIsInstance(parser, argparse.ArgumentParser)
        self.assertEqual(args.sampler, "dynesty")

    def test_no_arguments_gives_the_defaults(self):
        args, _ = multi_parsing.parse_generation_args([""])
        self.assertEqual(args.outdir, "outdir")
        self.assertEqual(args.label, "label")

    def test_command_line_arguments_are_applied_after_the_ini_placeholder(self):
        args, _ = multi_parsing.parse_generation_args(
            ["", "--label", "joint_run", "--n-check-point", "10"]
        )
        self.assertEqual(args.label, "joint_run")
        self.assertEqual(args.n_check_point, 10)

    def test_flags_without_an_ini_placeholder_are_refused(self):
        # The inherited ini positional is required, so the first element of
        # cli_args is always consumed as the config path.
        with self.assertRaises(SystemExit):
            multi_parsing.parse_generation_args(["--label", "joint_run"])

    def test_a_leading_ini_file_is_read_as_a_config(self):
        config = self.tmp_dir / "generation.ini"
        config.write_text("label=from_ini\ndetectors=[H1, L1]\n")
        args, parser = multi_parsing.parse_generation_args([str(config)])
        self.assertEqual(args.label, "from_ini")
        self.assertIsInstance(parser, configargparse.ArgumentParser)

    def test_the_command_line_overrides_the_config_file(self):
        config = self.tmp_dir / "generation.ini"
        config.write_text("label=from_ini\n")
        args, _ = multi_parsing.parse_generation_args(
            [str(config), "--label", "from_cli"]
        )
        self.assertEqual(args.label, "from_cli")


class TestCreateAnalysisParser(unittest.TestCase):
    """The analysis parser is the same base parser with the data-dump
    arguments added, and without bilby_pipe as a parent."""

    @classmethod
    def setUpClass(cls):
        cls.parser = multi_parsing.create_nmma_analysis_parser()
        cls.titles = [group.title for group in cls.parser._action_groups]

    def test_the_main_run_group_is_added(self):
        self.assertIn("Setting for the Main run", self.titles)

    def test_the_bilby_pipe_groups_are_not_inherited(self):
        self.assertNotIn("Detector arguments", self.titles)
        self.assertNotIn("Job submission arguments", self.titles)

    def test_the_messenger_groups_are_still_present(self):
        self.assertIn("GW input arguments", self.titles)
        self.assertIn("EOS input arguments", self.titles)

    def test_the_dynesty_groups_are_present_by_default(self):
        self.assertIn("Setting for the Dynesty Sampler", self.titles)

    def test_another_sampler_leaves_out_the_dynesty_groups(self):
        parser = multi_parsing.create_nmma_analysis_parser(sampler="pymultinest")
        titles = [group.title for group in parser._action_groups]
        self.assertNotIn("Setting for the Dynesty Sampler", titles)
        self.assertNotIn("Dynesty Settings", titles)

    def test_the_em_input_group_is_attached_twice(self):
        # em_settings_parsing and em_analysis_parsing each open a group of
        # this name, so the title appears twice in --help.
        self.assertEqual(self.titles.count("EM analysis input arguments"), 2)


class TestParseAnalysisArgs(unittest.TestCase):
    """Parsing the analysis arguments also rejects walk settings that would
    make the dynesty run ill-defined."""

    @classmethod
    def setUpClass(cls):
        cls.parser = multi_parsing.create_nmma_analysis_parser()

    def test_the_defaults_are_consistent_and_parse(self):
        args = multi_parsing.parse_analysis_args(self.parser, [])
        self.assertEqual(args.walks, 100)
        self.assertEqual(args.maxmcmc, 5000)
        self.assertEqual(args.nact, 2)

    def test_the_data_dump_is_read_from_the_command_line(self):
        args = multi_parsing.parse_analysis_args(
            self.parser, ["run_data_dump.pickle", "--label", "rerun"]
        )
        self.assertEqual(args.data_dump, "run_data_dump.pickle")
        self.assertEqual(args.label, "rerun")

    def test_more_walks_than_the_mcmc_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            multi_parsing.parse_analysis_args(
                self.parser, ["--walks", "500", "--maxmcmc", "100"]
            )

    def test_walks_equal_to_the_mcmc_limit_is_accepted(self):
        args = multi_parsing.parse_analysis_args(
            self.parser, ["--walks", "100", "--maxmcmc", "100"]
        )
        self.assertEqual(args.walks, 100)

    def test_fewer_than_one_autocorrelation_time_is_rejected(self):
        with self.assertRaises(ValueError):
            multi_parsing.parse_analysis_args(self.parser, ["--nact", "0"])

    def test_exactly_one_autocorrelation_time_is_accepted(self):
        args = multi_parsing.parse_analysis_args(self.parser, ["--nact", "1"])
        self.assertEqual(args.nact, 1)


if __name__ == "__main__":
    unittest.main()
