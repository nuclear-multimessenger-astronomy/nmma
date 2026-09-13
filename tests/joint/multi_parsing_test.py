import argparse
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import configargparse
import pytest

from nmma.joint import multi_parsing


class ParserMixin:
    def setup_method(self):
        self.parser = argparse.ArgumentParser()

    def group_titles(self, parser):
        return [group.title for group in parser._action_groups]


class TestEMSettingsParsing(ParserMixin):
    """The observed light curve is the one EM input the joint pipeline needs
    beyond what the shared EM parsing already provides."""

    def setup_method(self):
        super().setup_method()
        self.parser = multi_parsing.em_settings_parsing(self.parser)

    def test_the_group_is_attached(self):
        assert "EM analysis input arguments" in self.group_titles(self.parser)

    def test_the_light_curve_path_is_parsed(self):
        args = self.parser.parse_args(["--light-curve-data", "lc.dat"])
        assert args.light_curve_data == "lc.dat"

    def test_the_light_curve_path_is_optional(self):
        assert self.parser.parse_args([]).light_curve_data is None

    def test_the_group_is_populated_through_the_configargparse_alias(self):
        # The function calls group.add() rather than group.add_argument();
        # that short alias exists only because importing configargparse
        # patches it onto argparse's group class, so a plain argparse
        # installation would not be enough.
        group = argparse.ArgumentParser().add_argument_group("probe")
        assert group.add == group.add_argument


class TestMultiDynestyParsing(ParserMixin):
    """Settings for the parallelised dynesty run, several of which carry a
    --dynesty-prefixed alias so a bilby_pipe config keeps working."""

    def setup_method(self):
        super().setup_method()
        self.parser = multi_parsing.multi_dynesty_parsing(self.parser)

    def test_the_group_is_attached(self):
        assert "Setting for the Dynesty Sampler" in self.group_titles(self.parser)

    def test_the_sampler_defaults_to_dynesty(self):
        assert self.parser.parse_args([]).sampler == "dynesty"

    def test_the_bounding_method_defaults_to_live_points(self):
        assert self.parser.parse_args([]).bound == "live"

    def test_the_bounding_method_has_a_dynesty_prefixed_alias(self):
        assert self.parser.parse_args(["--bound", "multi"]).bound == "multi"
        assert (
            self.parser.parse_args(["--dynesty-bound", "multi"]).bound == "multi"
        )

    def test_the_sampling_method_defaults_to_an_acceptance_walk(self):
        assert self.parser.parse_args([]).sample == "acceptance-walk"

    def test_the_sampling_method_has_a_dynesty_prefixed_alias(self):
        assert self.parser.parse_args(["--sample", "rwalk"]).sample == "rwalk"
        assert (
            self.parser.parse_args(["--dynesty-sample", "rwalk"]).sample == "rwalk"
        )

    def test_saving_bounds_is_off_by_default_because_resume_files_grow(self):
        assert not self.parser.parse_args([]).save_bounds
        assert self.parser.parse_args(["--save-bounds"]).save_bounds

    def test_the_checkpoint_interval_is_an_integer_number_of_steps(self):
        assert self.parser.parse_args([]).n_check_point == 2000
        args = self.parser.parse_args(["--n-check-point", "500"])
        assert args.n_check_point == 500
        with pytest.raises(SystemExit):
            self.parser.parse_args(["--n-check-point", "often"])


class TestMiscSettings(ParserMixin):
    def setup_method(self):
        super().setup_method()
        self.parser = multi_parsing.add_misc_settings(self.parser)

    def test_the_group_is_attached(self):
        assert "Misc. Settings" in self.group_titles(self.parser)

    def test_clean_is_a_flag_with_a_short_alias(self):
        assert not self.parser.parse_args([]).clean
        assert self.parser.parse_args(["-c"]).clean
        assert self.parser.parse_args(["--clean"]).clean

    def test_plotting_is_off_by_default(self):
        assert not self.parser.parse_args([]).plot
        assert self.parser.parse_args(["--plot"]).plot


class TestRunParsing(ParserMixin):
    """The analysis stage is pointed at the data dump written by the
    generation stage, either positionally or by flag."""

    def setup_method(self):
        super().setup_method()
        self.parser = multi_parsing.run_parsing(self.parser)

    def test_the_group_is_attached(self):
        assert "Setting for the Main run" in self.group_titles(self.parser)

    def test_the_data_dump_can_be_given_positionally(self):
        args = self.parser.parse_args(["run_data_dump.pickle"])
        assert args.data_dump == "run_data_dump.pickle"

    def test_the_positional_and_the_flag_share_one_destination(self):
        dests = [
            action.dest for action in self.parser._actions if action.dest == "data_dump"
        ]
        assert len(dests) == 2

    def test_the_data_dump_flag_on_its_own_is_lost(self):
        # Sharing one destination between an optional positional
        # (nargs="?") and a flag means argparse fills the positional with
        # its None default *after* the flag has been stored, so
        # "--data-dump path" alone silently resolves to no data dump.
        args = self.parser.parse_args(["--data-dump", "run_data_dump.pickle"])
        assert args.data_dump is None

    def test_the_flag_wins_only_when_a_positional_is_also_present(self):
        args = self.parser.parse_args(
            ["placeholder", "--data-dump", "run_data_dump.pickle"]
        )
        assert args.data_dump == "run_data_dump.pickle"

    def test_the_data_dump_is_optional(self):
        assert self.parser.parse_args([]).data_dump is None

    def test_the_output_directory_and_label_override_the_dump(self):
        args = self.parser.parse_args(["--outdir", "here", "--label", "rerun"])
        assert args.outdir == "here"
        assert args.label == "rerun"

    def test_the_override_arguments_are_unset_by_default(self):
        args = self.parser.parse_args([])
        assert args.outdir is None
        assert args.label is None

    def test_the_result_format_defaults_to_hdf5(self):
        assert self.parser.parse_args([]).result_format == "hdf5"
        args = self.parser.parse_args(["--result-format", "json"])
        assert args.result_format == "json"


class TestRemoveArgumentFromParser:
    """bilby_pipe's parser carries arguments NMMA either replaces or does
    not honour, and they are removed by flag name, not destination."""

    def setup_method(self):
        self.parser = argparse.ArgumentParser(conflict_handler="resolve")
        self.parser.add_argument("--keep-me", default="kept")
        self.parser.add_argument("--remove-me", default="removed")

    def dests(self):
        return {action.dest for action in self.parser._actions}

    def test_an_argument_is_removed(self):
        multi_parsing.remove_argument_from_parser(self.parser, "remove-me")
        assert "remove_me" not in self.dests()

    def test_the_other_arguments_are_untouched(self):
        multi_parsing.remove_argument_from_parser(self.parser, "remove-me")
        assert "keep_me" in self.dests()
        assert self.parser.parse_args([]).keep_me == "kept"

    def test_a_removed_argument_is_no_longer_accepted(self):
        multi_parsing.remove_argument_from_parser(self.parser, "remove-me")
        with pytest.raises(SystemExit):
            self.parser.parse_args(["--remove-me", "value"])

    def test_the_dashed_flag_name_is_matched_against_the_underscored_dest(self):
        self.parser.add_argument("--two-words", default=1)
        multi_parsing.remove_argument_from_parser(self.parser, "two-words")
        assert "two_words" not in self.dests()

    def test_removing_an_absent_argument_is_not_an_error(self):
        multi_parsing.remove_argument_from_parser(self.parser, "never-existed")
        assert "keep_me" in self.dests()

    def test_a_removal_that_cannot_be_resolved_is_logged_not_raised(self):
        # Stripping bilby_pipe's arguments must never abort parser
        # construction, so a failed resolution only warns.
        with patch.object(
            self.parser, "_handle_conflict_resolve", side_effect=ValueError("nope")
        ):
            with patch.object(multi_parsing.logger, "warning") as warning:
                multi_parsing.remove_argument_from_parser(self.parser, "remove-me")
        warning.assert_called_once()
        assert "remove-me" in warning.call_args.args[0]


class TestReducedBilbyPipeParser:
    """NMMA reuses bilby_pipe's parser but strips the scheduler, plotting
    and sampler arguments it provides itself or cannot honour."""

    @classmethod
    def setup_class(cls):
        cls.parser = multi_parsing._create_reduced_bilby_pipe_parser()
        cls.dests = {action.dest for action in cls.parser._actions}

    def test_the_scheduler_arguments_are_gone(self):
        for dest in ["scheduler", "scheduler_args", "accounting", "request_memory"]:
            assert dest not in self.dests, dest

    def test_the_sampler_arguments_are_gone_because_nmma_adds_its_own(self):
        for dest in ["sampler", "sampling_seed", "sampler_kwargs"]:
            assert dest not in self.dests, dest

    def test_the_plotting_arguments_are_gone(self):
        for dest in ["plot_corner", "plot_skymap", "plot_waveform", "plot_format"]:
            assert dest not in self.dests, dest

    def test_the_local_running_arguments_are_gone(self):
        for dest in ["local", "local_generation", "local_plot", "osg", "email"]:
            assert dest not in self.dests, dest

    def test_the_data_and_detector_arguments_are_kept(self):
        for dest in ["detectors", "duration", "prior_file", "trigger_time"]:
            assert dest in self.dests, dest

    def test_the_ini_positional_is_inherited_and_still_required(self):
        # bilby_pipe marks ini as its config-file argument and keeps it
        # required even at top_level=False, which is why every
        # parse_generation_args call has to fill that slot.
        ini = next(action for action in self.parser._actions if action.dest == "ini")
        assert ini.option_strings == []
        assert ini.required


class TestCreateGenerationParser:
    """The generation parser is the bilby_pipe parser plus every NMMA
    messenger group, because generation has to build all of the data."""

    @classmethod
    def setup_class(cls):
        cls.parser = multi_parsing.create_nmma_generation_parser()
        cls.titles = [group.title for group in cls.parser._action_groups]

    def test_the_bilby_pipe_groups_are_inherited(self):
        for title in ["Detector arguments", "Waveform arguments", "Prior arguments"]:
            assert title in self.titles, title

    def test_every_messenger_group_is_present(self):
        for title in [
            "EM analysis input arguments",
            "GW input arguments",
            "EOS input arguments",
            "Tabulated EOS input arguments",
        ]:
            assert title in self.titles, title

    def test_the_dynesty_groups_are_present_because_the_sampler_is_all(self):
        assert "Dynesty Settings" in self.titles
        assert "Setting for the Dynesty Sampler" in self.titles

    def test_the_main_run_group_is_absent_because_generation_writes_the_dump(self):
        assert "Setting for the Main run" not in self.titles

    def test_conflicting_arguments_resolve_rather_than_raise(self):
        assert self.parser.conflict_handler == "resolve"

    def test_the_version_action_reports_both_nmma_and_bilby(self):
        action = next(
            action for action in self.parser._actions if action.dest == "version"
        )
        assert "bilby=" in action.version


class TestParseGenerationArgs:
    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    def test_the_parser_is_returned_alongside_the_arguments(self):
        args, parser = multi_parsing.parse_generation_args([""])
        assert isinstance(parser, argparse.ArgumentParser)
        assert args.sampler == "dynesty"

    def test_no_arguments_gives_the_defaults(self):
        args, _ = multi_parsing.parse_generation_args([""])
        assert args.outdir == "outdir"
        assert args.label == "label"

    def test_command_line_arguments_are_applied_after_the_ini_placeholder(self):
        args, _ = multi_parsing.parse_generation_args(
            ["", "--label", "joint_run", "--n-check-point", "10"]
        )
        assert args.label == "joint_run"
        assert args.n_check_point == 10

    def test_flags_without_an_ini_placeholder_are_refused(self):
        # The inherited ini positional is required, so the first element of
        # cli_args is always consumed as the config path.
        with pytest.raises(SystemExit):
            multi_parsing.parse_generation_args(["--label", "joint_run"])

    def test_a_leading_ini_file_is_read_as_a_config(self):
        config = self.tmp_dir / "generation.ini"
        config.write_text("label=from_ini\ndetectors=[H1, L1]\n")
        args, parser = multi_parsing.parse_generation_args([str(config)])
        assert args.label == "from_ini"
        assert isinstance(parser, configargparse.ArgumentParser)

    def test_the_command_line_overrides_the_config_file(self):
        config = self.tmp_dir / "generation.ini"
        config.write_text("label=from_ini\n")
        args, _ = multi_parsing.parse_generation_args(
            [str(config), "--label", "from_cli"]
        )
        assert args.label == "from_cli"


class TestCreateAnalysisParser:
    """The analysis parser is the same base parser with the data-dump
    arguments added, and without bilby_pipe as a parent."""

    @classmethod
    def setup_class(cls):
        cls.parser = multi_parsing.create_nmma_analysis_parser()
        cls.titles = [group.title for group in cls.parser._action_groups]

    def test_the_main_run_group_is_added(self):
        assert "Setting for the Main run" in self.titles

    def test_the_bilby_pipe_groups_are_not_inherited(self):
        assert "Detector arguments" not in self.titles
        assert "Job submission arguments" not in self.titles

    def test_the_messenger_groups_are_still_present(self):
        assert "GW input arguments" in self.titles
        assert "EOS input arguments" in self.titles

    def test_the_dynesty_groups_are_present_by_default(self):
        assert "Setting for the Dynesty Sampler" in self.titles

    def test_another_sampler_leaves_out_the_dynesty_groups(self):
        parser = multi_parsing.create_nmma_analysis_parser(sampler="pymultinest")
        titles = [group.title for group in parser._action_groups]
        assert "Setting for the Dynesty Sampler" not in titles
        assert "Dynesty Settings" not in titles

    def test_the_em_input_group_is_attached_twice(self):
        # em_settings_parsing and em_analysis_parsing each open a group of
        # this name, so the title appears twice in --help.
        assert self.titles.count("EM analysis input arguments") == 2


class TestParseAnalysisArgs:
    """Parsing the analysis arguments also rejects walk settings that would
    make the dynesty run ill-defined."""

    @classmethod
    def setup_class(cls):
        cls.parser = multi_parsing.create_nmma_analysis_parser()

    def test_the_defaults_are_consistent_and_parse(self):
        args = multi_parsing.parse_analysis_args(self.parser, [])
        assert args.walks == 100
        assert args.maxmcmc == 5000
        assert args.nact == 2

    def test_the_data_dump_is_read_from_the_command_line(self):
        args = multi_parsing.parse_analysis_args(
            self.parser, ["run_data_dump.pickle", "--label", "rerun"]
        )
        assert args.data_dump == "run_data_dump.pickle"
        assert args.label == "rerun"

    def test_more_walks_than_the_mcmc_limit_is_rejected(self):
        with pytest.raises(ValueError):
            multi_parsing.parse_analysis_args(
                self.parser, ["--walks", "500", "--maxmcmc", "100"]
            )

    def test_walks_equal_to_the_mcmc_limit_is_accepted(self):
        args = multi_parsing.parse_analysis_args(
            self.parser, ["--walks", "100", "--maxmcmc", "100"]
        )
        assert args.walks == 100

    def test_fewer_than_one_autocorrelation_time_is_rejected(self):
        with pytest.raises(ValueError):
            multi_parsing.parse_analysis_args(self.parser, ["--nact", "0"])

    def test_exactly_one_autocorrelation_time_is_accepted(self):
        args = multi_parsing.parse_analysis_args(self.parser, ["--nact", "1"])
        assert args.nact == 1
