import argparse
import shutil
import tempfile
from pathlib import Path

import pytest

from nmma.core.parsing import nmma_base_parsing
from nmma.eos import eos_parsing


class ParserMixin:
    """Each parsing function takes a parser and returns it with one more
    argument group attached, so a bare parser is enough to test them."""

    def setup_method(self):
        self.parser = argparse.ArgumentParser()

    def group_titles(self, parser):
        return [group.title for group in parser._action_groups]


class TestTabulatedEoSParsing(ParserMixin):
    """Arguments for running over a directory of precomputed EOSs."""

    def setup_method(self):
        super().setup_method()
        self.parser = eos_parsing.tabulated_eos_parsing(self.parser)

    def test_the_parser_is_returned_with_its_own_group(self):
        assert "Tabulated EOS input arguments" in self.group_titles(self.parser)

    def test_the_eos_directory_and_count_are_parsed(self):
        args = self.parser.parse_args(["--eos-data", "eos_dir", "--Neos", "100"])
        assert args.eos_data == "eos_dir"
        assert args.Neos == 100

    def test_the_number_of_equations_of_state_is_an_integer(self):
        with pytest.raises(SystemExit):
            self.parser.parse_args(["--Neos", "many"])

    def test_loading_to_ram_is_a_flag_that_defaults_to_false(self):
        assert not self.parser.parse_args([]).eos_to_ram
        assert self.parser.parse_args(["--eos-to-ram"]).eos_to_ram

    def test_the_weight_file_is_parsed(self):
        args = self.parser.parse_args(["--eos-weight", "weights.dat"])
        assert args.eos_weight == "weights.dat"

    def test_everything_defaults_to_none(self):
        args = self.parser.parse_args([])
        for attribute in ["eos_data", "Neos", "eos_weight"]:
            assert getattr(args, attribute) is None, attribute


class TestEoSParsing(ParserMixin):
    """Arguments for sampling an EOS model directly, plus the constraints
    that are imposed on it."""

    def setup_method(self):
        super().setup_method()
        self.parser = eos_parsing.eos_parsing(self.parser)

    def test_the_parser_is_returned_with_its_own_group(self):
        assert "EOS input arguments" in self.group_titles(self.parser)

    def test_the_default_micro_eos_model_is_the_five_parameter_nep_model(self):
        assert self.parser.parse_args([]).micro_eos_model == "nep-5"

    def test_the_micro_eos_model_can_be_chosen(self):
        args = self.parser.parse_args(["--micro-eos-model", "lec-13"])
        assert args.micro_eos_model == "lec-13"

    def test_the_emulator_metadata_is_parsed_as_a_plain_string(self):
        args = self.parser.parse_args(["--emulator-metadata", "metadata.json"])
        assert args.emulator_metadata == "metadata.json"

    def test_the_constraint_file_is_parsed(self):
        args = self.parser.parse_args(["--eos-constraint-json", "constraints.json"])
        assert args.eos_constraint_json == "constraints.json"

    def test_a_mass_limit_can_be_given_as_a_yaml_dictionary(self):
        args = self.parser.parse_args(
            ["--lower-mtov", "{J0740: {mass: 2.08, error: 0.07, arxiv: '2104.00880'}}"]
        )
        assert args.lower_mtov == {
            "J0740": {"mass": 2.08, "error": 0.07, "arxiv": "2104.00880"}
        }

    def test_a_mass_limit_can_be_given_as_parallel_lists(self):
        args = self.parser.parse_args(
            [
                "--lower-mtov-name",
                "J0740",
                "J0348",
                "--lower-mtov-mass",
                "2.08",
                "2.01",
                "--lower-mtov-error",
                "0.07",
                "0.04",
            ]
        )
        assert args.lower_mtov_name == ["J0740", "J0348"]
        assert args.lower_mtov_mass == ["2.08", "2.01"]
        assert args.lower_mtov_error == ["0.07", "0.04"]

    def test_both_mass_limits_take_the_same_set_of_arguments(self):
        args = self.parser.parse_args(
            [
                "--upper-mtov",
                "{GW170817: {mass: 2.3}}",
                "--upper-mtov-name",
                "GW170817",
                "--upper-mtov-mass",
                "2.3",
                "--upper-mtov-error",
                "0.1",
                "--upper-mtov-arxiv",
                "1710.05938",
            ]
        )
        assert args.upper_mtov == {"GW170817": {"mass": 2.3}}
        assert args.upper_mtov_name == ["GW170817"]
        assert args.upper_mtov_arxiv == ["1710.05938"]

    def test_mass_radius_posteriors_are_parsed(self):
        args = self.parser.parse_args(
            [
                "--mass-radius",
                "{NICER: {file_path: posterior.dat}}",
                "--mass-radius-name",
                "NICER",
                "--mass-radius-file-path",
                "posterior.dat",
            ]
        )
        assert args.mass_radius == {"NICER": {"file_path": "posterior.dat"}}
        assert args.mass_radius_file_path == ["posterior.dat"]

    def test_the_legacy_posterior_flag_is_an_alias(self):
        args = self.parser.parse_args(
            ["--mass-radius-posterior", "first.dat", "second.dat"]
        )
        assert args.mass_radius_file_path == ["first.dat", "second.dat"]

    def test_plot_keywords_are_parsed_as_dictionaries(self):
        args = self.parser.parse_args(
            [
                "--lower-plot-kwargs",
                "{color: red}",
                "--upper-plot-kwargs",
                "{color: blue}",
                "--mass-radius-plot-kwargs",
                "{manual: [12.0, 1.4]}",
            ]
        )
        assert args.lower_plot_kwargs == {"color": "red"}
        assert args.upper_plot_kwargs == {"color": "blue"}
        assert args.mass_radius_plot_kwargs == {"manual": [12.0, 1.4]}

    def test_the_constraint_arguments_default_to_none(self):
        args = self.parser.parse_args([])
        for kind in ["lower_mtov", "upper_mtov", "mass_radius"]:
            assert getattr(args, kind) is None, kind
            assert getattr(args, f"{kind}_name") is None, kind

    def test_the_argument_names_match_what_the_constraint_reader_expects(self):
        # read_constraint_from_args strips the constraint-kind prefix from the
        # namespace attributes, so the flags have to keep these exact names.
        args = self.parser.parse_args([])
        for kind in ["lower_mtov", "upper_mtov"]:
            for suffix in ["name", "mass", "error", "arxiv"]:
                assert hasattr(args, f"{kind}_{suffix}"), kind


class TestEoSAnalysisParsing:
    """The analysis parser composes the shared single-messenger arguments
    with both EOS groups, and is reached through nmma_base_parsing."""

    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    def parse(self, cli_args):
        return nmma_base_parsing(eos_parsing.eos_analysis_parsing, cli_args)

    def test_all_three_argument_groups_are_present(self):
        parser = nmma_base_parsing(
            eos_parsing.eos_analysis_parsing, [], return_parser=True
        )
        titles = [group.title for group in parser._action_groups]
        assert "Tabulated EOS input arguments" in titles
        assert "EOS input arguments" in titles
        assert "Dynesty Settings" in titles

    def test_the_shared_analysis_arguments_keep_their_defaults(self):
        args = self.parse([])
        assert args.outdir == "outdir"
        assert args.label == "nmma_transient"
        assert args.sampler == "pymultinest"

    def test_eos_and_sampling_arguments_are_parsed_together(self):
        args = self.parse(
            [
                "--eos-data",
                "eos_dir",
                "--Neos",
                "5000",
                "--eos-to-ram",
                "--micro-eos-model",
                "lec-7",
                "--label",
                "eos_run",
                "--sampler",
                "dynesty",
            ]
        )
        assert args.Neos == 5000
        assert args.eos_to_ram
        assert args.micro_eos_model == "lec-7"
        assert args.label == "eos_run"
        assert args.sampler == "dynesty"

    def test_arguments_can_be_read_from_a_config_file(self):
        config = self.tmp_dir / "eos.yaml"
        config.write_text("eos-data: eos_dir\nNeos: 42\nlabel: from_config\n")
        args = self.parse([str(config)])
        assert args.eos_data == "eos_dir"
        assert args.Neos == 42
        assert args.label == "from_config"

    def test_the_command_line_overrides_the_config_file(self):
        config = self.tmp_dir / "eos.yaml"
        config.write_text("label: from_config\nNeos: 42\n")
        args = self.parse([str(config), "--label", "from_cli"])
        assert args.label == "from_cli"
        assert args.Neos == 42

    def test_a_constraint_dictionary_survives_a_config_file(self):
        config = self.tmp_dir / "eos.yaml"
        config.write_text("lower-mtov: '{J0740: {mass: 2.08, error: 0.07}}'\n")
        args = self.parse([str(config)])
        assert args.lower_mtov == {"J0740": {"mass": 2.08, "error": 0.07}}

    def test_the_sweep_arguments_from_the_base_parser_are_available(self):
        args = self.parse(["--multi", "{run1: {Neos: 10}, run2: {Neos: 20}}"])
        assert args.multi == {"run1": {"Neos": 10}, "run2": {"Neos": 20}}

    def test_an_unknown_argument_is_rejected(self):
        with pytest.raises(SystemExit):
            self.parse(["--not-an-argument", "1"])
