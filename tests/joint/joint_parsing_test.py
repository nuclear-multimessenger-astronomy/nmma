import argparse
import shutil
import tempfile
from pathlib import Path

import pytest

from nmma.core.parsing import nmma_base_parsing
from nmma.joint import joint_parsing


class ParserMixin:
    """Each parsing function takes a parser and returns it with more
    arguments attached, so a bare parser is enough to test them."""

    def setup_method(self):
        self.parser = argparse.ArgumentParser()

    def group_titles(self, parser):
        return [group.title for group in parser._action_groups]


class TestJointLikelihoodParsing(ParserMixin):
    """The only argument the joint likelihood contributes of its own is the
    switch that turns on automatic ejecta conversion."""

    def setup_method(self):
        super().setup_method()
        self.parser = joint_parsing.joint_likelihood_parsing(self.parser)

    def test_ejecta_conversion_is_a_flag_that_defaults_to_false(self):
        assert not self.parser.parse_args([]).ejecta_conversion
        assert self.parser.parse_args(["--ejecta-conversion"]).ejecta_conversion

    def test_the_description_names_what_the_parser_sets_up(self):
        assert "joint NMMA likelihood" in self.parser.description

    def test_the_same_parser_object_is_returned(self):
        parser = argparse.ArgumentParser()
        assert joint_parsing.joint_likelihood_parsing(parser) is parser


class TestInjectionParsingComposition(ParserMixin):
    """injection_parsing composes the shared injection arguments with the
    EOS, EM, GW and joint-likelihood groups, so an injection can be
    specified for every messenger from a single parser."""

    def setup_method(self):
        super().setup_method()
        self.parser = joint_parsing.injection_parsing(self.parser)

    def test_the_eos_groups_are_pulled_in(self):
        titles = self.group_titles(self.parser)
        assert "Tabulated EOS input arguments" in titles
        assert "EOS input arguments" in titles

    def test_the_em_groups_are_pulled_in(self):
        titles = self.group_titles(self.parser)
        assert "EM model arguments" in titles
        assert "EM analysis time arguments" in titles

    def test_the_gw_injection_group_is_pulled_in(self):
        assert "Lightcurve injection arguments" in self.group_titles(self.parser)

    def test_the_joint_likelihood_argument_is_reachable(self):
        assert not self.parser.parse_args([]).ejecta_conversion

    def test_the_base_injection_arguments_are_reachable(self):
        args = self.parser.parse_args(["--n-injection", "7", "--extension", "csv"])
        assert args.n_injection == 7
        assert args.extension == "csv"

    def test_the_injection_description_is_overwritten_by_the_last_composed_parser(self):
        # injection_parsing sets a description about creating injections,
        # but joint_likelihood_parsing is composed last and overwrites it,
        # so this is what --help of the injection script actually shows.
        assert (
            self.parser.description
            == "Set up a joint NMMA likelihood from provided messengers "
            "and analysis modifiers"
        )


class TestInjectionParsingOwnArguments(ParserMixin):
    """The arguments injection_parsing adds itself drive the redraw loop,
    the tests applied to each draw and the post-processing afterwards."""

    def setup_method(self):
        super().setup_method()
        self.parser = joint_parsing.injection_parsing(self.parser)

    def test_the_redraw_limit_defaults_to_ten_draws(self):
        assert self.parser.parse_args([]).max_redraws == 10

    def test_the_redraw_limit_is_an_integer(self):
        assert self.parser.parse_args(["--max-redraws", "3"]).max_redraws == 3
        with pytest.raises(SystemExit):
            self.parser.parse_args(["--max-redraws", "many"])

    def test_simple_setup_is_a_flag_that_defaults_to_false(self):
        assert not self.parser.parse_args([]).simple_setup
        assert self.parser.parse_args(["--simple-setup"]).simple_setup

    def test_original_parameters_is_a_flag_that_defaults_to_false(self):
        assert not self.parser.parse_args([]).original_parameters
        assert self.parser.parse_args(["--original-parameters"]).original_parameters

    def test_tests_and_post_processing_default_to_empty_lists(self):
        args = self.parser.parse_args([])
        assert args.tests == []
        assert args.post_processing == []

    def test_several_tests_are_collected_into_a_list(self):
        args = self.parser.parse_args(["--tests", "snr>12", "ejecta"])
        assert args.tests == ["snr>12", "ejecta"]

    def test_several_post_processing_steps_are_collected_into_a_list(self):
        args = self.parser.parse_args(["--post-processing", "snr", "lightcurve"])
        assert args.post_processing == ["snr", "lightcurve"]

    def test_the_output_directory_has_a_short_alias(self):
        assert self.parser.parse_args([]).outdir == "outdir"
        assert self.parser.parse_args(["-o", "elsewhere"]).outdir == "elsewhere"
        assert (
            self.parser.parse_args(["--outdir", "elsewhere"]).outdir == "elsewhere"
        )

    def test_the_lightcurve_label_is_optional(self):
        assert self.parser.parse_args([]).lc_label is None
        assert self.parser.parse_args(["--lc-label", "lc"]).lc_label == "lc"

    def test_the_peak_magnitude_is_kept_as_a_string(self):
        # It is either 'any', 'all' or a filter dictionary, so the parser
        # hands it on untouched and the injection creator interprets it.
        args = self.parser.parse_args(["--peak-magnitude", "any"])
        assert args.peak_magnitude == "any"

    def test_the_population_model_defaults_to_uniform(self):
        assert self.parser.parse_args([]).population_model == "uniform"

    def test_the_cosmology_is_unset_so_the_default_is_chosen_downstream(self):
        assert self.parser.parse_args([]).cosmology is None
        assert (
            self.parser.parse_args(["--cosmology", "Planck15"]).cosmology
            == "Planck15"
        )


class TestInjectionParsingLegacyArguments(ParserMixin):
    """Two groups of arguments exist only to reproduce older behaviour:
    reading masses from an external GW injection file, and applying one
    binary type's ejecta formula to every injection."""

    def setup_method(self):
        super().setup_method()
        self.parser = joint_parsing.injection_parsing(self.parser)

    def test_an_external_gw_injection_file_can_be_given(self):
        args = self.parser.parse_args(["--gw-injection-file", "injections.xml"])
        assert args.gw_injection_file == "injections.xml"

    def test_the_reference_frequency_of_that_file_defaults_to_twenty_hertz(self):
        assert self.parser.parse_args([]).reference_frequency == 20

    def test_the_reference_frequency_is_a_float(self):
        args = self.parser.parse_args(["--reference-frequency", "50"])
        assert isinstance(args.reference_frequency, float)
        assert args.reference_frequency == 50.0

    def test_the_binary_type_and_its_eos_file_are_both_unset_by_default(self):
        args = self.parser.parse_args([])
        assert args.binary_type is None
        assert args.eos_file is None

    def test_the_binary_type_and_eos_file_are_parsed_as_strings(self):
        args = self.parser.parse_args(["--binary-type", "BNS", "--eos-file", "eos.dat"])
        assert args.binary_type == "BNS"
        assert args.eos_file == "eos.dat"

    def test_the_binary_type_help_explains_why_it_is_not_a_test(self):
        # The argument exists alongside --tests rather than inside it
        # because a mass read from a file can never be redrawn away.
        action = next(
            action for action in self.parser._actions if action.dest == "binary_type"
        )
        assert "--eos-file" in action.help


class TestInjectionParsingThroughTheBaseParser:
    """The console script reaches injection_parsing through
    nmma_base_parsing, which adds config-file and sweep support."""

    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    def parse(self, cli_args):
        return nmma_base_parsing(joint_parsing.injection_parsing, cli_args)

    def test_the_injection_arguments_keep_their_defaults(self):
        args = self.parse([])
        assert args.max_redraws == 10
        assert args.outdir == "outdir"
        assert args.population_model == "uniform"

    def test_arguments_can_be_read_from_a_config_file(self):
        config = self.tmp_dir / "injection.yaml"
        config.write_text("n-injection: 5\nmax-redraws: 3\nbinary-type: NSBH\n")
        args = self.parse([str(config)])
        assert args.n_injection == 5
        assert args.max_redraws == 3
        assert args.binary_type == "NSBH"

    def test_the_command_line_overrides_the_config_file(self):
        config = self.tmp_dir / "injection.yaml"
        config.write_text("max-redraws: 3\nn-injection: 5\n")
        args = self.parse([str(config), "--max-redraws", "99"])
        assert args.max_redraws == 99
        assert args.n_injection == 5

    def test_a_list_valued_argument_survives_a_config_file(self):
        config = self.tmp_dir / "injection.yaml"
        config.write_text("tests: [ejecta, snr>12]\n")
        args = self.parse([str(config)])
        assert args.tests == ["ejecta", "snr>12"]

    def test_an_unknown_argument_is_rejected(self):
        with pytest.raises(SystemExit):
            self.parse(["--not-an-argument", "1"])
