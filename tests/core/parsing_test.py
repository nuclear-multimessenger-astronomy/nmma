import argparse
import operator
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import configargparse

from nmma.core import parsing


class ConfigFileMixin:
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def write_config(self, name, content):
        path = self.tmp_dir / name
        path.write_text(content)
        return str(path)


class TestYamlParse(unittest.TestCase):
    def test_parses_a_mapping(self):
        self.assertEqual(parsing.yaml_parse("{a: 1, b: two}"), {"a": 1, "b": "two"})

    def test_parses_a_scalar(self):
        self.assertEqual(parsing.yaml_parse("3"), 3)

    def test_parses_a_nested_multi_run_specification(self):
        parsed = parsing.yaml_parse("{run1: {nlive: 100}, run2: {nlive: 200}}")
        self.assertEqual(parsed, {"run1": {"nlive": 100}, "run2": {"nlive": 200}})


class TestCheckForConfig(ConfigFileMixin, unittest.TestCase):
    def test_no_config_returns_a_plain_argparse_parser(self):
        parser, cli_args = parsing.check_for_config(["--nlive", "5"])
        self.assertNotIsInstance(parser, configargparse.ArgumentParser)
        self.assertIsInstance(parser, argparse.ArgumentParser)
        self.assertEqual(cli_args, ["--nlive", "5"])

    def test_empty_args_return_a_plain_parser(self):
        parser, cli_args = parsing.check_for_config([])
        self.assertNotIsInstance(parser, configargparse.ArgumentParser)
        self.assertEqual(cli_args, [])

    def test_leading_yaml_file_selects_the_yaml_config_parser(self):
        config = self.write_config("conf.yaml", "nlive: 7\n")
        parser, cli_args = parsing.check_for_config([config, "--label", "cli"])
        self.assertIsInstance(parser, configargparse.ArgumentParser)
        self.assertIs(
            parser._config_file_parser.__class__, configargparse.YAMLConfigFileParser
        )
        self.assertEqual(cli_args, ["--label", "cli"])

    def test_leading_ini_file_selects_the_default_config_parser(self):
        config = self.write_config("conf.ini", "nlive = 7\n")
        parser, _ = parsing.check_for_config([config])
        self.assertIs(
            parser._config_file_parser.__class__, configargparse.DefaultConfigFileParser
        )

    def test_toml_cfg_and_tml_suffixes_are_also_accepted(self):
        for name in ["conf.toml", "conf.cfg", "conf.tml"]:
            config = self.write_config(name, "nlive = 7\n")
            parser, _ = parsing.check_for_config([config])
            self.assertIsInstance(parser, configargparse.ArgumentParser, msg=name)

    def test_explicit_config_flag_is_stripped(self):
        config = self.write_config("conf.yaml", "nlive: 7\n")
        parser, cli_args = parsing.check_for_config(["--config", config, "--label", "x"])
        self.assertIsInstance(parser, configargparse.ArgumentParser)
        self.assertEqual(cli_args, ["--label", "x"])

    def test_short_config_flag_is_stripped(self):
        config = self.write_config("conf.yaml", "nlive: 7\n")
        parser, cli_args = parsing.check_for_config(["-c", config])
        self.assertIsInstance(parser, configargparse.ArgumentParser)
        self.assertEqual(cli_args, [])

    def test_ini_flag_is_stripped(self):
        config = self.write_config("conf.ini", "nlive = 7\n")
        parser, cli_args = parsing.check_for_config(["--ini", config])
        self.assertIsInstance(parser, configargparse.ArgumentParser)
        self.assertEqual(cli_args, [])

    def test_config_file_is_kept_in_the_args_when_drop_config_is_false(self):
        config = self.write_config("conf.yaml", "nlive: 7\n")
        _, cli_args = parsing.check_for_config([config], drop_config=False)
        self.assertEqual(cli_args, [config])

    def test_parents_produce_a_configargparse_parser_without_a_config_file(self):
        parent = argparse.ArgumentParser(add_help=False)
        parent.add_argument("--shared")
        parser, _ = parsing.check_for_config(["--shared", "x"], parents=[parent])
        self.assertIsInstance(parser, configargparse.ArgumentParser)
        self.assertEqual(parser.parse_args(["--shared", "x"]).shared, "x")

    def test_a_non_file_first_argument_is_left_alone(self):
        parser, cli_args = parsing.check_for_config(["not_a_file.yaml", "--label", "x"])
        self.assertNotIsInstance(parser, configargparse.ArgumentParser)
        self.assertEqual(cli_args, ["not_a_file.yaml", "--label", "x"])

    def test_an_existing_file_with_an_unparseable_extension_exits(self):
        # ".txt" matches none of the suffix branches, so the config file
        # parser class is never assigned and building the parser raises;
        # with an explicit --config flag that is reported and exits.
        config = self.write_config("conf.txt", "nlive: 7\n")
        with self.assertRaises(SystemExit):
            parsing.check_for_config(["--config", config])

    def test_an_unparseable_config_without_the_flag_falls_through_silently(self):
        config = self.write_config("conf.txt", "nlive: 7\n")
        parser, cli_args = parsing.check_for_config([config])
        self.assertNotIsInstance(parser, configargparse.ArgumentParser)
        self.assertEqual(cli_args, [config])

    def test_config_flag_pointing_at_a_missing_file_leaves_a_stray_positional(self):
        # The --config flag is popped before the file is checked for
        # existence, so a missing config file is not reported at all: the
        # path is left behind as a positional argument and only surfaces
        # later as an "unrecognized arguments" error from parse_args. This
        # test documents that known gap rather than asserting it is correct.
        missing = str(self.tmp_dir / "absent.yaml")
        parser, cli_args = parsing.check_for_config(["--config", missing, "--label", "x"])
        self.assertNotIsInstance(parser, configargparse.ArgumentParser)
        self.assertEqual(cli_args, [missing, "--label", "x"])

    def test_the_input_argument_list_is_consumed_in_place(self):
        config = self.write_config("conf.yaml", "nlive: 7\n")
        cli_args = ["--config", config, "--label", "x"]
        _, returned = parsing.check_for_config(cli_args)
        self.assertIs(returned, cli_args)
        self.assertEqual(cli_args, ["--label", "x"])


class TestNmmaBaseParsing(ConfigFileMixin, unittest.TestCase):
    def test_string_command_line_is_split(self):
        args = parsing.nmma_base_parsing(
            parsing.single_messenger_analysis_parsing, "--nlive 3 --label mylabel"
        )
        self.assertEqual(args.nlive, 3)
        self.assertEqual(args.label, "mylabel")

    def test_defaults_are_applied(self):
        args = parsing.nmma_base_parsing(parsing.single_messenger_analysis_parsing, [])
        self.assertEqual(args.nlive, 2048)
        self.assertEqual(args.sampler, "pymultinest")
        self.assertEqual(args.outdir, "outdir")
        self.assertEqual(args.sampling_seed, 42)
        self.assertEqual(args.sampler_kwargs, {})

    def test_values_are_read_from_a_yaml_config_file(self):
        config = self.write_config("conf.yaml", "nlive: 7\nlabel: from_config\n")
        args = parsing.nmma_base_parsing(parsing.single_messenger_analysis_parsing, [config])
        self.assertEqual(args.nlive, 7)
        self.assertEqual(args.label, "from_config")

    def test_command_line_overrides_the_config_file(self):
        config = self.write_config("conf.yaml", "nlive: 7\nlabel: from_config\n")
        args = parsing.nmma_base_parsing(
            parsing.single_messenger_analysis_parsing, [config, "--label", "from_cli"]
        )
        self.assertEqual(args.nlive, 7)
        self.assertEqual(args.label, "from_cli")

    def test_several_parsing_functions_are_composed(self):
        args = parsing.nmma_base_parsing(
            [parsing.base_analysis_parsing, parsing.base_injection_parsing], []
        )
        self.assertEqual(args.nlive, 2048)  # from base_analysis_parsing
        self.assertEqual(args.extension, "json")  # from base_injection_parsing

    def test_return_parser_gives_back_the_parser_itself(self):
        parser = parsing.nmma_base_parsing(
            parsing.single_messenger_analysis_parsing, [], return_parser=True
        )
        self.assertIsInstance(parser, argparse.ArgumentParser)
        self.assertEqual(parser.parse_args([]).nlive, 2048)

    def test_multi_and_matrix_are_parsed_as_yaml(self):
        args = parsing.nmma_base_parsing(
            parsing.single_messenger_analysis_parsing,
            ["--multi", "{run1: {nlive: 100}}", "--matrix", "{nlive: [1, 2]}"],
        )
        self.assertEqual(args.multi, {"run1": {"nlive": 100}})
        self.assertEqual(args.matrix, {"nlive": [1, 2]})

    def test_multi_and_matrix_default_to_none(self):
        args = parsing.nmma_base_parsing(parsing.single_messenger_analysis_parsing, [])
        self.assertIsNone(args.multi)
        self.assertIsNone(args.matrix)

    def test_cli_args_default_to_sys_argv(self):
        with patch.object(sys, "argv", ["lightcurve-analysis", "--label", "from_argv"]):
            args = parsing.nmma_base_parsing(parsing.single_messenger_analysis_parsing)
        self.assertEqual(args.label, "from_argv")


class TestParserFunctions(unittest.TestCase):
    """Each parser function adds flags to a parser it is handed, so they can
    be exercised one at a time against a bare ArgumentParser."""

    def parse(self, parser_func, cli_args=()):
        parser = parser_func(argparse.ArgumentParser())
        return parser.parse_args(list(cli_args))

    def test_base_analysis_parsing_defaults(self):
        args = self.parse(parsing.base_analysis_parsing)
        self.assertFalse(args.Hubble)
        self.assertIsNone(args.cosmology)
        self.assertEqual(args.dlogz, 0.1)
        self.assertEqual(args.cpus, 1)
        self.assertEqual(args.check_point_delta_t, 1800)
        self.assertFalse(args.soft_init)
        self.assertFalse(args.skip_sampling)

    def test_hubble_flag_aliases(self):
        for flag in ["--Hubble", "--with-Hubble", "--sample-over-Hubble"]:
            self.assertTrue(self.parse(parsing.base_analysis_parsing, [flag]).Hubble)

    def test_sampling_seed_alias(self):
        self.assertEqual(
            self.parse(parsing.base_analysis_parsing, ["--seed", "7"]).sampling_seed, 7
        )

    def test_sampler_kwargs_are_parsed_as_yaml(self):
        args = self.parse(parsing.base_analysis_parsing, ["--sampler-kwargs", "{nlive: 5}"])
        self.assertEqual(args.sampler_kwargs, {"nlive": 5})

    def test_nlive_aliases(self):
        for flag in ["-n", "--nlive", "--n-live"]:
            self.assertEqual(
                self.parse(parsing.base_analysis_parsing, [flag, "16"]).nlive, 16
            )

    def test_dynesty_parsing_defaults(self):
        args = self.parse(parsing.dynesty_parsing)
        self.assertEqual(args.walks, 100)
        self.assertEqual(args.maxmcmc, 5000)
        self.assertEqual(args.nact, 2)
        self.assertEqual(args.naccept, 60)
        self.assertEqual(args.facc, 0.5)
        self.assertEqual(args.enlarge, 1.5)
        self.assertEqual(args.n_check_point, 1000)

    def test_rejection_sample_posterior_defaults_to_true_and_flips(self):
        self.assertTrue(self.parse(parsing.dynesty_parsing).rejection_sample_posterior)
        self.assertFalse(
            self.parse(
                parsing.dynesty_parsing, ["--rejection-sample-posterior"]
            ).rejection_sample_posterior
        )

    def test_proposals_accumulate(self):
        args = self.parse(
            parsing.dynesty_parsing, ["--proposals", "diff", "--proposals", "volumetric"]
        )
        self.assertEqual(args.proposals, ["diff", "volumetric"])

    def test_single_messenger_analysis_parsing_includes_the_base_and_dynesty_groups(self):
        args = self.parse(parsing.single_messenger_analysis_parsing)
        self.assertEqual(args.nlive, 2048)  # base_analysis_parsing
        self.assertEqual(args.walks, 100)  # dynesty_parsing
        self.assertEqual(args.label, "nmma_transient")
        self.assertEqual(args.result_format, "json")
        self.assertFalse(args.plot)

    def test_prior_file_alias(self):
        args = self.parse(
            parsing.single_messenger_analysis_parsing, ["--prior", "my.prior"]
        )
        self.assertEqual(args.prior_file, "my.prior")

    def test_bestfit_alias(self):
        self.assertTrue(
            self.parse(parsing.single_messenger_analysis_parsing, ["--best-fit"]).bestfit
        )

    def test_base_injection_parsing_defaults(self):
        args = self.parse(parsing.base_injection_parsing)
        self.assertEqual(args.extension, "json")
        self.assertEqual(args.generation_seed, 42)
        self.assertEqual(args.injection_num, 0)
        self.assertFalse(args.injection)

    def test_base_injection_parsing_restricts_the_extension(self):
        with self.assertRaises(SystemExit):
            self.parse(parsing.base_injection_parsing, ["--extension", "hdf5"])

    def test_pipe_inj_parsing_defaults(self):
        args = self.parse(parsing.pipe_inj_parsing)
        self.assertEqual(args.n_injection, 20)
        self.assertEqual(args.trigger_time, 0.0)
        self.assertEqual(args.deltaT, 0.2)
        self.assertEqual(args.duration, 4.0)
        self.assertEqual(args.post_trigger_duration, 2.0)

    def test_slurm_setup_parser_requires_an_injection_and_analysis_file(self):
        with self.assertRaises(SystemExit):
            self.parse(parsing.slurm_setup_parser)
        args = self.parse(
            parsing.slurm_setup_parser,
            ["--injection-file", "inj.json", "--analysis-file", "run.sh"],
        )
        self.assertEqual(args.injection_file, "inj.json")
        self.assertEqual(args.n_per_job, 100)

    def test_slurm_analysis_parser_defaults(self):
        args = self.parse(parsing.slurm_analysis_parser)
        self.assertEqual(args.Ncore, 8)
        self.assertEqual(args.nodes, 1)
        self.assertEqual(args.gpus, 0)
        self.assertEqual(args.memory_GB, 64)
        self.assertEqual(args.cluster_name, "Expanse")
        self.assertEqual(args.base_dir, Path.cwd())


class TestProcessSamplerKwargs(unittest.TestCase):
    def make_args(self, **overrides):
        args = Namespace(sampler_kwargs={}, nlive=500)
        args.__dict__.update(overrides)
        return args

    def test_defaults_are_used_when_args_are_missing(self):
        init_kwargs, run_kwargs = parsing.process_sampler_kwargs(self.make_args())
        self.assertEqual(run_kwargs, {"dlogz": 0.1, "save_bounds": False})
        self.assertEqual(init_kwargs["sample"], "acceptance-walk")
        self.assertEqual(init_kwargs["bound"], "live")
        self.assertEqual(init_kwargs["walks"], 100)

    def test_namespace_values_override_the_defaults(self):
        init_kwargs, run_kwargs = parsing.process_sampler_kwargs(
            self.make_args(dlogz=0.5, walks=50, nlive=123)
        )
        self.assertEqual(run_kwargs["dlogz"], 0.5)
        self.assertEqual(init_kwargs["walks"], 50)
        self.assertEqual(init_kwargs["nlive"], 123)

    def test_sampler_kwargs_win_over_namespace_values(self):
        init_kwargs, _ = parsing.process_sampler_kwargs(
            self.make_args(sampler_kwargs={"nlive": 11, "sample": "rwalk"}, nlive=500)
        )
        self.assertEqual(init_kwargs["nlive"], 11)
        self.assertEqual(init_kwargs["sample"], "rwalk")

    def test_min_eff_is_folded_into_first_update(self):
        init_kwargs, _ = parsing.process_sampler_kwargs(self.make_args(min_eff=25))
        self.assertNotIn("min_eff", init_kwargs)
        self.assertEqual(init_kwargs["first_update"], {"min_eff": 25, "min_ncall": 1000})

    def test_min_ncall_follows_the_namespace_nlive_not_the_sampler_kwargs_one(self):
        # first_update uses args.nlive directly, so a sampler_kwargs override
        # of nlive does not propagate into min_ncall
        init_kwargs, _ = parsing.process_sampler_kwargs(
            self.make_args(sampler_kwargs={"nlive": 11}, nlive=500)
        )
        self.assertEqual(init_kwargs["first_update"]["min_ncall"], 1000)


class TestProcessMultiConditionString(unittest.TestCase):
    def test_comparison_operators_are_recognised(self):
        parsed = parsing.process_multi_condition_string("a==1,b!=2,c>=3,d>4,e<=5,f<6")
        self.assertEqual(parsed["a"], (operator.eq, 1.0))
        self.assertEqual(parsed["b"], (operator.ne, 2.0))
        self.assertEqual(parsed["c"], (operator.ge, 3.0))
        self.assertEqual(parsed["d"], (operator.gt, 4.0))
        self.assertEqual(parsed["e"], (operator.le, 5.0))
        self.assertEqual(parsed["f"], (operator.lt, 6.0))

    def test_the_parsed_operator_is_callable_on_the_threshold(self):
        op, value = parsing.process_multi_condition_string("mass>1.4")["mass"]
        self.assertTrue(op(2.0, value))
        self.assertFalse(op(1.0, value))

    def test_plain_assignment_gives_a_string(self):
        self.assertEqual(
            parsing.process_multi_condition_string("model=Bu2019lm")["model"], "Bu2019lm"
        )

    def test_a_bare_name_is_a_boolean_flag(self):
        self.assertIs(parsing.process_multi_condition_string("detection")["detection"], True)

    def test_whitespace_is_stripped(self):
        parsed = parsing.process_multi_condition_string(" mass > 1.4 , model = Bu2019lm ")
        self.assertEqual(parsed["mass"][1], 1.4)
        self.assertEqual(parsed["model"], "Bu2019lm")

    def test_a_list_of_conditions_is_accepted(self):
        self.assertEqual(
            parsing.process_multi_condition_string(["a>1", "b=2"]),
            parsing.process_multi_condition_string("a>1,b=2"),
        )

    def test_two_character_operators_are_matched_before_one_character_ones(self):
        # ">=" must not be read as ">" followed by a stray "="
        op, value = parsing.process_multi_condition_string("a>=3")["a"]
        self.assertIs(op, operator.ge)
        self.assertEqual(value, 3.0)


class TestParsingAndLogging(ConfigFileMixin, unittest.TestCase):
    def test_an_existing_namespace_is_passed_through(self):
        args = Namespace(outdir=str(self.tmp_dir / "out"), label="test")
        self.assertIs(parsing.parsing_and_logging(None, args), args)

    def test_the_output_directory_is_created(self):
        outdir = self.tmp_dir / "created"
        parsing.parsing_and_logging(None, Namespace(outdir=str(outdir), label="test"))
        self.assertTrue(outdir.is_dir())

    def test_a_command_line_is_parsed_when_no_namespace_is_given(self):
        args = parsing.parsing_and_logging(
            parsing.single_messenger_analysis_parsing,
            ["--outdir", str(self.tmp_dir / "out"), "--label", "mylabel"],
        )
        self.assertEqual(args.label, "mylabel")

    def test_pymultinest_rejects_a_long_output_directory(self):
        args = Namespace(sampler="pymultinest", outdir="x" * 65, label="test")
        with self.assertRaises(ValueError):
            parsing.parsing_and_logging(None, args)

    def test_other_samplers_accept_a_long_output_directory(self):
        outdir = str(self.tmp_dir / ("x" * 65))
        args = Namespace(sampler="dynesty", outdir=outdir, label="test")
        parsing.parsing_and_logging(None, args)
        self.assertTrue(Path(outdir).is_dir())

    def test_refresh_model_list_is_forwarded_to_gitlab(self):
        args = Namespace(
            outdir=str(self.tmp_dir / "out"),
            label="test",
            refresh_model_list=True,
            svd_path="/some/svd/path",
        )
        with patch.object(parsing, "refresh_models_list") as mock_refresh:
            parsing.parsing_and_logging(None, args)
        mock_refresh.assert_called_once_with("/some/svd/path")

    def test_model_list_is_not_refreshed_by_default(self):
        args = Namespace(outdir=str(self.tmp_dir / "out"), label="test")
        with patch.object(parsing, "refresh_models_list") as mock_refresh:
            parsing.parsing_and_logging(None, args)
        mock_refresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
