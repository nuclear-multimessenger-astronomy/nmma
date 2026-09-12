import argparse
import pickle
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import configargparse
import numpy as np
from bilby.core.prior import PriorDict, Uniform

from nmma.joint import generation
from nmma.joint.generation import NMMADataGenerationInput


class TestGetVersionInfo(unittest.TestCase):
    """Every dump records the versions of the stack that produced it, so a
    result can be traced back to the code that made it."""

    @classmethod
    def setUpClass(cls):
        cls.info = generation.get_version_info()

    def test_the_whole_inference_stack_is_recorded(self):
        for package in [
            "bilby_version",
            "bilby_pipe_version",
            "dynesty_version",
            "lalsimulation_version",
            "nmma_version",
        ]:
            self.assertIn(package, self.info, msg=package)

    def test_every_version_is_a_string(self):
        for package, version in self.info.items():
            self.assertIsInstance(version, str, msg=package)

    def test_nothing_else_is_recorded(self):
        self.assertEqual(len(self.info), 5)


class TestDetermineRequiredArgs(unittest.TestCase):
    """The written config keeps only the argument groups a run actually
    used, so the groups are chosen from the messengers in the dump."""

    def test_the_shared_groups_are_always_required(self):
        required = generation.determine_required_args([])
        for title in ["options", "Dynesty Settings", "Prior arguments"]:
            self.assertIn(title, required, msg=title)

    def test_a_gw_run_adds_the_detector_and_waveform_groups(self):
        required = generation.determine_required_args(["gw"])
        for title in [
            "Calibration arguments",
            "Waveform arguments",
            "Detector arguments",
            "Post processing arguments",
        ]:
            self.assertIn(title, required, msg=title)

    def test_an_em_run_adds_the_em_input_group(self):
        self.assertIn(
            "EM analysis input arguments", generation.determine_required_args(["em"])
        )

    def test_an_eos_run_adds_the_eos_input_group(self):
        self.assertIn(
            "EOS analysis input arguments", generation.determine_required_args(["eos"])
        )

    def test_a_hubble_run_adds_the_hubble_group(self):
        self.assertIn(
            "Hubble input arguments", generation.determine_required_args(["Hubble"])
        )

    def test_a_tabulated_eos_run_adds_its_own_group(self):
        self.assertIn(
            "Tabulated EOS input arguments",
            generation.determine_required_args(["tabulated_eos"]),
        )

    def test_the_messenger_groups_are_not_added_unless_asked_for(self):
        required = generation.determine_required_args([])
        for title in [
            "EM analysis input arguments",
            "EOS analysis input arguments",
            "Detector arguments",
        ]:
            self.assertNotIn(title, required, msg=title)

    def test_several_messengers_accumulate_their_groups(self):
        required = generation.determine_required_args(["gw", "em", "eos"])
        self.assertIn("Detector arguments", required)
        self.assertIn("EM analysis input arguments", required)
        self.assertIn("EOS analysis input arguments", required)


class TestRemoveExpandableArgs(unittest.TestCase):
    def parser_with(self, titles):
        parser = argparse.ArgumentParser()
        for title in titles:
            parser.add_argument_group(title=title)
        return parser

    def titles(self, parser):
        return [group.title for group in parser._action_groups]

    def test_a_group_that_is_not_required_is_removed(self):
        parser = self.parser_with(["keep", "drop"])
        generation.remove_expandable_args(
            parser, ["positional arguments", "options", "keep"]
        )
        self.assertNotIn("drop", self.titles(parser))

    def test_a_required_group_is_kept(self):
        parser = self.parser_with(["keep", "drop"])
        generation.remove_expandable_args(
            parser, ["positional arguments", "options", "keep"]
        )
        self.assertIn("keep", self.titles(parser))

    def test_the_parser_is_returned(self):
        parser = self.parser_with(["keep"])
        self.assertIs(
            generation.remove_expandable_args(parser, ["keep"]),
            parser,
        )

    def test_consecutive_unrequired_groups_are_not_all_removed(self):
        # The function removes from the same list it is iterating over, so
        # removing one group shifts the next into the slot already passed
        # and that group survives. Written config files therefore still
        # carry some groups the run never used.
        parser = self.parser_with(["first_drop", "second_drop"])
        generation.remove_expandable_args(parser, ["positional arguments", "options"])
        self.assertEqual(
            self.titles(parser),
            ["positional arguments", "options", "second_drop"],
        )


class TestWriteCompleteConfigFile(unittest.TestCase):
    """The complete config is the record of what was run, and the arguments
    are normalised into a form that can be read back as a config file."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.existing_file = self.tmp_dir / "lightcurve.dat"
        self.existing_file.write_text("data")

        self.parser = configargparse.ArgumentParser()
        self.parser.add_argument_group(title="Prior arguments")
        self.parser.add_argument_group(title="Detector arguments")
        self.parser.write_config_file = MagicMock()

        self.inputs = MagicMock()
        self.inputs.messengers = ["em"]
        self.inputs.analysis_modifiers = []
        self.inputs.complete_ini_file = str(self.tmp_dir / "complete.ini")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def write(self, remove_none=True, **kwargs):
        args = Namespace(**kwargs)
        generation.write_complete_config_file(
            self.parser, args, self.inputs, remove_none=remove_none
        )
        return args

    def test_the_config_is_written_to_the_complete_ini_path(self):
        args = self.write(label="run")
        self.parser.write_config_file.assert_called_once_with(
            args, [self.inputs.complete_ini_file]
        )

    def test_the_parallel_settings_are_pinned_on_the_inputs(self):
        self.write(label="run")
        self.assertEqual(self.inputs.request_cpus, 1)
        self.assertEqual(self.inputs.mpi_timing_interval, 0)
        self.assertIsNone(self.inputs.log_directory)

    def test_an_existing_path_is_rewritten_as_an_absolute_path(self):
        args = self.write(label="run", light_curve_data=str(self.existing_file))
        self.assertEqual(args.light_curve_data, str(self.existing_file.absolute()))

    def test_a_string_that_is_not_a_path_is_left_alone(self):
        args = self.write(label="run", em_model="Bu2019lm")
        self.assertEqual(args.em_model, "Bu2019lm")

    def test_the_label_is_never_treated_as_a_path(self):
        # A label can easily collide with a file in the working directory,
        # and turning it into an absolute path would rename the run.
        label_file = Path("lightcurve.dat")
        args = self.write(label=str(self.existing_file))
        self.assertEqual(args.label, str(self.existing_file))
        self.assertFalse(label_file.is_absolute())

    def test_an_empty_list_becomes_an_empty_bracket_string(self):
        args = self.write(label="run", filters=[])
        self.assertEqual(args.filters, "[]")

    def test_a_list_of_strings_becomes_a_bracketed_comma_separated_string(self):
        args = self.write(label="run", detectors=["H1", "L1"])
        self.assertEqual(args.detectors, "[H1, L1]")

    def test_a_list_of_numbers_is_left_alone(self):
        args = self.write(label="run", numbers=[1, 2])
        self.assertEqual(args.numbers, [1, 2])

    def test_the_sampler_keywords_are_stored_as_a_string(self):
        args = self.write(label="run")
        self.assertEqual(args.sampler_kwargs, "{}")

    def test_submission_is_switched_off_so_the_config_does_not_resubmit(self):
        args = self.write(label="run")
        self.assertFalse(args.submit)

    def test_unset_arguments_are_dropped_by_default(self):
        args = self.write(label="run", em_model=None)
        self.assertNotIn("em_model", vars(args))

    def test_unset_arguments_can_be_kept(self):
        args = self.write(label="run", em_model=None, remove_none=False)
        self.assertIn("em_model", vars(args))

    def test_the_argument_groups_are_pruned_to_the_messengers_in_use(self):
        self.inputs.messengers = ["em"]
        self.write(label="run")
        titles = [group.title for group in self.parser._action_groups]
        self.assertNotIn("Detector arguments", titles)

    def test_the_analysis_modifiers_also_decide_which_groups_survive(self):
        self.inputs.messengers = []
        self.inputs.analysis_modifiers = ["Hubble"]
        self.write(label="run")
        titles = [group.title for group in self.parser._action_groups]
        self.assertIn("Prior arguments", titles)


class TestCreateGenerationLogger(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_the_log_is_written_under_the_data_directory(self):
        with patch("bilby.core.utils.setup_logger") as setup_logger:
            generation.create_generation_logger(str(self.tmp_dir), "run")
        setup_logger.assert_called_once_with(
            outdir=str(Path(self.tmp_dir, "data")), label="run"
        )

    def test_the_bilby_pipe_generation_module_shares_the_logger(self):
        # bilby_pipe logs its own data generation steps, and they belong in
        # the same file as NMMA's.
        with patch("bilby.core.utils.setup_logger"):
            logger = generation.create_generation_logger(str(self.tmp_dir), "run")
        self.assertIs(generation.bilby_pipe.data_generation.logger, logger)


class TestSamplingSeed(unittest.TestCase):
    """The seed is set on the input object rather than passed around, and
    seeding numpy there makes the whole generation reproducible."""

    def setUp(self):
        self.inputs = object.__new__(NMMADataGenerationInput)

    def test_a_given_seed_is_kept(self):
        self.inputs.sampling_seed = 1234
        self.assertEqual(self.inputs.sampling_seed, 1234)

    def test_no_seed_is_drawn_at_random_so_the_run_is_still_recorded(self):
        self.inputs.sampling_seed = None
        self.assertIsInstance(self.inputs.sampling_seed, int)
        self.assertGreaterEqual(self.inputs.sampling_seed, 1)

    def test_two_unseeded_runs_get_different_seeds(self):
        self.inputs.sampling_seed = None
        first = self.inputs.sampling_seed
        second_inputs = object.__new__(NMMADataGenerationInput)
        second_inputs.sampling_seed = None
        self.assertNotEqual(first, second_inputs.sampling_seed)

    def test_setting_the_seed_makes_the_numpy_draws_reproducible(self):
        self.inputs.sampling_seed = 7
        first = np.random.rand(5)
        self.inputs.sampling_seed = 7
        np.testing.assert_array_equal(first, np.random.rand(5))


class TestGetPriors(unittest.TestCase):
    """Without gravitational waves there is no binary to assume, so the
    prior file is read as it stands rather than through a CBC prior."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.prior_file = self.tmp_dir / "test.prior"
        self.prior_file.write_text(
            "log10_mej_wind = Uniform(minimum=-3, maximum=-1, name='log10_mej_wind')\n"
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_the_prior_file_is_read_into_a_plain_prior_dict(self):
        inputs = object.__new__(NMMADataGenerationInput)
        inputs.prior_file = str(self.prior_file)
        priors = inputs._get_priors()
        self.assertIsInstance(priors, PriorDict)
        self.assertIn("log10_mej_wind", priors)

    def test_no_binary_parameters_are_invented(self):
        inputs = object.__new__(NMMADataGenerationInput)
        inputs.prior_file = str(self.prior_file)
        priors = inputs._get_priors()
        self.assertNotIn("chirp_mass", priors)
        self.assertNotIn("mass_ratio", priors)


class TestSaveDataDump(unittest.TestCase):
    """The dump is the only thing passed from generation to analysis, so it
    has to survive a pickle round trip."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.inputs = object.__new__(NMMADataGenerationInput)
        self.inputs.data_dump_file = str(self.tmp_dir / "run_data_dump.pickle")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def load(self):
        with open(self.inputs.data_dump_file, "rb") as handle:
            return pickle.load(handle)

    def test_the_dump_is_written_where_the_analysis_will_look_for_it(self):
        self.inputs.data_dump = {"messengers": ["em"]}
        self.inputs.save_data_dump()
        self.assertTrue(Path(self.inputs.data_dump_file).is_file())

    def test_the_contents_survive_the_round_trip(self):
        self.inputs.data_dump = {"messengers": ["em"], "analysis_modifiers": []}
        self.inputs.save_data_dump()
        self.assertEqual(self.load()["messengers"], ["em"])

    def test_the_parsed_arguments_survive_the_round_trip(self):
        self.inputs.data_dump = {"args": Namespace(label="run", sampler="dynesty")}
        self.inputs.save_data_dump()
        self.assertEqual(self.load()["args"].sampler, "dynesty")

    def test_priors_survive_the_round_trip(self):
        priors = PriorDict()
        priors["x"] = Uniform(0, 1, "x")
        self.inputs.data_dump = {"priors": priors}
        self.inputs.save_data_dump()
        self.assertIn("x", self.load()["priors"])

    def test_writing_twice_overwrites_rather_than_appends(self):
        self.inputs.data_dump = {"run": "first"}
        self.inputs.save_data_dump()
        self.inputs.data_dump = {"run": "second"}
        self.inputs.save_data_dump()
        self.assertEqual(self.load(), {"run": "second"})


class TestGenerateRunner(unittest.TestCase):
    """generate_runner is the python-level entry point: it parses, lets
    keyword arguments override, builds the input object and writes the
    complete config."""

    def setUp(self):
        self.args = Namespace(outdir="outdir", label="label")
        self.parser = MagicMock(name="parser")
        self.inputs = MagicMock(name="inputs")
        self.logger = MagicMock(name="logger")

        self.parse = patch.object(
            generation, "parse_generation_args", return_value=(self.args, self.parser)
        ).start()
        self.create_logger = patch.object(
            generation, "create_generation_logger", return_value=self.logger
        ).start()
        self.input_class = patch.object(
            generation, "NMMADataGenerationInput", return_value=self.inputs
        ).start()
        self.write_config = patch.object(
            generation, "write_complete_config_file"
        ).start()
        self.addCleanup(patch.stopall)

    def test_the_command_line_arguments_are_parsed(self):
        generation.generate_runner(["config.ini"])
        self.parse.assert_called_once_with(["config.ini"])

    def test_keyword_arguments_override_the_parsed_arguments(self):
        generation.generate_runner([""], label="overridden")
        self.assertEqual(self.args.label, "overridden")

    def test_a_keyword_argument_can_introduce_a_new_setting(self):
        generation.generate_runner([""], em_model="Bu2019lm")
        self.assertEqual(self.args.em_model, "Bu2019lm")

    def test_the_logger_is_set_up_before_anything_is_generated(self):
        generation.generate_runner([""])
        self.create_logger.assert_called_once_with(outdir="outdir", label="label")

    def test_the_overridden_output_directory_is_used_for_the_log(self):
        generation.generate_runner([""], outdir="elsewhere")
        self.create_logger.assert_called_once_with(outdir="elsewhere", label="label")

    def test_the_input_object_is_built_from_the_arguments_and_the_logger(self):
        generation.generate_runner([""])
        self.input_class.assert_called_once_with(self.args, [], self.logger)

    def test_the_complete_config_is_written_from_the_same_parser(self):
        generation.generate_runner([""])
        self.write_config.assert_called_once_with(
            parser=self.parser, args=self.args, inputs=self.inputs
        )

    def test_the_inputs_and_the_logger_are_returned(self):
        inputs, logger = generation.generate_runner([""])
        self.assertIs(inputs, self.inputs)
        self.assertIs(logger, self.logger)

    def test_the_stack_versions_are_logged(self):
        generation.generate_runner([""])
        logged = " ".join(str(call) for call in self.logger.info.call_args_list)
        self.assertIn("nmma_version", logged)


class TestNMMAGenerationEntryPoint(unittest.TestCase):
    def test_the_command_line_is_forwarded_without_the_program_name(self):
        with patch.object(generation, "generate_runner") as runner:
            with patch.object(
                generation.sys, "argv", ["nmma_generation", "config.ini"]
            ):
                generation.nmma_generation()
        runner.assert_called_once_with(cli_args=["config.ini"])

    def test_no_arguments_gives_an_empty_command_line(self):
        with patch.object(generation, "generate_runner") as runner:
            with patch.object(generation.sys, "argv", ["nmma_generation"]):
                generation.nmma_generation()
        runner.assert_called_once_with(cli_args=[])


class TestLatexIsDisabled(unittest.TestCase):
    def test_importing_the_module_turns_off_latex_rendering(self):
        # Generation plots have to work on a cluster node without a LaTeX
        # installation.
        import matplotlib

        self.assertFalse(matplotlib.rcParams["text.usetex"])


if __name__ == "__main__":
    unittest.main()
