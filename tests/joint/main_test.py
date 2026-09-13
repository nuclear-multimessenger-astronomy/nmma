import pickle
import shutil
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nmma.joint import main


class StubRepr:
    """Stands in for an object that cannot be stored in a result file and is
    therefore reduced to its representation."""

    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return self.text


class AnalysisRunnerMixin:
    """analysis_runner reads a generation data dump, rebuilds the likelihood
    from it and hands both to a sampler, so the sampler and the likelihood
    setup are replaced and only that wiring is checked."""

    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.likelihood = MagicMock(name="likelihood")
        self.priors = MagicMock(name="priors")

        self.setup_from_args = patch.object(
            main.MultiMessengerLikelihood,
            "setup_from_args",
            return_value=self.likelihood,
        ).start()
        self.prior_dict = patch.object(main, "PriorDict").start()
        self.prior_dict.from_json.return_value = self.priors
        self.pbilby = patch.object(
            main, "pbilby_sampling", return_value="pbilby_result"
        ).start()
        self.bilby = patch.object(
            main, "bilby_sampling", return_value="bilby_result"
        ).start()

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)
        patch.stopall()

    def write_dump(self, name="run_data_dump.pickle", in_data_dir=False, **extra):
        args = Namespace(
            sampler=extra.pop("sampler", "dynesty"),
            outdir="original_outdir",
            label="original_label",
            plot=None,
        )
        dump = dict(
            args=args,
            prior_file=str(self.tmp_dir / "prior.json"),
            messengers=["em"],
            analysis_modifiers=[],
        )
        dump.update(extra)
        directory = self.tmp_dir / "data" if in_data_dir else self.tmp_dir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        with open(path, "wb") as handle:
            pickle.dump(dump, handle)
        self.args = args
        return path

    def sampler_args(self):
        """The positional arguments the selected sampler was called with."""
        call = self.pbilby.call_args or self.bilby.call_args
        return call


class TestAnalysisRunnerDataDumpLocation(AnalysisRunnerMixin):
    def test_a_dump_file_is_loaded_directly(self):
        path = self.write_dump()
        main.analysis_runner(str(path))
        self.setup_from_args.assert_called_once()

    def test_a_run_directory_is_searched_for_the_dump(self):
        # The analysis script is usually pointed at the run directory, and
        # the generation stage always writes the dump under its data/ folder.
        self.write_dump(in_data_dir=True)
        main.analysis_runner(str(self.tmp_dir))
        self.setup_from_args.assert_called_once()

    def test_a_directory_without_a_dump_is_an_error(self):
        (self.tmp_dir / "data").mkdir()
        with pytest.raises(StopIteration):
            main.analysis_runner(str(self.tmp_dir))

    def test_a_missing_run_directory_is_an_error(self):
        with pytest.raises((FileNotFoundError, StopIteration)):
            main.analysis_runner(str(self.tmp_dir / "absent"))


class TestAnalysisRunnerArgumentOverrides(AnalysisRunnerMixin):
    """The dump carries the arguments of the original run, and a handful of
    them can be overridden when the analysis is repeated."""

    def test_the_output_directory_is_overridden_when_given(self):
        path = self.write_dump()
        main.analysis_runner(str(path), outdir="new_outdir")
        assert self.setup_from_args.call_args.args[2].outdir == "new_outdir"

    def test_the_output_directory_is_kept_when_not_given(self):
        path = self.write_dump()
        main.analysis_runner(str(path))
        assert (
            self.setup_from_args.call_args.args[2].outdir == "original_outdir"
        )

    def test_the_label_is_overridden_when_given(self):
        path = self.write_dump()
        main.analysis_runner(str(path), label="new_label")
        assert self.setup_from_args.call_args.args[2].label == "new_label"

    def test_the_label_is_kept_when_not_given(self):
        path = self.write_dump()
        main.analysis_runner(str(path))
        assert self.setup_from_args.call_args.args[2].label == "original_label"

    def test_plotting_is_always_taken_from_the_call_not_the_dump(self):
        path = self.write_dump()
        main.analysis_runner(str(path), plot=True)
        assert self.setup_from_args.call_args.args[2].plot

    def test_plotting_defaults_to_off(self):
        path = self.write_dump()
        main.analysis_runner(str(path))
        assert not self.setup_from_args.call_args.args[2].plot


class TestAnalysisRunnerLikelihoodSetup(AnalysisRunnerMixin):
    def test_the_priors_are_read_from_the_file_named_in_the_dump(self):
        path = self.write_dump()
        main.analysis_runner(str(path))
        self.prior_dict.from_json.assert_called_once_with(
            str(self.tmp_dir / "prior.json")
        )

    def test_the_likelihood_is_built_from_the_dump_and_the_priors(self):
        path = self.write_dump()
        main.analysis_runner(str(path))
        dump, priors, _args, _logger = self.setup_from_args.call_args.args
        assert dump["messengers"] == ["em"]
        assert priors is self.priors

    def test_the_shared_logger_is_passed_so_every_stage_logs_together(self):
        path = self.write_dump()
        main.analysis_runner(str(path))
        assert self.setup_from_args.call_args.args[3] is main.logger


class TestAnalysisRunnerMetaData(AnalysisRunnerMixin):
    """The dump holds live objects that cannot be serialised into a result
    file, so the metadata copy keeps only their representations."""

    def test_the_waveform_generator_is_reduced_to_its_representation(self):
        path = self.write_dump(waveform_generator=StubRepr("a waveform generator"))
        main.analysis_runner(str(path))
        meta_data = self.pbilby.call_args.kwargs["meta_data"]
        assert meta_data["waveform_generator"] == "a waveform generator"

    def test_each_interferometer_is_reduced_to_its_representation(self):
        path = self.write_dump(ifo_list=[StubRepr("H1"), StubRepr("L1")])
        main.analysis_runner(str(path))
        meta_data = self.pbilby.call_args.kwargs["meta_data"]
        assert meta_data["ifo_list"] == ["H1", "L1"]

    def test_an_em_only_dump_has_neither_key(self):
        path = self.write_dump()
        main.analysis_runner(str(path))
        meta_data = self.pbilby.call_args.kwargs["meta_data"]
        assert "waveform_generator" not in meta_data
        assert "ifo_list" not in meta_data

    def test_the_rest_of_the_dump_is_carried_into_the_metadata(self):
        path = self.write_dump()
        main.analysis_runner(str(path))
        meta_data = self.pbilby.call_args.kwargs["meta_data"]
        assert meta_data["messengers"] == ["em"]


class TestAnalysisRunnerSamplerChoice(AnalysisRunnerMixin):
    """Only dynesty is run through the parallel bilby path; every other
    sampler goes through the plain bilby wrapper."""

    def test_dynesty_uses_the_parallel_bilby_path(self):
        path = self.write_dump(sampler="dynesty")
        result = main.analysis_runner(str(path))
        self.pbilby.assert_called_once()
        self.bilby.assert_not_called()
        assert result == "pbilby_result"

    def test_another_sampler_uses_the_plain_bilby_path(self):
        path = self.write_dump(sampler="pymultinest")
        result = main.analysis_runner(str(path))
        self.bilby.assert_called_once()
        self.pbilby.assert_not_called()
        assert result == "bilby_result"

    def test_the_injection_parameters_are_forwarded_when_present(self):
        path = self.write_dump(injection_parameters={"mass_1": 1.4})
        main.analysis_runner(str(path))
        assert self.pbilby.call_args.args[3] == {"mass_1": 1.4}

    def test_a_run_on_real_data_forwards_no_injection_parameters(self):
        path = self.write_dump()
        main.analysis_runner(str(path))
        assert self.pbilby.call_args.args[3] is None

    def test_the_mpi_rank_is_forwarded_so_only_one_rank_writes_output(self):
        path = self.write_dump()
        main.analysis_runner(str(path))
        assert self.pbilby.call_args.args[4] == main.rank

    def test_extra_keyword_arguments_reach_the_parallel_sampler(self):
        path = self.write_dump()
        main.analysis_runner(str(path), nlive=500)
        assert self.pbilby.call_args.kwargs["nlive"] == 500

    def test_the_plain_sampler_is_called_with_the_positional_arguments_only(self):
        path = self.write_dump(sampler="pymultinest")
        main.analysis_runner(str(path))
        assert len(self.bilby.call_args.args) == 5
        assert self.bilby.call_args.kwargs == {}


class TestNMMAAnalysisEntryPoint:
    """The console script only builds the parser and forwards the parsed
    arguments as keywords."""

    def test_the_parser_is_built_for_the_parallel_dynesty_sampler(self):
        with patch.object(main, "create_nmma_analysis_parser") as create:
            with patch.object(
                main, "parse_analysis_args", return_value=Namespace(data_dump="d")
            ):
                with patch.object(main, "analysis_runner"):
                    main.nmma_analysis()
        create.assert_called_once_with(sampler="dynesty")

    def test_the_parsed_arguments_are_forwarded_as_keywords(self):
        parsed = Namespace(data_dump="dump.pickle", outdir="out", label="run")
        with patch.object(main, "create_nmma_analysis_parser"):
            with patch.object(main, "parse_analysis_args", return_value=parsed):
                with patch.object(main, "analysis_runner") as runner:
                    main.nmma_analysis()
        runner.assert_called_once_with(
            data_dump="dump.pickle", outdir="out", label="run"
        )

    def test_the_parser_is_handed_to_the_argument_parsing_helper(self):
        with patch.object(
            main, "create_nmma_analysis_parser", return_value="the_parser"
        ):
            with patch.object(
                main, "parse_analysis_args", return_value=Namespace()
            ) as parse:
                with patch.object(main, "analysis_runner"):
                    main.nmma_analysis()
        parse.assert_called_once_with("the_parser")


class TestModuleSetup:
    """Importing the analysis module pins the thread count, because each MPI
    rank runs its own process and must not oversubscribe the node."""

    def test_the_thread_count_is_pinned_to_one(self):
        import os

        assert os.environ["OMP_NUM_THREADS"] == "1"

    def test_a_rank_is_always_defined_even_without_mpi(self):
        assert isinstance(main.rank, int)
