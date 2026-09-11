import json
import logging
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from astropy import time

from nmma.core import utils


class TestSetupLogger(unittest.TestCase):
    def setUp(self):
        self.original_level = utils.logger.level
        self.original_handlers = list(utils.logger.handlers)

    def tearDown(self):
        utils.logger.handlers = self.original_handlers
        utils.logger.setLevel(self.original_level)

    def test_default_level_is_info(self):
        utils.setup_logger()
        self.assertEqual(utils.logger.level, logging.INFO)

    def test_level_is_case_insensitive(self):
        utils.setup_logger("debug")
        self.assertEqual(utils.logger.level, logging.DEBUG)
        utils.setup_logger("WARNING")
        self.assertEqual(utils.logger.level, logging.WARNING)

    def test_unknown_level_raises(self):
        with self.assertRaises(ValueError):
            utils.setup_logger("not_a_level")

    def test_stream_handler_is_added_only_once(self):
        utils.logger.handlers = []
        utils.setup_logger()
        utils.setup_logger()
        stream_handlers = [
            h for h in utils.logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        self.assertEqual(len(stream_handlers), 1)

    def test_existing_handlers_follow_the_new_level(self):
        utils.setup_logger("info")
        utils.setup_logger("debug")
        for handler in utils.logger.handlers:
            self.assertEqual(handler.level, logging.DEBUG)


class TestNumpyEncoder(unittest.TestCase):
    def test_encodes_numpy_arrays_as_lists(self):
        encoded = json.dumps({"a": np.array([1.0, 2.0])}, cls=utils.NumpyEncoder)
        self.assertEqual(json.loads(encoded), {"a": [1.0, 2.0]})

    def test_falls_back_to_the_default_encoder(self):
        with self.assertRaises(TypeError):
            json.dumps({"a": object()}, cls=utils.NumpyEncoder)


class TestLoadYaml(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_loads_a_mapping(self):
        path = self.tmp_dir / "conf.yaml"
        path.write_text("label: test\nnlive: 32\n")
        self.assertEqual(utils.load_yaml(path), {"label": "test", "nlive": 32})

    def test_expands_environment_variables(self):
        import os

        path = self.tmp_dir / "conf.yaml"
        path.write_text("outdir: $NMMA_TEST_OUTDIR\n")
        os.environ["NMMA_TEST_OUTDIR"] = "/some/where"
        try:
            self.assertEqual(utils.load_yaml(path), {"outdir": "/some/where"})
        finally:
            del os.environ["NMMA_TEST_OUTDIR"]


class TestReadTriggerTime(unittest.TestCase):
    def test_reads_mjd_from_parameters(self):
        self.assertAlmostEqual(
            utils.read_trigger_time(parameters={"trigger_time": 59000.0}), 59000.0
        )

    def test_reads_geocent_time_as_gps(self):
        gps = 1187008882.43
        expected = time.Time(gps, format="gps").mjd
        self.assertAlmostEqual(
            utils.read_trigger_time(parameters={"geocent_time": gps}), expected
        )

    def test_geocent_time_x_takes_precedence_over_geocent_time(self):
        # a joint run carries the GW trigger as geocent_time_x
        parameters = {"geocent_time_x": 1187008882.43, "geocent_time": 0.0}
        expected = time.Time(1187008882.43, format="gps").mjd
        self.assertAlmostEqual(utils.read_trigger_time(parameters), expected)

    def test_gps_output_format(self):
        gps = 1187008882.43
        self.assertAlmostEqual(
            utils.read_trigger_time(parameters={"geocent_time": gps}, out_format="gps"),
            gps,
            places=3,
        )

    def test_args_gps_attribute_wins(self):
        args = Namespace(gps=1187008882.43, trigger_time=None)
        expected = time.Time(1187008882.43, format="gps").mjd
        self.assertAlmostEqual(utils.read_trigger_time(args=args), expected)

    def test_args_trigger_time_is_read_as_mjd(self):
        args = Namespace(gps=None, trigger_time=59000.0)
        self.assertAlmostEqual(utils.read_trigger_time(args=args), 59000.0)

    def test_missing_trigger_time_returns_none(self):
        self.assertIsNone(utils.read_trigger_time(parameters={}))

    def test_parameters_take_precedence_and_are_written_back_to_args(self):
        args = Namespace(gps=None, trigger_time=None)
        result = utils.read_trigger_time(parameters={"trigger_time": 59000.0}, args=args)
        self.assertAlmostEqual(result, 59000.0)
        self.assertAlmostEqual(args.trigger_time, 59000.0)


class InjectionFileMixin:
    """Writes the bilby-style injection JSON that read_injection_file expects:
    a {"injections": <bilby-encoded DataFrame>} document."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.injections = pd.DataFrame(
            {"luminosity_distance": [40.0, 100.0], "log10_mej": [-2.0, -1.5]}
        )
        self.injection_file = self.tmp_dir / "injections.json"
        self.injection_file.write_text(
            json.dumps(
                {
                    "injections": {
                        "__dataframe__": True,
                        "content": self.injections.to_dict(orient="list"),
                    }
                }
            )
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)


class TestReadInjectionFile(InjectionFileMixin, unittest.TestCase):
    def test_reads_a_path(self):
        df = utils.read_injection_file(str(self.injection_file))
        pd.testing.assert_frame_equal(df, self.injections)

    def test_reads_from_a_namespace(self):
        args = Namespace(
            injection=str(self.injection_file),
            injection_file=str(self.injection_file),
            outdir=str(self.tmp_dir),
        )
        df = utils.read_injection_file(args)
        pd.testing.assert_frame_equal(df, self.injections)

    def test_namespace_without_json_suffix_is_resolved_against_outdir(self):
        args = Namespace(
            injection="injections", injection_file="injections", outdir=str(self.tmp_dir)
        )
        df = utils.read_injection_file(args)
        pd.testing.assert_frame_equal(df, self.injections)
        self.assertEqual(Path(args.injection_file), self.injection_file)


class TestInjectionFromArgs(InjectionFileMixin, unittest.TestCase):
    def test_injection_from_file_selects_by_injection_num(self):
        args = Namespace(
            injection=str(self.injection_file),
            injection_file=str(self.injection_file),
            outdir=str(self.tmp_dir),
            injection_num=1,
        )
        self.assertEqual(
            utils.injection_from_file(args),
            {"luminosity_distance": 100.0, "log10_mej": -1.5},
        )

    def test_injection_from_args_dispatches_to_the_file(self):
        args = Namespace(
            injection=True,
            injection_file=str(self.injection_file),
            outdir=str(self.tmp_dir),
            injection_num=0,
        )
        self.assertEqual(
            utils.injection_from_args(args),
            {"luminosity_distance": 40.0, "log10_mej": -2.0},
        )

    def test_injection_from_args_dispatches_to_the_prior(self):
        prior_file = self.tmp_dir / "test.prior"
        prior_file.write_text(
            "log10_mej = Uniform(minimum=-3, maximum=-1, name='log10_mej')\n"
        )
        args = Namespace(
            injection=True,
            injection_file=None,
            prior_file=str(prior_file),
            prior=None,
            generation_seed=42,
            outdir=str(self.tmp_dir),
        )
        sample = utils.injection_from_args(args)
        self.assertIn("log10_mej", sample)
        self.assertTrue(-3 <= sample["log10_mej"] <= -1)

    def test_injection_from_prior_is_reproducible_for_a_fixed_seed(self):
        prior_file = self.tmp_dir / "test.prior"
        prior_file.write_text(
            "log10_mej = Uniform(minimum=-3, maximum=-1, name='log10_mej')\n"
        )
        args = Namespace(
            prior_file=str(prior_file), prior=None, generation_seed=7, outdir=str(self.tmp_dir)
        )
        first = utils.injection_from_prior(args)
        second = utils.injection_from_prior(args)
        self.assertEqual(first, second)


class TestGetPosteriors(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.posterior = pd.DataFrame(
            {
                "log10_mej": [-2.0, -1.5, -1.0],
                "log_likelihood": [1.0, 3.0, 2.0],
                "log_prior": [0.0, -1.0, 1.5],
            }
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_dataframe_is_passed_through(self):
        self.assertIs(utils.get_posteriors(self.posterior), self.posterior)

    def test_dict_is_converted_to_a_dataframe(self):
        result = utils.get_posteriors({"a": [1, 2]})
        pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [1, 2]}))

    def test_reads_a_csv(self):
        path = self.tmp_dir / "samples.dat"
        self.posterior.to_csv(path, sep=" ", index=False)
        pd.testing.assert_frame_equal(utils.get_posteriors(path), self.posterior)

    def test_reads_a_json(self):
        path = self.tmp_dir / "samples.json"
        path.write_text(json.dumps({"posterior": self.posterior.to_dict(orient="list")}))
        self.assertEqual(
            utils.get_posteriors(path)["log10_mej"], self.posterior["log10_mej"].tolist()
        )

    def test_reads_an_hdf5(self):
        path = self.tmp_dir / "samples.hdf5"
        with h5py.File(path, "w") as f:
            group = f.create_group("posterior")
            for key, values in self.posterior.items():
                group.create_dataset(key, data=values.to_numpy())
        result = utils.get_posteriors(path)
        pd.testing.assert_frame_equal(
            result[self.posterior.columns], self.posterior, check_dtype=False
        )

    def test_relative_path_is_resolved_against_outdir(self):
        path = self.tmp_dir / "samples.dat"
        self.posterior.to_csv(path, sep=" ", index=False)
        result = utils.get_posteriors("samples.dat", outdir=self.tmp_dir)
        pd.testing.assert_frame_equal(result, self.posterior)

    def test_unsupported_suffix_raises(self):
        path = self.tmp_dir / "samples.nonsense"
        path.touch()
        with self.assertRaises(ValueError):
            utils.get_posteriors(path)

    def test_missing_file_raises(self):
        with self.assertRaises(AssertionError):
            utils.get_posteriors("does_not_exist.dat", outdir=self.tmp_dir)

    def test_namespace_without_a_result_file_raises(self):
        args = Namespace(label="missing", outdir=str(self.tmp_dir))
        with self.assertRaises(FileNotFoundError):
            utils.get_posteriors(args)


class TestSetFilename(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.args = Namespace(outdir=str(self.tmp_dir))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_extensionless_name_uses_the_default_extension(self):
        self.assertEqual(
            utils.set_filename("bestfit", self.args), self.tmp_dir / "bestfit.json"
        )

    def test_extensionless_name_honours_the_extension_argument(self):
        self.args.extension = "csv"
        self.assertEqual(
            utils.set_filename("bestfit", self.args), self.tmp_dir / "bestfit.csv"
        )

    def test_identifier_is_inserted_before_the_suffix(self):
        self.assertEqual(
            utils.set_filename("bestfit.json", self.args, identifier="_0"),
            self.tmp_dir / "bestfit_0.json",
        )

    def test_a_name_with_a_parent_directory_bypasses_outdir(self):
        self.assertEqual(
            utils.set_filename("/elsewhere/bestfit.json", self.args),
            Path("/elsewhere/bestfit.json"),
        )

    def test_unsupported_suffix_raises(self):
        with self.assertRaises(ValueError):
            utils.set_filename("bestfit.hdf5", self.args)

    def test_outdir_is_created(self):
        outdir = self.tmp_dir / "new" / "nested"
        utils.set_filename("bestfit.json", Namespace(outdir=str(outdir)))
        self.assertTrue(outdir.is_dir())


class TestReadBestfit(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.posterior = pd.DataFrame(
            {
                "log10_mej": [-2.0, -1.5, -1.0],
                "log_likelihood": [1.0, 3.0, 2.0],
                "log_prior": [0.0, -1.0, 1.5],
            }
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_max_likelihood_picks_the_highest_log_likelihood(self):
        bestfit = utils.read_bestfit_from_posterior(self.posterior)
        self.assertAlmostEqual(bestfit["log10_mej"], -1.5)
        self.assertEqual(bestfit["best_fit_index"], 1)

    def test_max_posterior_adds_the_log_prior(self):
        bestfit = utils.read_bestfit_from_posterior(self.posterior, mode="max_posterior")
        self.assertAlmostEqual(bestfit["log10_mej"], -1.0)
        self.assertEqual(bestfit["best_fit_index"], 2)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            utils.read_bestfit_from_posterior(self.posterior, mode="max_nonsense")

    def test_return_posterior_also_returns_the_samples(self):
        bestfit, posterior = utils.read_bestfit_from_posterior(
            self.posterior, return_posterior=True
        )
        self.assertIn("best_fit_index", bestfit)
        pd.testing.assert_frame_equal(posterior, self.posterior)

    def test_read_bestfit_from_json_selects_the_requested_columns(self):
        path = self.tmp_dir / "bestfit.json"
        path.write_text(json.dumps({"log10_mej": -2.0, "log10_vej": -1.0, "extra": 3.0}))
        truths = utils.read_bestfit_from_json(path, ["log10_mej", "log10_vej"])
        np.testing.assert_allclose(truths, [-2.0, -1.0])

    def test_read_bestfit_from_json_ignores_missing_columns(self):
        path = self.tmp_dir / "bestfit.json"
        path.write_text(json.dumps({"log10_mej": -2.0}))
        truths = utils.read_bestfit_from_json(path, ["log10_mej", "not_sampled"])
        np.testing.assert_allclose(truths, [-2.0])


class TestRejectionSample(unittest.TestCase):
    def test_uniform_weights_keep_roughly_half_the_samples(self):
        rng = np.random.default_rng(42)
        posterior = np.arange(1000.0)
        weights = np.ones(1000)
        kept, keep = utils.rejection_sample(posterior, weights, rng)
        self.assertEqual(len(kept), keep.sum())
        self.assertGreater(len(kept), 400)

    def test_zero_weights_are_always_rejected(self):
        rng = np.random.default_rng(42)
        weights = np.array([0.0, 1.0, 0.0, 1.0])
        _, keep = utils.rejection_sample(np.arange(4.0), weights, rng)
        self.assertFalse(keep[0])
        self.assertFalse(keep[2])

    def test_kept_samples_match_the_mask(self):
        rng = np.random.default_rng(0)
        posterior = np.arange(100.0)
        weights = rng.uniform(0, 1, 100)
        kept, keep = utils.rejection_sample(posterior, weights, rng)
        np.testing.assert_allclose(kept, posterior[keep])


class TestSigLims(unittest.TestCase):
    def test_returns_a_latex_string_with_asymmetric_errors(self):
        rng = np.random.default_rng(0)
        label = utils.sig_lims(rng.normal(100.0, 3.0, 100000))
        self.assertTrue(label.startswith("$") and label.endswith("$"))
        self.assertIn("_{-", label)
        self.assertIn("^{+", label)

    def test_quantiles_can_be_overridden(self):
        values = np.linspace(0.0, 1.0, 10001)
        default = utils.sig_lims(values)
        narrow = utils.sig_lims(values, quantiles=[0.25, 0.5, 0.75])
        self.assertNotEqual(default, narrow)

    def test_large_values_are_rounded_to_integers(self):
        rng = np.random.default_rng(0)
        label = utils.sig_lims(rng.normal(1e6, 3e4, 100000))
        # ord_error < 0 here, so the branch that rounds to whole numbers runs
        self.assertNotIn(".", label)

    def test_significant_digits_can_be_widened(self):
        rng = np.random.default_rng(0)
        values = rng.normal(100.0, 3.0, 100000)
        self.assertNotEqual(
            utils.sig_lims(values, sig_unc=2), utils.sig_lims(values, sig_unc=3)
        )


class TestInputObjToStr(unittest.TestCase):
    def test_string_is_passed_through(self):
        self.assertEqual(utils.input_obj_to_str("a_file.dat"), "a_file.dat")

    def test_namespace_attribute_is_read_by_name(self):
        args = Namespace(prior_file="my.prior")
        self.assertEqual(utils.input_obj_to_str(args, "prior_file"), "my.prior")

    def test_missing_namespace_attribute_gives_none(self):
        self.assertIsNone(utils.input_obj_to_str(Namespace(), "prior_file"))

    def test_dict_lookup_by_reference_name(self):
        self.assertEqual(utils.input_obj_to_str({"prior_file": "my.prior"}, "prior_file"), "my.prior")

    def test_dict_without_the_reference_name_falls_back_to_the_first_value(self):
        self.assertEqual(utils.input_obj_to_str({"other": "my.prior"}, "prior_file"), "my.prior")

    def test_list_falls_back_to_the_first_entry(self):
        self.assertEqual(utils.input_obj_to_str(["first.dat", "second.dat"]), "first.dat")

    def test_unidentifiable_input_raises(self):
        with self.assertRaises(TypeError):
            utils.input_obj_to_str(3.0)


class TestNanLevel(unittest.TestCase):
    def test_credible_interval_without_nans(self):
        data = np.linspace(0.0, 1.0, 1001)
        low, high = utils.nan_level(data, 0.9)
        self.assertAlmostEqual(low, 0.05, places=2)
        self.assertAlmostEqual(high, 0.95, places=2)

    def test_nans_narrow_the_interval(self):
        data = np.linspace(0.0, 1.0, 1000)
        clean = utils.nan_level(data, 0.9)
        with_nans = utils.nan_level(np.concatenate([data[:900], np.full(100, np.nan)]), 0.9)
        self.assertGreater(with_nans[0], clean[0])

    def test_too_many_nans_gives_nan_bounds(self):
        data = np.array([1.0, 2.0, np.nan, np.nan])
        self.assertTrue(np.all(np.isnan(utils.nan_level(data, 0.4))))

    def test_weights_shift_the_interval(self):
        data = np.linspace(0.0, 1.0, 1000)
        unweighted = utils.nan_level(data, 0.5)
        weighted = utils.nan_level(data, 0.5, weights=np.linspace(0.0, 1.0, 1000))
        self.assertGreater(weighted[0], unweighted[0])

    def test_weights_are_masked_alongside_the_nans(self):
        data = np.concatenate([np.linspace(0.0, 1.0, 900), np.full(100, np.nan)])
        low, high = utils.nan_level(data, 0.5, weights=np.ones(1000))
        self.assertTrue(np.isfinite(low) and np.isfinite(high))


if __name__ == "__main__":
    unittest.main()
