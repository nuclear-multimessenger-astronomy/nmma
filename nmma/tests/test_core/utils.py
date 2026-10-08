import json
import logging
import shutil
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from astropy import time
from bilby.core.utils import BilbyJsonEncoder

from nmma.core import utils


class TestSetupLogger:
    """Check log-level parsing and the connection to bilby"""

    outdir = Path(__file__).with_name("setup_logger_outdir")
    label = "test_run"

    def setup_method(self):
        """
        logging.getLogger are process-wide singletons, so
        setup_method/teardown_method snapshot and restore both loggers'
        handlers and levels around every test, keeping this suite from
        leaking state into other tests that log through nmma or bilby.
        """
        self.nmma_logger = logging.getLogger("nmma")
        self.bilby_logger = logging.getLogger("bilby")
        self._saved_state = [
            (logger, list(logger.handlers), logger.level)
            for logger in (self.nmma_logger, self.bilby_logger)
        ]
        for logger in (self.nmma_logger, self.bilby_logger):
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()

    def teardown_method(self):
        shutil.rmtree(self.outdir, ignore_errors=True)
        for logger, handlers, level in self._saved_state:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()
            for handler in handlers:
                logger.addHandler(handler)
            logger.setLevel(level)

    def test_rejects_an_unrecognised_log_level(self):
        with pytest.raises(ValueError):
            utils.setup_logger(log_level="bogus")

    def test_log_level(self):
        utils.setup_logger(log_level="warning", outdir=self.outdir, label=self.label)

        assert self.nmma_logger.level == logging.WARNING
        assert self.bilby_logger.level == logging.WARNING

        self.nmma_logger.info("this should be filtered out")
        self.nmma_logger.warning("nmma warning")
        self.bilby_logger.info("this should also be filtered out")
        self.bilby_logger.warning("bilby warning")

        content = (self.outdir / f"{self.label}.log").read_text()
        assert "filtered out" not in content
        assert "nmma warning" in content
        assert "bilby warning" in content

    def test_updates_nmmas_own_handlers(self):
        utils.setup_logger(log_level="DEBUG", outdir=self.outdir, label=self.label)
        utils.setup_logger(log_level="ERROR", outdir=self.outdir, label=self.label)

        assert self.nmma_logger.level == logging.ERROR
        for handler in self.nmma_logger.handlers:
            assert handler.level == logging.ERROR

        assert self.bilby_logger.level == logging.ERROR
        for handler in self.bilby_logger.handlers:
            assert handler.level == logging.ERROR


class TestReadTriggerTime:
    trigger_time_mjd = 58000.0
    trigger_time_gps = 1187008882.4

    def test_read_trigger_time(self):
        parameters = {
            "trigger_time": self.trigger_time_mjd,
            "geocent_time_x": self.trigger_time_gps,
            "geocent_time": self.trigger_time_gps,
        }

        assert utils.read_trigger_time(parameters=parameters) == self.trigger_time_mjd

        geocent_only = {"geocent_time_x": self.trigger_time_gps}
        assert utils.read_trigger_time(parameters=geocent_only) == pytest.approx(
            time.Time(self.trigger_time_gps, format="gps").mjd
        )

        args_gps = Namespace(gps=self.trigger_time_gps, trigger_time=None)
        assert utils.read_trigger_time(args=args_gps) == pytest.approx(
            time.Time(self.trigger_time_gps, format="gps").mjd
        )
        assert utils.read_trigger_time(
            args=args_gps, out_format="gps"
        ) == pytest.approx(self.trigger_time_gps)

        args_mjd = Namespace(gps=None, trigger_time=self.trigger_time_mjd)
        assert utils.read_trigger_time(args=args_mjd) == self.trigger_time_mjd
        assert args_mjd.trigger_time == self.trigger_time_mjd

        assert utils.read_trigger_time() is None
        no_time_args = Namespace(gps=None, trigger_time=None)
        assert utils.read_trigger_time(args=no_time_args) is None

    def test_read_zero_trigger_time(self):
        parameters = {"trigger_time": 0.0}
        test_args = Namespace()
        trigger_time = utils.read_trigger_time(parameters=parameters, args=test_args)
        assert trigger_time == test_args.trigger_time

        test_args_1 = Namespace(trigger_time=0.0, time_format="mjd")
        test_args_2 = Namespace(trigger_time=0.0, time_format="gps")
        trigger_time_mjd = utils.read_trigger_time(args=test_args_1)
        assert trigger_time_mjd == test_args_1.trigger_time

        trigger_time_gps = utils.read_trigger_time(args=test_args_2, out_format="gps")
        assert trigger_time_gps == test_args_2.trigger_time

        trigger_time_mix = utils.read_trigger_time(args=test_args_2)
        assert trigger_time_mix != test_args_2.trigger_time


class TestInjectionHandling:
    @classmethod
    def setup_class(cls):
        cls.outdir = Path(__file__).with_name("utils_injection_outdir")
        cls.outdir.mkdir(parents=True, exist_ok=True)
        cls.injection = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        cls.seed = 5

        cls.injection_path = cls.outdir / "injection.json"
        with open(cls.injection_path, "w") as f:
            json.dump({"injections": cls.injection}, f, cls=BilbyJsonEncoder)

        cls.prior_path = cls.outdir / "test.prior"
        cls.prior_path.write_text("a = Uniform(minimum=0, maximum=1, name='a')\n")

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.outdir, ignore_errors=True)

    def setup_method(self):
        self.args = Namespace(
            injection=False,
            injection_file=None,
            outdir=str(self.outdir),
            injection_num=0,
            generation_seed=self.seed,
            prior_file=str(self.prior_path),
        )

    # test read_injection_file
    def test_read_injection_file_from_path(self):
        from_path = utils.read_injection_file(str(self.injection_path))
        pd.testing.assert_frame_equal(from_path, self.injection)

    def test_read_injection_file_from_namespace(self):
        self.args.injection_file = str(self.injection_path)
        from_namespace = utils.read_injection_file(self.args)
        pd.testing.assert_frame_equal(from_namespace, self.injection)

    def test_read_injection_file_from_incomplete_args(self):
        self.args.injection = str(self.injection_path.stem)
        pd.testing.assert_frame_equal(
            utils.read_injection_file(self.args), self.injection
        )

    def test_injection_from_file(self):
        self.args.injection_file = str(self.injection_path)
        assert utils.injection_from_file(self.args) == self.injection.iloc[0].to_dict()

    def test_injection_from_prior(self):
        injection_1 = utils.injection_from_prior(self.args)
        injection_2 = utils.injection_from_prior(self.args)

        assert injection_1 == injection_2

    def test_injection_from_args_prior_branch(self):
        self.args.injection = True
        assert utils.injection_from_args(self.args) == utils.injection_from_prior(
            self.args
        )

    def test_injection_from_args_file_branch(self):
        self.args.injection_file = str(self.injection_path)
        expected = self.injection.iloc[0].to_dict()
        assert utils.injection_from_args(self.args) == expected


class TestPosteriorHandling:
    outdir = Path(__file__).with_name("utils_posterior_outdir")
    posterior = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [10.0, 20.0, 30.0],
            "log_likelihood": [-1.0, -0.5, -2.0],
            "log_prior": [0.1, 0.2, 0.05],
        }
    )
    bestfit_path = outdir / "bestfit.json"

    @classmethod
    def setup_class(cls):
        cls.outdir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.outdir, ignore_errors=True)

    def test_get_posteriors_accepts_for_dfs_and_dicts(self):
        assert utils.get_posteriors(self.posterior) is self.posterior

        as_dict = utils.get_posteriors(self.posterior.to_dict(orient="list"))
        pd.testing.assert_frame_equal(as_dict, self.posterior)

    def test_get_posteriors_for_files(self):
        csv_path = self.outdir / "post.csv"
        self.posterior.to_csv(csv_path, sep=" ", index=False)
        pd.testing.assert_frame_equal(utils.get_posteriors(csv_path), self.posterior)

        hdf5_path = self.outdir / "post.hdf5"
        with h5py.File(hdf5_path, "w") as f:
            group = f.create_group("posterior")
            for column in self.posterior.columns:
                group.create_dataset(column, data=self.posterior[column].to_numpy())
        pd.testing.assert_frame_equal(utils.get_posteriors(hdf5_path), self.posterior)

        json_path = self.outdir / "post.json"
        with open(json_path, "w") as f:
            json.dump({"posterior": self.posterior.to_dict(orient="list")}, f)

        pd.testing.assert_frame_equal(utils.get_posteriors(json_path), self.posterior)

        with pytest.raises(AssertionError):
            utils.get_posteriors("does_not_exist.csv", outdir=str(self.outdir))
        bad_path = self.outdir / "post.xyz"
        bad_path.write_text("not a real posterior file")
        with pytest.raises(ValueError):
            utils.get_posteriors(bad_path)

    def test_bestfits(self):
        by_likelihood = utils.read_bestfit_from_posterior(
            self.posterior, mode="max_likelihood"
        )
        likelihood_index = self.posterior.log_likelihood.idxmax()
        assert by_likelihood["best_fit_index"] == likelihood_index

        combined = self.posterior.log_likelihood + self.posterior.log_prior
        by_posterior = utils.read_bestfit_from_posterior(
            self.posterior, mode="max_posterior"
        )
        assert by_posterior["best_fit_index"] == combined.idxmax()

        with pytest.raises(ValueError):
            utils.read_bestfit_from_posterior(self.posterior, mode="unknown")


class TestRestOfCoreUtils:
    @classmethod
    def setup_class(cls):
        cls.outdir = Path(__file__).with_name("utils_outdir")
        cls.outdir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.outdir, ignore_errors=True)

    def test_input_obj_to_str_resolves_various_container_types(self):
        assert utils.input_obj_to_str(Namespace(a="hello"), "a") == "hello"
        assert utils.input_obj_to_str({"a": "hi"}, "a") == "hi"
        assert utils.input_obj_to_str({"a": "hi"}, "missing") == "hi"
        assert utils.input_obj_to_str(["only"], None) == "only"
        assert utils.input_obj_to_str("plain", None) == "plain"

        with pytest.raises(TypeError):
            utils.input_obj_to_str(42, None)

    def test_set_filename(self):
        args = Namespace(outdir=self.outdir)
        assert utils.set_filename("result", args) == self.outdir / "result.json"
        args.extension = "csv"
        assert utils.set_filename("result", args) == self.outdir / "result.csv"
        assert utils.set_filename("sub/result.csv", args) == Path("sub/result.csv")

        with pytest.raises(ValueError):
            utils.set_filename("result.bad", args)

    def test_sig_lims(self):
        # test a few explicit examples
        test_values_1 = np.geomspace(1.0, 100.0)
        assert utils.sig_lims(test_values_1) == "${10}_{-8}^{+38}$"
        test_values_2 = np.linspace(1.0, 9.0)
        assert utils.sig_lims(test_values_1, sig_unc=4) == "${10.01}_{-7.92}^{+37.88}$"
        assert utils.sig_lims(test_values_2) == "${5.0}_{-2.7}^{+2.7}$"
        test_values_3 = np.linspace(1.0, 12.0)
        assert utils.sig_lims(test_values_3) == "${6}_{-4}^{+4}$"
        test_values_4 = np.geomspace(-1.0, -12.0)
        assert utils.sig_lims(test_values_4) == "${-3.5}_{-4.6}^{+2.0}$"
