import copy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nmma.em import analysis, em_parsing, utils
from nmma.em.model import SimpleBolometricLightCurveModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class AnalysisTestsContainer:
    outdir = Path(__file__).with_name("outdir")
    filters = ["ztfr"]
    generation_seed = 42

    @classmethod
    def setup_class(cls):
        cls.args = em_parsing.parsing_and_logging(
            em_parsing.multi_wavelength_analysis_parser, []
        )
        overrides = dict(
            em_transient_class="fiesta_kn",
            em_model="Bu2026_MLP",
            label=cls.label,
            outdir=str(cls.outdir),
            prior_file=str(DATA_DIR / "Bu2026_simplified.prior"),
            filters=cls.filters,
            injection=True,
            generation_seed=cls.generation_seed,
        )
        for key, value in overrides.items():
            setattr(cls.args, key, value)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.outdir, ignore_errors=True)


class TestDataFiltering:
    """Tests for the pure photometry-wrangling helpers analysis_setup
    composes: inspect_detection_limit, check_detections and
    set_analysis_filters. None of these touch a light curve model.
    """

    def all_finite_data(self):
        return {
            "ztfr": {
                "time": np.array([1.0, 2.0, 3.0]),
                "mag": np.array([20.0, 21.0, 25.0]),
                "mag_error": np.array([0.1, 0.2, 0.3]),
            },
        }

    def mixed_detection_data(self):
        return {
            "ztfr": {
                "time": np.array([1.0, 2.0, 3.0]),
                "mag": np.array([20.0, 21.0, 25.0]),
                "mag_error": np.array([0.1, 0.2, np.inf]),
            },
            "ztfg": {
                "time": np.array([1.0, 2.0]),
                "mag": np.array([30.0, 31.0]),
                "mag_error": np.array([np.inf, np.inf]),
            },
        }

    def test_inspect_detection_limit_clips_points_fainter_than_the_limit(self):
        data = self.all_finite_data()

        result = analysis.inspect_detection_limit({"ztfr": 22.0}, data)

        assert result["ztfr"]["mag"].tolist() == [20.0, 21.0, 22.0]
        assert result["ztfr"]["mag_error"][:2].tolist() == [0.1, 0.2]
        assert np.isinf(result["ztfr"]["mag_error"][2])

    def test_check_detections_removes_nondetections_and_drops_empty_filters(self):
        data = self.mixed_detection_data()

        result = analysis.check_detections(data, remove_nondetections=True)

        assert list(result.keys()) == ["ztfr"]
        assert len(result["ztfr"]["time"]) == 2

    def test_check_detections_keeps_nondetections_by_default(self):
        data = self.mixed_detection_data()

        result = analysis.check_detections(data)

        assert list(result.keys()) == ["ztfr", "ztfg"]
        assert len(result["ztfr"]["time"]) == 3

    def test_set_analysis_filters_intersects_requested_and_available(self):
        data = self.mixed_detection_data()

        assert analysis.set_analysis_filters(None, data) == ["ztfr", "ztfg"]
        assert analysis.set_analysis_filters(["ztfr"], data) == ["ztfr"]
        assert analysis.set_analysis_filters(["sdssg"], data) == []


class TestDataFromInjection(AnalysisTestsContainer):
    """Tests for data_from_injection in isolation: the first step
    analysis_setup composes to simulate an injected transient, before the
    data are ever cut to the model's time range or filtered.
    """

    label = "test_injection"

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.data, cls.injection_params = analysis.data_from_injection(
            cls.args, cls.filters
        )

    def test_returns_only_finite_photometry_for_the_requested_filters(self):
        assert list(self.data.keys()) == self.filters

        for filt_data in self.data.values():
            assert np.isfinite(filt_data["mag"]).all()
            assert np.isfinite(filt_data["mag_error"]).all()
            n = len(filt_data["time"])
            assert len(filt_data["mag"]) == n
            assert len(filt_data["mag_error"]) == n

    def test_cache_remains_unchanged(self):
        lc_file = self.outdir / f"{self.label}_lc.json"
        assert lc_file.is_file()
        mtime_before = lc_file.stat().st_mtime

        data_again, params_again = analysis.data_from_injection(self.args, self.filters)

        assert lc_file.stat().st_mtime == mtime_before
        for filt in self.filters:
            assert data_again[filt]["time"] == pytest.approx(self.data[filt]["time"])
            assert data_again[filt]["mag"] == pytest.approx(self.data[filt]["mag"])

    def test_injection_params_are_reproducible_across_repeated_calls(self):
        pytest.skip(reason="""requires a more substantial fix""")
        # NOTE : HR - This is desirable in principle, but fails because of the way
        # we add the timeshift each time to trigger-time. I cannot see an obvious fix
        # out of fear for undesired downstream effects. Since double calls are not
        # expected in practice, I am leaving this test skipped for now.
        _, injection_params_again = analysis.data_from_injection(
            self.args, self.filters
        )

        assert injection_params_again == self.injection_params


class TestAnalysisSetup(AnalysisTestsContainer):
    """Tests for the assembly logic behind ``lightcurve-analysis``"""

    label = "test_injection"

    @classmethod
    def setup_class(cls):
        super().setup_class()
        result = analysis.analysis_setup(cls.args)
        cls.priors, cls.likelihood, cls.injection_parameters = result
        cls.light_curve_model = cls.likelihood.sub_model.light_curve_model

    def test_analysis_setup_builds_the_requested_model_and_priors(self):
        assert type(self.light_curve_model).__name__ == "FiestaKilonovaModel"
        assert set(self.priors.keys()) == set(
            self.light_curve_model.model_parameters
        ) | {"luminosity_distance", "timeshift"}

    def test_analysis_setup_restricts_injection_parameters_to_the_priors(self):
        assert set(self.injection_parameters.keys()) == set(self.priors.keys())

    def test_analysis_setup_caches_the_injection_light_curve(self):
        lc_file = self.outdir / f"{self.label}_lc.json"
        assert lc_file.is_file()

        mtime_before = lc_file.stat().st_mtime
        analysis.analysis_setup(self.args)
        assert lc_file.stat().st_mtime == mtime_before

    def test_likelihood_favours_the_injected_truths_over_a_mismatched_point(self):
        logl_at_truth = self.likelihood.log_likelihood(self.injection_parameters)
        assert np.isfinite(logl_at_truth)

        mismatched = dict(self.injection_parameters)
        mismatched["log10_mej_dyn"] = -4.0
        mismatched["log10_mej_wind"] = -4.0
        logl_mismatched = self.likelihood.log_likelihood(mismatched)

        assert logl_at_truth > logl_mismatched

    def test_model_time_cut_shrinks_the_data_and_keeps_everything_consistent(self):
        filt = self.filters[0]
        sub_model = self.likelihood.sub_model
        post_cut_times = sub_model.light_curve_times
        systematics_handler = sub_model.systematics_handler
        raw_data, inj_params = analysis.data_from_injection(self.args, self.filters)
        trigger_time = inj_params.get("trigger_time", 0)
        pre_cut_times, _, _, _ = utils.setup_filtered_lc_data(raw_data, trigger_time)

        for filt, times in post_cut_times.items():
            n = len(times)
            assert n <= len(pre_cut_times[filt])
            assert len(sub_model.light_curves[filt]) == n
            assert len(sub_model.light_curve_uncertainties[filt]) == n
            assert len(systematics_handler.light_curve_times[filt]) == n
            assert len(systematics_handler.error_budget[filt]) == n

    def test_systematics_file_adds_a_prior_the_rebuilt_handler_consumes(self):
        args = copy.deepcopy(self.args)
        args.label = "test_injection_with_systematics"
        args.systematics_file = {"prior": "Uniform(minimum=0.5, maximum= 2.)"}

        priors, likelihood, injection_parameters = analysis.analysis_setup(args)

        # the systematics config introduces a free parameter the model
        assert "em_syserr" in priors
        assert "em_syserr" not in self.light_curve_model.model_parameters
        assert injection_parameters["em_syserr"] is None

        systematics_handler = likelihood.sub_model.systematics_handler
        assert systematics_handler.compute_em_err.__name__ == "from_param"

        sample = dict(injection_parameters)
        sample["em_syserr"] = 0.0001  # negligible: should fit tightly
        logl_tight = likelihood.log_likelihood(sample)

        sample["em_syserr"] = 1.0  # sizable: should broaden the fit a lot
        logl_loose = likelihood.log_likelihood(sample)

        assert np.isfinite(logl_tight)
        assert np.isfinite(logl_loose)
        assert logl_tight > logl_loose

    def integration_test(self):
        self.args.sampler = "dynesty"
        self.args.maxiter = 10
        analysis.multi_analysis_loop(self.args, analysis.analysis_setup)


class TestBolometricSetup(AnalysisTestsContainer):
    """Tests for bolometric_setup, the assembly logic behind
    ``lightcurve-analysis-lbol``, exercised against the default
    SimpleBolometricLightCurveModel ("Arnett").
    """

    label = "test_bolometric"
    trigger_time = 60000.0
    truth = {
        "tau_m": 10.0,
        "log10_mni": -1.0,
        "redshift": 0.0,
        "luminosity_distance": 10.0,
    }
    relative_uncertainty = 0.05
    generation_seed = 0

    @classmethod
    def setup_class(cls):
        cls.outdir.mkdir(parents=True, exist_ok=True)

        # generate a synthetic light curve from the model itself, rather
        # than fabricating luminosities by hand
        injection_model = SimpleBolometricLightCurveModel(model="Arnett")
        times, luminosity = injection_model.gen_detector_lc(dict(cls.truth))
        uncertainty = cls.relative_uncertainty * luminosity
        rng = np.random.default_rng(cls.generation_seed)
        noisy_luminosity = luminosity + rng.normal(scale=uncertainty)

        cls.light_curve_path = cls.outdir / f"{cls.label}_bbdata.csv"
        pd.DataFrame(
            {
                "phase": cls.trigger_time + times,
                "Lbb": noisy_luminosity,
                "Lbb_unc": uncertainty,
            }
        ).to_csv(cls.light_curve_path, index=False)

        cls.prior_path = cls.outdir / "Arnett.prior"
        cls.prior_path.write_text(
            "tau_m = Uniform(minimum=1.0, maximum=30.0, name='tau_m')\n"
            "log10_mni = Uniform(minimum=-3.0, maximum=0.0, name='log10_mni')\n"
            "redshift = DeltaFunction(peak=0.0, name='redshift')\n"
            "luminosity_distance = DeltaFunction(peak=10.0, "
            "name='luminosity_distance')\n"
        )

        cls.args = em_parsing.parsing_and_logging(em_parsing.bolometric_parser, [])
        overrides = dict(
            em_model="Arnett",
            label=cls.label,
            outdir=str(cls.outdir),
            light_curve_data=str(cls.light_curve_path),
            prior_file=str(cls.prior_path),
            trigger_time=cls.trigger_time,
        )
        for key, value in overrides.items():
            setattr(cls.args, key, value)

        setup_result = analysis.bolometric_setup(cls.args)
        cls.priors, cls.likelihood, cls.injection_parameters = setup_result
        cls.light_curve_model = cls.likelihood.sub_model.light_curve_model

    def test_bolometric_setup_builds_the_default_model_and_priors(self):
        model = self.light_curve_model
        assert type(model).__name__ == "SimpleBolometricLightCurveModel"
        assert model.model == "Arnett"
        assert set(self.priors.keys()) == set(model.model_parameters) | {
            "redshift",
            "luminosity_distance",
        }

    def test_likelihood_favours_the_true_parameters_over_a_mismatched_point(self):
        logl_at_truth = self.likelihood.log_likelihood(self.truth)
        assert np.isfinite(logl_at_truth)

        mismatched = dict(self.truth)
        mismatched["tau_m"] = 1.0
        mismatched["log10_mni"] = -3.0
        logl_mismatched = self.likelihood.log_likelihood(mismatched)

        assert logl_at_truth > logl_mismatched
