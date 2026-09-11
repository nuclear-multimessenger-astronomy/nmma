import os
import unittest
from pathlib import Path
from unittest.mock import patch

import bilby
import numpy as np

from nmma.core.conversion import get_redshift
from nmma.em import lightcurve_generation as lc_gen
from nmma.em import model

NMMA_FIESTA_SURROGATES = os.environ.get("NMMA_FIESTA_SURROGATES")


class TestLightCurveModelContainer(unittest.TestCase):
    """Unit tests for the base class, using a lightweight analytic model
    ("Me2017") to instantiate it directly."""

    def setUp(self):
        self.filters = ["sdssg", "sdssr"]
        self.container = model.LightCurveModelContainer(
            "Me2017", filters=self.filters
        )

    def tearDown(self):
        del self.container
        del self.filters

    def test_init_sets_expected_attributes(self):
        self.assertEqual(self.container.model, "Me2017")
        self.assertEqual(self.container.filters, self.filters)
        self.assertEqual(
            self.container.model_parameters,
            model.model_parameters_dict["Me2017"],
        )
        self.assertTrue(self.container.good_parameters)
        self.assertIsNone(self.container.extinction_model)
        self.assertIsNone(self.container.extinction_frame)
        self.assertEqual(len(self.container.lambdas), len(self.filters))
        self.assertEqual(len(self.container.nu_0s), len(self.filters))

    def test_init_splits_comma_separated_filter_string(self):
        container = model.LightCurveModelContainer(
            "Me2017", filters="sdssg,sdssr"
        )
        self.assertEqual(container.filters, ["sdssg", "sdssr"])

    def test_init_with_no_filters_uses_all_available_filters(self):
        container = model.LightCurveModelContainer("Me2017", filters=None)
        self.assertIsNone(container.filters)
        self.assertGreater(len(container.default_filts), 0)
        self.assertGreater(len(container.lambdas), 0)

    def test_repr(self):
        self.assertEqual(repr(self.container), "LightCurveModelContainer(model=Me2017)")

    def test_identify_model_parameters_uses_explicit_list(self):
        container = model.LightCurveModelContainer(
            "Me2017", filters=self.filters, model_parameters=["a", "b"]
        )
        self.assertEqual(container.model_parameters, ["a", "b"])

    def test_identify_model_parameters_raises_for_unknown_model(self):
        with self.assertRaises(AssertionError):
            model.LightCurveModelContainer("NotAModel", filters=self.filters)

    def test_setup_model_times_default_range(self):
        times = self.container.setup_model_times()
        self.assertAlmostEqual(times[0], 0.01)
        self.assertAlmostEqual(times[-1], 14.0)
        self.assertEqual(len(times), 150)

    def test_sanity_checks_defaults_to_good_parameters(self):
        self.container.good_parameters = False
        self.container.sanity_checks({})
        self.assertTrue(self.container.good_parameters)

    def test_parameter_conversion_fills_missing_log10_parameters(self):
        parameters = dict(mej=0.01, vej=0.1, beta=3.0, kappa_r=1.0)
        new_parameters = self.container.parameter_conversion(parameters)
        self.assertAlmostEqual(new_parameters["log10_mej"], np.log10(0.01))
        self.assertAlmostEqual(new_parameters["log10_vej"], np.log10(0.1))
        self.assertAlmostEqual(new_parameters["log10_kappa_r"], np.log10(1.0))

    def test_parameter_conversion_fills_missing_linear_parameters(self):
        # HoNa2020 has some model parameters that are not log10-prefixed
        # (e.g. "vej_max"), so it exercises the branch of parameter_conversion
        # that derives a linear parameter from its log10 counterpart.
        container = model.LightCurveModelContainer("HoNa2020", filters=self.filters)
        parameters = dict(
            log10_mej=-2.0,
            log10_vej_max=-1.0,
            log10_vej_min=-2.0,
            log10_vej_frac=-0.5,
            log10_kappa_low_vej=0.0,
            log10_kappa_high_vej=0.5,
        )
        new_parameters = container.parameter_conversion(parameters)
        self.assertAlmostEqual(new_parameters["vej_max"], 10 ** -1.0)
        self.assertAlmostEqual(new_parameters["vej_min"], 10 ** -2.0)
        self.assertAlmostEqual(new_parameters["vej_frac"], 10 ** -0.5)

    def test_em_parameter_setup_and_combine_lc_params(self):
        parameters = dict(
            log10_mej=-2.0,
            log10_vej=-1.0,
            beta=3.0,
            log10_kappa_r=0.0,
            luminosity_distance=40.0,
        )
        combined = self.container.em_parameter_setup(dict(parameters))
        self.assertEqual(combined, {k: parameters[k] for k in self.container.model_parameters})
        self.assertAlmostEqual(self.container.luminosity_distance, 40.0)
        self.assertAlmostEqual(self.container.distmod, 5.0 * np.log10(40.0 * 1e6 / 10.0))
        self.assertAlmostEqual(self.container.timeshift, 0.0)

    def test_em_parameter_setup_default_luminosity_distance(self):
        self.container.em_parameter_setup(
            dict(log10_mej=-2.0, log10_vej=-1.0, beta=3.0, log10_kappa_r=0.0)
        )
        self.assertAlmostEqual(self.container.luminosity_distance, 1e-5)

    def test_check_vs_priors_switches_to_interpolated_redshift(self):
        default_redshift_func = self.container.redshift_func
        priors = bilby.core.prior.PriorDict()
        priors["luminosity_distance"] = bilby.core.prior.Uniform(
            10, 100, "luminosity_distance"
        )
        self.container.check_vs_priors(priors)
        self.assertIsNot(self.container.redshift_func, default_redshift_func)

        approx = float(self.container.redshift_func(dict(luminosity_distance=40.0)))
        exact = float(get_redshift(dict(luminosity_distance=40.0)))
        self.assertAlmostEqual(approx, exact, delta=1e-3 * exact)

    def test_check_vs_priors_keeps_default_redshift_func_with_redshift_prior(self):
        default_redshift_func = self.container.redshift_func
        priors = bilby.core.prior.PriorDict()
        priors["redshift"] = bilby.core.prior.Uniform(0, 1, "redshift")
        self.container.check_vs_priors(priors)
        self.assertIs(self.container.redshift_func, default_redshift_func)

    def test_check_vs_priors_sets_up_extinction(self):
        priors = bilby.core.prior.PriorDict()
        priors["Ebv"] = bilby.core.prior.Uniform(0, 1, "Ebv")
        self.container.check_vs_priors(priors)
        self.assertEqual(self.container.extinction_frame, "rest")
        self.assertIsNotNone(self.container.extinction_model)
        self.assertEqual(
            self.container.extinction_wavenumbers, self.container.rest_wavenumbers
        )

    def test_rest_and_obs_wavenumbers(self):
        self.container.wavenumbers = np.array([1.0, 2.0])
        self.container.redshift = 1.0
        np.testing.assert_allclose(self.container.obs_wavenumbers(), [1.0, 2.0])
        np.testing.assert_allclose(self.container.rest_wavenumbers(), [2.0, 4.0])

    def test_extinction_correction_applies_positive_offset_within_range(self):
        priors = bilby.core.prior.PriorDict()
        priors["Ebv"] = bilby.core.prior.Uniform(0, 1, "Ebv")
        self.container.check_vs_priors(priors)
        self.container.redshift = 0.0
        self.container.Ebv = 0.5

        original_mags = {filt: np.array([20.0, 21.0]) for filt in self.container.default_filts}
        model_mags = {filt: mags.copy() for filt, mags in original_mags.items()}
        corrected = self.container.extinction_correction(model_mags)
        for filt in self.container.default_filts:
            np.testing.assert_array_less(original_mags[filt], corrected[filt])

    def test_generate_lightcurve_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.container.generate_lightcurve(self.container.model_times, {})

    def test_combine_detector_data_applies_distance_modulus_and_redshift(self):
        self.container.redshift = 0.0
        self.container.distmod = 5.0
        self.container.extinction_frame = None
        observable_times = np.array([1.0, 2.0, 3.0])
        model_lc = {"sdssg": np.array([1.0, np.nan, 3.0])}

        times, lc_data = self.container.combine_detector_data(model_lc, observable_times)
        np.testing.assert_allclose(times, observable_times)
        np.testing.assert_allclose(lc_data["sdssg"], [6.0, np.nan, 8.0])

    def test_combine_detector_data_all_nonfinite_returns_inf(self):
        self.container.redshift = 0.0
        self.container.distmod = 5.0
        self.container.extinction_frame = None
        observable_times = np.array([1.0, 2.0, 3.0])
        model_lc = {"sdssg": np.array([np.nan, np.nan, np.nan])}

        _, lc_data = self.container.combine_detector_data(model_lc, observable_times)
        self.assertTrue(np.all(np.isinf(lc_data["sdssg"])))

    def test_citation_property(self):
        self.assertEqual(
            self.container.citation, {"Me2017": model.citation_dict["Me2017"]}
        )


def _require_fiesta_surrogates():
    """fiesta-backed tests need fiesta itself and a real checkout of the
    fiesta-surrogates HuggingFace repo, pointed to by $NMMA_FIESTA_SURROGATES
    (CI downloads this before running the test suite; see the 'Download
    fiesta-surrogates from HuggingFace' step in
    .github/workflows/continous_integration.yml, which sets this variable
    only after a successful download -- so in CI it is always set, and a
    missing/broken directory here means the download itself is broken and
    must fail loudly, not be skipped over).

    The only legitimate skip is $NMMA_FIESTA_SURROGATES being unset entirely,
    which happens when a developer hasn't opted into downloading the ~280MB
    of surrogate data locally."""
    try:
        import fiesta  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("fiesta not installed; surrogate pipeline untested")
    if not NMMA_FIESTA_SURROGATES:
        raise unittest.SkipTest(
            "NMMA_FIESTA_SURROGATES not set; skipping fiesta surrogate tests locally. "
            "Set it to a fiesta-surrogates checkout (e.g. `hf download "
            "nuclear-multimessenger-astronomy/fiesta-surrogates --repo-type model "
            "--local-dir fiesta-surrogates`) to run them."
        )
    if not Path(NMMA_FIESTA_SURROGATES).is_dir():
        raise AssertionError(
            f"NMMA_FIESTA_SURROGATES={NMMA_FIESTA_SURROGATES!r} is set but is not a "
            "directory -- the fiesta-surrogates download must have failed."
        )


def _load_fiesta_kilonova_model(filters):
    last_err = None
    for model_name in ("Bu2026_MLP", "Bu2025_MLP"):
        try:
            return model.FiestaKilonovaModel(
                model=model_name, filters=filters, surrogate_dir=NMMA_FIESTA_SURROGATES
            )
        except OSError as e:
            last_err = e
    raise AssertionError(
        f"None of the expected kilonova surrogates (Bu2026_MLP, Bu2025_MLP) were "
        f"found under NMMA_FIESTA_SURROGATES={NMMA_FIESTA_SURROGATES!r}: {last_err}. "
        "The fiesta-surrogates download must be incomplete or corrupted."
    )


def _load_fiesta_grb_model(filters):
    try:
        return model.FiestaGRBModel(
            model="blastwave_gaussian_CVAE",
            filters=filters,
            surrogate_dir=NMMA_FIESTA_SURROGATES,
        )
    except OSError as e:
        raise AssertionError(
            f"Expected GRB surrogate 'blastwave_gaussian_CVAE' not found under "
            f"NMMA_FIESTA_SURROGATES={NMMA_FIESTA_SURROGATES!r}: {e}. The "
            "fiesta-surrogates download must be incomplete or corrupted."
        )


class TestFiestaKilonovaModelDownloadSmoke(unittest.TestCase):
    """Smoke test for the fiesta-surrogates pipeline: instantiate
    FiestaKilonovaModel against a locally-available copy of the
    fiesta-surrogates HuggingFace repo. Carried over from the retired
    nmma/tests/test_em/fiesta_smoke.py."""

    @classmethod
    def setUpClass(cls):
        _require_fiesta_surrogates()

    def test_fiesta_kilonova_loads(self):
        kn_model = _load_fiesta_kilonova_model(filters=None)
        self.assertTrue(kn_model.model_parameters, "fiesta model did not expose any parameters")
        self.assertTrue(kn_model.filters, "fiesta model did not advertise any filters")


class TestFiestaModel(unittest.TestCase):
    """Exercises FiestaModel's own orchestration logic (parameter
    bookkeeping, prior checks, detector/source-frame conversion) using the
    real Bu2026_MLP/Bu2025_MLP surrogate loaded by FiestaKilonovaModel, since
    FiestaModel is not meant to be driven with a fake surrogate stand-in."""

    @classmethod
    def setUpClass(cls):
        _require_fiesta_surrogates()
        cls.kn_model = _load_fiesta_kilonova_model(filters=["sdssg"])
        cls.surrogate = cls.kn_model.fiesta_model
        cls.bounds = cls.surrogate.parameter_distributions
        cls.mid_parameters = {key: 0.5 * (lo + hi) for key, (lo, hi, *_) in cls.bounds.items()}

    def test_init_uses_fiesta_model_filters_and_parameters_by_default(self):
        lc_model = model.FiestaModel(self.surrogate, filters=None)
        self.assertEqual(lc_model.filters, self.surrogate.filters)
        self.assertEqual(lc_model.model_parameters, self.surrogate.parameter_names)
        self.assertEqual(lc_model.model, self.surrogate.name)

    def test_setup_model_times_returns_fiesta_model_times(self):
        lc_model = model.FiestaModel(self.surrogate, filters=None)
        np.testing.assert_allclose(lc_model.setup_model_times(), self.surrogate.times)

    def test_check_vs_priors_accepts_prior_within_bounds(self):
        lc_model = model.FiestaModel(self.surrogate, filters=None)
        key = next(iter(self.bounds))
        lo, hi = self.bounds[key][0], self.bounds[key][1]
        margin = 0.1 * (hi - lo)
        priors = bilby.core.prior.PriorDict()
        priors[key] = bilby.core.prior.Uniform(lo + margin, hi - margin, key)
        lc_model.check_vs_priors(priors)  # should not raise

    def test_check_vs_priors_rejects_prior_outside_bounds(self):
        lc_model = model.FiestaModel(self.surrogate, filters=None)
        key = next(iter(self.bounds))
        lo, hi = self.bounds[key][0], self.bounds[key][1]
        priors = bilby.core.prior.PriorDict()
        priors[key] = bilby.core.prior.Uniform(lo - 1.0, hi, key)
        with self.assertRaises(ValueError):
            lc_model.check_vs_priors(priors)

    def test_combine_lc_params_augments_parameters_with_container_state(self):
        lc_model = model.FiestaModel(self.surrogate, filters=None)
        lc_model.redshift = 0.01
        lc_model.luminosity_distance = 40.0
        lc_model.timeshift = 0.0
        key = next(iter(self.mid_parameters))
        combined = lc_model.combine_lc_params({key: self.mid_parameters[key]})
        self.assertEqual(
            combined,
            {
                key: self.mid_parameters[key],
                "redshift": 0.01,
                "luminosity_distance": 40.0,
                "timeshift": 0.0,
            },
        )

    def test_gen_detector_lc_returns_finite_magnitudes_when_good_parameters(self):
        lc_model = model.FiestaModel(self.surrogate, filters=None)
        parameters = dict(self.mid_parameters, luminosity_distance=40.0)
        times, mag = lc_model.gen_detector_lc(parameters)
        self.assertTrue(lc_model.good_parameters)
        np.testing.assert_allclose(times[0], self.surrogate.times[0], rtol=1e-2)
        self.assertIn("sdssg", mag)
        self.assertTrue(np.isfinite(mag["sdssg"]).any())

    def test_gen_detector_lc_skips_predict_when_parameters_are_bad(self):
        lc_model = model.FiestaModel(self.surrogate, filters=None)
        lc_model.good_parameters = False
        parameters = dict(self.mid_parameters, luminosity_distance=40.0)
        times, mag = lc_model.gen_detector_lc(parameters)
        np.testing.assert_allclose(times, self.surrogate.times)
        self.assertEqual(mag, {})

    def test_generate_lightcurve_returns_finite_source_frame_magnitudes(self):
        lc_model = model.FiestaModel(self.surrogate, filters=None)
        parameters = dict(self.mid_parameters, luminosity_distance=40.0)
        abs_mags = lc_model.generate_lightcurve(self.surrogate.times, parameters)
        self.assertTrue(np.isfinite(abs_mags["sdssg"]).any())


class TestFiestaKilonovaModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _require_fiesta_surrogates()

    def test_default_filters_are_used_when_none_given(self):
        kn_model = _load_fiesta_kilonova_model(filters=None)
        expected_default_filters = [
            "sdssg", "sdssr", "sdssi", "sdssz", "ztfg", "ztfr", "ztfi",
            "2massj", "2massh", "2massks",
        ]
        self.assertEqual(kn_model.filters, expected_default_filters)

    def test_explicit_filters_are_passed_through(self):
        kn_model = _load_fiesta_kilonova_model(filters=["sdssg", "sdssr"])
        self.assertEqual(kn_model.filters, ["sdssg", "sdssr"])

    def test_model_parameters_come_from_surrogate(self):
        kn_model = _load_fiesta_kilonova_model(filters=["sdssg"])
        self.assertEqual(kn_model.model_parameters, kn_model.fiesta_model.parameter_names)

    def test_loads_directly_from_the_model_subdirectory_too(self):
        # Besides the repo root (which triggers the OSError fallback to
        # "<surrogate_dir>/KN/<model>/model"), FiestaKilonovaModel should
        # also load when pointed straight at that model subdirectory.
        assert NMMA_FIESTA_SURROGATES is not None  # guaranteed by setUpClass
        model_dir = Path(NMMA_FIESTA_SURROGATES) / "KN" / "Bu2026_MLP" / "model"
        self.assertTrue(
            model_dir.is_dir(),
            f"{model_dir} not found -- the fiesta-surrogates download must be incomplete.",
        )
        kn_model = model.FiestaKilonovaModel(
            model="Bu2026_MLP", filters=["sdssg"], surrogate_dir=model_dir
        )
        self.assertEqual(kn_model.model, "Bu2026_MLP")


class TestFiestaGRBModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _require_fiesta_surrogates()

    def test_construction_without_explicit_filters_currently_raises(self):
        # Unlike FiestaKilonovaModel, FiestaGRBModel does not fall back to a
        # default filter list when filters=None: it forwards None straight
        # to fiesta's AfterglowFlux.load_filters, which does `for filter in
        # filters` and blows up with a TypeError. This test documents that
        # known gap rather than asserting it is correct.
        with self.assertRaises(TypeError):
            _load_fiesta_grb_model(filters=None)

    def test_construction_with_explicit_filters(self):
        grb_model = _load_fiesta_grb_model(filters=["sdssg"])
        self.assertEqual(grb_model.filters, ["sdssg"])
        self.assertEqual(grb_model.model_parameters, grb_model.fiesta_model.parameter_names)
        self.assertEqual(grb_model.resolution, 12)
        self.assertIsInstance(grb_model, model.GRBMixin)

    def test_gen_detector_lc_returns_finite_magnitudes(self):
        grb_model = _load_fiesta_grb_model(filters=["sdssg"])
        bounds = grb_model.fiesta_model.parameter_distributions
        parameters = {key: 0.5 * (lo + hi) for key, (lo, hi, *_) in bounds.items()}
        parameters["luminosity_distance"] = 40.0
        _, mag = grb_model.gen_detector_lc(parameters)
        self.assertTrue(grb_model.good_parameters)
        self.assertTrue(np.isfinite(mag["sdssg"]).any())


class FakeSVDMagModel(dict):
    """A stand-in for the dict-of-filter-metadata that SVDLightCurveModel
    loads from a joblib file, so its own dispatch/bookkeeping logic can be
    unit tested without a real (trained) SVD model."""


class TestSVDLightCurveModel(unittest.TestCase):
    """SVDLightCurveModel.__init__ downloads and loads real trained models
    from GitLab; the project retired testing that download path directly
    (see the retired nmma/tests/test_core/models.py), so these tests build
    instances via __new__ and exercise each method's own logic in isolation,
    mocking only the external GitLab/joblib/lc_gen calls."""

    def test_get_model_data_defaults_filters_from_model_metadata(self):
        svd_model = object.__new__(model.SVDLightCurveModel)
        svd_model.svd_path = Path("/fake/svd/path")
        svd_model.model_specifier = ""

        with patch.object(
            model, "get_model", return_value=(None, ["sdss_g", "sdss_r"])
        ) as mock_get_model:
            filters = svd_model.get_model_data("Ka2017", None)

        self.assertEqual(filters, ["sdss:g", "sdss:r"])
        mock_get_model.assert_called_once_with(
            svd_model.svd_path, "Ka2017", filters=None
        )

    def test_get_model_data_translates_explicit_filters_for_gitlab(self):
        svd_model = object.__new__(model.SVDLightCurveModel)
        svd_model.svd_path = Path("/fake/svd/path")
        svd_model.model_specifier = "_tf"

        with patch.object(
            model, "get_model", return_value=(None, ["unused"])
        ) as mock_get_model:
            filters = svd_model.get_model_data("Ka2017", "sdss:g,sdss:r")

        self.assertEqual(filters, ["sdss:g", "sdss:r"])
        mock_get_model.assert_called_once_with(
            svd_model.svd_path, "Ka2017_tf", filters=["sdss_g", "sdss_r"]
        )

    def test_load_filt_model_loads_available_filters_and_warns_on_missing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            outdir = Path(tmp_dir) / "Ka2017"
            outdir.mkdir()
            (outdir / "sdss_g.joblib").touch()

            svd_model = object.__new__(model.SVDLightCurveModel)
            svd_model.svd_path = Path(tmp_dir)
            svd_model.model_specifier = ""
            svd_model.filters = ["sdss:g", "sdss:r"]
            svd_model.svd_mag_model = {"sdss:g": {}, "sdss:r": {}}

            loaded = {}

            def fake_load(path):
                loaded[path.name] = "loaded"
                return "loaded"

            svd_model.load_filt_model("Ka2017", fake_load, fn_ext="joblib", target_name="gps")

        self.assertEqual(svd_model.svd_mag_model["sdss:g"]["gps"], "loaded")
        self.assertNotIn("gps", svd_model.svd_mag_model["sdss:r"])
        self.assertEqual(list(loaded), ["sdss_g.joblib"])

    def test_load_filt_model_raises_when_no_filter_file_is_found(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "Ka2017").mkdir()

            svd_model = object.__new__(model.SVDLightCurveModel)
            svd_model.svd_path = Path(tmp_dir)
            svd_model.model_specifier = ""
            svd_model.filters = ["sdss:i"]
            svd_model.svd_mag_model = {"sdss:i": {}}

            with self.assertRaises(ValueError):
                svd_model.load_filt_model("Ka2017", lambda path: None, fn_ext="joblib")

    def test_combine_lc_params_returns_ordered_list(self):
        svd_model = object.__new__(model.SVDLightCurveModel)
        svd_model.model_parameters = ["log10_mej", "log10_vej"]
        svd_model.log10_vej = -1.0

        result = svd_model.combine_lc_params({"log10_mej": -2.0})
        self.assertEqual(result, [-2.0, -1.0])

    def _make_svd_model_for_generate_lightcurve(self):
        svd_model = object.__new__(model.SVDLightCurveModel)
        svd_model.model = "Ka2017"
        svd_model.model_parameters = ["log10_mej", "log10_vej"]
        svd_model.filters = ["sdssg", "sdssr"]
        svd_model.svd_mag_model = {"fake": True}
        svd_model.svd_lbol_model = None
        svd_model.mag_ncoeff = 10
        svd_model.lbol_ncoeff = 5
        svd_model.redshift_func = get_redshift
        return svd_model

    def test_generate_lightcurve_dispatches_to_calc_svd_lc_with_all_filters(self):
        svd_model = self._make_svd_model_for_generate_lightcurve()
        sample_times = np.array([1.0, 2.0])
        parameters = dict(log10_mej=-2.0, log10_vej=-1.0, luminosity_distance=40.0)

        with patch.object(lc_gen, "calc_svd_lc", return_value="LC_RESULT") as mock_lc:
            result = svd_model.generate_lightcurve(sample_times, dict(parameters))

        self.assertEqual(result, "LC_RESULT")
        args, kwargs = mock_lc.call_args
        np.testing.assert_allclose(args[0], sample_times)
        self.assertEqual(args[1], [-2.0, -1.0])
        self.assertEqual(kwargs["filters"], ["sdssg", "sdssr"])

    def test_generate_lightcurve_dispatches_to_calc_svd_lc_with_explicit_filters(self):
        svd_model = self._make_svd_model_for_generate_lightcurve()
        sample_times = np.array([1.0, 2.0])
        parameters = dict(log10_mej=-2.0, log10_vej=-1.0, luminosity_distance=40.0)

        with patch.object(lc_gen, "calc_svd_lc", return_value="LC_RESULT") as mock_lc:
            svd_model.generate_lightcurve(sample_times, dict(parameters), filters=["sdssg"])

        self.assertEqual(mock_lc.call_args.kwargs["filters"], ["sdssg"])

    def test_generate_lightcurve_dispatches_to_calc_svd_lbol_when_filters_none(self):
        svd_model = self._make_svd_model_for_generate_lightcurve()
        sample_times = np.array([1.0, 2.0])
        parameters = dict(log10_mej=-2.0, log10_vej=-1.0, luminosity_distance=40.0)

        with patch.object(lc_gen, "calc_svd_lbol", return_value="LBOL_RESULT") as mock_lbol:
            result = svd_model.generate_lightcurve(sample_times, dict(parameters), filters=None)

        self.assertEqual(result, "LBOL_RESULT")
        mock_lbol.assert_called_once()
        self.assertEqual(mock_lbol.call_args.kwargs["svd_lbol_model"], None)
        self.assertEqual(mock_lbol.call_args.kwargs["lbol_ncoeff"], 5)

    def test_generate_spectra_delegates_to_generate_lightcurve_with_wavelengths_as_filters(self):
        svd_model = self._make_svd_model_for_generate_lightcurve()
        sample_times = np.array([1.0, 2.0])
        wavelengths = [500.0, 600.0]
        parameters = dict(log10_mej=-2.0, log10_vej=-1.0)

        with patch.object(
            model.SVDLightCurveModel, "generate_lightcurve", return_value="SPECTRA"
        ) as mock_generate:
            result = svd_model.generate_spectra(sample_times, wavelengths, parameters)

        self.assertEqual(result, "SPECTRA")
        mock_generate.assert_called_once_with(sample_times, parameters, filters=wavelengths)

    def test_repr_includes_model_name_and_svd_path(self):
        svd_model = object.__new__(model.SVDLightCurveModel)
        svd_model.model = "Ka2017"
        svd_model.svd_path = Path("/some/path")
        self.assertEqual(
            repr(svd_model), "SVDLightCurveModel(model=Ka2017)(model=Ka2017, svd_path=/some/path)"
        )


class TestSimpleBolometricLightCurveModel(unittest.TestCase):
    def setUp(self):
        self.parameters = dict(
            tau_m=5.0, log10_mni=-1.5, luminosity_distance=40.0
        )

    def test_arnett_model_selects_arnett_lc_function(self):
        lc_model = model.SimpleBolometricLightCurveModel(model="Arnett", filters=["sdssg"])
        from nmma.em import lightcurve_generation as lc_gen

        self.assertIs(lc_model.lc_func, lc_gen.arnett_lc)

    def test_arnett_modified_model_selects_arnett_modified_lc_function(self):
        lc_model = model.SimpleBolometricLightCurveModel(
            model="Arnett_modified", filters=["sdssg"]
        )
        from nmma.em import lightcurve_generation as lc_gen

        self.assertIs(lc_model.lc_func, lc_gen.arnett_modified_lc)

    def test_setup_model_times_range(self):
        lc_model = model.SimpleBolometricLightCurveModel(model="Arnett", filters=["sdssg"])
        times = lc_model.setup_model_times()
        self.assertAlmostEqual(times[0], 0.005)
        self.assertAlmostEqual(times[-1], 20.0)
        self.assertEqual(len(times), 40)

    def test_combine_detector_data_uses_luminosity_redshift_correction(self):
        lc_model = model.SimpleBolometricLightCurveModel(model="Arnett", filters=["sdssg"])
        lc_model.redshift = 1.0
        _, lbol = lc_model.combine_detector_data(np.array([8.0]), np.array([1.0]))
        np.testing.assert_allclose(lbol, [2.0])

    def test_generate_lightcurve_returns_finite_bolometric_luminosity(self):
        lc_model = model.SimpleBolometricLightCurveModel(model="Arnett", filters=["sdssg"])
        _, lbol = lc_model.gen_detector_lc(self.parameters)
        self.assertTrue(np.all(np.isfinite(lbol)))
        self.assertTrue(np.all(np.array(lbol) > 0))


class TestSimpleKilonovaLightCurveModel(unittest.TestCase):
    def test_lc_func_selected_from_lc_dict(self):
        for model_name in ["Me2017", "PL_BB_fixedT", "blackbody_fixedT", "synchrotron_powerlaw"]:
            lc_model = model.SimpleKilonovaLightCurveModel(
                model=model_name, filters=["sdssg"]
            )
            self.assertIs(lc_model.lc_func, model.SimpleKilonovaLightCurveModel.lc_dict[model_name])

    def test_me2017_model_times_are_trimmed_above_0_05_days(self):
        lc_model = model.SimpleKilonovaLightCurveModel(model="Me2017", filters=["sdssg"])
        self.assertGreaterEqual(np.min(lc_model.model_times), 5e-2)

    def test_pl_bb_fixedt_model_times_are_not_trimmed(self):
        lc_model = model.SimpleKilonovaLightCurveModel(
            model="PL_BB_fixedT", filters=["sdssg"]
        )
        self.assertAlmostEqual(np.min(lc_model.model_times), 0.01)

    def test_generate_lightcurve_returns_finite_magnitudes(self):
        lc_model = model.SimpleKilonovaLightCurveModel(model="Me2017", filters=["sdssg"])
        parameters = dict(
            log10_mej=-2.0,
            log10_vej=-1.0,
            beta=3.0,
            log10_kappa_r=0.0,
            luminosity_distance=40.0,
        )
        _, lc_data = lc_model.gen_detector_lc(parameters)
        self.assertIn("sdssg", lc_data)
        self.assertTrue(np.isfinite(lc_data["sdssg"]).any())


class TestGRBLightCurveModel(unittest.TestCase):
    """Also covers the GRBMixin logic, since GRBMixin is only ever used
    together with a LightCurveModelContainer subclass."""

    def setUp(self):
        self.lc_model = model.GRBLightCurveModel(filters=["sdssg"])
        self.parameters = dict(
            inclination_EM=0.1,
            thetaCore=0.1,
            thetaWing=0.3,
            log10_E0=52.0,
            b=1.0,
            L0=0.0,
            q=0.0,
            ts=1.0,
            log10_n0=-3.0,
            p=2.2,
            log10_epsilon_e=-1.0,
            log10_epsilon_B=-3.0,
            xi_N=1.0,
            d_L=100.0,
        )

    def tearDown(self):
        del self.lc_model
        del self.parameters

    def test_default_parameters(self):
        self.assertEqual(self.lc_model.default_parameters["jetType"], 0)
        self.assertEqual(self.lc_model.default_parameters["specType"], 0)
        self.assertAlmostEqual(self.lc_model.default_parameters["xi_N"], 1.0)
        self.assertAlmostEqual(self.lc_model.default_parameters["d_L"], 3.086e19)

    def test_setup_model_times_range(self):
        times = self.lc_model.setup_model_times()
        self.assertAlmostEqual(times[0], 1.0e-5)
        self.assertAlmostEqual(times[-1], 200.0)
        self.assertEqual(len(times), 201)

    def test_em_parameter_setup_maps_log_sampling_and_viewing_angle(self):
        grb_param_dict = self.lc_model.em_parameter_setup(dict(self.parameters))
        self.assertAlmostEqual(grb_param_dict["E0"], 10 ** 52.0)
        self.assertAlmostEqual(grb_param_dict["n0"], 10 ** -3.0)
        self.assertAlmostEqual(grb_param_dict["epsilon_e"], 10 ** -1.0)
        self.assertAlmostEqual(grb_param_dict["epsilon_B"], 10 ** -3.0)
        self.assertAlmostEqual(grb_param_dict["thetaObs"], self.parameters["inclination_EM"])
        self.assertAlmostEqual(grb_param_dict["d_L"], self.parameters["d_L"])

    def test_em_parameter_setup_prefers_linear_over_log_sampling_keys(self):
        parameters = dict(self.parameters)
        parameters["E0"] = 5e51
        grb_param_dict = self.lc_model.em_parameter_setup(parameters)
        self.assertAlmostEqual(grb_param_dict["E0"], 5e51)

    def test_parameter_conversion_from_alpha_wing(self):
        parameters = dict(
            alphaWing=2.0, thetaCore=0.1, epsilon_e=0.05, epsilon_B=0.01
        )
        new_parameters = self.lc_model.parameter_conversion(dict(parameters))
        self.assertAlmostEqual(new_parameters["thetaWing"], 0.2)
        self.assertAlmostEqual(self.lc_model.resolution, 2.0, places=6)
        self.assertAlmostEqual(new_parameters["epsilon_tot"], 0.06)

    def test_parameter_conversion_epsilon_tot_from_log10_values(self):
        parameters = dict(
            thetaWing=0.3, thetaCore=0.1, log10_epsilon_e=-1.0, log10_epsilon_B=-3.0
        )
        new_parameters = self.lc_model.parameter_conversion(dict(parameters))
        self.assertAlmostEqual(
            new_parameters["epsilon_tot"], 10 ** -1.0 + 10 ** -3.0
        )

    def test_sanity_checks_good_parameters(self):
        self.lc_model.sanity_checks(
            dict(thetaWing=0.3, thetaCore=0.1, epsilon_tot=0.5)
        )
        self.assertTrue(self.lc_model.good_parameters)

    def test_sanity_checks_rejects_wide_opening_angle(self):
        self.lc_model.sanity_checks(
            dict(thetaWing=2.0, thetaCore=0.1, epsilon_tot=0.5)
        )
        self.assertFalse(self.lc_model.good_parameters)

    def test_sanity_checks_rejects_excess_efficiency(self):
        self.lc_model.sanity_checks(
            dict(thetaWing=0.3, thetaCore=0.1, epsilon_tot=1.5)
        )
        self.assertFalse(self.lc_model.good_parameters)

    def test_sanity_checks_rejects_resolution_violation(self):
        self.lc_model.resolution = 1.0
        self.lc_model.sanity_checks(
            dict(thetaWing=0.5, thetaCore=0.1, epsilon_tot=0.5)
        )
        self.assertFalse(self.lc_model.good_parameters)


class TestHostGalaxyLightCurveModel(unittest.TestCase):
    def test_host_mag_array_is_kept_as_is(self):
        host_mag = np.array([21.0, 23.0])
        lc_model = model.HostGalaxyLightCurveModel(
            filters=["sdssg", "sdssr"], host_mag=host_mag
        )
        np.testing.assert_allclose(lc_model.host_mag, host_mag)

    def test_scalar_host_mag_is_broadcast_per_filter(self):
        lc_model = model.HostGalaxyLightCurveModel(
            filters=["sdssg", "sdssr"], host_mag=22.0
        )
        self.assertEqual(len(lc_model.host_mag), 2)

    def test_check_vs_priors_rejects_ebv(self):
        lc_model = model.HostGalaxyLightCurveModel(filters=["sdssg"])
        priors = bilby.core.prior.PriorDict()
        priors["Ebv"] = bilby.core.prior.Uniform(0, 1, "Ebv")
        with self.assertRaises(ValueError):
            lc_model.check_vs_priors(priors)


class TestShockCoolingLightCurveModel(unittest.TestCase):
    def test_model_parameters(self):
        lc_model = model.ShockCoolingLightCurveModel(filters=["sdssg"])
        self.assertEqual(
            lc_model.model_parameters, model.model_parameters_dict["Piro2021"]
        )

    def test_setup_model_times_range(self):
        lc_model = model.ShockCoolingLightCurveModel(filters=["sdssg"])
        times = lc_model.setup_model_times()
        self.assertAlmostEqual(times[0], 1.0 / 24.0)
        self.assertAlmostEqual(times[-1], 3.5)
        self.assertEqual(len(times), 100)


class TestSupernovaLightCurveModel(unittest.TestCase):
    def setUp(self):
        self.lc_model = model.SupernovaLightCurveModel(
            model="nugent-hyper", filters=["sdssg"]
        )

    def tearDown(self):
        del self.lc_model

    def test_identify_model_parameters_rejects_explicit_list(self):
        with self.assertRaises(ValueError):
            model.SupernovaLightCurveModel(
                model="nugent-hyper", filters=["sdssg"], model_parameters=["a"]
            )

    def test_model_parameters_come_from_sncosmo_model(self):
        self.assertEqual(self.lc_model.model_parameters, self.lc_model.sn_model.param_names)

    def test_combine_lc_params_defaults_t0_and_uses_redshift(self):
        self.lc_model.em_parameter_setup(dict(luminosity_distance=40.0, amplitude=2.0))
        self.assertAlmostEqual(self.lc_model.sn_model.get("z"), self.lc_model.redshift)
        self.assertAlmostEqual(self.lc_model.sn_model.get("t0"), 0.0)
        self.assertAlmostEqual(self.lc_model.sn_model.get("amplitude"), 2.0)

    def test_anchor_amplitude_sets_mag_ref_and_amplitude(self):
        self.lc_model._anchor_amplitude(mag_ref=-19.35)
        self.assertAlmostEqual(self.lc_model.mag_ref, -19.35)
        self.assertGreater(self.lc_model.sn_model.get("amplitude"), 0.0)

    def test_check_vs_priors_rejects_amplitude_and_mag_boost_together(self):
        self.lc_model.model = "salt2"
        priors = bilby.core.prior.PriorDict()
        priors["supernova_mag_boost"] = bilby.core.prior.Uniform(
            -1, 1, "supernova_mag_boost"
        )
        priors["amplitude"] = bilby.core.prior.Uniform(0, 1, "amplitude")
        with self.assertRaises(ValueError):
            self.lc_model.check_vs_priors(priors)


class TestCombinedLightCurveModelContainer(unittest.TestCase):
    def setUp(self):
        self.kn_model = model.SimpleKilonovaLightCurveModel(
            model="Me2017", filters=["sdssg"]
        )
        self.host_model = model.HostGalaxyLightCurveModel(
            filters=["sdssg"], host_mag=22.0
        )
        self.combined = model.CombinedLightCurveModelContainer(
            [self.kn_model, self.host_model]
        )

    def tearDown(self):
        del self.kn_model
        del self.host_model
        del self.combined

    def test_model_is_list_of_submodel_names(self):
        self.assertEqual(self.combined.model, ["Me2017", "Sr2023"])

    def test_all_filters_is_union_of_submodel_filters(self):
        self.assertEqual(self.combined.all_filters, {"sdssg"})

    def test_model_times_is_sorted_union_of_submodel_times(self):
        expected = np.array(
            sorted(set().union(self.kn_model.model_times, self.host_model.model_times))
        )
        np.testing.assert_allclose(self.combined.model_times, expected)

    def test_citation_merges_submodel_citations(self):
        citation = self.combined.citation
        self.assertEqual(citation["Me2017"], model.citation_dict["Me2017"])
        self.assertEqual(citation["Sr2023"], model.citation_dict["Sr2023"])

    def test_good_parameters_getter_reflects_submodels(self):
        self.kn_model.good_parameters = True
        self.host_model.good_parameters = True
        self.assertTrue(self.combined.good_parameters)

        self.host_model.good_parameters = False
        self.assertFalse(self.combined.good_parameters)

    def test_good_parameters_setter_propagates_to_submodels(self):
        self.combined.good_parameters = False
        self.assertFalse(self.kn_model.good_parameters)
        self.assertFalse(self.host_model.good_parameters)

    def test_stack_magnitudes_combines_flux_of_overlapping_filters(self):
        mags_per_model = [
            {"sdssg": np.array([20.0, 20.0])},
            {"sdssg": np.array([21.0, 21.0])},
        ]
        stacked = self.combined.stack_magnitudes(mags_per_model)
        flux_sum = 10 ** (-0.4 * 20.0) + 10 ** (-0.4 * 21.0)
        expected_mag = -2.5 * np.log10(flux_sum)
        np.testing.assert_allclose(stacked["sdssg"], [expected_mag, expected_mag])

    def test_stack_magnitudes_returns_inf_when_no_model_has_the_filter(self):
        combined = model.CombinedLightCurveModelContainer.__new__(
            model.CombinedLightCurveModelContainer
        )
        combined.all_filters = {"sdssr"}
        combined.compatible_filters = {"sdssr": "sdssr"}
        combined.model_times = np.array([1.0, 2.0])
        stacked = combined.stack_magnitudes([{"sdssg": np.array([20.0, 20.0])}])
        self.assertTrue(np.all(np.isinf(stacked["sdssr"])))


class TestSingleModelFromMapping(unittest.TestCase):
    def test_transient_class_lookup_is_case_insensitive(self):
        self.assertIs(model.single_model_from_mapping("svd"), model.SVDLightCurveModel)
        self.assertIs(model.single_model_from_mapping("SVD"), model.SVDLightCurveModel)
        self.assertIs(model.single_model_from_mapping("grb"), model.GRBLightCurveModel)

    def test_model_name_lookup_via_model_name_map(self):
        self.assertIs(
            model.single_model_from_mapping("TrPi2018"), model.GRBLightCurveModel
        )
        self.assertIs(
            model.single_model_from_mapping("Me2017"),
            model.SimpleKilonovaLightCurveModel,
        )

    def test_sncosmo_source_name_maps_to_supernova_model(self):
        self.assertIs(
            model.single_model_from_mapping("nugent-hyper"),
            model.SupernovaLightCurveModel,
        )

    def test_unknown_identifier_falls_back_to_svd_model(self):
        self.assertIs(
            model.single_model_from_mapping("totally_unknown_model"),
            model.SVDLightCurveModel,
        )

    def test_unknown_identifier_raises_when_class_enforced(self):
        with self.assertRaises(ValueError):
            model.single_model_from_mapping("totally_unknown_model", enfore_class=True)

    def test_existing_model_instance_is_returned_unchanged(self):
        lc_model = model.SimpleKilonovaLightCurveModel(model="Me2017", filters=["sdssg"])
        self.assertIs(model.single_model_from_mapping(lc_model), lc_model)


class TestLcModelFromTransientClass(unittest.TestCase):
    def test_matches_model_names_to_transient_classes_in_order(self):
        model_classes, model_names = model.lc_model_from_transient_class(
            ["grb", "supernova"], "TrPi2018,nugent-hyper"
        )
        self.assertEqual(model_classes, [model.GRBLightCurveModel, model.SupernovaLightCurveModel])
        self.assertEqual(model_names, ["TrPi2018", "nugent-hyper"])

    def test_missing_model_name_defaults_to_none(self):
        model_classes, model_names = model.lc_model_from_transient_class(["grb"], None)
        self.assertEqual(model_classes, [model.GRBLightCurveModel])
        self.assertEqual(model_names, [None])

    def test_unknown_transient_class_raises(self):
        with self.assertRaises(ValueError):
            model.lc_model_from_transient_class(["not_a_class"], None)


if __name__ == "__main__":
    unittest.main()
