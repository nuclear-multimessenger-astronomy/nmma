import warnings
from copy import copy
from pathlib import Path

import joblib
import numpy as np
import pytest
import sncosmo
from astropy import units as u
from bilby.core.prior import PriorDict, Uniform
from dust_extinction.parameter_averages import G23

from nmma.core.constants import c_SI, default_cosmology as ref_cosmo
from nmma.em import lightcurve_generation as lc_gen, model

PRIOR_DIR = Path(__file__).resolve().parent.parent.parent.parent / "priors"
MODELS_DIR = PRIOR_DIR.parent / "nmma_models"


class LightCurveModelTestContainer:
    filters = ["ztfr", "ztfg"]

    min_lightcurve = {k: np.linspace(18, 22, 5) for k in filters}
    transient_class = model.LightCurveModelContainer
    init_kwargs = {}

    alt_tmin = 0.1  # days
    alt_tmax = 10.0  # days
    alt_sample_times = np.geomspace(alt_tmin, alt_tmax, 50)
    min_lightcurve_times = np.geomspace(alt_tmin, alt_tmax, 5)  # for min_lightcurve
    test_redshift = 0.05
    test_timeshift = 1.5
    test_distance = ref_cosmo.luminosity_distance(test_redshift).value
    test_Ebv = 0.2
    share_params = {"redshift": test_redshift, "Ebv": test_Ebv, "timeshift": 1.5}
    extra_timeshift = 2.0  # days

    def setup_method(self):
        self.model = self.build_model()
        self.setup_prior()
        self.base_params = self.priors.sample()
        use_params = self.base_params | self.share_params.copy()
        self.lc_params = self.model.parameter_conversion(use_params)

    def build_model(self, **kwargs):
        use_kwargs = dict(model=self.model_name, filters=self.filters)
        use_kwargs |= self.init_kwargs | kwargs
        return self.transient_class(**use_kwargs)

    def setup_prior(self):
        prior_name = getattr(self, "prior_name", self.model_name)
        prior_file = PRIOR_DIR / f"{prior_name}.prior"
        if prior_file.exists():
            self.priors = PriorDict(filename=str(prior_file))
        else:
            assert hasattr(self, "priors"), (
                "No prior file found for model "
                f"{self.model_name}, and no dummy_prior defined in the test class."
            )

    def test_init_sets_up(self):
        assert self.model.default_filts == self.filters
        assert self.model.nu_0s == pytest.approx(c_SI / self.model.lambdas)

    def test_sample_time_initialisation(self):
        sample_times = self.model.model_times
        assert sample_times[0] == pytest.approx(self.model.tmin)
        assert sample_times[-1] == pytest.approx(self.model.tmax)

        self.model.setup_model_times(self.alt_sample_times)
        assert self.model.model_times[0] == pytest.approx(self.alt_tmin)
        assert self.model.model_times[-1] == pytest.approx(self.alt_tmax)

    # -- em_parameter_setup / set_distance_parameters / combine_lc_params ---

    def test_em_parameter_setup_for_ebv_and_timeshift(self):
        self.model.em_parameter_setup(self.lc_params)
        assert self.model.Ebv == self.test_Ebv
        assert self.model.redshift == pytest.approx(self.test_redshift)

        self.model.em_parameter_setup(
            self.base_params | {"redshift": self.test_redshift}
        )
        assert self.model.Ebv == 0.0
        if "timeshift" not in self.priors:
            assert self.model.timeshift == 0.0
        else:
            assert self.model.timeshift == pytest.approx(self.base_params["timeshift"])
        assert self.model.redshift == pytest.approx(self.test_redshift)

    def test_set_distance_parameters_for_base_cosmology(self):
        result = self.model.set_distance_parameters({"redshift": self.test_redshift})
        assert result["luminosity_distance"] == pytest.approx(self.test_distance)
        assert result["redshift"] == self.test_redshift
        assert self.model.luminosity_distance == pytest.approx(self.test_distance)
        assert self.model.redshift == self.test_redshift
        assert self.model.distmod == pytest.approx(
            5.0 * (5 + np.log10(self.test_distance))
        )

    def test_combine_detector_data(self):
        self.model.em_parameter_setup(self.lc_params)
        offset = self.model.frame_correction()
        first = self.filters[0]

        model_lc = {k: v.copy() for k, v in self.min_lightcurve.items()}
        model_lc[first][0] = np.nan  # bad output
        model_lc[first][-1] = np.inf  # too faint
        times, lc = self.model.combine_detector_data(
            model_lc, self.min_lightcurve_times
        )
        assert times is self.min_lightcurve_times
        # both are reported as non-detections
        assert lc[first][0] == np.inf
        assert lc[first][-1] == np.inf
        for filt in self.filters:
            assert lc[filt][1:-1] == pytest.approx(
                self.min_lightcurve[filt][1:-1] + offset
            )

    def test_combine_detector_data_with_extinction(self):
        self.model.em_parameter_setup(self.lc_params)
        model_lc = {k: v.copy() for k, v in self.min_lightcurve.items()}
        t, non_ext_lc = self.model.combine_detector_data(
            model_lc, self.min_lightcurve_times
        )
        self.model.setup_extinction()
        dimming = self.model.extinction_correction(
            {filt: np.zeros(1) for filt in self.filters}
        )
        model_lc = {k: v.copy() for k, v in self.min_lightcurve.items()}
        _, extinct_lc = self.model.combine_detector_data(
            model_lc, self.min_lightcurve_times
        )
        for filt in self.filters:
            filt_dimming = dimming[filt][0]
            assert filt_dimming > 0
            assert extinct_lc[filt] == pytest.approx(non_ext_lc[filt] + filt_dimming)

    def test_gen_detector_lc(self):
        times, mags = self.model.gen_detector_lc(self.lc_params)

        # check time shifted correctly
        assert np.asarray(times) == pytest.approx(
            self.model.model_times * (1 + self.test_redshift) + self.test_timeshift
        )
        # check magnitudes are finite and have the right shape
        assert set(mags) == set(self.filters)
        for filt in self.filters:
            assert np.shape(mags[filt]) == np.shape(times)
            assert not np.any(np.isnan(mags[filt]))

        # check timeshift only moves the light curve
        shifted_times, shifted_mags = self.model.gen_detector_lc(
            self.lc_params | {"timeshift": self.test_timeshift + self.extra_timeshift}
        )
        assert shifted_times == pytest.approx(times + self.extra_timeshift)
        for filt in self.filters:
            assert shifted_mags[filt] == pytest.approx(mags[filt])

    def test_citation_is_available(self):
        # fixme: find meaningful test
        assert self.model.citation is not None


class TestLightCurveModelContainer(LightCurveModelTestContainer):
    """
    Dummy class to test general behaviour irrespective of the specific model.
    """

    model_name = "dummy_test"
    transient_class = model.LightCurveModelContainer
    dummy_prior = dict(
        a=Uniform(0, 1, name="a"),
        b=Uniform(0, 1, name="b"),
        c=Uniform(0, 1, name="c"),
    )
    dummy_prior = PriorDict(dummy_prior)
    init_kwargs = {"model_parameters": list(dummy_prior.keys())}

    def setup_prior(self):
        self.priors = self.dummy_prior.copy()

    def test_identify_model_parameters_rejects_unknown_models(self):
        with pytest.raises(AssertionError):
            self.init_kwargs = {}
            super().setup_method()

    def test_inclination_conversion(self):
        parameters = {"KNtheta": 90.0}
        result = model.observation_angle_conversion(parameters)
        assert result["inclination_EM"] == pytest.approx(np.pi / 2)

        parameters = {"inclination_EM": np.pi / 2}
        result = model.observation_angle_conversion(parameters)
        assert result["KNtheta"] == pytest.approx(90.0)

    def test_reject_two_inclination_priors(self):
        prior = self.priors
        prior["KNtheta"] = Uniform(0, 180, name="KNtheta")
        prior["inclination_EM"] = Uniform(0, np.pi / 2, name="inclination_EM")
        with pytest.raises(ValueError):
            self.model.check_vs_priors(prior)

    def test_degree_inclination_prior(self):
        prior = self.priors
        prior["KNtheta"] = Uniform(0, 180, name="KNtheta")
        self.model.check_vs_priors(prior)
        prior["KNtheta"] = Uniform(0, np.pi / 2, name="KNtheta", unit="deg")
        self.model.check_vs_priors(prior)
        prior["KNtheta"] = Uniform(0, np.pi / 2, name="KNtheta")
        with pytest.raises(ValueError):
            self.model.check_vs_priors(prior)

    def test_rad_inclination_prior(self):
        prior = self.priors
        prior["inclination_EM"] = Uniform(0, np.pi / 2, name="inclination_EM")
        self.model.check_vs_priors(prior)
        prior["inclination_EM"] = Uniform(0, 45, name="inclination_EM", unit="rad")
        self.model.check_vs_priors(prior)
        prior["inclination_EM"] = Uniform(0, 45, name="inclination_EM")
        with pytest.raises(ValueError):
            self.model.check_vs_priors(prior)

    def test_cosmology_conversion_from_prior_check(self):
        self.model.check_vs_priors(self.priors)
        result = self.model.cosmo_converter({})
        assert result["luminosity_distance"] == pytest.approx(1e-5)  # 10 pc in Mpc
        assert result["redshift"] == 0.0

    def test_extinction_from_prior_check(self):
        self.model.check_vs_priors(self.priors)
        assert self.model.extinction_frame is None

        self.priors["Ebv"] = Uniform(0, 1, name="Ebv")
        self.model.check_vs_priors(self.priors)
        assert self.model.extinction_frame == "rest"

    def test_setup_extinction(self):
        self.model = self.build_model(extinction_model="G23_mw")
        self.model.setup_extinction()
        assert isinstance(self.model.extinction_model, G23)
        assert self.model.extinction_wavenumbers == self.model.obs_wavenumbers
        assert self.model.wavenumbers == pytest.approx(
            1.0 / (self.model.lambdas * u.meter.to(u.micron))
        )
        with pytest.raises(ValueError):
            bad_frame_model = "G23_host"
            self.model.extinction_model = bad_frame_model
            self.model.setup_extinction()

    def test_rest_and_obs_wavenumbers(self):
        self.model.setup_extinction()
        self.model.redshift = 0.05

        assert self.model.obs_wavenumbers() == pytest.approx(self.model.wavenumbers)
        assert self.model.rest_wavenumbers() == pytest.approx(
            self.model.wavenumbers * 1.05
        )

    def test_sanity_checks(self):
        parameters = self.priors.sample()
        self.model.sanity_checks(parameters)
        assert self.model.good_parameters is True

    def test_parameter_conversion(self):
        sample = self.priors.sample()
        log_sample = {f"log10_{k}": np.log10(v) for k, v in sample.items()}
        converted = self.model.parameter_conversion(log_sample)
        converted = {k: v for k, v in converted.items() if k in self.priors}
        assert converted == pytest.approx(sample)

    def test_handle_missing_parameters(self):
        model = self.build_model(model_parameters=["a", "b", "missing_param"])
        sample = self.priors.sample()
        with pytest.raises(ValueError):
            model.parameter_conversion(sample)

    def test_combine_lc_params(self):
        params = self.priors.sample()
        result = self.model.combine_lc_params(params)
        assert result == pytest.approx(params)

        self.model.model_parameters.append("redshift")
        with pytest.raises(AttributeError):
            self.model.combine_lc_params(params)

        self.model.redshift = self.test_redshift
        result = self.model.combine_lc_params(params)
        assert result["redshift"] == pytest.approx(self.test_redshift)

        params["redshift"] = self.test_redshift + 0.1
        result = self.model.combine_lc_params(params)
        assert result["redshift"] == pytest.approx(self.test_redshift + 0.1)
        assert self.model.redshift != result["redshift"]

    def test_extinction_correction(self):
        self.model.setup_extinction()
        self.model.redshift = self.test_redshift

        mags = self.min_lightcurve
        ref = {k: v.copy() for k, v in mags.items()}
        self.model.Ebv = 0.0  # this should do nothing
        result = self.model.extinction_correction(dict(mags))
        for filt in self.filters:
            assert ref[filt] == pytest.approx(result[filt])

        mags = self.min_lightcurve
        self.model.Ebv = 0.4  # with extinct
        result = self.model.extinction_correction(dict(mags))
        for filt in self.filters:
            assert ref[filt] != pytest.approx(result[filt])

    def test_gen_detector_lc(self):
        pass


class FiestaModelTestContainer(LightCurveModelTestContainer):
    """Behaviour shared by every fiesta-backed model.

    FiestaModel itself cannot run without a surrogate, so these tests run
    through its concrete subclasses.
    """

    prior_overshoot = 0.5

    @classmethod
    def setup_class(cls):
        # loading a surrogate is slow, so it is loaded once per test class
        cls.loaded_model = cls.transient_class(
            model=cls.model_name, filters=cls.filters, **cls.init_kwargs
        )

    def build_model(self):
        # every test gets its own wrapper around the shared, read-only surrogate
        return copy(self.loaded_model)

    def setup_prior(self):
        self.priors = PriorDict(
            {
                k: Uniform(v[0], v[1], name=k)
                for k, v in self.model.fiesta_model.parameter_distributions.items()
            }
        )

    def test_init(self):
        # fallback to default
        assert self.model.model == self.model_name
        assert self.model.model_parameters == self.model.fiesta_model.parameter_names
        assert self.model.fiesta_model.filters == self.filters
        assert self.model.model_times == pytest.approx(self.model.fiesta_model.times)
        assert self.model.default_filts == self.model.fiesta_model.filters

        # read times correctly
        surrogate_times = np.asarray(self.model.fiesta_model.times)
        assert np.asarray(self.model._default_model_times()) == pytest.approx(
            surrogate_times
        )
        assert self.model.tmin == pytest.approx(surrogate_times[0])
        assert self.model.tmax == pytest.approx(surrogate_times[-1])

        # model path
        model_dir = Path(self.model.fiesta_model.directory)
        surrogate_root = model_dir.parents[1]
        assert model_dir == (
            surrogate_root / self.model.load_dir_string / self.model_name
        )

        root_model = self.transient_class(
            filters=self.filters, surrogate_dir=surrogate_root
        )
        assert Path(root_model.fiesta_model.directory) == model_dir

    def test_check_vs_priors_respects_training_range(self):
        self.model.check_vs_priors(self.priors)  # within the training range

        low, high = self.model.fiesta_model.parameter_distributions[
            self.checked_parameter
        ][:2]
        self.priors[self.checked_parameter] = Uniform(
            low, high + self.prior_overshoot, name=self.checked_parameter
        )
        with pytest.raises(ValueError):
            self.model.check_vs_priors(self.priors)

        self.priors[self.checked_parameter] = Uniform(
            low - self.prior_overshoot, high, name=self.checked_parameter
        )
        with pytest.raises(ValueError):
            self.model.check_vs_priors(self.priors)

    def test_combine_lc_params(self):
        self.model.em_parameter_setup(self.share_params)
        parameters = self.lc_params | {"redshift": self.test_redshift + 0.1}
        result = self.model.combine_lc_params(parameters)
        assert result is parameters  # updated in place

        # fiesta takes the frame from what em_parameter_setup stored
        assert result["redshift"] == self.test_redshift
        assert result["timeshift"] == self.test_timeshift
        assert result["luminosity_distance"] == pytest.approx(self.test_distance)
        for key in self.model.model_parameters:
            assert result[key] == self.lc_params[key]

    def test_gen_detector_lc_applies_extinction(self):
        plain_times, plain_mags = self.model.gen_detector_lc(self.lc_params.copy())
        self.model.setup_extinction()
        ext_times, extinct_mags = self.model.gen_detector_lc(self.lc_params.copy())
        assert ext_times == pytest.approx(plain_times)
        for filt in self.filters:
            dimming = extinct_mags[filt] - plain_mags[filt]
            assert np.all(dimming > 0)

    def test_gen_detector_lc_rejects_bad_parameters(self):
        self.model.good_parameters = False
        times, mags = self.model.gen_detector_lc(self.lc_params.copy())
        assert mags == {}
        assert np.asarray(times) == pytest.approx(self.model.fiesta_model.times)

    def test_generate_lightcurve_inverts_detector_frame(self):
        detector_times, detector_mags = self.model.gen_detector_lc(
            self.lc_params.copy()
        )
        source_mags = self.model.generate_lightcurve(
            self.model.model_times, self.lc_params.copy()
        )
        # the base-class route back to the detector must reproduce fiesta's output
        _, roundtrip_mags = self.model.combine_detector_data(
            source_mags, detector_times
        )
        for filt in self.filters:
            assert roundtrip_mags[filt] == pytest.approx(detector_mags[filt])


class TestFiestaKilonovaModel(FiestaModelTestContainer):
    transient_class = model.FiestaKilonovaModel
    model_name = "Bu2026_MLP"
    checked_parameter = "log10_mej_dyn"


class GRBMixinTestContainer:
    """Behaviour GRBMixin adds to every GRB model it is mixed into."""

    max_log10_epsilon = -1.0  # keeps epsilon_e + epsilon_B safely below 1
    test_resolution = 6
    jet_core = 0.1  # rad
    wing_ratio = 3.0

    def setup_prior(self):
        super().setup_prior()
        for key in ("log10_epsilon_e", "log10_epsilon_B"):
            self.priors[key] = Uniform(
                self.priors[key].minimum, self.max_log10_epsilon, name=key
            )

    def test_init_sets_resolution(self):
        resolved_model = self.transient_class(
            filters=self.filters, resolution=self.test_resolution
        )
        assert resolved_model.resolution == self.test_resolution

    def test_parameter_conversion_derives_wing_from_ratio(self):
        converted = self.model.parameter_conversion(
            self.lc_params | {"thetaCore": self.jet_core, "alphaWing": self.wing_ratio}
        )
        assert converted["thetaWing"] == pytest.approx(self.jet_core * self.wing_ratio)
        assert self.model.resolution == pytest.approx(self.wing_ratio)

    def test_sanity_checks(self):
        self.model.sanity_checks(self.lc_params)
        assert self.model.good_parameters

        unphysical_jets = (
            {"thetaCore": 1.0, "thetaWing": 2.0},  # wing beyond the hemisphere
            {"thetaCore": 0.01, "thetaWing": 0.5},  # beyond resolution
            {"thetaCore": 1e-4, "thetaWing": 1e-4},  # core too narrow
            {"epsilon_tot": 1.5},  # too much energy
        )
        for jet in unphysical_jets:
            self.model.sanity_checks(self.lc_params | jet)
            assert not self.model.good_parameters


class TestFiestaGRBModel(GRBMixinTestContainer, FiestaModelTestContainer):
    transient_class = model.FiestaGRBModel
    model_name = "afgpy_gaussian_CVAE"
    checked_parameter = "log10_E0"
    max_theta_core = 0.3  # rad; keeps alphaWing * thetaCore within the hemisphere

    def setup_prior(self):
        super().setup_prior()
        self.priors["thetaCore"] = Uniform(
            self.priors["thetaCore"].minimum, self.max_theta_core, name="thetaCore"
        )


class TestGRBLightCurveModel(GRBMixinTestContainer, LightCurveModelTestContainer):
    transient_class = model.GRBLightCurveModel
    model_name = "TrPi2018"
    prior_name = "TrPi2018_onaxis"
    custom_xi_N = 0.5

    def test_init_sets_afterglowpy_defaults(self):
        assert self.model.jet_type == 0
        assert self.model.default_parameters["jetType"] == self.model.jet_type
        assert self.model.default_parameters["xi_N"] == 1.0

    def test_handle_missing_parameters_tolerates_gaps(self):
        incomplete = {k: v for k, v in self.lc_params.items() if k != "p"}
        self.model.parameter_conversion(incomplete)  # filled in later, no error

    def test_combine_lc_params(self):
        custom_model = self.build_model(xi_N=self.custom_xi_N)
        assert custom_model.default_parameters["xi_N"] == self.custom_xi_N

        self.model.redshift = self.test_redshift
        grb_params = self.model.combine_lc_params(self.lc_params.copy())
        assert grb_params["z"] == self.test_redshift
        assert grb_params["thetaObs"] == self.lc_params["inclination_EM"]
        assert grb_params["E0"] == pytest.approx(10 ** self.lc_params["log10_E0"])
        assert grb_params["n0"] == pytest.approx(10 ** self.lc_params["log10_n0"])
        assert grb_params["jetType"] == self.model.jet_type

    def test_prior_check_adopts_energy_injection_approach(self):
        assert self.model.flux_func is lc_gen.flux_density_on_time_array
        E_injection_prior = {
            key: Uniform(0, 1, name=key) for key in self.model.energy_injection_params
        }
        self.model.check_vs_priors(self.priors | E_injection_prior)
        assert self.model.flux_func is lc_gen.flux_density_on_E0_array
        assert "E0" not in self.model.log_sampling_keys

    def test_em_parameter_setup_passes_b_for_structured_jets(self):
        structured_jet_type = 4  # afgpy convention
        structured_model = self.build_model(jet_type=structured_jet_type)
        grb_params = structured_model.em_parameter_setup(self.lc_params.copy())
        assert grb_params["b"] == self.lc_params["b"]

    def test_generate_lightcurve_rejects_bad_parameters(self):
        self.model.good_parameters = False
        lc = self.model.generate_lightcurve(
            self.model.model_times, self.lc_params.copy()
        )
        assert lc == {}


class TestSimpleBolometricLightCurveModel(LightCurveModelTestContainer):
    transient_class = model.SimpleBolometricLightCurveModel
    model_name = "Arnett"
    min_luminosity = np.geomspace(1e40, 1e42, 5)  # erg/s, for min_lightcurve_times

    def setup_prior(self):
        self.priors = PriorDict(
            {
                "tau_m": Uniform(5.0, 20.0, name="tau_m"),
                "log10_mni": Uniform(-2.0, 0.0, name="log10_mni"),
            }
        )

    def test_init_selects_lightcurve_function(self):
        assert self.model.lc_func is lc_gen.arnett_lc
        modified_model = self.build_model(model="Arnett_modified")
        assert modified_model.lc_func is lc_gen.arnett_modified_lc

    def test_gen_detector_lc(self):
        # a bolometric luminosity, rather than magnitudes per filter
        times, lbol = self.model.gen_detector_lc(self.lc_params.copy())
        assert np.asarray(times) == pytest.approx(
            self.model.model_times * (1 + self.test_redshift) + self.test_timeshift
        )
        assert np.shape(lbol) == np.shape(times)
        assert np.all(np.isfinite(lbol))

        # neither the timeshift nor the distance changes the luminosity
        shifted_times, shifted_lbol = self.model.gen_detector_lc(
            self.lc_params | {"timeshift": self.test_timeshift + self.extra_timeshift}
        )
        assert shifted_times == pytest.approx(times + self.extra_timeshift)
        assert shifted_lbol == pytest.approx(lbol)

        _, far_lbol = self.model.gen_detector_lc(
            self.lc_params | {"luminosity_distance": 10 * self.test_distance}
        )
        assert far_lbol == pytest.approx(lbol)

    def test_combine_detector_data(self):
        self.model.em_parameter_setup(self.lc_params)
        times, lbol = self.model.combine_detector_data(
            self.min_luminosity, self.min_lightcurve_times
        )
        assert times is self.min_lightcurve_times
        # energy and time bin are both redshifted
        assert lbol == pytest.approx(
            self.min_luminosity / (1 + self.test_redshift) ** 2
        )

    def test_combine_detector_data_with_extinction(self):
        pass  # bolometric luminosity has no extinction correction


class TestSVDLightCurveModel(LightCurveModelTestContainer):
    """The keras path, reached through the tensorflow-specific model name."""

    transient_class = model.SVDLightCurveModel
    model_name = "Bu2019lm_tf"
    core_model_name = "Bu2019lm"
    prior_name = "Bu2019lm"
    init_kwargs = {
        "svd_path": MODELS_DIR / "svdmodels",
        "interpolation_type": "tensorflow",
        "local_only": False,
    }
    estimator_key = "model"  # where the trained predictor sits, per filter
    unknown_interpolation = "spline"
    missing_extension = "missing"

    @classmethod
    def setup_class(cls):
        if not cls.init_kwargs["svd_path"].exists():
            pytest.skip("SVD models are not available in the repo.")
        super().setup_class()

    def setup_method(self):
        super().setup_method()
        # the SVD models typically sample the inclination in degrees, but the priors are in radians
        self.base_params = model.observation_angle_conversion(self.base_params)

    def test_init_drops_tf_suffix(self):
        assert self.model.model == self.core_model_name

    def test_init_attaches_estimator_per_filter(self):
        for filt in self.filters:
            assert self.estimator_key in self.model.svd_mag_model[filt]

    def test_init_rejects_unknown_interpolation(self):
        with pytest.raises(ValueError):
            self.transient_class(
                model=self.model_name,
                filters=self.filters,
                **self.init_kwargs | {"interpolation_type": self.unknown_interpolation},
            )

    def test_default_model_times(self):
        training_times = self.model.svd_mag_model[self.filters[0]]["tt"]
        assert self.model._default_model_times() == pytest.approx(training_times)
        assert self.model.model_times == pytest.approx(training_times)

    def test_load_filt_model_requires_model_files(self):
        with pytest.raises(ValueError):
            self.model.load_filt_model(
                self.core_model_name, joblib.load, fn_ext=self.missing_extension
            )

    def test_repr_names_svd_path(self):
        assert str(self.model.svd_path) in repr(self.model)

    def test_combine_lc_params_orders_parameters(self):
        ordered = self.model.combine_lc_params(self.lc_params)
        assert ordered == [self.lc_params[k] for k in self.model.model_parameters]

        # a parameter that was not sampled is read from the model
        first = self.model.model_parameters[0]
        setattr(self.model, first, self.lc_params[first])
        partial = {k: v for k, v in self.lc_params.items() if k != first}
        assert self.model.combine_lc_params(partial) == ordered

    def test_generate_lightcurve_selects_filters(self):
        all_mags = self.model.generate_lightcurve(
            self.alt_sample_times, self.lc_params.copy()
        )
        assert set(all_mags) == set(self.filters)

        first = self.filters[0]
        first_mags = self.model.generate_lightcurve(
            self.alt_sample_times, self.lc_params.copy(), filters=[first]
        )
        assert set(first_mags) == {first}
        assert first_mags[first] == pytest.approx(all_mags[first])

    def test_generate_spectra_uses_wavelengths_as_filters(self):
        wavelengths = [self.filters[0]]
        spectra = self.model.generate_spectra(
            self.alt_sample_times, wavelengths, self.lc_params.copy()
        )
        lightcurve = self.model.generate_lightcurve(
            self.alt_sample_times, self.lc_params.copy(), filters=wavelengths
        )
        for wavelength in wavelengths:
            assert spectra[wavelength] == pytest.approx(lightcurve[wavelength])


class TestSVDLightCurveModelWithGP(TestSVDLightCurveModel):
    """The sklearn Gaussian-process path, with a model name kept as given."""

    model_name = "Ka2017"
    core_model_name = "Ka2017"
    prior_name = "Ka2017"
    filters = ["sdssu"]  # the only filter with a local GP
    min_lightcurve = {k: np.linspace(18, 22, 5) for k in filters}
    init_kwargs = {
        "svd_path": MODELS_DIR,
        "interpolation_type": "sklearn_gp",
        "local_only": False,
    }
    estimator_key = "gps"


class TestHostGalaxyLightCurveModel(LightCurveModelTestContainer):
    transient_class = model.HostGalaxyLightCurveModel
    model_name = "Sr2023"
    default_host_mag = 23.9  # AB zero point of a flux in muJy
    custom_host_mags = [22.0, 23.0]  # one per filter

    def test_init_fills_host_magnitude_per_filter(self):
        assert self.model.host_mag == pytest.approx(
            [self.default_host_mag] * len(self.filters)
        )

        custom_model = self.transient_class(
            filters=self.filters, host_mag=self.custom_host_mags
        )
        assert custom_model.host_mag == self.custom_host_mags

    def test_check_vs_priors_rejects_extinction(self):
        self.model.check_vs_priors(self.priors)
        self.priors["Ebv"] = Uniform(0, 1, name="Ebv")
        with pytest.raises(ValueError):
            self.model.check_vs_priors(self.priors)

    def test_generate_lightcurve_fades_onto_host(self):
        mags = self.model.generate_lightcurve(
            self.alt_sample_times, self.lc_params.copy()
        )
        for filt in self.filters:
            flux = (
                self.lc_params[f"a_AG_{filt}"]
                * self.alt_sample_times ** -self.lc_params["alpha_AG"]
                + self.lc_params[f"f_nu_{filt}"]
            )
            assert mags[filt] == pytest.approx(
                self.default_host_mag - 2.5 * np.log10(flux)
            )


class TestSupernovaLightCurveModel(LightCurveModelTestContainer):
    """An sncosmo source given by name, defined from the explosion onwards."""

    transient_class = model.SupernovaLightCurveModel
    model_name = "nugent-hyper"
    prior_name = "sncosmo-generic"
    anchor_mag = -19.35  # fiducial peak magnitude NMMA anchors to
    anchor_band = ("bessellv", "vega")  # first choice for anchoring
    test_stretch = 1.2
    test_boost = 0.5  # mag
    boost_prior = Uniform(-5.0, 5.0, name="supernova_mag_boost")
    amplitude_prior = Uniform(0.0, 1.0, name="amplitude")

    def test_init_takes_parameters_from_sncosmo(self):
        assert self.model.model == self.model.source.name
        assert list(self.model.model_parameters) == list(
            self.model.sn_model.param_names
        )
        with pytest.raises(ValueError):
            self.transient_class(
                model=self.model_name,
                filters=self.filters,
                model_parameters=self.model.model_parameters,
            )

    def test_default_model_times_start_at_explosion(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fresh_model = self.build_model()
        source = fresh_model.source
        assert fresh_model.model_times[0] == 0.0
        assert fresh_model.model_times[-1] == pytest.approx(
            source.maxphase() - source.minphase()
        )
        # only models relative to the peak are shifted, with a warning
        warned = any("explosion time" in str(w.message) for w in caught)
        assert warned == (source.minphase() < 0)

    def test_check_vs_priors_anchors_amplitude(self):
        self.priors["supernova_mag_boost"] = self.boost_prior
        self.model.check_vs_priors(self.priors)
        assert self.model.mag_ref == self.anchor_mag
        assert self.model.sn_model.source.peakmag(*self.anchor_band) == pytest.approx(
            self.anchor_mag
        )

        # the anchored model keeps NMMA's own stretch-and-boost evaluation
        self.model.em_parameter_setup(self.lc_params.copy())
        nmma_mags = self.model.nmma_lc(self.alt_sample_times, self.lc_params)
        mags = self.model.get_lc(self.alt_sample_times, self.lc_params)
        for filt in self.filters:
            assert mags[filt] == pytest.approx(nmma_mags[filt])

        self.priors["amplitude"] = self.amplitude_prior
        with pytest.raises(ValueError):  # amplitude and boost say the same thing
            self.model.check_vs_priors(self.priors)

    def test_check_vs_priors_without_boost_reads_sncosmo(self):
        self.priors.pop("supernova_mag_boost", None)
        self.model.check_vs_priors(self.priors)
        assert self.model.get_lc == self.model.sncosmo_lc

    def test_em_parameter_setup_configures_sncosmo(self):
        result = self.model.em_parameter_setup(self.lc_params.copy())
        assert result is None  # sncosmo keeps the state
        assert self.model.sn_model.get("z") == self.test_redshift
        assert self.model.sn_model.get("t0") == 0.0

    def test_combine_lc_params_keeps_sncosmo_defaults(self):
        self.model.set_distance_parameters({"redshift": self.test_redshift})
        lc_pars = self.model.combine_lc_params({})
        assert lc_pars["z"] == self.test_redshift
        assert lc_pars["t0"] == 0.0
        for key in set(self.model.model_parameters) - {"z", "t0"}:
            assert lc_pars[key] == self.model.sn_model.get(key)

    def test_nmma_lc_stretches_and_boosts(self):
        self.model.em_parameter_setup(self.lc_params.copy())
        reference = self.model.sncosmo_lc(self.alt_sample_times / self.test_stretch)
        mags = self.model.nmma_lc(
            self.alt_sample_times,
            {
                "supernova_mag_stretch": self.test_stretch,
                "supernova_mag_boost": self.test_boost,
            },
        )
        for filt in self.filters:
            ref = np.asarray(reference[filt])
            finite = np.isfinite(ref)
            assert mags[filt][finite] == pytest.approx(
                ref[finite] + self.test_boost + self.model.distmod
            )


class TestSupernovaLightCurveModelFromSncosmo(TestSupernovaLightCurveModel):
    """A ready-made sncosmo model, relative to peak and amplitude-free (SALT)."""

    source_name = "salt2"
    prior_name = "salt2"

    def build_model(self):
        # sncosmo models are stateful, so every test gets a fresh one
        return self.transient_class(
            model=sncosmo.Model(source=self.source_name),
            filters=self.filters,
            **self.init_kwargs,
        )

    def test_init_takes_parameters_from_sncosmo(self):
        assert self.model.model == self.source_name
        assert list(self.model.model_parameters) == list(
            self.model.sn_model.param_names
        )

    def test_check_vs_priors_anchors_amplitude(self):
        # SALT carries its amplitude in x0, so refuse anchor
        self.priors["supernova_mag_boost"] = self.boost_prior
        with pytest.raises(ValueError):
            self.model.check_vs_priors(self.priors)


class TestShockCoolingLightCurveModel(LightCurveModelTestContainer):
    transient_class = model.ShockCoolingLightCurveModel
    model_name = "Piro2021"

    def test_generate_lightcurve_bolometric(self):
        lbol = self.model.generate_lightcurve(
            self.model.model_times, self.lc_params.copy(), filters=None
        )
        assert np.shape(lbol) == np.shape(self.model.model_times)
        assert np.all(lbol > 0)

    def test_generate_lightcurve_selects_filters(self):
        all_mags = self.model.generate_lightcurve(
            self.model.model_times, self.lc_params.copy()
        )
        assert set(all_mags) == set(self.filters)

        first = self.filters[0]
        first_mags = self.model.generate_lightcurve(
            self.model.model_times, self.lc_params.copy(), filters=[first]
        )
        assert set(first_mags) == {first}


class TestSimpleKilonovaLightCurveModel(LightCurveModelTestContainer):
    transient_class = model.SimpleKilonovaLightCurveModel
    model_name = "Me2017"
    min_valid_time = 5e-2  # days, below which Me2017 is not valid

    def test_init_selects_lightcurve_function(self):
        assert self.model.lc_func is self.model.lc_dict[self.model_name]

    def test_init_restricts_times_to_validity(self):
        assert np.min(self.model.model_times) >= self.min_valid_time


class TestCombinedLightCurveModelContainer(LightCurveModelTestContainer):
    """A kilonova on top of a GRB afterglow, both from fiesta."""

    transient_class = model.CombinedLightCurveModelContainer
    component_classes = (model.FiestaKilonovaModel, model.FiestaGRBModel)
    # the GRB component needs the same physical restrictions as on its own
    physical_maxima = {
        "thetaCore": TestFiestaGRBModel.max_theta_core,
        "log10_epsilon_e": GRBMixinTestContainer.max_log10_epsilon,
        "log10_epsilon_B": GRBMixinTestContainer.max_log10_epsilon,
    }
    out_of_range_parameter = "log10_E0"  # checked by the GRB component only
    prior_overshoot = FiestaModelTestContainer.prior_overshoot

    @classmethod
    def setup_class(cls):
        # loading surrogates is slow, so they are loaded once per test class
        cls.loaded_components = [
            component(filters=cls.filters) for component in cls.component_classes
        ]

    def build_model(self):
        return self.transient_class(
            [copy(component) for component in self.loaded_components]
        )

    def setup_prior(self):
        self.priors = PriorDict()
        for lc_model in self.model.lc_models:
            distributions = lc_model.fiesta_model.parameter_distributions
            self.priors.update(
                {k: Uniform(v[0], v[1], name=k) for k, v in distributions.items()}
            )
        for key, maximum in self.physical_maxima.items():
            self.priors[key] = Uniform(self.priors[key].minimum, maximum, name=key)

    def test_init_sets_up(self):
        assert self.model.model == [c.model for c in self.loaded_components]
        assert self.model.all_filters == set(self.filters)

        default_model = self.transient_class(
            self.component_classes, model_args=[(), ()]
        )
        for lc_model, component in zip(default_model.lc_models, self.component_classes):
            assert isinstance(lc_model, component)

    def test_sample_time_initialisation(self):
        component_times = [np.asarray(c.model_times) for c in self.model.lc_models]
        assert self.model.model_times == pytest.approx(
            np.unique(np.concatenate(component_times))
        )

    def test_em_parameter_setup_for_ebv_and_timeshift(self):
        # the container has no frame of its own; each component sets up its own
        self.model.gen_detector_lc(self.lc_params.copy())
        for lc_model in self.model.lc_models:
            assert lc_model.Ebv == self.test_Ebv
            assert lc_model.timeshift == self.test_timeshift
            assert lc_model.redshift == self.test_redshift

    def test_set_distance_parameters_for_base_cosmology(self):
        self.model.gen_detector_lc(self.lc_params.copy())
        true_distance = ref_cosmo.luminosity_distance(self.test_redshift).value
        for lc_model in self.model.lc_models:
            assert lc_model.luminosity_distance == pytest.approx(true_distance)

    def test_combine_detector_data(self):
        pytest.skip("Each component combines its own detector data.")

    def test_citation_is_available(self):
        assert set(self.model.citation) == set(self.model.model)

    def test_repr_names_components(self):
        for lc_model in self.model.lc_models:
            assert repr(lc_model) in repr(self.model)

    def test_check_vs_priors_reaches_every_component(self):
        self.model.check_vs_priors(self.priors)

        low, high = (
            self.priors[self.out_of_range_parameter].minimum,
            self.priors[self.out_of_range_parameter].maximum,
        )
        self.priors[self.out_of_range_parameter] = Uniform(
            low, high + self.prior_overshoot, name=self.out_of_range_parameter
        )
        with pytest.raises(ValueError):
            self.model.check_vs_priors(self.priors)

    def test_good_parameters_need_every_component(self):
        assert self.model.good_parameters
        self.model.lc_models[-1].good_parameters = False
        assert not self.model.good_parameters

        self.model.good_parameters = True
        for lc_model in self.model.lc_models:
            assert lc_model.good_parameters

    def test_parameter_conversion_chains_components(self):
        converted = self.model.parameter_conversion(
            self.priors.sample() | self.share_params
        )
        assert "KNtheta" in converted  # from the kilonova
        assert "thetaWing" in converted and "epsilon_tot" in converted  # from the GRB

    def test_gen_detector_lc_returns_all_components(self):
        times_per_model, mags_per_model = self.model.gen_detector_lc(
            self.lc_params.copy(), return_all=True
        )
        assert len(mags_per_model) == len(self.model.lc_models)
        for lc_model, times, mags in zip(
            self.model.lc_models, times_per_model, mags_per_model
        ):
            own_times, own_mags = lc_model.gen_detector_lc(self.lc_params.copy())
            assert np.asarray(times) == pytest.approx(np.asarray(own_times))
            for filt in self.filters:
                assert np.asarray(mags[filt]) == pytest.approx(
                    np.asarray(own_mags[filt])
                )

    def test_gen_detector_lc_adds_component_fluxes(self):
        times, mags = self.model.gen_detector_lc(self.lc_params.copy())
        times_per_model, mags_per_model = self.model.gen_detector_lc(
            self.lc_params.copy(), return_all=True
        )
        flux = {filt: np.zeros(np.shape(times)) for filt in self.filters}
        for component_times, component_mags in zip(times_per_model, mags_per_model):
            for filt in self.filters:
                # a component contributes nothing outside its own time range
                component_mag = np.interp(
                    times,
                    np.asarray(component_times),
                    np.asarray(component_mags[filt]),
                    left=np.inf,
                    right=np.inf,
                )
                flux[filt] += 10 ** (-0.4 * component_mag)
        for filt in self.filters:
            assert mags[filt] == pytest.approx(-2.5 * np.log10(flux[filt]))

    def test_gen_detector_lc_rejects_bad_parameters(self):
        self.model.lc_models[-1].good_parameters = False
        _, mags = self.model.gen_detector_lc(self.lc_params.copy())
        assert mags == {}

    def test_stack_magnitudes_adds_fluxes(self):
        identical_components = [self.min_lightcurve] * 2
        stacked = self.model.stack_magnitudes(identical_components)
        for filt in self.filters:
            assert stacked[filt] == pytest.approx(
                self.min_lightcurve[filt] - 2.5 * np.log10(len(identical_components))
            )

        # a filter no component provides is a non-detection
        stacked = self.model.stack_magnitudes([{}, {}])
        for filt in self.filters:
            assert np.all(stacked[filt] == np.inf)

    def test_combine_detector_data_with_extinction(self):
        pass  # does only apply to  component models
