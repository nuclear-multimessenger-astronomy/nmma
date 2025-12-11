from __future__ import division

import copy
import os
import joblib
import numpy as np
from scipy.special import logsumexp
import scipy.constants
from sncosmo.models import _SOURCES

from . import utils

from ..utils.models import get_models_home, get_model

ln10 = np.log(10)

# As different KN models have very different parameters,
# we need a dict to keep track for them. Remember, the order matters

model_parameters_dict = {
    "Bu2019nsbh": ["log10_mej_dyn", "log10_mej_wind", "KNtheta"],
    "Bu2019lm": ["log10_mej_dyn", "log10_mej_wind", "KNphi", "KNtheta"],
    "Bu2019lm_sparse": ["log10_mej_dyn", "log10_mej_wind"],
    "Ka2017": ["log10_mej", "log10_vej", "log10_Xlan"],
    "TrPi2018": [
        "inclination_EM",
        "log10_E0",
        "thetaCore",
        "thetaWing",
        "b",
        "L0",
        "q",
        "ts",
        "log10_n0",
        "p",
        "log10_epsilon_e",
        "log10_epsilon_B",
        "xi_N",
        "d_L",
    ],
    "Piro2021": ["log10_Menv", "log10_Renv", "log10_Ee"],
    "Me2017": ["log10_mej", "log10_vej", "beta", "log10_kappa_r"],
    "Bu2022mv": ["log10_mej_dyn", "vej_dyn", "log10_mej_wind", "vej_wind", "KNtheta"],
    "PL_BB_fixedT": ["bb_luminosity", "temperature", "beta", "powerlaw_mag"],
    "blackbody_fixedT": ["bb_luminosity", "temperature"],
    "synchrotron_powerlaw": ["alpha_time", "beta_freq", "F_ref"],
    "CV": ["example_num"],
    "AnBa2022_sparse": ["mrp", "xmix"],
    "AnBa2022_log": ["log10_mtot", "log10_mni", "vej", "log10_mrp", "xmix"],
    "AnBa2022_linear": ["mtot", "mni", "vej", "mrp", "xmix"],
    "salt2": ["x0", "x1", "c"],
    "nugent-hyper": ["supernova_mag_boost", "supernova_mag_stretch"],
    # for Sr2023, the following array is not used
    "Sr2023": ["a_AG", "alpha_AG", "f_nu_host"],
    "Bu2022Ye": [
        "log10_mej_dyn",
        "vej_dyn",
        "Yedyn",
        "log10_mej_wind",
        "vej_wind",
        "KNtheta",
    ],
    "Bu2023Ye": [
        "log10_mej_dyn",
        "vej_dyn",
        "Yedyn",
        "log10_mej_wind",
        "vej_wind",
        "Yewind",
        "KNtheta",
    ],
    "LANLTP1": [
        "log10_mej_dyn",
        "vej_dyn",
        "log10_mej_wind",
        "vej_wind",
        "KNtheta",
    ],
    "LANLTP2": [
        "log10_mej_dyn",
        "vej_dyn",
        "log10_mej_wind",
        "vej_wind",
        "KNtheta",
    ],
    "LANLTS1": [
        "log10_mej_dyn",
        "vej_dyn",
        "log10_mej_wind",
        "vej_wind",
        "KNtheta",
    ],
    "LANLTS2": [
        "log10_mej_dyn",
        "vej_dyn",
        "log10_mej_wind",
        "vej_wind",
        "KNtheta",
    ],
    "HoNa2020": [
        "log10_mej",
        "vej_max",
        "vej_min",
        "vej_frac",
        "log10_kappa_low_vej",
        "log10_kappa_high_vej",
    ],
}


class LightCurveMixin:
    @property
    def citation(self):
        citation_dict = {
            **dict.fromkeys(
                ["LANLTP1", "LANLTP2", "LANLTS1", "LANLTS2"],
                ["https://arxiv.org/abs/2105.11543"],
            ),
            "Ka2017": ["https://arxiv.org/abs/1710.05463"],
            **dict.fromkeys(
                ["Bu2019lm", "Bu2019lm_sparse"],
                [
                    "https://arxiv.org/abs/2002.11355",
                    "https://arxiv.org/abs/1906.04205",
                ],
            ),
            **dict.fromkeys(
                [
                    "AnBa2022_sparse",
                    "AnBa2022_log",
                    "AnBa2022_linear",
                ],
                [
                    "https://arxiv.org/abs/2302.09226",
                    "https://arxiv.org/abs/2205.10421",
                ],
            ),
            "Bu2019nsbh": [
                "https://arxiv.org/abs/2009.07210",
                "https://arxiv.org/abs/1906.04205",
            ],
            **dict.fromkeys(
                ["Bu2022Ye", "Bu2023Ye", "Bu2022mv"],
                [
                    "https://arxiv.org/abs/2307.11080",
                    "https://arxiv.org/abs/1906.04205",
                ],
            ),
            "TrPi2018": ["https://arxiv.org/abs/1909.11691"],
            "Piro2021": ["https://arxiv.org/abs/2007.08543"],
            "Me2017": ["https://arxiv.org/abs/1910.01617"],
            "HoNa2020": [
                "https://arxiv.org/abs/1909.02581",
                "https://arxiv.org/abs/1206.2379",
            ],
            "Sr2023": [None],  # TODO: add citation,
            "nugent-hyper": [
                "https://sncosmo.readthedocs.io/en/stable/source-list.html"
            ],
            **dict.fromkeys(
                ["PL_BB_fixedT", "blackbody_fixedT", "synchrotron_powerlaw"],
                ["Analytical models"],
            ),
        }

        return {self.model: citation_dict[self.model]}


class GenericCombineLightCurveModel(LightCurveMixin):
    def __init__(self, models, sample_times):
        self.models = models
        self.sample_times = sample_times

    @property
    def citation(self):
        citations = []
        for model in self.models:
            citations.append(model.citation)

        return citations

    def generate_lightcurve(self, sample_times, parameters, return_all=False):
        total_lbol = np.zeros(len(sample_times))
        total_mag = {}
        mag_per_model = []
        lbol_per_model = []

        for model in self.models:
            lbol, mag = model.generate_lightcurve(sample_times, parameters)

            if np.sum(lbol) == 0.0 or len(np.isfinite(lbol)) == 0:
                return np.zeros(len(sample_times)), {}

            else:
                total_lbol += lbol
                mag_per_model.append(mag)
                lbol_per_model.append(lbol)

        filts = mag_per_model[0].keys()  # just get the first one

        for filt in filts:
            mAB_list = []
            for mag in mag_per_model:
                try:
                    mag_per_filt = utils.getFilteredMag(mag, filt)
                    # check if the mag_flit is valid, if it is 0
                    # meaning the filter is unknown to getFilteredMag
                    # and therefore, we have the mag_per_filt set to inf (0 flux)
                    if isinstance(mag_per_filt, int) and mag_per_filt == 0:
                        mag_per_filt = np.ones(len(sample_times)) * np.inf

                except KeyError:
                    mag_per_filt = np.ones(len(sample_times)) * np.inf

                mAB_list.append(-2.0 / 5.0 * ln10 * np.array(mag_per_filt))

            total_mag[filt] = -5.0 / 2.0 * logsumexp(mAB_list, axis=0) / ln10

        if return_all:
            return lbol_per_model, mag_per_model
        else:
            return total_lbol, total_mag


class SVDLightCurveModel(LightCurveMixin):
    """A light curve model object

    An object to evaluate the light curve across filters
    with a set of parameters given based on a prebuilt SVD

    Parameters
    ----------
    model: str
        Name of the model
    sample_times: np.array
        An arry of time for the light curve to be evaluted on
    svd_path: str
        Path to the svd directory
    parameter_conversion: func
        Function to convert from sampled parameters to parameters of the
        light curve model
    mag_ncoeff: int
        mag_ncoeff highest eigenvalues to be taken for mag's SVD evaluation
    lbol_ncoeff: int
        lbol_ncoeff highest eigenvalues to be taken for lbol's SVD evaluation
    filters : List[str]
        List of filters to create model for.
        Defaults to all available filters.

    Returns
    -------
    LightCurveModel: `nmma.em.model.SVDLightCurveModel`
        A light curve model object, able to evalute the light curve
        give a set of parameters
    """

    def __init__(
        self,
        model,
        sample_times,
        svd_path=None,
        parameter_conversion=None,
        mag_ncoeff=None,
        lbol_ncoeff=None,
        interpolation_type="sklearn_gp",
        model_parameters=None,
        filters=None,
        local_only=False,
    ):
        if model_parameters is None:
            assert (
                model in model_parameters_dict.keys()
            ), "Unknown model,please update model_parameters_dict at em/model.py"
            self.model_parameters = model_parameters_dict[model]
        else:
            self.model_parameters = model_parameters

        self.model = model
        self.sample_times = sample_times
        self.parameter_conversion = parameter_conversion

        self.mag_ncoeff = mag_ncoeff
        self.lbol_ncoeff = lbol_ncoeff

        self.interpolation_type = interpolation_type
        self.filters = filters

        self.svd_path = get_models_home(svd_path)

        # Some models have underscores. Keep those, but drop '_tf' if it exists
        model_name_components = model.split("_")
        if "tf" in model_name_components:
            model_name_components.remove("tf")
        core_model_name = "_".join(model_name_components)

        modelfile = os.path.join(self.svd_path, f"{core_model_name}.joblib")

        if self.interpolation_type == "sklearn_gp":
            if not local_only:
                _, model_filters = get_model(
                    self.svd_path, f"{self.model}", filters=filters
                )
                if filters is None and model_filters is not None:
                    self.filters = model_filters

            if os.path.isfile(modelfile):
                self.svd_mag_model = joblib.load(modelfile)

                if self.filters is None:
                    self.filters = list(self.svd_mag_model.keys())

                outdir = os.path.join(self.svd_path, f"{model}")
                for filt in self.filters:
                    outfile = os.path.join(outdir, f"{filt}.joblib")
                    if not os.path.isfile(outfile):
                        print(f"Could not find model file for filter {filt}")
                        if filt not in self.svd_mag_model:
                            self.svd_mag_model[filt] = {}
                        self.svd_mag_model[filt]["gps"] = None
                    else:
                        print(f"Loaded filter {filt}")
                        self.svd_mag_model[filt]["gps"] = joblib.load(outfile)
                self.svd_lbol_model = None
            else:
                if local_only:
                    raise ValueError(
                        f"Model file not found: {modelfile}\n If possible, try removing the --local-only flag and rerunning."
                    )
                else:
                    raise ValueError(f"Model file not found: {modelfile}")
        elif self.interpolation_type == "api_gp":
            from .training import load_api_gp_model

            if os.path.isfile(modelfile):
                self.svd_mag_model = joblib.load(modelfile)
                for filt in self.filters:
                    for ii in range(len(self.svd_mag_model[filt]["gps"])):
                        self.svd_mag_model[filt]["gps"][ii] = load_api_gp_model(
                            self.svd_mag_model[filt]["gps"][ii]
                        )
                self.svd_lbol_model = None
        elif self.interpolation_type == "tensorflow":
            import tensorflow as tf

            tf.get_logger().setLevel("ERROR")
            from keras.models import load_model

            if not local_only:
                _, model_filters = get_model(
                    self.svd_path, f"{self.model}_tf", filters=filters
                )
                if filters is None:
                    self.filters = model_filters

            if os.path.isfile(modelfile):
                self.svd_mag_model = joblib.load(modelfile)

                if self.filters is None:
                    self.filters = list(self.svd_mag_model.keys())

                outdir = os.path.join(self.svd_path, f"{model}_tf")
                for filt in self.filters:
                    outfile = os.path.join(outdir, f"{filt}.h5")
                    if not os.path.isfile(outfile):
                        print(f"Could not find model file for filter {filt}")
                        if filt not in self.svd_mag_model:
                            self.svd_mag_model[filt] = {}
                        self.svd_mag_model[filt]["model"] = None
                    else:
                        print(f"Loaded filter {filt}")
                        self.svd_mag_model[filt]["model"] = load_model(
                            outfile, compile=False
                        )
                        self.svd_mag_model[filt]["model"].compile(
                            optimizer="adam", loss="mse"
                        )
                self.svd_lbol_model = None
            else:
                if local_only:
                    raise ValueError(
                        f"Model file not found: {modelfile}\n If possible, try removing the --local-only flag and rerunning."
                    )
                else:
                    raise ValueError(f"Model file not found: {modelfile}")
        else:
            return ValueError("--interpolation-type must be sklearn_gp or tensorflow")

    def __repr__(self):
        return self.__class__.__name__ + "(model={0}, svd_path={1})".format(
            self.model, self.svd_path
        )

    def observation_angle_conversion(self, parameters):
        if "KNtheta" not in parameters:
            parameters["KNtheta"] = (
                parameters.get("inclination_EM", 0.0) * 180.0 / np.pi
            )
        return parameters

    def generate_lightcurve(self, sample_times, parameters):
        if self.parameter_conversion:
            new_parameters = parameters.copy()
            new_parameters, _ = self.parameter_conversion(new_parameters)
        else:
            new_parameters = parameters.copy()

        new_parameters = self.observation_angle_conversion(new_parameters)

        parameters_list = []
        for parameter_name in self.model_parameters:
            try:
                parameters_list.append(new_parameters[parameter_name])
            except KeyError:
                if "log10" in parameter_name:
                    parameters_list.append(
                        np.log10(new_parameters[parameter_name.replace("log10_", "")])
                    )
                else:
                    parameters_list.append(
                        10 ** new_parameters[f"log10_{parameter_name}"]
                    )

        z = utils.getRedShift(new_parameters)

        _, lbol, mag = utils.calc_lc(
            sample_times / (1.0 + z),
            parameters_list,
            svd_mag_model=self.svd_mag_model,
            svd_lbol_model=self.svd_lbol_model,
            mag_ncoeff=self.mag_ncoeff,
            lbol_ncoeff=self.lbol_ncoeff,
            interpolation_type=self.interpolation_type,
            filters=self.filters,
        )
        lbol *= 1.0 + z
        for filt in mag.keys():
            mag[filt] -= 2.5 * np.log10(1.0 + z)

        # calculate extinction
        filts = mag.keys()
        _, lambdas = utils.get_default_filts_lambdas(filters=filts)
        nu_0s = scipy.constants.c / lambdas

        try:
            Ebv = new_parameters["Ebv"]
            if Ebv != 0.0:
                ext = utils.extinctionFactorP92SMC(nu_0s, Ebv, z)
                ext_mag = -2.5 * np.log10(ext)
            else:
                ext_mag = np.zeros(len(nu_0s))

            # apply extinction
            for ext_mag_per_filt, filt in zip(ext_mag, filts):
                mag[filt] += ext_mag_per_filt
        except KeyError:
            pass

        return lbol, mag

    def generate_spectra(self, sample_times, wavelengths, parameters):
        if self.parameter_conversion:
            new_parameters = parameters.copy()
            new_parameters, _ = self.parameter_conversion(new_parameters, [])
        else:
            new_parameters = parameters.copy()

        new_parameters = self.observation_angle_conversion(new_parameters)

        parameters_list = []
        for parameter_name in self.model_parameters:
            try:
                parameters_list.append(new_parameters[parameter_name])
            except KeyError:
                parameters_list.append(10 ** new_parameters[f"log10_{parameter_name}"])

        z = utils.getRedShift(new_parameters)

        _, _, spec = utils.calc_lc(
            sample_times / (1.0 + z),
            parameters_list,
            svd_mag_model=self.svd_mag_model,
            svd_lbol_model=self.svd_lbol_model,
            mag_ncoeff=self.mag_ncoeff,
            lbol_ncoeff=self.lbol_ncoeff,
            interpolation_type=self.interpolation_type,
            filters=wavelengths,
        )

        return spec


class GRBLightCurveModel(LightCurveMixin):
    def __init__(
        self,
        sample_times,
        parameter_conversion=None,
        model="TrPi2018",
        resolution=12,
        jetType=0,
        filters=None,
        energy_injection=False,
    ):
        """A light curve model object

        An object to evaluted the GRB light curve across filters
        with a set of parameters given based on afterglowpy

        Parameters
        ----------
        sample_times: np.array
            An arry of time for the light curve to be evaluted on

        Returns
        -------
        LightCurveModel: `nmma.em.model.GRBLightCurveModel`
            A light curve model onject, able to evaluted the light curve
            give a set of parameters
        """

        assert (
            model in model_parameters_dict.keys()
        ), "Unknown model,please update model_parameters_dict at em/model.py"
        self.model = model
        self.model_parameters = model_parameters_dict[model]
        self.sample_times = sample_times
        self.parameter_conversion = parameter_conversion
        self.resolution = resolution
        self.jetType = jetType
        self.filters = filters
        self.energy_injection = energy_injection

    def __repr__(self):
        return self.__class__.__name__ + "(model={0})".format(self.model)

    def generate_lightcurve(self, sample_times, parameters):
        if self.parameter_conversion:
            new_parameters = parameters.copy()
            new_parameters, _ = self.parameter_conversion(new_parameters)
        else:
            new_parameters = parameters.copy()

        z = utils.getRedShift(new_parameters)

        default_parameters = {"xi_N": 1.0, "d_L": 3.086e19}  # 10pc in cm

        grb_param_dict = {}

        for key in default_parameters.keys():
            if key not in new_parameters.keys():
                grb_param_dict[key] = default_parameters[key]
            else:
                grb_param_dict[key] = new_parameters[key]

        grb_param_dict["jetType"] = self.jetType
        grb_param_dict["specType"] = 0

        grb_param_dict["thetaObs"] = new_parameters["inclination_EM"]
        grb_param_dict["thetaCore"] = new_parameters["thetaCore"]
        grb_param_dict["n0"] = 10 ** new_parameters["log10_n0"]
        grb_param_dict["p"] = new_parameters["p"]
        grb_param_dict["epsilon_e"] = 10 ** new_parameters["log10_epsilon_e"]
        grb_param_dict["epsilon_B"] = 10 ** new_parameters["log10_epsilon_B"]
        grb_param_dict["z"] = z

        # energy handling
        if not self.energy_injection:
            grb_param_dict["E0"] = 10 ** new_parameters["log10_E0"]
        else:
            # additional parameters
            energy_injection_params = [
                "energy_exponential",
                "log10_Eend",
                "t_start",
                "injection_duration",
            ]
            assert all(key in new_parameters for key in energy_injection_params)
            # fetch parameters
            log10_Eend = new_parameters["log10_Eend"]
            t_start = new_parameters["t_start"]
            t_end = new_parameters["t_start"] + new_parameters["injection_duration"]
            energy_exponential = new_parameters["energy_exponential"]
            # populate the E0 along the sample_times
            log10_Estart = log10_Eend + energy_exponential * np.log10(t_start / t_end)
            log10_E0 = log10_Eend * np.ones(len(sample_times))
            # now adjust the log10_E0
            log10_E0[sample_times <= t_start] = log10_Estart
            log10_E0[sample_times >= t_end] = log10_Eend
            mask = (sample_times > t_start) * (sample_times < t_end)
            time_scale = np.log10(sample_times / t_end)
            log10_E0[mask] = log10_Eend + energy_exponential * time_scale[mask]
            # now place the array into the param_dict
            grb_param_dict["E0"] = 10**log10_E0
        # make sure L0, q and ts are also passed
        for param in ["L0", "q", "ts"]:
            if param in new_parameters:
                grb_param_dict[param] = new_parameters[param]

        if "thetaWing" in new_parameters:
            grb_param_dict["thetaWing"] = new_parameters["thetaWing"]
            if (
                new_parameters["thetaWing"] / new_parameters["thetaCore"]
                > self.resolution
            ):
                return np.zeros(len(sample_times)), {}

        if grb_param_dict["epsilon_e"] + grb_param_dict["epsilon_B"] > 1.0:
            return np.zeros(len(sample_times)), {}

        if self.jetType == 1 or self.jetType == 4:
            grb_param_dict["b"] = new_parameters["b"]
            if "thetaWing" in new_parameters:
                grb_param_dict["thetaWing"] = new_parameters["thetaWing"]

        Ebv = new_parameters.get("Ebv", 0.0)

        _, lbol, mag = utils.grb_lc(
            sample_times, Ebv, grb_param_dict, filters=self.filters
        )
        return lbol, mag


class KilonovaGRBLightCurveModel(LightCurveMixin):
    def __init__(
        self,
        sample_times,
        kilonova_kwargs,
        parameter_conversion=None,
        GRB_resolution=12,
        jetType=0,
    ):
        self.sample_times = sample_times
        self.parameter_conversion = kilonova_kwargs["parameter_conversion"]

        kilonova_kwargs["parameter_conversion"] = parameter_conversion
        kilonova_kwargs["sample_times"] = sample_times

        self.kilonova_lightcurve_model = SVDLightCurveModel(**kilonova_kwargs)
        self.grb_lightcurve_model = GRBLightCurveModel(
            sample_times,
            parameter_conversion,
            resolution=GRB_resolution,
            jetType=jetType,
        )

    def __repr__(self):
        details = "(grb model using afterglowpy with kilonova model {0})".format(
            self.kilonova_lightcurve_model
        )
        return self.__class__.__name__ + details

    @property
    def citation(self):
        citations = [self.grb_lightcurve_model.citation]
        citations.append(self.kilonova_lightcurve_model.citation)

        return citations

    def observation_angle_conversion(self, parameters):
        parameters["KNtheta"] = parameters["inclination_EM"] * 180.0 / np.pi

        return parameters

    def generate_lightcurve(self, sample_times, parameters):
        total_lbol = np.zeros(len(sample_times))
        total_mag = {}

        if self.parameter_conversion:
            new_parameters = parameters.copy()
            new_parameters, _ = self.parameter_conversion(new_parameters)
        else:
            new_parameters = parameters.copy()

        new_parameters = self.observation_angle_conversion(new_parameters)

        grb_lbol, grb_mag = self.grb_lightcurve_model.generate_lightcurve(
            sample_times, new_parameters
        )

        if np.sum(grb_lbol) == 0.0 or len(np.isfinite(grb_lbol)) == 0:
            return total_lbol, total_mag

        (
            kilonova_lbol,
            kilonova_mag,
        ) = self.kilonova_lightcurve_model.generate_lightcurve(
            sample_times, new_parameters
        )

        for filt in grb_mag.keys():
            grb_mAB = grb_mag[filt]

            try:
                kilonova_mAB = utils.getFilteredMag(kilonova_mag, filt)

                # check if the kilnova_mAB is valid, if it is 0
                # meaning the filter is unknown to getFilteredMag
                # and therefore, we have the kilonova_mAB set to inf (0 flux)

                if isinstance(kilonova_mAB, int) and kilonova_mAB == 0:
                    kilonova_mAB = np.inf

                total_mag[filt] = (
                    -5.0
                    / 2.0
                    * logsumexp(
                        [-2.0 / 5.0 * ln10 * grb_mAB, -2.0 / 5.0 * ln10 * kilonova_mAB],
                        axis=0,
                    )
                    / ln10
                )
            except KeyError:
                total_mag[filt] = copy.deepcopy(grb_mAB)

        total_lbol = grb_lbol + kilonova_lbol

        return total_lbol, total_mag


class HostGalaxyLightCurveModel(LightCurveMixin):
    def __init__(
        self,
        sample_times,
        model="Sr2023",
        parameter_conversion=None,
        filters=None,
    ):
        """A light curve model object

        An object to evaluted the host galaxy light curve across filters
        with a set of parameters given

        Based on arxiv:2303.12849

        Parameters
        ----------
        sample_times: np.array
            An arry of time for the light curve to be evaluted on

        Returns
        -------
        LightCurveModel: `nmma.em.model.HostGalaxyLightCurveModel`
            A light curve model onject, able to evaluted the light curve
            give a set of parameters
        """

        self.sample_times = sample_times
        self.parameter_conversion = parameter_conversion
        self.model = model
        self.filters = filters

    def __repr__(self):
        return self.__class__.__name__ + "(model={self.model})"

    def generate_lightcurve(self, sample_times, parameters):
        if self.parameter_conversion:
            new_parameters = parameters.copy()
            new_parameters, _ = self.parameter_conversion(new_parameters, [])
        else:
            new_parameters = parameters.copy()

        mag = {}
        lbol = [1e33]  # just a random number
        alpha = new_parameters["alpha_AG"]
        for filt in self.filters:
            # assumed to be in unit of muJy
            a_AG = new_parameters[f"a_AG_{filt}"]
            f_nu_filt = new_parameters[f"f_nu_{filt}"]
            flux_per_filt = a_AG * np.power(sample_times, -alpha) + f_nu_filt

            mag[filt] = -2.5 * np.log10(flux_per_filt) + 23.9

        return lbol, mag


class SupernovaLightCurveModel(LightCurveMixin):
    def __init__(
        self,
        sample_times,
        parameter_conversion=None,
        model="nugent-hyper",
        filters=None,
    ):
        """A light curve model object

        An object to evaluted the supernova light curve across filters
        with a set of parameters given based on sncosmo

        Parameters
        ----------
        sample_times: np.array
            An arry of time for the light curve to be evaluted on

        Returns
        -------
        LightCurveModel: `nmma.em.model.GRBLightCurveModel`
            A light curve model onject, able to evaluted the light curve
            give a set of parameters
        """

        self.sample_times = sample_times
        self.parameter_conversion = parameter_conversion
        self.model = model
        self.filters = filters

    def __repr__(self):
        return self.__class__.__name__ + "(model={self.model})"

    def generate_lightcurve(self, sample_times, parameters):
        if self.parameter_conversion:
            new_parameters = parameters.copy()
            new_parameters, _ = self.parameter_conversion(new_parameters, [])
        else:
            new_parameters = parameters.copy()

        z = utils.getRedShift(new_parameters)

        Ebv = new_parameters.get("Ebv", 0.0)
        sketch = new_parameters.get("supernova_mag_stretch", 1)

        tt, lbol, mag = utils.sn_lc(
            sample_times / sketch,
            z,
            Ebv,
            model_name=self.model,
            parameters=new_parameters,
            filters=self.filters,
        )

        if "supernova_mag_boost" in parameters:
            for filt in mag.keys():
                mag[filt] += parameters["supernova_mag_boost"]

        return lbol, mag


class SupernovaGRBLightCurveModel(LightCurveMixin):
    def __init__(
        self,
        sample_times,
        parameter_conversion=None,
        SNmodel="nugent-hyper",
        GRB_resolution=12,
        jetType=0,
    ):
        self.sample_times = sample_times

        self.grb_lightcurve_model = GRBLightCurveModel(
            sample_times,
            parameter_conversion,
            resolution=GRB_resolution,
            jetType=jetType,
        )
        self.supernova_lightcurve_model = SupernovaLightCurveModel(
            sample_times, parameter_conversion, model=SNmodel
        )

    def __repr__(self):
        details = "(grb model using afterglowpy with supernova model nugent-hyper)"
        return self.__class__.__name__ + details

    @property
    def citation(self):
        citations = [
            self.grb_lightcurve_model.citation,
            self.supernova_lightcurve_model.citation,
        ]

        return citations

    def generate_lightcurve(self, sample_times, parameters):
        total_lbol = np.zeros(len(sample_times))
        total_mag = {}

        grb_lbol, grb_mag = self.grb_lightcurve_model.generate_lightcurve(
            sample_times, parameters
        )

        if np.sum(grb_lbol) == 0.0 or len(np.isfinite(grb_lbol)) == 0:
            return total_lbol, total_mag

        (
            supernova_lbol,
            supernova_mag,
        ) = self.supernova_lightcurve_model.generate_lightcurve(
            sample_times, parameters
        )

        if np.sum(supernova_lbol) == 0.0 or len(np.isfinite(supernova_lbol)) == 0:
            return total_lbol, total_mag

        for filt in grb_mag.keys():
            grb_mAB = grb_mag[filt]
            supernova_mAB = supernova_mag[filt]

            total_mag[filt] = (
                -5.0
                / 2.0
                * logsumexp(
                    [-2.0 / 5.0 * ln10 * grb_mAB, -2.0 / 5.0 * ln10 * supernova_mAB],
                    axis=0,
                )
                / ln10
            )

        total_lbol = grb_lbol + supernova_lbol

        return total_lbol, total_mag


class ShockCoolingLightCurveModel(LightCurveMixin):
    def __init__(
        self, sample_times, parameter_conversion=None, model="Piro2021", filters=None
    ):
        """A light curve model object

        An object to evaluted the shock cooling light curve across filters

        Parameters
        ----------
        sample_times: np.array
            An arry of time for the light curve to be evaluted on

        Returns
        -------
        LightCurveModel: `nmma.em.model.ShockCoolingLightCurveModel`
            A light curve model onject, able to evaluted the light curve
            give a set of parameters
        """

        assert (
            model in model_parameters_dict.keys()
        ), "Unknown model,please update model_parameters_dict at em/model.py"
        self.model = model
        self.model_parameters = model_parameters_dict[model]
        self.sample_times = sample_times
        self.parameter_conversion = parameter_conversion
        self.filters = filters

    def __repr__(self):
        return self.__class__.__name__ + "(model={0})".format(self.model)

    def generate_lightcurve(self, sample_times, parameters):
        if self.parameter_conversion:
            new_parameters = parameters.copy()
            new_parameters, _ = self.parameter_conversion(new_parameters, [])
        else:
            new_parameters = parameters.copy()

        z = utils.getRedShift(new_parameters)
        Ebv = new_parameters.get("Ebv", 0.0)

        param_dict = {}
        for key in self.model_parameters:
            param_dict[key] = new_parameters[key]
        param_dict["z"] = z
        param_dict["Ebv"] = Ebv

        _, lbol, mag = utils.sc_lc(sample_times, param_dict, filters=self.filters)
        return lbol, mag


class SupernovaShockCoolingLightCurveModel(LightCurveMixin):
    def __init__(self, sample_times, parameter_conversion=None, filters=None):
        self.sample_times = sample_times

        self.sc_lightcurve_model = ShockCoolingLightCurveModel(
            sample_times, parameter_conversion
        )
        self.supernova_lightcurve_model = SupernovaLightCurveModel(
            sample_times, parameter_conversion
        )
        self.filters = filters

    def __repr__(self):
        details = (
            "(shock cooling model using Piro2021 with supernova model nugent-hyper)"
        )
        return self.__class__.__name__ + details

    @property
    def citation(self):
        citations = [
            self.sc_lightcurve_model.citation,
            self.supernova_lightcurve_model.citation,
        ]

        return citations

    def generate_lightcurve(self, sample_times, parameters):
        total_lbol = np.zeros(len(sample_times))
        total_mag = {}

        sc_lbol, sc_mag = self.sc_lightcurve_model.generate_lightcurve(
            sample_times,
            parameters,
            filters=self.filters,
        )

        if np.sum(sc_lbol) == 0.0 or len(np.isfinite(sc_lbol)) == 0:
            return total_lbol, total_mag

        (
            supernova_lbol,
            supernova_mag,
        ) = self.supernova_lightcurve_model.generate_lightcurve(
            sample_times,
            parameters,
            filters=self.filters,
        )

        if np.sum(supernova_lbol) == 0.0 or len(np.isfinite(supernova_lbol)) == 0:
            return total_lbol, total_mag

        for filt in sc_mag.keys():
            sc_mAB = sc_mag[filt]
            supernova_mAB = supernova_mag[filt]

            total_mag[filt] = (
                -5.0
                / 2.0
                * logsumexp(
                    [-2.0 / 5.0 * ln10 * sc_mAB, -2.0 / 5.0 * ln10 * supernova_mAB],
                    axis=0,
                )
                / ln10
            )

        total_lbol = sc_lbol + supernova_lbol

        return total_lbol, total_mag


class SimpleKilonovaLightCurveModel(LightCurveMixin):
    def __init__(
        self, sample_times, parameter_conversion=None, model="Me2017", filters=None
    ):
        """A light curve model object

        An object to evaluted the kilonova (with Me2017 or HoNa2020) light curve across filters

        Parameters
        ----------
        sample_times: np.array
            An arry of time for the light curve to be evaluted on

        Returns
        -------
        LightCurveModel: `nmma.em.model.SimpleKilonovaLightCurveModel`
            A light curve model onject, able to evaluted the light curve
            give a set of parameters
        """

        assert (
            model in model_parameters_dict.keys()
        ), "Unknown model,please update model_parameters_dict at em/model.py"
        self.model = model
        self.model_parameters = model_parameters_dict[model]
        self.sample_times = sample_times
        self.parameter_conversion = parameter_conversion
        self.filters = filters

    def __repr__(self):
        return self.__class__.__name__ + "(model={0})".format(self.model)

    def generate_lightcurve(self, sample_times, parameters):
        if self.parameter_conversion:
            new_parameters = parameters.copy()
            new_parameters, _ = self.parameter_conversion(new_parameters, [])
        else:
            new_parameters = parameters.copy()

        z = utils.getRedShift(new_parameters)
        Ebv = new_parameters.get("Ebv", 0.0)

        param_dict = {}
        for key in self.model_parameters:

            param_dict[key] = new_parameters[key]
        param_dict["z"] = z
        param_dict["Ebv"] = Ebv

        if self.model == "Me2017":
            _, lbol, mag = utils.metzger_lc(
                sample_times, param_dict, filters=self.filters
            )
        elif self.model == "HoNa2020":
            default_parameters = {"n": 4.5}
            for key in default_parameters.keys():
                if key not in param_dict.keys():
                    param_dict[key] = default_parameters[key]
            # now handle the velocities
            vej_max = param_dict["vej_max"]
            vej_min = param_dict["vej_min"]
            vej_range = vej_max - vej_min
            vej = param_dict["vej_frac"] * vej_range + vej_min
            # calculate the temperature and luminosity to feed into the blackbody radiation calculation
            L, T, _ = utils.lightcurve_HoNa(
                sample_times,
                10 ** param_dict["log10_mej"],
                [param_dict["vej_min"], vej, param_dict["vej_max"]],
                [
                    10 ** param_dict["log10_kappa_low_vej"],
                    10 ** param_dict["log10_kappa_high_vej"],
                ],
                param_dict["n"],
            )
            param_dict["bb_luminosity"] = L.cgs.value
            param_dict["temperature"] = T.si.value
            _, lbol, mag = utils.blackbody_constant_temperature(
                sample_times, param_dict, filters=self.filters
            )
        elif self.model == "PL_BB_fixedT":
            _, lbol, mag = utils.powerlaw_blackbody_constant_temperature_lc(
                sample_times, param_dict, filters=self.filters
            )
        elif self.model == "blackbody_fixedT":
            _, lbol, mag = utils.blackbody_constant_temperature(
                sample_times, param_dict, filters=self.filters
            )
        elif self.model == "synchrotron_powerlaw":
            _, lbol, mag = utils.synchrotron_powerlaw(
                sample_times, param_dict, filters=self.filters
            )
            # remove the distance modulus for the synchrotron powerlaw
            # as the reference flux is defined at the observer
            dist_mod = 5.0 * np.log10(new_parameters["luminosity_distance"] * 1e6 / 10)
            for filt in mag.keys():
                mag[filt] -= dist_mod

        return lbol, mag


def create_light_curve_model_from_args(
    model_name_arg,
    args,
    sample_times,
    filters=None,
    sample_over_Hubble=False,
):
    # check if sampling over Hubble,
    # if so define the parameter_conversion accordingly
    if sample_over_Hubble:

        def parameter_conversion(converted_parameters, added_keys):
            if "luminosity_distance" not in converted_parameters:
                Hubble_constant = converted_parameters["Hubble_constant"]
                redshift = converted_parameters["redshift"]
                # redshift is supposed to be dimensionless
                # Hubble constant is supposed to be km/s/Mpc
                distance = redshift / Hubble_constant * scipy.constants.c / 1e3
                converted_parameters["luminosity_distance"] = distance
                added_keys = added_keys + ["luminosity_distance"]
            return converted_parameters, added_keys

    else:
        parameter_conversion = None

    models = []
    # check if there are more than one model
    if "," in model_name_arg:
        print("Running with combination of multiple light curve models")
        model_names = model_name_arg.split(",")
    else:
        model_names = [model_name_arg]

    sncosmo_names = [val["name"] for val in _SOURCES.get_loaders_metadata()]

    for model_name in model_names:
        if model_name == "TrPi2018":
            lc_model = GRBLightCurveModel(
                sample_times=sample_times,
                resolution=args.grb_resolution,
                jetType=args.jet_type,
                parameter_conversion=parameter_conversion,
                filters=filters,
                energy_injection=args.energy_injection,
            )

        elif model_name in sncosmo_names:
            lc_model = SupernovaLightCurveModel(
                sample_times=sample_times,
                model=model_name,
                parameter_conversion=parameter_conversion,
                filters=filters,
            )

        elif model_name == "Piro2021":
            lc_model = ShockCoolingLightCurveModel(
                sample_times=sample_times,
                parameter_conversion=parameter_conversion,
                filters=filters,
            )

        elif (
            model_name == "Me2017"
            or model_name == "PL_BB_fixedT"
            or model_name == "HoNa2020"
        ):
            lc_model = SimpleKilonovaLightCurveModel(
                sample_times=sample_times,
                model=model_name,
                parameter_conversion=parameter_conversion,
                filters=filters,
            )

        elif model_name == "Sr2023":
            lc_model = HostGalaxyLightCurveModel(
                sample_times=sample_times,
                parameter_conversion=parameter_conversion,
                filters=filters,
            )

        else:
            if hasattr(args, "local_only"):
                local_only = args.local_only
            else:
                local_only = False

            lc_kwargs = dict(
                model=model_name,
                sample_times=sample_times,
                svd_path=args.svd_path,
                mag_ncoeff=args.svd_mag_ncoeff,
                lbol_ncoeff=args.svd_lbol_ncoeff,
                interpolation_type=args.interpolation_type,
                parameter_conversion=parameter_conversion,
                filters=filters,
                local_only=local_only,
            )
            lc_model = SVDLightCurveModel(**lc_kwargs)

        models.append(lc_model)

    if len(models) > 1:
        light_curve_model = GenericCombineLightCurveModel(models, sample_times)
    else:
        light_curve_model = models[0]

    return model_names, models, light_curve_model
