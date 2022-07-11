from __future__ import division

import os
import pickle

import numpy as np
from scipy.special import logsumexp

from . import utils

ln10 = np.log(10)

# As different KN models have very different parameters,
# we need a dict to keep track for them. Remember, the order matters

model_parameters_dict = {
    "Bu2019nsbh": ["log10_mej_dyn", "log10_mej_wind", "KNtheta"],
    "Bu2019lm": ["log10_mej_dyn", "log10_mej_wind", "KNphi", "KNtheta"],
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
    "Me2017": ["log10_Mej", "log10_vej", "beta", "log10_kappa_r"],
    "Bu2022mv": ["log10_mej_dyn", "vej_dyn", "log10_mej_wind", "vej_wind", "KNtheta"],
    "PL_BB_fixedT": ["bb_luminosity", "temperature", "beta", "powerlaw_mag"],
    "CV": ["example_num"],
    "AnBa2022_sparse": ["mrp", "xmix"],
    "salt2": ["x0", "x1", "c"],
    "nugent-hyper": ["supernova_mag_boost"],
}


class GenericCombineLightCurveModel(object):
    def __init__(self, models, sample_times):
        self.models = models
        self.sample_times = sample_times

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
                mAB_list.append(-2.0 / 5.0 * ln10 * np.array(mag[filt]))

            total_mag[filt] = -5.0 / 2.0 * logsumexp(mAB_list, axis=0) / ln10

        if return_all:
            return lbol_per_model, mag_per_model
        else:
            return total_lbol, total_mag


class SVDLightCurveModel(object):
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
    ):

        if model_parameters is None:
            assert model in model_parameters_dict.keys(), (
                "Unknown model," "please update model_parameters_dict at em/model.py"
            )
            self.model_parameters = model_parameters_dict[model]
        else:
            self.model_parameters = model_parameters

        self.model = model
        self.sample_times = sample_times
        self.parameter_conversion = parameter_conversion

        self.mag_ncoeff = mag_ncoeff
        self.lbol_ncoeff = lbol_ncoeff

        self.interpolation_type = interpolation_type

        if svd_path is None:
            self.svd_path = os.path.join(os.path.dirname(__file__), "svdmodels")
        else:
            self.svd_path = svd_path

        if self.interpolation_type == "sklearn_gp":
            modelfile = os.path.join(self.svd_path, "{0}.pkl".format(model))
            if os.path.isfile(modelfile):
                with open(modelfile, "rb") as handle:
                    self.svd_mag_model = pickle.load(handle)
                self.svd_lbol_model = None
            else:
                mag_modelfile = os.path.join(self.svd_path, "{0}_mag.pkl".format(model))
                with open(mag_modelfile, "rb") as handle:
                    self.svd_mag_model = pickle.load(handle)
                lbol_modelfile = os.path.join(
                    self.svd_path, "{0}_lbol.pkl".format(model)
                )
                with open(lbol_modelfile, "rb") as handle:
                    self.svd_lbol_model = pickle.load(handle)
        elif self.interpolation_type == "api_gp":
            from .training import load_api_gp_model

            modelfile = os.path.join(self.svd_path, "{0}_api.pkl".format(model))
            if os.path.isfile(modelfile):
                with open(modelfile, "rb") as handle:
                    self.svd_mag_model = pickle.load(handle)
                for filt in self.svd_mag_model.keys():
                    for ii in range(len(self.svd_mag_model[filt]["gps"])):
                        self.svd_mag_model[filt]["gps"][ii] = load_api_gp_model(
                            self.svd_mag_model[filt]["gps"][ii]
                        )
                self.svd_lbol_model = None
        elif self.interpolation_type == "tensorflow":
            import tensorflow as tf

            tf.get_logger().setLevel("ERROR")
            from tensorflow.keras.models import load_model

            modelfile = os.path.join(self.svd_path, "{0}_tf.pkl".format(model))
            if os.path.isfile(modelfile):
                with open(modelfile, "rb") as handle:
                    self.svd_mag_model = pickle.load(handle)
                outdir = modelfile.replace(".pkl", "")
                for filt in self.svd_mag_model.keys():
                    outfile = os.path.join(outdir, f"{filt}.h5")
                    self.svd_mag_model[filt]["model"] = load_model(outfile)
                self.svd_lbol_model = None
            else:
                mag_modelfile = os.path.join(
                    self.svd_path, "{0}_mag_tf.pkl".format(model)
                )
                with open(mag_modelfile, "rb") as handle:
                    self.svd_mag_model = pickle.load(handle)
                outdir = mag_modelfile.replace(".pkl", "")
                for filt in self.svd_mag_model.keys():
                    outfile = os.path.join(outdir, f"{filt}.h5")
                    self.svd_mag_model[filt]["model"] = load_model(outfile)

                lbol_modelfile = os.path.join(
                    self.svd_path, "{0}_lbol_tf.pkl".format(model)
                )
                with open(lbol_modelfile, "rb") as handle:
                    self.svd_lbol_model = pickle.load(handle)

                outdir = lbol_modelfile.replace(".pkl", "")
                outfile = os.path.join(outdir, "model.h5")
                self.svd_lbol_model["model"] = load_model(outfile)
        else:
            return ValueError(
                "self.interpolation_type must be sklearn_gp or tensorflow"
            )

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
            parameters_list.append(new_parameters[parameter_name])

        z = utils.getRedShift(new_parameters)

        _, lbol, mag = utils.calc_lc(
            sample_times / (1.0 + z),
            parameters_list,
            svd_mag_model=self.svd_mag_model,
            svd_lbol_model=self.svd_lbol_model,
            mag_ncoeff=self.mag_ncoeff,
            lbol_ncoeff=self.lbol_ncoeff,
            interpolation_type=self.interpolation_type,
        )
        lbol *= 1.0 + z
        for filt in mag.keys():
            mag[filt] -= 2.5 * np.log10(1.0 + z)

        return lbol, mag

    def generate_spectra(self, sample_times, wavelengths, parameters):
        if self.parameter_conversion:
            new_parameters = parameters.copy()
            new_parameters, _ = self.parameter_conversion(new_parameters)
        else:
            new_parameters = parameters.copy()

        new_parameters = self.observation_angle_conversion(new_parameters)

        parameters_list = []
        for parameter_name in self.model_parameters:
            parameters_list.append(new_parameters[parameter_name])

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


class GRBLightCurveModel(object):
    def __init__(
        self,
        sample_times,
        parameter_conversion=None,
        model="TrPi2018",
        resolution=12,
        jetType=0,
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

        assert model in model_parameters_dict.keys(), (
            "Unknown model," "please update model_parameters_dict at em/model.py"
        )
        self.model = model
        self.model_parameters = model_parameters_dict[model]
        self.sample_times = sample_times
        self.parameter_conversion = parameter_conversion
        self.resolution = resolution
        self.jetType = jetType

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
        grb_param_dict["E0"] = 10 ** new_parameters["log10_E0"]
        grb_param_dict["thetaCore"] = new_parameters["thetaCore"]
        grb_param_dict["thetaWing"] = new_parameters["thetaWing"]
        grb_param_dict["n0"] = 10 ** new_parameters["log10_n0"]
        grb_param_dict["p"] = new_parameters["p"]
        grb_param_dict["epsilon_e"] = 10 ** new_parameters["log10_epsilon_e"]
        grb_param_dict["epsilon_B"] = 10 ** new_parameters["log10_epsilon_B"]
        grb_param_dict["z"] = z

        if self.jetType == 1 or self.jetType == 4:
            grb_param_dict["b"] = new_parameters["b"]

        if new_parameters["thetaWing"] / new_parameters["thetaCore"] > self.resolution:
            return np.zeros(len(sample_times)), {}

        Ebv = new_parameters.get("Ebv", 0.0)

        _, lbol, mag = utils.grb_lc(sample_times, Ebv, grb_param_dict)
        return lbol, mag


class KilonovaGRBLightCurveModel(object):
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
            kilonova_mAB = kilonova_mag[filt]

            total_mag[filt] = (
                -5.0
                / 2.0
                * logsumexp(
                    [-2.0 / 5.0 * ln10 * grb_mAB, -2.0 / 5.0 * ln10 * kilonova_mAB],
                    axis=0,
                )
                / ln10
            )

        total_lbol = grb_lbol + kilonova_lbol

        return total_lbol, total_mag


class SupernovaLightCurveModel(object):
    def __init__(self, sample_times, parameter_conversion=None, model="nugent-hyper"):
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

    def __repr__(self):
        return self.__class__.__name__ + "(model={self.model})"

    def generate_lightcurve(self, sample_times, parameters):

        if self.parameter_conversion:
            new_parameters = parameters.copy()
            new_parameters, _ = self.parameter_conversion(new_parameters)
        else:
            new_parameters = parameters.copy()

        z = utils.getRedShift(new_parameters)

        Ebv = new_parameters.get("Ebv", 0.0)

        _, lbol, mag = utils.sn_lc(
            sample_times, z, Ebv, model_name=self.model, parameters=new_parameters
        )

        if self.model == "nugent-hyper":
            for filt in mag.keys():
                mag[filt] += parameters["supernova_mag_boost"]

        return lbol, mag


class SupernovaGRBLightCurveModel(object):
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


class ShockCoolingLightCurveModel(object):
    def __init__(self, sample_times, parameter_conversion=None, model="Piro2021"):
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

        assert model in model_parameters_dict.keys(), (
            "Unknown model," "please update model_parameters_dict at em/model.py"
        )
        self.model = model
        self.model_parameters = model_parameters_dict[model]
        self.sample_times = sample_times
        self.parameter_conversion = parameter_conversion

    def __repr__(self):
        return self.__class__.__name__ + "(model={0})".format(self.model)

    def generate_lightcurve(self, sample_times, parameters):

        if self.parameter_conversion:
            new_parameters = parameters.copy()
            new_parameters, _ = self.parameter_conversion(new_parameters)
        else:
            new_parameters = parameters.copy()

        z = utils.getRedShift(new_parameters)
        Ebv = new_parameters.get("Ebv", 0.0)

        param_dict = {}
        for key in self.model_parameters:
            param_dict[key] = new_parameters[key]
        param_dict["z"] = z
        param_dict["Ebv"] = Ebv

        _, lbol, mag = utils.sc_lc(sample_times, param_dict)
        return lbol, mag


class SupernovaShockCoolingLightCurveModel(object):
    def __init__(self, sample_times, parameter_conversion=None):

        self.sample_times = sample_times

        self.sc_lightcurve_model = ShockCoolingLightCurveModel(
            sample_times, parameter_conversion
        )
        self.supernova_lightcurve_model = SupernovaLightCurveModel(
            sample_times, parameter_conversion
        )

    def __repr__(self):
        details = (
            "(shock cooling model using Piro2021 with supernova model nugent-hyper)"
        )
        return self.__class__.__name__ + details

    def generate_lightcurve(self, sample_times, parameters):

        total_lbol = np.zeros(len(sample_times))
        total_mag = {}

        sc_lbol, sc_mag = self.sc_lightcurve_model.generate_lightcurve(
            sample_times, parameters
        )

        if np.sum(sc_lbol) == 0.0 or len(np.isfinite(sc_lbol)) == 0:
            return total_lbol, total_mag

        (
            supernova_lbol,
            supernova_mag,
        ) = self.supernova_lightcurve_model.generate_lightcurve(
            sample_times, parameters
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


class SimpleKilonovaLightCurveModel(object):
    def __init__(self, sample_times, parameter_conversion=None, model="Me2017"):
        """A light curve model object

        An object to evaluted the kilonova (with Me2017) light curve across filters

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

        assert model in model_parameters_dict.keys(), (
            "Unknown model," "please update model_parameters_dict at em/model.py"
        )
        self.model = model
        self.model_parameters = model_parameters_dict[model]
        self.sample_times = sample_times
        self.parameter_conversion = parameter_conversion

    def __repr__(self):
        return self.__class__.__name__ + "(model={0})".format(self.model)

    def generate_lightcurve(self, sample_times, parameters):

        if self.parameter_conversion:
            new_parameters = parameters.copy()
            new_parameters, _ = self.parameter_conversion(new_parameters)
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
            _, lbol, mag = utils.metzger_lc(sample_times, param_dict)
        elif self.model == "PL_BB_fixedT":
            _, lbol, mag = utils.powerlaw_blackbody_constant_temperature_lc(
                sample_times, param_dict
            )
        return lbol, mag
