from __future__ import division

import copy
import os
import joblib
import numpy as np
from scipy.special import logsumexp
import scipy.constants
from sncosmo.models import _SOURCES

from . import utils
from . import lightcurve_generation as lc_gen

from ..joint.conversion import observation_angle_conversion, get_redshift, Hubble_constant_to_distance, default_cosmology
from ..utils.models import get_models_home, get_model

ln10 = np.log(10)


# As different KN models have very different parameters,
# we need a dict to keep track for them. Remember, the order matters

model_parameters_dict = { 
    ## bolometric models
    "Arnett": ["tau_m", "log10_mni"],
    "Arnett_modified": ["tau_m", "log10_mni", "t_0"],
    ## kilonova models
    "Bu2019nsbh": ["log10_mej_dyn", "log10_mej_wind", "KNtheta"],
    "Bu2019lm": ["log10_mej_dyn", "log10_mej_wind", "KNphi", "KNtheta"],
    "Bu2019lm_sparse": ["log10_mej_dyn", "log10_mej_wind"],
    "Ka2017": ["log10_mej", "log10_vej", "log10_Xlan"],
    ## GRB models
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
    ## Shock cooling models
    "Piro2021": ["log10_Menv", "log10_Renv", "log10_Ee"],
    "Me2017": ["log10_mej", "log10_vej", "beta", "log10_kappa_r"],
    "Bu2022mv": ["log10_mej_dyn", "vej_dyn", "log10_mej_wind", "vej_wind", "KNtheta"],
    "PL_BB_fixedT": ["bb_luminosity", "temperature", "beta", "powerlaw_mag"],
    "blackbody_fixedT": ["bb_luminosity", "temperature"],
    "synchrotron_powerlaw": ["alpha_time", "beta_freq", "F_ref", "luminosity_distance"],
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
    "LANL2022": [
        "log10_mej_dyn",
        "vej_dyn",
        "log10_mej_wind",
        "vej_wind",
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
}

citation_dict = {
            **dict.fromkeys(["LANLTP1", "LANLTP2", "LANLTS1", "LANLTS2"], ["https://arxiv.org/abs/2105.11543"]),
            "Ka2017": ["https://arxiv.org/abs/1710.05463"],
            **dict.fromkeys(
                ["Bu2019lm", "Bu2019lm_sparse"], ["https://arxiv.org/abs/2002.11355", "https://arxiv.org/abs/1906.04205"]
            ),
            **dict.fromkeys(
                [
                    "AnBa2022_sparse",
                    "AnBa2022_log",
                    "AnBa2022_linear",
                ],
                ["https://arxiv.org/abs/2302.09226", "https://arxiv.org/abs/2205.10421"],
            ),
            "Bu2019nsbh": ["https://arxiv.org/abs/2009.07210", "https://arxiv.org/abs/1906.04205"],
            **dict.fromkeys(
                ["Bu2022Ye", "Bu2023Ye", "Bu2022mv"], ["https://arxiv.org/abs/2307.11080", "https://arxiv.org/abs/1906.04205"]
            ),
            "TrPi2018": ["https://arxiv.org/abs/1909.11691"],
            "Piro2021": ["https://arxiv.org/abs/2007.08543"],
            "Me2017": ["https://arxiv.org/abs/1910.01617"],
            "Sr2023": [None],  # TODO: add citation,
            "nugent-hyper": ["https://sncosmo.readthedocs.io/en/stable/source-list.html"],
            **dict.fromkeys(["PL_BB_fixedT", "blackbody_fixedT", "synchrotron_powerlaw"], ["Analytical models"]),
        }

class LightCurveModelContainer(object):
    """A parent-class for light curve model objects to evaluate lightcurves
    across filters with a set of parameters given

    Parameters
    ----------
    model: str
        Name of the model
    sample_times: np.array
        An array of time for the light curve to be evaluated on
    parameter_conversion: func
        Function to convert from sampled parameters to parameters of the
        light curve model. By default, no conversion takes place.
    model_parameters: list, None
        list of alternative model parameters, if not speciefied default will be used.
    filters : List[str]
        List of filters to create model for.
        Defaults to all available filters.

    Returns
    -------
    LightCurveModel: `nmma.em.model.SVDLightCurveModel`
        A light curve model object to evaluate the light curve
        from a set of parameters
    """

    def __init__(
        self, model, 
        parameter_conversion=None,
        filters=None,
        model_parameters=None
    ):

        if model_parameters is None:
            assert model in model_parameters_dict.keys(), (
                f"{model} unknown," "please update model_parameters_dict at em/model.py"
            )
            self.model_parameters = model_parameters_dict[model]
        else:
            self.model_parameters = model_parameters

        self.model = model
        if parameter_conversion is None:
            parameter_conversion = lambda *x: x
        self.parameter_conversion = parameter_conversion
        self.filters = filters

    def __repr__(self):
        return self.__class__.__name__ + f"(model={self.model})"
    
    def em_parameter_setup(self, parameters, extinction = False):
        new_parameters = parameters.copy()
        new_parameters, _ = self.parameter_conversion(new_parameters, [])
        new_parameters = observation_angle_conversion(new_parameters)

        param_dict = {}
        for key in self.model_parameters:
            try:
                param_dict[key] = new_parameters[key]
            except KeyError:
                if key.lstrip('log10_') in new_parameters.keys():
                    param_dict[key] = np.log10(new_parameters[key.lstrip('log10_')])
                elif "log10_"+key in new_parameters.keys():
                    param_dict[key] = 10**new_parameters["log10_"+key]
                else:
                    pass ## Unclean fix, allows later addition of required params

        if extinction:
            param_dict["Ebv"] = new_parameters.get("Ebv", 0.0)
        param_dict['redshift'] = get_redshift(new_parameters)

        return param_dict
    
    def get_extinction_mags(self, redshift, Ebv, filts= None, out= 'nus'):
        default_filts, lambdas = utils.get_default_filts_lambdas(filters=filts)
        nu_0s = scipy.constants.c / lambdas

        if Ebv != 0.0:
            ext = utils.extinctionFactorP92SMC(nu_0s, Ebv, redshift)
            ext_mag = -2.5 * np.log10(ext)
        else:
            ext_mag = np.zeros(len(nu_0s))
        if out == 'lambdas':
            return default_filts, lambdas, ext_mag
        else:
            return default_filts, nu_0s, ext_mag

    def apply_extinction_correction(self, mag, ext_mags, filters):
        for ext_mag, filt in zip(ext_mags, filters):
            try:
                mag[filt] += ext_mag
            except:
                continue
        return mag  

    def extinction_correction(self, mag, redshift, Ebv, filts=None):
        def_filts ,_,  ext_mag = self.get_extinction_mags(redshift, Ebv, filts)
        return self.apply_extinction_correction(mag, ext_mag, def_filts)

    @property
    def citation(self):
        return {self.model: citation_dict[self.model]}
    

class SimpleBolometricLightCurveModel(LightCurveModelContainer):
    def __init__(self,  parameter_conversion=None, model="Arnett"):
        """A light curve model object

        An object to evaluted the kilonova Arnett light curve across filters

        Parameters
        ----------
        model: string
            Name of the model. Can be either "Arnett" (default) or "Arnett_modified"

        Returns
        -------
        LightCurveModel: `nmma.em.model.SimpleBolometricLightCurveModel`
            A light curve model object to evaluate the light curve
            given a set of parameters
        """
        super().__init__(model, parameter_conversion)
        if model == "Arnett":
            self.lc_func = lc_gen.arnett_lc
        elif model == "Arnett_modified":
            self.lc_func = lc_gen.arnett_modified_lc

    def generate_lightcurve(self, sample_times, parameters):
        new_parameters = self.em_parameter_setup(parameters)
        lbol = self.lc_func(sample_times, new_parameters)
        return lbol, {}


class SVDLightCurveModel(LightCurveModelContainer):
    """A light curve model object

    An object to evaluate the light curve across filters
    with a set of parameters given based on a prebuilt SVD

    Parameters
    ----------
    model: str
        Name of the model
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
        A light curve model object to evaluate the light curve
        from a set of parameters
    """

    def __init__(
        self,
        model,
        svd_path=None,
        parameter_conversion=None,
        mag_ncoeff=None,
        lbol_ncoeff=None,
        interpolation_type="sklearn_gp",
        model_parameters=None,
        filters=None,
        local_only=False,
    ):
        super().__init__(model, parameter_conversion, filters, model_parameters)

        self.mag_ncoeff = mag_ncoeff
        self.lbol_ncoeff = lbol_ncoeff
        self.interpolation_type = interpolation_type
        self.svd_path = get_models_home(svd_path)

        # Some models have underscores. Keep those, but drop '_tf' if it exists
        model_name_components = model.split("_")
        if "tf" in model_name_components:
            model_name_components.remove("tf")

        core_model_name = "_".join(model_name_components)

        modelfile   = os.path.join(self.svd_path, f"{core_model_name}.joblib")


        if interpolation_type == "tensorflow":
            self.model_specifier = "_tf"
        else:
            self.model_specifier = ""
        if not local_only:
            ##FIXME Does this make sense for api_gp, too?
            self.get_model_data(filters)
        try:
            self.svd_mag_model = joblib.load(modelfile)
            self.svd_lbol_model= None
        except ValueError(
                        f"Model file not found: {modelfile}\n If possible, try removing the --local-only flag and rerun."):
            exit()


        if self.filters is None:
            self.filters = list(self.svd_mag_model.keys())


        if self.interpolation_type == "sklearn_gp":
            self.load_filt_model(model,joblib.load, fn_ext='joblib', target_name='gps')

        elif self.interpolation_type == "api_gp":
            from .training import load_api_gp_model
            for filt in self.filters:
                for ii, gp_model in enumerate(self.svd_mag_model[filt]["gps"]):
                    self.svd_mag_model[filt]["gps"][ii] = load_api_gp_model(gp_model)
            
        elif self.interpolation_type == "tensorflow":
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            from tensorflow.keras.models import load_model
            def tensorflow_load_model(model_file):
                return load_model(model_file, compile=False)
            self.load_filt_model(model, tensorflow_load_model, fn_ext= 'h5', target_name='model')

        else:
            return ValueError("--interpolation-type must be sklearn_gp, api_gp or tensorflow")
        

    def get_model_data(self, filters):
        _, model_filters = get_model(self.svd_path, f"{self.model}{self.model_specifier}", filters=filters)
        if filters is None and model_filters is not None:
            self.filters = model_filters
        
    def load_filt_model(self, model, load_method, fn_ext='joblib', target_name='gps'):
        
        outdir = os.path.join(self.svd_path, f"{model}{self.model_specifier}")
        for filt in self.filters:
            outfile = os.path.join(outdir, f"{filt}.{fn_ext}")
            if os.path.isfile(outfile):
                print(f"Load filter {filt}")
                self.svd_mag_model[filt][target_name] = load_method(outfile)
            else:
                print(f"Could not find model file for filter {filt}")
                if filt not in self.svd_mag_model:
                    self.svd_mag_model[filt] = {}
                self.svd_mag_model[filt][target_name] = None
        
    def __repr__(self):
        return super().__repr__() + f"(model={self.model}, svd_path={self.svd_path})"

    def generate_lightcurve(self, sample_times, parameters, filters = None):
        param_dict = self.em_parameter_setup(parameters)
        z= param_dict['redshift']
        parameters_list = [param_dict[key] for key in self.model_parameters]
        if filters is None:
            filters = self.filters
        lbol, mag = lc_gen.calc_lc(
            sample_times / (1.0 + z),
            parameters_list,
            svd_mag_model=self.svd_mag_model,
            svd_lbol_model=self.svd_lbol_model,
            mag_ncoeff=self.mag_ncoeff,
            lbol_ncoeff=self.lbol_ncoeff,
            interpolation_type=self.interpolation_type,
            filters=filters,
        )
        lbol *= 1.0 + z

        filts = mag.keys()
        _, _, ext_mags = self.get_extinction_mags(z, parameters.get("Ebv", 0.0), filts)
        for i, filt in enumerate(filts):
            mag[filt] -= 2.5 * np.log10(1.0 + z)
            mag[filt] += ext_mags[i]

        return lbol, mag
    
    def generate_spectra(self, sample_times, wavelengths, parameters):
        _, spec = self.generate_lightcurve(sample_times, parameters, filters=wavelengths)
        return spec


class GRBLightCurveModel(LightCurveModelContainer):
    def __init__(
        self,
        parameter_conversion=None,
        model="TrPi2018",
        model_parameters = None,
        resolution=12,
        jetType=0,
        filters=None,
    ):
        """A light curve model object

        An object to evaluate the GRB light curve across filters
        with a set of parameters given based on afterglowpy

        Parameters
        ----------
        sample_times: np.array
            An array of timesteps for  lightcurve evaluation

        Returns
        -------
        LightCurveModel: `nmma.em.model.GRBLightCurveModel`
            A light curve model object to evaluate the light curve
            from a set of parameters
        """
        super().__init__(model, parameter_conversion, filters, model_parameters)
        self.resolution = resolution
        self.jetType = jetType
        self.default_parameters = {"xi_N": 1.0, "d_L": 3.086e19, "jetType": jetType, "specType": 0}  # d_L=10pc in cm
        self.def_keys = self.default_parameters.keys()
        #keys we typically sample in log space, but need to convert to linear space
        self.log_sampling_keys = ["E0", "n0", "epsilon_e", "epsilon_B"]
    
    def grb_parameter_setup(self, new_parameters):

        # set the default parameters preferentially from sampling
        grb_param_dict = {k: new_parameters.get(k, self.default_parameters[k]) for k in self.def_keys}

        grb_param_dict["z"] = new_parameters['redshift']
        grb_param_dict["thetaObs"] = new_parameters["inclination_EM"]

        for key in self.log_sampling_keys:
            try:
                grb_param_dict[key] = new_parameters[key]
            except KeyError:
                grb_param_dict[key] = 10 ** new_parameters[f"log10_{key}"]

        # make sure L0, q and ts are also passed
        for param in ["thetaCore","thetaWing", "p", 'L0', 'q', 'ts']:
            try:
                grb_param_dict[param] = new_parameters[param]
            except KeyError:
                pass

        if self.jetType == 1 or self.jetType == 4:
            grb_param_dict["b"] = new_parameters["b"]

    def generate_lightcurve(self, sample_times, parameters):
        new_parameters = self.em_parameter_setup(parameters, extinction=True)
        grb_param_dict = self.grb_parameter_setup(new_parameters)

        #sanity checks
        if "thetaWing" in grb_param_dict.keys():
            if grb_param_dict["thetaWing"] / grb_param_dict["thetaCore"] > self.resolution:
                return np.zeros(len(sample_times)), {}

        if grb_param_dict["epsilon_e"] + grb_param_dict["epsilon_B"] > 1.0:
            return np.zeros(len(sample_times)), {}
        
        filts, nu_0s, ext_mag = self.get_extinction_mags(new_parameters['redshift'], new_parameters.get("Ebv", 0.0), self.filters)
        lbol, mag = lc_gen.grb_lc(
            sample_times, grb_param_dict, filters=filts, obs_frequencies= nu_0s
        )
        mag = self.apply_extinction_correction(mag, ext_mag, filts )
        return lbol, mag


class HostGalaxyLightCurveModel(LightCurveModelContainer):
    def __init__(
        self,
        model="Sr2023",
        parameter_conversion=None,
        filters=None,
        host_mag=23.9, ##value for case of arxiv:2303.12849
        model_parameters= None
    ):
        """A light curve model object

        An object to evaluate the host galaxy light curve across filters
        with a set of parameters given

        Based on arxiv:2303.12849

        Parameters
        ----------
        sample_times: np.array
            An array of timesteps for  lightcurve evaluation

        Returns
        -------
        LightCurveModel: `nmma.em.model.HostGalaxyLightCurveModel`
            A light curve model object to evaluate the light curve
            from a set of parameters
        """
        super().__init__(model,  parameter_conversion, filters, model_parameters)
        if isinstance(host_mag, (float, int)):
            self.host_mag = np.full_like(self.filters, host_mag)

    def generate_lightcurve(self, sample_times, parameters):
        new_parameters = self.em_parameter_setup(parameters)
        lbol, mag = lc_gen.host_lc(sample_times, new_parameters, self.filters, self.host_mag)

        return lbol, mag


class SupernovaLightCurveModel(LightCurveModelContainer):
    def __init__(
        self,
        parameter_conversion=None,
        model="nugent-hyper",
        filters=None,
        model_parameters = None
    ):
        """A light curve model object

        An object to evaluate the supernova light curve across filters
        with a set of parameters given based on sncosmo

        Parameters
        ----------
        sample_times: np.array
            An array of timesteps for  lightcurve evaluation

        Returns
        -------
        LightCurveModel: `nmma.em.model.GRBLightCurveModel`
            A light curve model object to evaluate the light curve
            from a set of parameters
        """
        super().__init__(model, parameter_conversion, filters, model_parameters)

    def generate_lightcurve(self, sample_times, parameters):
        em_param_dict = self.em_parameter_setup(parameters, extinction=True)
        filts, lambdas, ext_mag = self.get_extinction_mags(em_param_dict['redshift'], em_param_dict.get("Ebv", 0.0), self.filters, out='lambdas')
        
        stretch = em_param_dict.get("supernova_mag_stretch", 1)

        lbol, mag = lc_gen.sn_lc(
            sample_times / stretch,
            em_param_dict,
            cosmology=default_cosmology,
            model_name=self.model,
            filters=filts,
            lambdas=lambdas
        )
        mag_boost= em_param_dict.get("supernova_mag_boost", 0.0)
        for i, filt in enumerate(filts):
            mag[filt]+= ext_mag[i] + mag_boost

        return lbol, mag


class ShockCoolingLightCurveModel(LightCurveModelContainer):
    def __init__(
        self, parameter_conversion=None, model="Piro2021", filters=None, model_parameters=None
    ):
        """A light curve model object

        An object to evaluted the shock cooling light curve across filters

        Parameters
        ----------
        sample_times: np.array
            An array of timesteps for  lightcurve evaluation

        Returns
        -------
        LightCurveModel: `nmma.em.model.ShockCoolingLightCurveModel`
            A light curve model object to evaluate the light curve
            from a set of parameters
        """
        super().__init__(model, parameter_conversion, filters, model_parameters)


    def generate_lightcurve(self, sample_times, parameters):
        lc_param_dict = self.em_parameter_setup(parameters)
        filts, nus, ext_mag = self.get_extinction_mags(lc_param_dict['redshift'], lc_param_dict.get("Ebv", 0.0), self.filters)

        lbol, mag = lc_gen.sc_lc(sample_times, lc_param_dict, nus, filters=filts)

        mag = self.apply_extinction_correction(mag, ext_mag, filts)
        return lbol, mag


class SimpleKilonovaLightCurveModel(LightCurveModelContainer):
    def __init__(
        self, parameter_conversion=None, model="Me2017", filters=None
    ):
        """A light curve model object

        An object to evaluted the kilonova (with Me2017) light curve across filters

        Parameters
        ----------
        sample_times: np.array
            An array of timesteps for  lightcurve evaluation

        Returns
        -------
        LightCurveModel: `nmma.em.model.SimpleKilonovaLightCurveModel`
            A light curve model object to evaluate the light curve
            from a set of parameters
        """
        super().__init__(model, parameter_conversion, filters)
        lc_dict={
            "Me2017"        : lc_gen.metzger_lc,
            "PL_BB_fixedT"  :  lc_gen.powerlaw_blackbody_constant_temperature_lc,
            "blackbody_fixedT": lc_gen.blackbody_constant_temperature,
            "synchrotron_powerlaw": lc_gen.synchrotron_powerlaw
        }
        self.lc_func = lc_dict[model]

    def generate_lightcurve(self, sample_times, parameters):
        param_dict = self.em_parameter_setup(parameters, extinction=True)

        filts, nu_obs, ext_mags = self.get_extinction_mags(param_dict['redshift'], param_dict.get("Ebv", 0.0), self.filters)

        # prevent the output message flooded by these warning messages
        old = np.seterr()
        np.seterr(invalid="ignore")
        np.seterr(divide="ignore")
        lbol, mag = self.lc_func(sample_times, param_dict, nu_obs, self.filters)
        np.seterr(**old)
        if self.model == "synchrotron_powerlaw":
            # remove the distance modulus for the synchrotron powerlaw
            # as the reference flux is defined at the observer
            dist_mod = 5.0 * np.log10(param_dict["luminosity_distance"] * 1e6 / 10)
            for filt in mag.keys():
                mag[filt] -= dist_mod
        mag = self.apply_extinction_correction(mag, ext_mags, filts)

        return lbol, mag
    

class CombinedLightCurveModelContainer(object):
    """A parent-class for combining light curve model objects
    Parameters
    ----------
    sample_times: np.array
        An array of time for the lightcurves to be evaluated on
    models: list
        A list of LightCurveModel-instances
    model_args: list, None
        If not None (default), this is assumed a list of arg_lists 
        with which the respective submodels will be initialised.

    Returns
    -------
    LightCurveModel: `nmma.em.CombinedLightCurveModelContainer`
        A light curve model object to evaluate the combined 
        lightcurve from a set of parameters.
    """
    def __init__(self, models, model_args=None):
        if model_args is None:
            self.models= models
        else:
            self.models = [model(*model_args[i]) for i, model in enumerate(models)]

    def __repr__(self):
        return f"Combination of {', '.join(map(str, self.models[:-1].__repr__()))} and {self.models[-1].__repr__()}"

    @property
    def citation(self):
        citations = {}
        for model in self.models:
            citations.update(model.citation)
        return citations

    def generate_lightcurve(self, sample_times, parameters, return_all=False):

        total_lbol = np.zeros(len(sample_times))
        total_mag = {}
        mag_per_model = []
        lbol_per_model = []

        for model in self.models:
            lbol, mag = model.generate_lightcurve(sample_times, parameters)

            if np.sum(lbol) == 0.0 or len(np.isfinite(lbol)) == 0:
                return total_lbol, total_mag

            else:
                total_lbol += lbol
                mag_per_model.append(mag)
                lbol_per_model.append(lbol)

        # filts = mag_per_model[0].keys()  # just get the first one
        filts = set().union(*[mags.keys() for mags in mag_per_model])

        for filt in filts:
            mAB_list = []
            for mag in mag_per_model:
                try:
                    mag_per_filt = utils.getFilteredMag(mag, filt)
                    # check if the mag_flit is valid, if it is 0
                    # meaning the filter is unknown to getFilteredMag
                    # and therefore, we have the mag_per_filt set to inf (0 flux)
                    if isinstance(mag_per_filt, int) and mag_per_filt == 0:
                        continue
                except KeyError:
                    continue
                mAB_list.append(-2.0 / 5.0 * ln10 * np.array(mag_per_filt))
            if not mAB_list:
                ## return no magnitude if no good value in either model
                total_mag[filt] = np.full_like(sample_times, np.inf)
            else:
                total_mag[filt] = -5.0 / 2.0 * logsumexp(mAB_list, axis=0) / ln10

        if return_all:
            return lbol_per_model, mag_per_model
        else:
            return total_lbol, total_mag
        
class GenericCombineLightCurveModel(CombinedLightCurveModelContainer):
    "A synonym for CombinedLightCurveModelContainer"


class KilonovaGRBLightCurveModel(CombinedLightCurveModelContainer):
    def __init__(
        self,
        kilonova_kwargs,
        parameter_conversion=None,
        GRB_resolution=12,
        jetType=0,
    ):  
        ## FIXME in what context would this be meaningful?
        self.parameter_conversion = kilonova_kwargs["parameter_conversion"]
        kilonova_kwargs["parameter_conversion"] = parameter_conversion

        kn_model = SVDLightCurveModel(**kilonova_kwargs)
        grb_model = GRBLightCurveModel(
            parameter_conversion,
            resolution=GRB_resolution,
            jetType=jetType,
        )
        super().__init__([grb_model, kn_model])


class SupernovaGRBLightCurveModel(CombinedLightCurveModelContainer):
    def __init__(
        self,
        parameter_conversion=None,
        SNmodel="nugent-hyper",
        GRB_resolution=12,
        jetType=0,
    ):
        grb_model = GRBLightCurveModel(
            parameter_conversion,
            resolution=GRB_resolution,
            jetType=jetType,
        )
        sn_model = SupernovaLightCurveModel( parameter_conversion, model=SNmodel
        )
        super().__init__([grb_model, sn_model])


class SupernovaShockCoolingLightCurveModel(CombinedLightCurveModelContainer):
    def __init__(self, parameter_conversion=None, filters=None):
        super().__init__([
            ShockCoolingLightCurveModel(parameter_conversion, filters),
            SupernovaLightCurveModel(parameter_conversion, filters)
        ])

def create_light_curve_model_from_args(
    model_name_arg,
    args,
    filters=None,
    sample_over_Hubble=False
):
    # check if sampling over Hubble,
    # if so define the parameter_conversion accordingly
    if sample_over_Hubble:
        parameter_conversion = Hubble_constant_to_distance
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
                resolution=args.grb_resolution,
                jetType=args.jet_type,
                parameter_conversion=parameter_conversion,
                filters=filters,
            )

        elif model_name in sncosmo_names:
            lc_model = SupernovaLightCurveModel(
                model=model_name,
                parameter_conversion=parameter_conversion,
                filters=filters,
            )

        elif model_name == "Piro2021":
            lc_model = ShockCoolingLightCurveModel(
                parameter_conversion=parameter_conversion,
                filters=filters,
            )

        elif model_name == "Me2017" or model_name == "PL_BB_fixedT":
            lc_model = SimpleKilonovaLightCurveModel(
                model=model_name,
                parameter_conversion=parameter_conversion,
                filters=filters,
            )

        elif model_name == "Sr2023":
            lc_model = HostGalaxyLightCurveModel(
                parameter_conversion=parameter_conversion,
                filters=filters,
            )

        else:
            local_only = getattr(args, 'local_only', False)
            lc_kwargs = dict(
                model=model_name,
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
        light_curve_model = CombinedLightCurveModelContainer(models)
    else:
        light_curve_model = models[0]

    return model_names, models, light_curve_model
