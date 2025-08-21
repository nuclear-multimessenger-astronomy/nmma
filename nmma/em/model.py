from __future__ import division

import os
from copy import copy
import joblib
import numpy as np
from scipy.special import logsumexp
from sncosmo.models import _SOURCES
from bilby_pipe.utils import convert_string_to_dict

from . import utils
from . import lightcurve_generation as lc_gen

from nmma.joint.base import initialisation_args_from_signature_and_namespace
from nmma.joint.constants import default_cosmology, c_SI
from nmma.joint.conversion import observation_angle_conversion, get_redshift, cosmology_to_distance, distance_modulus_nmma
from nmma.utils.models import get_models_home, get_model

ln10 = np.log(10)


# As different KN models have very different parameters,
# we need a dict to keep track for them. Remember, the order matters

model_parameters_dict = { 
    ## bolometric models
    "Arnett": ["tau_m", "log10_mni"],
    "Arnett_modified": ["tau_m", "log10_mni", "t_0"],
    ## kilonova SVD models
    "Bu2019nsbh": ["log10_mej_dyn", "log10_mej_wind", "KNtheta"],
    "Bu2019lm": ["log10_mej_dyn", "log10_mej_wind", "KNphi", "KNtheta"],
    "Bu2019lm_sparse": ["log10_mej_dyn", "log10_mej_wind"],
    "Ka2017": ["log10_mej", "log10_vej", "log10_Xlan"],
    ## GRB model
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
    ## analytical kilonova models
    "Me2017": ["log10_mej", "log10_vej", "beta", "log10_kappa_r"],
    "Bu2022mv": ["log10_mej_dyn", "vej_dyn", "log10_mej_wind", "vej_wind", "KNtheta"],
    "HoNa2020": ["log10_Mej", "vej_max", "vej_min", "vej_frac", "log10_kappa_low_vej", "log10_kappa_high_vej"],
    ## supernova models
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
    ## Supernova models
    **dict.fromkeys(["AnBa2022_sparse", "AnBa2022_log", "AnBa2022_linear"],
        ["https://arxiv.org/abs/2302.09226", "https://arxiv.org/abs/2205.10421"]),
    "nugent-hyper": ["https://arxiv.org/abs/astro-ph/0403450"],

    ## SVD kilonova models
    **dict.fromkeys( ["Bu2019lm", "Bu2019lm_sparse"], 
        ["https://arxiv.org/abs/2002.11355", "https://arxiv.org/abs/1906.04205"] ),
    "Bu2019nsbh": ["https://arxiv.org/abs/2009.07210", "https://arxiv.org/abs/1906.04205"],
    **dict.fromkeys(["Bu2022Ye", "Bu2023Ye", "Bu2022mv"], 
        ["https://arxiv.org/abs/2307.11080", "https://arxiv.org/abs/1906.04205"] ),
    "Ka2017": ["https://arxiv.org/abs/1710.05463"],
    **dict.fromkeys(["LANLTP1", "LANLTP2", "LANLTS1", "LANLTS2"], 
        ["https://arxiv.org/abs/2105.11543"]),

    # analytical kilonova models
    "Me2017": ["https://arxiv.org/abs/1910.01617"],
    "HoNa2020": ["https://arxiv.org/abs/1909.02581", "https://arxiv.org/abs/1206.2379"],
    **dict.fromkeys(["PL_BB_fixedT", "blackbody_fixedT", "synchrotron_powerlaw"], ["Analytical models"]),

    # Shock cooling model
    "Piro2021": ["https://arxiv.org/abs/2007.08543"],

    #Host Galaxy model
    "Sr2023": ['https://arxiv.org/pdf/2303.12849'],

    #GRB model
    "TrPi2018": ["https://arxiv.org/abs/1909.11691"],

}

class LightCurveModelContainer(object):
    """A parent-class for light curve model objects to evaluate lightcurves
    across filters with a set of parameters given

    Parameters
    ----------
    model: str
        Name of the model
    parameter_conversion: func
        Function to convert from sampled parameters to parameters of the
        light curve model. By default, no conversion takes place.
    model_parameters: list, None
        list of alternative model parameters, if not specified default will be used.
    filters : List[str]
        List of filters to create model for.
        Defaults to all available filters.
    sample_times: array_like, optional
        Times at which to sample the light curve. If None, sets model default.

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
        model_parameters=None,
        sample_times=None,
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

        if isinstance(filters, str):
            filters = filters.split(',')
        self.filters = filters
        self.default_filts, self.lambdas = utils.get_default_filts_lambdas(self.filters)
        self.nu_0s = c_SI / self.lambdas

        # sample times are used as nodes to generate the light curve, 
        # characterising the model's validity range and the resolution
        # below which interpolation should not be performed.
        self.model_times = sample_times if sample_times is not None else self.setup_model_times()

    def __repr__(self):
        return self.__class__.__name__ + f"(model={self.model})"

    def setup_model_times(self):
        tmin = 0.01  # minimum time in days
        tmax = 14.0  # maximum time in days
        nsteps = 150  # number of time steps
        return np.geomspace(tmin, tmax, nsteps)
    
    def check_vs_priors(self, priors):
        """Check if the parameters are in the priors of the model."""
        for key in self.model_parameters:
            if key not in priors:
                print(f"Parameter {key} not found in priors, might fail.")

    def sanity_checks(self, parameters):
        return True
        
    def em_parameter_setup(self, parameters):
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
        
        # read here, but used later for correction that is observation-dependent
        self.Ebv = new_parameters.get("Ebv", 0.0)
        d_lum = new_parameters.get("luminosity_distance", 1e-5) ## default 10pc = 1e-5 Mpc
        self.distmod = distance_modulus_nmma(d_lum) ## default 10pc = 1e-5 Mpc
        param_dict['luminosity_distance'] = d_lum
        self.timeshift = new_parameters.get('timeshift', 0.)
        
        # redshift computation can be expensive, so we only do it 
        # once and store it for conversion to detector frame
        self.redshift = get_redshift(new_parameters)
        param_dict['redshift'] = self.redshift

        return param_dict

    def get_extinction_mags(self, redshift=None, Ebv=None):
        if redshift is None:
            redshift = self.redshift
        if Ebv is None:
            Ebv = self.Ebv
        ext_mag = np.zeros_like(self.nu_0s)

        if Ebv != 0.0:
            ext = utils.extinctionFactorP92SMC(self.nu_0s, Ebv, redshift)
            ext_mag = -2.5 * np.log10(ext)

        return ext_mag

    def apply_extinction_correction(self, mag, ext_mags, filters):
        for ext_mag, filt in zip(ext_mags, filters):
            try:
                mag[filt] += ext_mag
            except KeyError: # this catches key error if ext mag also considers filters that are not given in the lc
                continue
        return mag  
    
    def gen_detector_lc(self, parameters, sample_times=None):
        """Generate a light curve for given parameter as observable in detector frame.

        Parameters
        ----------
        parameters: dict
            Parameters of the light curve model.
        filters: str or list of str, optional
            Filters to use for the light curve. Defaults to 'all'.

        Returns
        -------
        dict
            Light curve magnitudes across the specified filters.
        """
        if sample_times is None:
            sample_times = self.model_times
                    
        model_lc = self.generate_lightcurve(sample_times, parameters)

        # redshift has been set in em_parameter_setup
        # timeshift is a detector-frame correction parameter
        observable_times = sample_times * (1 + self.redshift) + self.timeshift

        return self.combine_detector_data(model_lc, observable_times)

    def generate_lightcurve(self, sample_times, parameters):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def combine_detector_data(self, model_lc, observable_times):
        
        ext_mag = self.get_extinction_mags()
        model_lc = self.apply_extinction_correction(model_lc, ext_mag, self.default_filts)

        # abs_mags consider source frame fluxes, so we have to correct 
        # for the fact that we integrate over the 'wrong' luminosity window 
        redshift_correction = -2.5 * np.log10(1 + self.redshift)

        lc_data = {}
        for filt, mags in model_lc.items():
            use_mask = np.isfinite(mags)
            if np.sum(use_mask) >=2: 

                apparent_magnitude = mags + self.distmod + redshift_correction
                # apparent_magnitude = utils.autocomplete_data(
                #      observable_times,  observable_times[use_mask], apparent_magnitude[use_mask])
            else: #no meaningful inter-/extrapolation possible
                apparent_magnitude =  np.full_like(self.model_times, np.inf)
            lc_data[filt] = apparent_magnitude

        return (observable_times, lc_data)

    @property
    def citation(self):
        return {self.model: citation_dict[self.model]}


class FiestaModel(LightCurveModelContainer):    
    def __init__(self, fiesta_model, parameter_conversion, filters, sample_times = None):
        """A light curve model object for evaluating light curves using fiesta.

        Parameters
        ----------
        fiesta_model: fiesta.inference.lightcurve_model.SurrogateModel
            The fiesta model to use.
        parameter_conversion: func, optional
            Function to convert from sampled parameters to parameters of the
            light curve model. By default, no conversion takes place.
        filters: str or list of str, optional
            Filters to use for the light curve. Defaults to all trained filters.
        sample_times: array_like, optional
            Unused, included for compatibility with other Models.
        """
        self.fiesta_model = fiesta_model
        if filters is None:
            filters = fiesta_model.filters
        if sample_times is not None:
            print('Warning: sample_times are not used in FiestaModel, ignoring.')
        super().__init__(fiesta_model.name, parameter_conversion, fiesta_model.parameter_names, filters)
        
    
    def setup_model_times(self):
        return self.fiesta_model.times # default sample times for fiesta model
        
    def check_vs_priors(self, priors):
        for key in self.model_parameters:
            try:
                prior = priors[key]
                param_info = self.fiesta_model.parameter_distributions[key]
                if (prior.minimum > param_info[0]) or (prior.maximum < param_info[1]):
                    raise ValueError(f"Parameter {key} has bounds {param_info}, but prior was {prior}.")

            except KeyError:
                print(f"Parameter {key} not found in priors, might fail.")

    def gen_detector_lc(self, parameters, sample_times=None):
        """Generate a light curve for given parameter as observable in detector frame.
        Parameters
        ----------
        parameters: dict
            Parameters of the light curve model.
        sample_times: Unused, included for compatibility with other Models."""
        # convert the parameters to the fiesta model parameters
        new_parameters = self.em_parameter_setup(parameters)
        
        if not self.sanity_checks(new_parameters):
            return {}

        # generate the light curve using fiesta
        time_range, mag = self.fiesta_model.predict(parameters)

        # apply the extinction correction
        ext_mag = self.get_extinction_mags()
        obs_mags = self.apply_extinction_correction(mag, ext_mag, self.default_filts)
        
        # we are in observer frame, but still need to add the timeshift
        #time_range = is in jax-specific format that we need to convert
        return (np.array(time_range) + self.timeshift, obs_mags)
    
    
    def generate_lightcurve(self, sample_times, parameters):
        """Generate a light curve for given parameter as observable in detector frame."""
        obs_times, obs_mags = self.gen_detector_lc(parameters)
        # NOTE: we should almost always prefer gen_detector_lc!
        
        # obs_times were redshift corrected; have to reverse this:
        source_times = (obs_times - self.timeshift) / (1+self.redshift)
        abs_mags = {filt: 
                    np.interp(sample_times, source_times, obs_mag -self.distmod)
                    for filt, obs_mag in obs_mags.items()}
        
        return abs_mags


class SimpleBolometricLightCurveModel(LightCurveModelContainer):
    """A light curve model object for Arnett's bolometric supernova light curves.

    Parameters
    ----------
    parameter_conversion: func, optional
        Function to convert from sampled parameters to parameters of the
        light curve model. By default, no conversion takes place.
    model: string, optional
        Name of the model. Can be either "Arnett" (default) or "Arnett_modified"

    Returns
    -------
    LightCurveModel: `nmma.em.model.SimpleBolometricLightCurveModel`
        A light curve model object to evaluate the light curve
        from a set of parameters
    """
    def __init__(self,  parameter_conversion=None, model="Arnett", sample_times=None):
        super().__init__(model, parameter_conversion, sample_times=sample_times)
        if model == "Arnett":
            self.lc_func = lc_gen.arnett_lc
        elif model == "Arnett_modified":
            self.lc_func = lc_gen.arnett_modified_lc

    def setup_model_times(self):
        tmin = 0.005  # minimum time in days, arbitrary
        tmax = 20.0  # NOTE: The underlying integrals tend to diverge at later times
        nsteps = 40  # number of time steps

        return np.linspace(tmin, tmax, nsteps)

    def combine_detector_data(self, model_lc, observable_times):
        ## lBol is essentially energy per time bin, both quantities
        #  get a redshift correction in detector frame 
        return (observable_times, model_lc/(1+self.redshift)**2)
    

    def generate_lightcurve(self, sample_times, parameters):
        new_parameters = self.em_parameter_setup(parameters)
        lbol = self.lc_func(sample_times, new_parameters)
        return lbol


class SVDLightCurveModel(LightCurveModelContainer):
    """A light curve model object for evaluating light curves using prebuilt SVD models.

    An object to evaluate the light curve across filters
    with a set of parameters given based on a prebuilt SVD.

    Parameters
    ----------
    model: str
        Name of the model.
    svd_path: str, optional
        Path to the SVD directory.
    parameter_conversion: func, optional
        Function to convert from sampled parameters to parameters of the
        light curve model. By default, no conversion takes place.
    svd_mag_ncoeff: int, optional
        Number of highest eigenvalues to be taken for magnitude's SVD evaluation.
    svd_lbol_ncoeff: int, optional
        Number of highest eigenvalues to be taken for bolometric luminosity's SVD evaluation.
    interpolation_type: str, optional
        Type of interpolation to use. Can be 'keras','sklearn_gp', 'api_gp', or a keras-backend.
    model_parameters: list, optional
        List of alternative model parameters. If not specified, default will be used.
    filters: list of str, optional
        List of filters to create model for. Defaults to all available filters.
    local_only: bool, optional
        If True, only local models will be used.

    Returns
    -------
    LightCurveModel: `nmma.em.model.SVDLightCurveModel`
        A light curve model object to evaluate the light curve
        from a set of parameters.
    """

    def __init__(
        self,
        model,
        svd_path=None,
        parameter_conversion=None,
        svd_mag_ncoeff=None,
        svd_lbol_ncoeff=None,
        interpolation_type="keras",
        model_parameters=None,
        filters=None,
        sample_times = None,
        local_only=False,
    ):
        # Some models have underscores. Keep those, but drop '_tf' if it exists
        model_name_components = model.split("_")
        if "tf" in model_name_components:
            model_name_components.remove("tf")

        core_model_name = "_".join(model_name_components)


        self.mag_ncoeff = svd_mag_ncoeff
        self.lbol_ncoeff = svd_lbol_ncoeff
        self.interpolation_type = interpolation_type
        self.svd_path = get_models_home(svd_path)

        modelfile   = os.path.join(self.svd_path, f"{core_model_name}.joblib")
        if interpolation_type == "tensorflow":
            self.model_specifier = "_tf"
        else:
            self.model_specifier = ""
        if not local_only:
            ##FIXME Does this make sense for api_gp, too?
            filters = self.get_model_data(core_model_name, filters)
        try:
            svd_mag_model = joblib.load(modelfile)
            # temporary fix, moving towards permament setting of sncosmo filter names
            self.svd_mag_model = {k.replace("_", ":"): v for k, v in svd_mag_model.items()}
            self.svd_lbol_model= None  # FIXME: this is not yet implemented
        except ValueError:
            raise ValueError("Model file not found: {modelfile}\n \
                If possible, try removing the --local-only flag and rerun.")
        
        ## need to have read the model before identifying the model_times
        super().__init__(core_model_name, parameter_conversion, filters, model_parameters, sample_times)

        if self.filters is None:
            try:
                self.filters = list(self.svd_mag_model.keys())
            except TypeError: ## if using a bol_model, we keep None
                pass

        if self.interpolation_type == "sklearn_gp":
            self.load_filt_model(model,joblib.load, fn_ext='joblib', target_name='gps')

        elif self.interpolation_type == "api_gp":
            from .training import load_api_gp_model
            for filt in self.filters:
                for ii, gp_model in enumerate(self.svd_mag_model[filt]["gps"]):
                    self.svd_mag_model[filt]["gps"][ii] = load_api_gp_model(gp_model)
            
        elif self.interpolation_type in ("keras", "tensorflow", "torch", 'jax'):

            import keras as k
            def keras_load_model(model_file):
                return k.saving.load_model(model_file, compile=False)
            try:
                self.load_filt_model(self.model, keras_load_model, fn_ext='keras', target_name='model')
            # if no filter- model is found, try to load the legacy h5 model
            except ValueError:
                self.load_filt_model(self.model, keras_load_model, fn_ext= 'h5', target_name='model')

        else:
            raise ValueError("--interpolation-type must be sklearn_gp, api_gp or tensorflow")
        
    def setup_model_times(self):
        ## reset model_times to the model's training times if possible
        try:
            return next(iter(self.svd_mag_model.values()))['tt'] 
        except:
            return super().setup_model_times()
        

    def get_model_data(self, model, filters):
        # we need to map the filter names to repo names!
        search_filters = None
        if isinstance(filters, str):
            filters = filters.split(',')
        if filters is not None:
            search_filters  = [filt.replace(":", "_") for filt in filters]
        _, model_filters = get_model(self.svd_path, f"{model}{self.model_specifier}", filters=search_filters)
        if filters is None and model_filters is not None:
            filters = [filt.replace("_", ":") for filt in model_filters]
        return filters

    def load_filt_model(self, model, load_method, fn_ext='joblib', target_name='gps'):
        """Subroutine to load the filter models from the SVD path.
        While svd_mag_model as loaded from the corresponding model file is only
        a dictionary with some model metadata, this step includes the actual 
        ml-model and makes it available in the filter-specific sub-dictionary."""
        outdir = os.path.join(self.svd_path, f"{model}{self.model_specifier}")
        found_any_model = False
        not_found = []
        for filt in self.filters:
            outfile = os.path.join(outdir, f"{filt.replace(':', '_')}.{fn_ext}")
            if os.path.isfile(outfile):
                self.svd_mag_model[filt][target_name] = load_method(outfile)
                found_any_model = True
            else:
                not_found.append(filt)
        if not found_any_model:
            raise ValueError(f"No {fn_ext}-model files found for {model} in {outdir}")
        elif len(not_found) > 0:
            print(f"Warning: No {fn_ext}-model files found for filters: {not_found} at {outdir}")   
        
    def __repr__(self):
        return super().__repr__() + f"(model={self.model}, svd_path={self.svd_path})"
    
    def em_parameter_setup(self, parameters):
        param_dict = super().em_parameter_setup(parameters)
        return [param_dict[key] for key in self.model_parameters]

    def generate_lightcurve(self, sample_times, parameters, filters = 'all'):
        parameters_list = self.em_parameter_setup(parameters)
        # parameters_list.append(param_dict["redshift"])

        # if filters is set to None, we assume we want the bolometric luminosity
        if filters is None:
            return lc_gen.calc_svd_lbol(
                sample_times, parameters_list,
                svd_lbol_model=self.svd_lbol_model,
                lbol_ncoeff=self.lbol_ncoeff,
            )  ## original code provides a redshift correction of (1+z), but should this not rather be (1+z)**2
        
        elif filters =='all':
            filters = self.filters
        return lc_gen.calc_svd_lc(
            sample_times, parameters_list,
            self.svd_mag_model,
            mag_ncoeff=self.mag_ncoeff,
            filters=filters,
        )
    
    def generate_spectra(self, sample_times, wavelengths, parameters):
        return self.generate_lightcurve(sample_times, parameters, filters=wavelengths)


class FiestaKilonovaModel(FiestaModel):
    """A light curve model object for GRB light curves using fiesta

    An object to evaluate the GRB light curve across filters
    with a set of parameters given based on fiesta.

    Parameters
    ----------
    parameter_conversion: func, optional
        Function to convert from sampled parameters to parameters of the
        light curve model. By default, no conversion takes place.
    model: str, optional
        Name of the model. Default is "Bu2025_CVAE".
    filters: list of str, optional
        List of filters to create model for. Defaults to all available filters.
    surrogate_dir: str, optional
        path to the directory containing the surrogate models.

    Returns
    -------
    LightCurveModel: `nmma.em.model.GRBFiestaModel`
        A light curve model object to evaluate the light curve
        from a set of parameters.
    """
    def __init__(self, parameter_conversion=None, model="Bu2025_CVAE", filters=None, surrogate_dir=None, **kwargs):
        from fiesta.inference.lightcurve_model import BullaLightcurveModel
        kwargs.update(dict( name=model, filters=filters, directory=surrogate_dir,))
        try:
            fiesta_model = BullaLightcurveModel(**kwargs)
        except OSError:
            kwargs['surrogate_dir'] = f'{surrogate_dir}/KN/{model}/model'
            fiesta_model = BullaLightcurveModel(**kwargs)
            

        super().__init__(fiesta_model, parameter_conversion, filters, sample_times=kwargs.get('sample_times', None))


class FiestaGRBModel(FiestaModel):
    """A light curve model object for GRB light curves using fiesta

    An object to evaluate the GRB light curve across filters
    with a set of parameters given based on fiesta.

    Parameters
    ----------
    parameter_conversion: func, optional
        Function to convert from sampled parameters to parameters of the
        light curve model. By default, no conversion takes place.
    model: str, optional
        Name of the model. Default is "afgpy_gaussian_CVAE".
    filters: list of str, optional
        List of filters to create model for. Defaults to all available filters.
    surrogate_dir: str, optional
        path to the directory containing the surrogate models.

    Returns
    -------
    LightCurveModel: `nmma.em.model.GRBFiestaModel`
        A light curve model object to evaluate the light curve
        from a set of parameters.
    """
    def __init__(self, parameter_conversion=None, model="afgpy_gaussian_CVAE", filters=None, surrogate_dir=None, **kwargs):
        from fiesta.inference.lightcurve_model import AfterglowFlux
        kwargs.update(dict( name=model, filters=filters, directory=surrogate_dir))
        try:
            self.fiesta_model = AfterglowFlux(**kwargs)
        except OSError:
            kwargs['surrogate_dir'] = f'{surrogate_dir}/GRB/{model}/model'
            self.fiesta_model = AfterglowFlux(**kwargs)

        super().__init__(model, parameter_conversion, filters, model_parameters= self.fiesta_model.parameter_names, sample_times=kwargs.get('sample_times', None))
    
    def em_parameter_setup(self, parameters):
        new_parameters =super().em_parameter_setup(parameters)
        try:
            new_parameters["thetaWing"] = new_parameters["alphaWing"] * new_parameters['thetaCore']
        except KeyError:
            new_parameters["thetaWing"] = new_parameters["thetaWing"]

        try:
            new_parameters['epsilon_tot'] = new_parameters['epsilon_e'] + new_parameters['epsilon_B']
        except KeyError:
            new_parameters['epsilon_tot'] = 10**new_parameters['log10_epsilon_e'] + 10**new_parameters['log10_epsilon_B']

        return new_parameters
    
    def sanity_checks(self, fiesta_parameters):
        """Perform sanity checks on the GRB parameters."""
        if fiesta_parameters["thetaWing"] > np.pi / 2:
            return False
        
        # core opening angle must be greater than 0.1 degree for proper integration
        if fiesta_parameters["thetaCore"] < np.pi / 1800.:
            return False

        if fiesta_parameters['epsilon_tot'] > 1.0:
            return False
        
        return True


class GRBLightCurveModel(LightCurveModelContainer):
    """A light curve model object for GRB light curves using afterglowpy

    An object to evaluate the GRB light curve across filters
    with a set of parameters given based on afterglowpy.

    Parameters
    ----------
    parameter_conversion: func, optional
        Function to convert from sampled parameters to parameters of the
        light curve model. By default, no conversion takes place.
    model: str, optional
        Name of the model. Default is "TrPi2018".
    model_parameters: list, optional
        List of alternative model parameters. If not specified, default will be used.
    resolution: int, optional
        Resolution for the GRB model. Default is 12.
    jet_type: int, optional
        Type of jet for the GRB model. Default is 0.
    filters: list of str, optional
        List of filters to create model for. Defaults to all available filters.

    Returns
    -------
    LightCurveModel: `nmma.em.model.GRBLightCurveModel`
        A light curve model object to evaluate the light curve
        from a set of parameters.
    """
    def __init__(
        self,
        parameter_conversion=None,
        model="TrPi2018",
        model_parameters = None,
        resolution=12,
        jet_type=0,
        filters=None,
        sample_times=None,
    ):
        super().__init__(model, parameter_conversion, filters, model_parameters, sample_times)
        self.resolution = resolution
        self.jet_type = jet_type
        self.default_parameters = {"xi_N": 1.0, "d_L": 3.086e19, "jetType": jet_type, "specType": 0}  # d_L=10pc in cm
        self.def_keys = self.default_parameters.keys()
        #keys we typically sample in log space, but need to convert to linear space
        self.log_sampling_keys = ["E0", "n0", "epsilon_e", "epsilon_B"]
        self.energy_injection_params = ['energy_exponential', 'log10_Eend', 't_start', 'injection_duration']
        self.flux_func = None  # flux function to be set later, if needed

    
    def setup_model_times(self):
        tmin = 1.e-5  # minimum time in days
        tmax = 200  # maximum time in days
        nsteps = 201  # number of time steps
        return np.geomspace(tmin, tmax, nsteps)
    
    def em_parameter_setup(self, parameters):
        
        ## set on first call
        if self.flux_func is None: 
            # case 1: use energy injection approach
            if all(key in parameters for key in self.energy_injection_params):
                self.flux_func = lc_gen.flux_density_on_E0_array
                self.log_sampling_keys.remove("E0")
            else: #case 2
                self.flux_func = lc_gen.flux_density_on_time_array

        new_parameters = super().em_parameter_setup(parameters)

        # set the default parameters, preferentially from sampling
        grb_param_dict = {k: new_parameters.get(k, self.default_parameters[k]) for k in self.def_keys}

        grb_param_dict["z"] = new_parameters['redshift']
        grb_param_dict["thetaObs"] = new_parameters["inclination_EM"]# energy handling

        for key in self.log_sampling_keys:
            try:
                grb_param_dict[key] = new_parameters[key]
            except KeyError:
                grb_param_dict[key] = 10 ** new_parameters[f"log10_{key}"]

        # it is beneficial to sample the ratio of angles alpha instead of checking later whether this can be resolved        
        try:
            grb_param_dict["thetaWing"] = new_parameters["alphaWing"] * new_parameters['thetaCore']
        except KeyError:
            grb_param_dict["thetaWing"] = new_parameters["thetaWing"]

        try:
            grb_param_dict['epsilon_tot'] = new_parameters['epsilon_e'] + new_parameters['epsilon_B']
        except KeyError:
            grb_param_dict['epsilon_tot'] = 10**new_parameters['log10_epsilon_e'] + 10**new_parameters['log10_epsilon_B']

        # make sure L0, q and ts are also passed
        for param in ["thetaCore","p", 'L0', 'q', 'ts']:
            try:
                grb_param_dict[param] = new_parameters[param]
            except KeyError:
                pass

        if self.jet_type == 1 or self.jet_type == 4:
            grb_param_dict["b"] = new_parameters["b"]
        return grb_param_dict
    
    def sanity_checks(self, grb_param_dict):
        """Perform sanity checks on the GRB parameters."""
        if "thetaWing" in grb_param_dict.keys():
            if grb_param_dict["thetaWing"] > np.pi / 2:
                return False
            elif grb_param_dict["thetaWing"] / grb_param_dict["thetaCore"] > self.resolution:
                return False
        
        # core opening angle must be greater than 0.1 degree for proper integration
        if grb_param_dict["thetaCore"] < np.pi / 1800.:
            return False

        if grb_param_dict.pop('epsilon_tot') > 1.0:
            return False
        
        return True

    def generate_lightcurve(self, sample_times, parameters):
        grb_param_dict = self.em_parameter_setup(parameters)

        #sanity checks
        if not self.sanity_checks(grb_param_dict):
            return {}
        
        return lc_gen.afterglowpy_lc(
            sample_times, grb_param_dict, filters=self.default_filts, 
            obs_frequencies= self.nu_0s, flux_func= self.flux_func
        )
    

class HostGalaxyLightCurveModel(LightCurveModelContainer):
    """
    A light curve model object for a simple lightcurve model that includes the transient's host galaxy.

    An object to evaluate the host galaxy light curve across filters
    with a set of parameters given based on arxiv:2303.12849.

    Parameters
    ----------
    model: str, optional
        Name of the model. Default is "Sr2023".
    parameter_conversion: func, optional
        Function to convert from sampled parameters to parameters of the
        light curve model. By default, no conversion takes place.
    filters: list of str, optional
        List of filters to create model for. Defaults to all available filters.
    host_mag: float or int, optional
        Magnitude of the host galaxy. Default is 23.9.
    model_parameters: list, optional
        List of alternative model parameters. If not specified, default will be used.

    Returns
    -------
    LightCurveModel: `nmma.em.model.HostGalaxyLightCurveModel`
        A light curve model object to evaluate the light curve
        from a set of parameters.
    """
    def __init__(
        self,
        model="Sr2023",
        parameter_conversion=None,
        filters=None,
        sample_times=None,
        # host_mag is the magnitude of the host galaxy in the filters
        host_mag=23.9,  # value for case of arxiv:2303.12849
        model_parameters=None
    ):
        super().__init__(model, parameter_conversion, filters, model_parameters, sample_times=sample_times)
        if isinstance(host_mag, (float, int)):
            self.host_mag = np.full_like(self.filters, host_mag)

    def generate_lightcurve(self, sample_times, parameters):
        new_parameters = self.em_parameter_setup(parameters)
        self.extinction = 0. # extinction correction implicit in host galaxy model, will not be applied even when combined with extinction-sensitive models
        return lc_gen.host_lc(sample_times, new_parameters, self.filters, self.host_mag)


class SupernovaLightCurveModel(LightCurveModelContainer):
    """
    A light curve model object for supernova light curves using sncosmo

    An object to evaluate the supernova light curve across filters
    with a set of parameters given based on sncosmo.

    Parameters
    ----------
    parameter_conversion: func, optional
        Function to convert from sampled parameters to parameters of the
        light curve model. By default, no conversion takes place.
    model: str, optional
        Name of the model. Default is "nugent-hyper".
    filters: list of str, optional
        List of filters to create model for. Defaults to all available filters.
    model_parameters: list, optional
        List of alternative model parameters. If not specified, default will be used.

    Returns
    -------
    LightCurveModel: `nmma.em.model.SupernovaLightCurveModel`
        A light curve model object to evaluate the light curve
        from a set of parameters.
    """
    def __init__(
        self,
        parameter_conversion=None,
        model="nugent-hyper",
        filters=None,
        sample_times=None,
        model_parameters=None,
        cosmology=default_cosmology
    ):
        super().__init__(model, parameter_conversion, filters, model_parameters, sample_times)
        self.cosmology = cosmology

    def generate_lightcurve(self, sample_times, parameters):
        em_param_dict = self.em_parameter_setup(parameters)
        
        stretch = em_param_dict.get("supernova_mag_stretch", 1.)

        mag = lc_gen.sn_lc(
            sample_times / stretch,
            em_param_dict,
            cosmology=self.cosmology,
            model_name=self.model,
            filters=self.default_filts,
            lambdas=self.lambdas
        )
        mag_boost = em_param_dict.get("supernova_mag_boost", 0.0)
        return {filt: filt_mag + mag_boost for filt, filt_mag in mag.items()}
    

class ShockCoolingLightCurveModel(LightCurveModelContainer):
    def __init__(
        self, parameter_conversion=None, model="Piro2021", filters=None, 
        model_parameters=None, sample_times=None
    ):
        """A light curve model object

        An object to evaluted the shock cooling light curve across filters, particularly suited for descriptions of lightcurves at early times (hours to few days)

        Parameters
        ----------
    parameter_conversion: func, optional
        Function to convert from sampled parameters to parameters of the
        light curve model. By default, no conversion takes place.
    model: str, optional
        Name of the model. Default is "Piro2021".
    filters: list of str, optional
        List of filters to create model for. Defaults to all available filters.

        Returns
        -------
        LightCurveModel: `nmma.em.model.ShockCoolingLightCurveModel`
            A light curve model object to evaluate the light curve
            from a set of parameters
        """
        super().__init__(model, parameter_conversion, filters, 
                         model_parameters, sample_times)

    def setup_model_times(self):
        # model is suitable on the order of hours to a few days
        # this limit is somewhat arbitrary, but should be sufficient for most cases
        return np.geomspace(1./24., 3.5, 100)

    def generate_lightcurve(self, sample_times, parameters, filters = 'all'):
        """Generate an absolute-magnitude light curve for given parameters on sample times.
        Contrary to earlier behaviour of this function, we asume sample_times 
        to be measured in source frame"""
        lc_param_dict = self.em_parameter_setup(parameters)

        if filters is None:
            return lc_gen.sc_bol_lc(sample_times, lc_param_dict, compute_Rs=False)
            # would need a (1+z)**2 correction in observ. frame luminosity
        
        elif filters == 'all':
            filters = self.default_filts
        lbol, Rs = lc_gen.sc_bol_lc(sample_times, lc_param_dict, compute_Rs=True)

        nu_host = self.nu_0s * (1 + lc_param_dict['redshift'])  # convert to host frame
        return lc_gen.sc_lc(lbol, Rs, nu_host, filters)



class SimpleKilonovaLightCurveModel(LightCurveModelContainer):
    """
    A light curve model object for simple kilonova light curves

    An object to evaluate the kilonova light curve across filters
    with a set of parameters given based on simple analytical models.

    Parameters
    ----------
    parameter_conversion: func, optional
        Function to convert from sampled parameters to parameters of the
        light curve model. By default, no conversion takes place.
    model: str, optional
        Name of the model. Default is "Me2017".
    filters: list of str, optional
        List of filters to create model for. Defaults to all available filters.

    Returns
    -------
    LightCurveModel: `nmma.em.model.SimpleKilonovaLightCurveModel`
        A light curve model object to evaluate the light curve
        from a set of parameters.
    """
    def __init__(
        self, parameter_conversion=None, model="Me2017", filters=None, sample_times=None
    ):
        super().__init__(model, parameter_conversion, filters, sample_times=sample_times)
        lc_dict={
            "HoHa2020"      : lc_gen.HoNa_lc,
            "Me2017"        : lc_gen.eff_metzger_lc,
            "PL_BB_fixedT"  : lc_gen.powerlaw_blackbody_constant_temperature_lc,
            "blackbody_fixedT"      : lc_gen.blackbody_constant_temperature,
            "synchrotron_powerlaw"  : lc_gen.synchrotron_powerlaw
        }
        self.lc_func = lc_dict[model]


    def generate_lightcurve(self, sample_times, parameters):
        """Generate an absolute-magnitude light curve for given parameters on sample times.
        Contrary to earlier behaviour of this function, we asume sample_times 
        to be measured in source frame"""
        param_dict = self.em_parameter_setup(parameters)
        param_dict['distance_modulus'] = self.distmod

        nu_host = self.nu_0s * (1 + param_dict['redshift'])  # convert to host frame
        # prevent the output message flooded by these warning messages
        old = np.seterr()
        np.seterr(invalid="ignore")
        np.seterr(divide="ignore")
        mag = self.lc_func(sample_times, param_dict, nu_host,  self.default_filts)
        np.seterr(**old)

        return mag


class CombinedLightCurveModelContainer(object):
    """
    An object to evaluate the combined light curve from a set of parameters
    using multiple light curve models.

    Parameters
    ----------
    models: list
        A list of LightCurveModel instances.
    model_args: list, optional
        If not None (default), this is assumed to be a list of argument lists 
        with which the respective submodels will be initialized.

    Returns
    -------
    LightCurveModel: `nmma.em.CombinedLightCurveModelContainer`
        A light curve model object to evaluate the combined 
        light curve from a set of parameters.
    """
    def __init__(self, models, model_args=None):
        if model_args is None:
            self.models= models
        else:
            self.models = [model(*model_args[i]) for i, model in enumerate(models)]
        
        self.all_filters = set().union(*[model.filters for model in self.models])
        ## FIXME: Better treatment for synonymous or equivalent filters?
        self.compatible_filters, _ = utils.get_filter_name_mapping(self.all_filters)

        self.model_times = np.array(sorted(set().union(
            *[model.model_times for model in self.models]
            )))

    def __repr__(self):
        return f"Combination of {', '.join(map(str, self.models[:-1].__repr__()))} and {self.models[-1].__repr__()}"

    def check_vs_priors(self, priors):
        for model in self.models:
            model.check_vs_priors(priors)

    @property
    def citation(self):
        citations = {}
        for model in self.models:
            citations.update(model.citation)
        return citations
        
    def gen_detector_lc(self, parameters, sample_times=None, return_all=False):
        """
        Generate the detector light curve for the combined model.
        This is a convenience function that calls the generate_lightcurve
        method of each submodel and combines the results.
        """

        lc_per_model = []
        time_per_model = []
        for model in self.models:
            ref_times, lc = model.gen_detector_lc(parameters, sample_times)

            if not lc:
                # if the model returns False/empty, it means that the model
                # could not generate a light curve for the given parameters
                return lc
            else:
                lc_per_model.append(lc)
                time_per_model.append(ref_times)

        if return_all:
            return (time_per_model, lc_per_model)
        if sample_times is None:
            connected_times = np.array(sorted(set().union(*time_per_model)))
        else:
            connected_times = ref_times #if we have sample_times, the ref_times are the same for all models 
        if isinstance(lc, dict):
            # if the (last) model returns a dictionary, we have light curves per filter

            lcs_on_joint_time_frame = [{filt: utils.autocomplete_data(connected_times, ref_times, filt_lc, extrapolate=np.inf)
                                    for filt, filt_lc in lc.items()}
                                    for ref_times, lc in zip(time_per_model, lc_per_model)]
            total_lc = self.stack_magnitudes(lcs_on_joint_time_frame)

        elif isinstance(lc, (list,np.ndarray)):
            # if the model returns array-like, this was a bolometric light curve
            lcs_on_joint_time_frame = [
                utils.autocomplete_data(connected_times, ref_times, lc, extrapolate=0.) 
                for ref_times, lc in zip(time_per_model, lc_per_model)]
            total_lc = np.sum(lcs_on_joint_time_frame, axis=0)

        return connected_times, total_lc

    def generate_lightcurve(self, sample_times, parameters, return_all=False):

        lc_per_model = []
        for model in self.models:
            lc = model.generate_lightcurve(sample_times, parameters)

            if not lc:
                # if the model returns False/empty, it means that the model
                # could not generate a light curve for the given parameters
                return lc
            else:
                lc_per_model.append(lc)

        if return_all:
            return lc_per_model
        
        if isinstance(lc, (list,np.ndarray)):
            # if the model returns array-like, this was a bolometric light curve
            total_lc = np.sum(lc_per_model, axis=0)

        elif isinstance(lc, dict):
            total_lc = self.stack_magnitudes(lc_per_model)

        return total_lc
            
    def stack_magnitudes(self, mags_per_model):
        """
        Stack the magnitudes from the light curves of each model.
        This is a helper function to combine the magnitudes from different models.
        """
        stacked_mags = {}
        for filt in self.all_filters:
            mAB_list = []
            for mag in mags_per_model:
                try:
                    mag_per_filt = mag[self.compatible_filters[filt]]
                except KeyError:
                    # if the filter is not available in the model,
                    # the submodels may still be compatible
                    try:
                        mag_per_filt = utils.average_mags(mag, filt)
                    except ValueError:
                        continue
                mAB_list.append(-2.0 / 5.0 * ln10 * np.array(mag_per_filt))
            if not mAB_list:
                ## return no magnitude if no good value in either model
                stacked_mags[filt] = np.full_like(self.model_times, np.inf)
            else:
                stacked_mags[filt] = -5.0 / 2.0 * logsumexp(mAB_list, axis=0) / ln10
        return stacked_mags
    
class GenericCombineLightCurveModel(CombinedLightCurveModelContainer):
    "A legacy synonym for CombinedLightCurveModelContainer"


class KilonovaGRBLightCurveModel(CombinedLightCurveModelContainer):
    """
    A combined light curve model for Kilonova and GRB (Gamma-Ray Burst) events.

    This model integrates the light curves from both Kilonova and GRB models
    to provide a comprehensive representation of the observed phenomena.

    Parameters
    ----------
    kilonova_kwargs : dict
        Dictionary of keyword arguments for the Kilonova light curve model.
    parameter_conversion : callable, optional
        Function for converting parameters between different representations.
        If not provided, the default conversion from `kilonova_kwargs` will be used.
    grb_resolution : int, optional
        Resolution parameter for the GRB light curve model. Default is 12.
    jet_type : int, optional
        Type of jet model to use for the GRB light curve. Default is 0.

    """
    def __init__(
        self,
        kilonova_kwargs,
        parameter_conversion=None,
        grb_resolution=12,
        jet_type=0,
    ):  
        ## FIXME in what context would this be meaningful?
        self.parameter_conversion = kilonova_kwargs["parameter_conversion"]
        kilonova_kwargs["parameter_conversion"] = parameter_conversion

        kn_model = SVDLightCurveModel(**kilonova_kwargs)
        grb_model = GRBLightCurveModel(
            parameter_conversion,
            resolution=grb_resolution,
            jet_type=jet_type,
        )
        super().__init__([grb_model, kn_model])

class SupernovaGRBLightCurveModel(CombinedLightCurveModelContainer):
    def __init__(
        self,
        supernova_kwargs,
        parameter_conversion=None,
        grb_resolution=12,
        jet_type=0,
    ):
        grb_model = GRBLightCurveModel(
            parameter_conversion,
            resolution=grb_resolution,
            jet_type=jet_type,
        )
        sn_model = SupernovaLightCurveModel( parameter_conversion = parameter_conversion, **supernova_kwargs
        )
        super().__init__([grb_model, sn_model])

class SupernovaShockCoolingLightCurveModel(CombinedLightCurveModelContainer):
    def __init__(self, parameter_conversion=None, filters=None):
        super().__init__([
            ShockCoolingLightCurveModel(parameter_conversion, filters),
            SupernovaLightCurveModel(parameter_conversion, filters)
        ])

def lc_model_class_from_str(class_names):
    transient_class_map = {
        "svd"           : SVDLightCurveModel,
        "fiesta_grb"    : FiestaGRBModel,
        "grb"           : GRBLightCurveModel,
        "host_galaxy"   : HostGalaxyLightCurveModel,
        "supernova"     : SupernovaLightCurveModel,
        "shock"         : ShockCoolingLightCurveModel,
        "simple_kilonova":SimpleKilonovaLightCurveModel,
        "combined"      : CombinedLightCurveModelContainer,
        "kilonova_grb"  : KilonovaGRBLightCurveModel,
        "supernova_grb" : SupernovaGRBLightCurveModel,
        "supernova_shock":SupernovaShockCoolingLightCurveModel,
    }
    if len(class_names) ==1:
        class_names = class_names[0].lower().split(",")
    ##FIXME get more robust handling of aliases and typos
    try:
        classes = [transient_class_map[cn.strip()] for cn in class_names]
    except KeyError:
        raise KeyError(f"EM transient classes must be in {transient_class_map.keys()}, but was {class_names}")
    return classes

def get_lc_model_from_modelname(model_name):
    #FIXME This is incomplete, but identical to handling in NMMA 0.2.2
    model_name_mapping = {
        "TrPi2018"      : GRBLightCurveModel,
        "Piro2021"      : ShockCoolingLightCurveModel,
        "Me2017"        : SimpleKilonovaLightCurveModel,
        "PL_BB_fixedT"  : SimpleKilonovaLightCurveModel,
        "Sr2023"        : HostGalaxyLightCurveModel, 
        "Arnett"        : SimpleBolometricLightCurveModel, ## Addition
    }
    if model_name in model_name_mapping.keys():
        return model_name_mapping[model_name]
    elif model_name in [val["name"] for val in _SOURCES.get_loaders_metadata()]:
        return SupernovaLightCurveModel
    else:
        # FIXME This is an unclean default, should be more explicit!
        return SVDLightCurveModel

def single_model_from_args(model_class, model_name, args, 
                           filters, prefixes = ['grb_', 'em_']):
    # parameter conversion as used in EM-only sector
    if getattr(args, 'Hubble', False):
        parameter_conversion = cosmology_to_distance 
    else:
        parameter_conversion = None

    # populate model-args from default and parsed args
    default_model_args = initialisation_args_from_signature_and_namespace(
        model_class, args, prefixes= prefixes
    )

    # update explicit args
    model_args = default_model_args | dict(filters=filters,
        parameter_conversion=parameter_conversion,
        sample_times=utils.setup_sample_times(args),
    )
    if model_name is not None: 
        model_args["model"] = model_name.strip()
    return model_class(**model_args)

def create_light_curve_model_from_args(
    em_transient, args, filters=None
):      
    if filters is None:
        filters = utils.set_filters(args)
    #case 1: we have the model_names and need to find the classes first
    # this is equivalent to the previous behaviour of this function for em-only analysis
    if isinstance(em_transient, str):   
        model_names = em_transient.split(",")
        model_classes = [get_lc_model_from_modelname(model_name) for model_name in model_names]


    # case 2, we have transient classes, need to identify the corresponding models
    elif isinstance(em_transient, list):
        model_classes = em_transient
        prel_model_names = args.em_model.split(",")
        model_names = []
        for i, model_class in enumerate(model_classes):
            try:
                model_names.append(prel_model_names[i])
            except IndexError:
                print(f"Warning: No model name found for {model_class}. Will try using the default model")
                model_names.append(None)
    
    models_list = [single_model_from_args( mc, mn, args, filters)
                    for mc, mn in zip(model_classes, model_names) ]
    
    if len(models_list)==1: ## if we only have one model, return it directly
        return models_list[0]
    print("Running with combination of multiple light curve models")
    return CombinedLightCurveModelContainer(models_list)

def identify_model_type(args):    
    ## identify what kind of transient we are dealing with
    try:
        # preferred method is to explicitly pass the desired class
        lc_model= lc_model_class_from_str(args.em_transient_class)
    except AttributeError:
        # if no class is given, we try to infer it from the model names
        lc_model = args.em_model
    return lc_model
        
def create_injection_model(args, filters=None):
    # step 0: by default, injection is created with the same args as the main model
    injection_args = copy(args)
    
    # step 1: find args that were parsed as "injection_..."
    #  and place them in the injection-Namespace
    for arg, val in args.__dict__.items():
        if arg == "injection_model_args":
            if val is None:
                injection_dict = {}
            else:
                injection_dict = convert_string_to_dict(val)
        else:
            arg = arg.replace("injection_", "") # replace 'injection_' prefix if necessary
            setattr(injection_args, arg, val)

    # step 2: we have set aside the values of injection_args, now fill them in
    for arg, val in injection_dict.items():
            arg = arg.lstrip('--').replace('-','_').replace("injection_", "")
            setattr(injection_args, arg, val)


    # step 3: identify the model type and create the model
    lc_model = identify_model_type(injection_args)
    return create_light_curve_model_from_args(lc_model, injection_args, filters)