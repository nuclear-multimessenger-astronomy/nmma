from __future__ import division
import inspect
import numpy as np
from scipy.stats import norm, truncnorm

from ..joint.base import NMMABaseLikelihood
from .model import *
from . import utils, systematics



def setup_em_kwargs(param_conv_func, data_dump, args, logger, **kwargs):
    # get lc_data and filters
    light_curve_data= data_dump["light_curve_data"]
    filters=args.filters
    if not filters:
        filters = list(light_curve_data.keys())

    # identify what kind of transient we are dealing with
    em_transient_class= get_em_transient_class(args.em_transient_class)

    # setup the light curve model for this transient class and filters
    light_curve_model = setup_lightcurve_model(em_transient_class, args, filters, param_conv_func)



    # inspect the signature of the EM-likelihood class
    em_transient_signature = inspect.signature(EMTransientLikelihood) 
    ## this provides all arguments that the likelihood takes from args or from the default value
    em_kwargs = {key : getattr(args, key, val.default) for key, val in em_transient_signature.parameters.items() } 


    
    em_likelihood_kwargs = dict(
        light_curve_model=light_curve_model,
        filters=filters,light_curve_data=light_curve_data,
        trigger_time=args.em_transient_trigger_time,
        error_budget=args.em_transient_error,
    )
    return em_kwargs | em_likelihood_kwargs

def get_em_transient_class(class_name = 'svd'):
    transient_class_map = {
        "svd": SVDLightCurveModel,
        "grb": GRBLightCurveModel,
        "host_galaxy": HostGalaxyLightCurveModel,
        "supernova": SupernovaLightCurveModel,
        "shock": ShockCoolingLightCurveModel,
        "simple_kilonova": SimpleKilonovaLightCurveModel,
        "combined": CombinedLightCurveModelContainer,
        "kilonova_grb": KilonovaGRBLightCurveModel,
        "supernova_grb": SupernovaGRBLightCurveModel,
        "supernova_shock": SupernovaShockCoolingLightCurveModel,
    }
    ##FIXME get more robust handling of aliases and typos
    try:
        return transient_class_map[class_name.lower()]
    except KeyError(f"EM transient class must be one of {transient_class_map.keys()}, but was {class_name}"):
        exit()

def setup_lightcurve_model(em_transient_class, args, filters, param_conv_func = None):

    ## parse args into light curve model
    # FIXME: this routine needs an update!
    # create_light_curve_model_from_args()

    signature = inspect.signature(em_transient_class) 
    default_transient_kwargs= {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}
    # case1: individual transient classes
    if isinstance(em_transient_class, LightCurveModelContainer):
        
        light_curve_model_kwargs = dict(
        # model=args.kilonova_model,
        # svd_path=args.lc_model_svd_path,
        # parameter_conversion=param_conv_func,
        # mag_ncoeff=args.svd_mag_ncoeff,
        # lbol_ncoeff=args.svd_lbol_ncoeff,
        # interpolation_type=args.transient_emulator_type,
        # filters=filters,
        # local_only=args.local_model_only,
        )
        light_curve_model = SVDLightCurveModel(**light_curve_model_kwargs)
        ##case 1.5: manually combine GRB and nuclear transient 
        ## FIXME this is a bit of an ugly hack, should be handled more elegantly
        if args.with_grb:
            grb_model = GRBLightCurveModel(
                    filters = filters,
                    resolution=args.grb_resolution, 
                    parameter_conversion=param_conv_func
                    ) 
            light_curve_model = CombinedLightCurveModelContainer([light_curve_model, grb_model])

    if isinstance(em_transient_class, CombinedLightCurveModelContainer):
        pass


    return light_curve_model



class EMTransientLikelihood(NMMABaseLikelihood):
    """A generic EM transient likelihood object

    Parameters
    ----------
    light_curve_model: `nmma.em.SVDLightCurveModel`
        And object which computes the light curve of a kilonova-like signal,
        given a set of parameters
    light_curve_data: dict
        Dictionary of light curve data returned from nmma.em.utils.loadEvent
    filters: list, str, None
        A list of filters to be taken for analysis
        E.g. "u", "g", "r", "i", "z", "y", "J", "H", "K"
    trigger_time: float
        Time of the kilonova trigger in Modified Julian Day
    error_budget: float (default:1)
        Additionally introduced statistical error on the light curve data,
        so as to keep the systematic error under control. This will only be used if the parameters-dict does not containt a 'sys_err' sampling parameter.
    tmin: float (default:0)
        Days from trigger_time to be started analysing
    tmax: float (default:14)
        Days from trigger_time to be ended analysing

    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        a set of  model parameters

    """

    def __init__(self, 
        light_curve_model,
        light_curve_data,
        priors = None,
        filters=None,
        detection_limit=np.inf,
        em_transient_trigger_time=0.,
        error_budget=1.0,
        em_transient_tmin=0.0,
        em_transient_tmax=14.0,
        verbose=False,
        param_conv_func = None, **kwargs
        
    ):  
        sample_times = np.arange(em_transient_tmin, em_transient_tmax, 0.1)
        ### FIXME add better criterion to switch modes
        if filters:
            model_type = OpticalTransient
        else:
            model_type=BolometricTransient
        sub_model = model_type(
                light_curve_model, sample_times, priors, light_curve_data, filters, em_transient_trigger_time, error_budget, detection_limit, verbose)


        # self.light_curve_model = light_curve_model
        # self.verbose = verbose
        #FIXME priors seems unnecessary here
        super().__init__(sub_model=sub_model, priors=priors, param_conversion_func=param_conv_func, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__} (light_curve_model={self.light_curve_model}) \
                based on {self.sub_model.__repr__()}"
    
       

class BaseEMTransient(object):
    """A generic EM transient object

    Parameters
    ----------
    
    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        a set of  model parameters

    """


    def __init__(self, light_curve_model,
                 sample_times, verbose = False
                 ):
        self.light_curve_model = light_curve_model
        self.sample_times = sample_times
        self.verbose = verbose
    
    def log_likelihood(self):
        model_lbol, model_mags = self.light_curve_model.generate_lightcurve(
            self.sample_times, self.parameters
        )        
        # sanity checking
        if len(np.isfinite(model_lbol)) == 0:
            return np.nan_to_num(-np.inf)
        if np.sum(model_lbol) == 0.0:
            return np.nan_to_num(-np.inf)
        

        # retrieve usable lightcurve data
        lc_data = self.update_lightcurve_reference(model_lbol, model_mags)

        # compare the estimated light curve and the measured data
        logL_model = self.band_log_likelihood(lc_data)
        if self.verbose:
            print(self.parameters, logL_model)
        return logL_model

    def update_lightcurve_reference(self, model_lbol, model_mags):
        raise NotImplementedError("This method should be implemented in the subclass")
    
    def band_log_likelihood(self, lc_data):
        raise NotImplementedError("This method should be implemented in the subclass")

class OpticalTransient(BaseEMTransient):
    """A optical kilonova / GRB / kilonova-GRB afterglow likelihood object
    see line 1221 gwemlightcurves/sampler/loglike.py

    Parameters
    ----------
    filters: list, str
        A list of filters to be taken for analysis
        E.g. "u", "g", "r", "i", "z", "y", "J", "H", "K"
    light_curve_data: dict
        Dictionary of light curve data returned from nmma.em.utils.loadEvent
    trigger_time: float
        Time of the kilonova trigger in Modified Julian Day
    error_budget: float (default:1)
        Additionally introduced statistical error on the light curve data,
        so as to keep the systematic error in control
    tmin: float (default:0)
        Days from trigger_time to be started analysing
    tmax: float (default:14)
        Days from trigger_time to be ended analysing
    """

    def __init__(
        self,
        light_curve_model, sample_times, light_curve_data,
        filters,
        trigger_time,
        error_budget=1.0,
        detection_limit=np.inf,
        verbose = False,
        systematics_file=None,
    ):  
        super().__init__(light_curve_model, sample_times, verbose)
        self.filters = filters

        ##setup light curve data
        self.light_curve_data = utils.dataProcess(
            light_curve_data, self.filters, trigger_time, sample_times[0], sample_times[-1])

        # setup detection limit
        self.detection_limit = {}
        if isinstance(detection_limit, (int, float)):
            self.detection_limit = {filt: detection_limit for filt in self.filters}
        elif isinstance(detection_limit, dict):
            self.detection_limit = {filt: detection_limit.get(filt, np.inf) for filt in self.filters}


        #determine_systematic_error_handling
        ## case 1: use systematics_file
        if systematics_file:
            yaml_dict = systematics.load_yaml(systematics_file)
            systematics.validate_only_one_true(yaml_dict)
            time_dep_sys_dict = yaml_dict["config"]["withTime"]
            # case 1a: time-dependent systematics
            if time_dep_sys_dict['value']:

                #get the time nodes and the filters
                self.systematics_time_nodes = np.round(
                    np.linspace(self.sample_times[0], self.sample_times[-1], time_dep_sys_dict["time_nodes"]),
                    2)
                yaml_filters = list(time_dep_sys_dict["filters"])
                systematics.validate_filters(yaml_filters)

                #iterate over the filters and assign them to a systematics filter group
                systematics_filters = {}
                for filter_group in yaml_filters:
                    #this should only be the case if no filters are specified
                    if filter_group is None:
                        systematics_filters = {filt: 'all' for filt in self.filters}
                        break
                    elif isinstance(filter_group, list):
                        for filt in filter_group:
                            systematics_filters[filt] = "___".join(filter_group)
                    else:
                        #this should mean that the filter_group is in fact a single filter
                        systematics_filters[filter_group] = filter_group
                ## By this procedure, every filter should immediately be assigned to a systematics filter-group that we can use to calculate the systematics error       
                self.systematics_filters = systematics_filters  

                self.compute_em_err = self.em_err_from_systematics_sampling

            # case 1b: no time-dependency 
            else:
                # sample with time-independent error, that is case 2
                ## FIXME would it not be more naturally to still have a filter dependent error, even if it does not vary in time?
                self.compute_em_err = self.em_err_from_parameters
                
        
        # case 2: sample over general limit
        elif 'em_syserr' in self.parameters.keys():
            self.compute_em_err = self.em_err_from_parameters
        
        #case 3: preset general limit
        else:
            #3a: shared value for all filters
            if isinstance(error_budget, (int, float, complex)) and not isinstance(
                error_budget, bool
            ):
                self.error_budget = {filt:error_budget for filt in self.filters}

            #3b: specific values in each filter
            elif isinstance(error_budget, dict):
                for filt in self.filters:
                    if filt not in error_budget:
                        raise ValueError(f"filter {filt} missing from error_budget")
                self.error_budget = error_budget
            self.compute_em_err = self.em_err_from_budget



    def em_err_from_parameters(self, *_):
        return self.parameters['em_syserr']
    
    def em_err_from_budget(self, filt, _):
        return self.error_budget[filt]
    
    def em_err_from_systematics_sampling(self, filt, data_time):
        systematics_filt = self.systematics_filters[filt]
        sampled_filter_systematics = [self.parameters[f"sys_err_{systematics_filt}{i}"] for i in range(len(self.systematics_time_nodes))]
        return utils.autocomplete_data(data_time, self.systematics_time_nodes, sampled_filter_systematics)


             ##FIXME Check if this is the right way to handle the error budget
    def update_lightcurve_reference(self, _, model_mags):
        lc_data ={}
        t0 = self.parameters["timeshift"]
        for filt in model_mags.keys():
            mag_abs_filt = utils.getFilteredMag(model_mags[filt], filt)
            if self.parameters["luminosity_distance"] > 0.0:
                mag_app_filt = mag_abs_filt + 5.0 * np.log10(
                    self.parameters["luminosity_distance"] * 1e6 / 10.0
                )
            else:
                mag_app_filt = mag_abs_filt

            usedIdx = np.where(np.isfinite(mag_app_filt))[0]
            if len(usedIdx)<2:
                #no meaningful inter-/extrapolation possible
                lc_data[filt] = (self.sample_times + t0, np.full_like(self.sample_times, np.inf))
                continue
            sample_times_used = self.sample_times[usedIdx]
            mag_app_used = mag_app_filt[usedIdx]
            lc_data[filt] = (sample_times_used + t0, mag_app_used)
        return lc_data

    def band_log_likelihood(self, lc_data):
        minus_chisquare_total = 0.0
        gaussprob_total = 0.0
        for filt in self.filters:
            # decompose the data
            data_time, data_mag, data_sigma  = self.light_curve_data[filt].T
            systematic_em_err = self.get_em_err(filt, data_time)
            data_sigma = np.sqrt(data_sigma**2 + systematic_em_err**2)
            model_time, model_mags = lc_data[filt]
            # evaluate the light curve magnitude at the data points
            est_mag = utils.autocomplete_data(data_time, model_time, model_mags)
            minus_chisquare, gaussprob = chisquare_gaussianlog_from_lc_data( 
                est_mag, data_mag , data_sigma,  systematic_em_err, 
                lim=self.detection_limit[filt]
            )
            if isinstance(minus_chisquare, bool):
                return np.nan_to_num(-np.inf)
            else:
                minus_chisquare_total+=minus_chisquare
                gaussprob_total += gaussprob
        return minus_chisquare_total + gaussprob_total


class BolometricTransient(BaseEMTransient):
    """A bolometric Transient object

    Parameters
    ----------
    light_curve_data: dict
        Dictionary of light curve data returned from nmma.em.utils.loadEvent
    error_budget: float (default:1)
        Additionally introduced statistical error on the light curve data,
        so as to keep the systematic error in control


    """

    def __init__(
        self,
        light_curve_model,
        sample_times,
        light_curve_data,
        filters = None, 
        trigger_time =0.0,
        error_budget=1.0,
        detection_limit = np.inf,
        verbose = False
    ):
        super().__init__(light_curve_model, sample_times, verbose)
        self.error_budget = error_budget
        data_time = light_curve_data['phase'].to_numpy() 
        self.data_time = data_time - trigger_time
        self.data_lum = light_curve_data['Lbb'].to_numpy()
        self.data_sigma = light_curve_data['Lbb_unc'].to_numpy()
        self.detection_limit = detection_limit

    def __repr__(self):
        return self.__class__.__name__
    
    def update_lightcurve_reference(self,lbol, _):
        return (self.sample_times + self.parameters["timeshift"], lbol)
    
    def log_likelihood(self, lc_data):
        em_err_param = self.parameters.get('em_syserr', self.error_budget)
        
        data_sigma = np.sqrt(self.data_sigma**2 + em_err_param**2)
        est_lum = utils.autocomplete_data(self.data_time, *lc_data )
        minus_chisquare, gaussprob = chisquare_gaussianlog_from_lc_data(
                est_lum, self.data_lum, data_sigma, em_err_param, lim=self.detection_limit)
        if isinstance(minus_chisquare, bool):
            return np.nan_to_num(-np.inf)
        else:            
            return minus_chisquare + gaussprob


def truncated_gaussian(m_det, m_err, m_est, lim):
    a, b = (-np.inf - m_est) / m_err, (lim - m_est) / m_err
    return truncnorm.logpdf(m_det, a, b, loc=m_est, scale=m_err)


def chisquare_gaussianlog_from_lc_data(est_mag, data_mag, data_sigma, upperlim_sigma, lim=np.inf):

    # seperate the data into bounds (inf err) and actual measurement
    infIdx = np.where(~np.isfinite(data_sigma))[0]
    finiteIdx = np.where(np.isfinite(data_sigma))[0]

    # evaluate the chisquare
    if len(finiteIdx) >= 1:
        minus_chisquare = np.sum(
            truncated_gaussian(data_mag[finiteIdx], data_sigma[finiteIdx],
                               est_mag[finiteIdx], lim)
                )
    else:
        minus_chisquare = 0.0

    if np.isnan(minus_chisquare):
        sanity_check_passed = False 
        return sanity_check_passed, -np.inf

    # evaluate the data with infinite error
    gausslogsf=np.zeros(2) ##hack if len(infIdx)==0
    if len(infIdx) > 0:
        gausslogsf = norm.logsf(
                data_mag[infIdx], est_mag[infIdx], upperlim_sigma
            )
    return minus_chisquare, np.sum(gausslogsf)


class OpticalLightCurve(EMTransientLikelihood):
    """legacy class for optical kilonova / GRB / kilonova-GRB afterglow likelihood object

    Parameters
    ----------
    light_curve_model: `nmma.em.SVDLightCurveModel`
        And object which computes the light curve of a kilonova signal,
        given a set of parameters
    filters: list, str
        A list of filters to be taken for analysis
        E.g. "u", "g", "r", "i", "z", "y", "J", "H", "K"
    light_curve_data: dict
        Dictionary of light curve data returned from nmma.em.utils.loadEvent
    trigger_time: float
        Time of the kilonova trigger in Modified Julian Day
    error_budget: float (default:1)
        Additionally introduced statistical error on the light curve data,
        so as to keep the systematic error in control
    tmin: float (default:0)
        Days from trigger_time to be started analysing
    tmax: float (default:14)
        Days from trigger_time to be ended analysing

    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        a set of  model parameters

    """

    def __init__(
        self,
        light_curve_model,
        filters,
        light_curve_data,
        trigger_time,
        detection_limit=None,
        error_budget=1.0,
        tmin=0.0,
        tmax=14.0,
        verbose=False,
    ):
        super().__init__(
                light_curve_model=light_curve_model,
                light_curve_data = light_curve_data,
                trigger_time = trigger_time,
                filters= filters,
                detection_limit = detection_limit,
                error_budget = error_budget,
                tmin = tmin,
                tmax = tmax, 
                verbose = verbose
                )
class BolometricLightCurve(EMTransientLikelihood):
    """Legacy class for bolometric likelihood objects

    Parameters
    ----------
    light_curve_model: `nmma.em.SimpleBolometricLightCurveModel`
        And object which computes the light curve of a kilonova signal,
        given a set of parameters
    light_curve_data: dict
        Dictionary of light curve data returned from nmma.em.utils.loadEvent
    error_budget: float (default:1)
        Additionally introduced statistical error on the light curve data,
        so as to keep the systematic error in control
    tmin: float (default:0)
        Days from trigger_time to be started analysing
    tmax: float (default:14)
        Days from trigger_time to be ended analysing

    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        a set of  model parameters

    """

    def __init__(
        self,
        light_curve_model,
        light_curve_data,
        detection_limit=None,
        error_budget=1.0,
        tmin=0.0,
        tmax=14.0,
        verbose=False
    ):
        super().__init__(
                light_curve_model=light_curve_model,
                light_curve_data = light_curve_data,
                detection_limit = detection_limit,
                error_budget = error_budget,
                tmin = tmin,
                tmax = tmax, 
                verbose = verbose
                )