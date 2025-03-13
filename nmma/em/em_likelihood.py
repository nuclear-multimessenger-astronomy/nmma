from __future__ import division
import numpy as np
from scipy.stats import norm, truncnorm

from ..joint.base import NMMABaseLikelihood, initialisation_args_from_signature_and_namespace
from ..joint.conversion import distance_modulus_nmma
from . import model, utils, systematics



def setup_em_kwargs(priors, data_dump, args,  logger=None):
    #Prerequisites
    ## get lc_data and filters
    light_curve_data= data_dump["light_curve_data"]
    filters=args.filters
    if not filters:
        filters = list(light_curve_data.keys())

    ## identify what kind of transient we are dealing with
    lc_model_class= model.lc_model_class_from_str(args.em_transient_class)
    ## setup the light curve model for this transient class and filters
    light_curve_model = model.create_light_curve_model_from_args(lc_model_class, args, filters)



    em_kwargs = initialisation_args_from_signature_and_namespace(
        EMTransientLikelihood, args, ['em_transient_', 'kilonova_']
        )

    # add kwargs manually
    em_likelihood_kwargs = dict(
        light_curve_model=light_curve_model,light_curve_data=light_curve_data,
        priors = priors, filters=filters,
        error_budget=args.em_transient_error,
    )
    return em_kwargs | em_likelihood_kwargs


   


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
        so as to keep the systematic error under control. This will only be used if the parameters-dict does not containt a 'em_syserr' sampling parameter.
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
        detection_limit=None,
        trigger_time=0.,
        error_budget=1.0,
        tmin=0.0,
        tmax=14.0,
        verbose=False,
        param_conv_func = None, 
        **kwargs
    ):  
        sample_times = kwargs.get('sample_times', np.arange(tmin, tmax, 0.1))
        ### FIXME add better criterion to switch modes
        if filters:
            model_type = OpticalTransient
        else:
            model_type=BolometricTransient

        sub_model = model_type(
                light_curve_model, sample_times, light_curve_data, 
                filters = filters, 
                trigger_time=trigger_time,
                error_budget= error_budget, 
                detection_limit= detection_limit, 
                verbose=verbose,
                priors=priors)


        # self.light_curve_model = light_curve_model
        # self.verbose = verbose
        #FIXME priors seems unnecessary here
        super().__init__(sub_model=sub_model, priors=priors, param_conversion_func=param_conv_func, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__} based on {self.sub_model.__repr__()}"
    
       

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


    def __init__(self, light_curve_model, sample_times, 
                 detection_limit, error_budget, verbose
                 ):
        self.light_curve_model = light_curve_model
        self.sample_times = sample_times
        self.error_budget = error_budget
        
        self.verbose = verbose
        self.set_detection_limit(detection_limit)

    def set_detection_limit(self, detection_limit):
        #FIXME this is more of a legacy convenience, probably better to
        # initialise with np.inf in the first place?
        if detection_limit is None:
            self.detection_limit = np.inf
        else: 
            self.detection_limit = detection_limit 

    def __repr__(self):
        return f"{self.__class__.__name__} (light_curve_model={self.light_curve_model})"
               
               
    
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

    def truncated_gaussian(self, m_det, m_err, m_est, lim):
        a, b = (-np.inf - m_est) / m_err, (lim - m_est) / m_err
        return truncnorm.logpdf(m_det, a, b, loc=m_est, scale=m_err)


    def chisquare_gaussianlog_from_lc_data(self, est_mag, data_mag, data_sigma, upperlim_sigma, lim=np.inf):

        # seperate the data into bounds (inf err) and actual measurement
        infIdx = np.where(~np.isfinite(data_sigma))[0]
        finiteIdx = np.where(np.isfinite(data_sigma))[0]

        # evaluate the chisquare
        if len(finiteIdx) >= 1:
            minus_chisquare = np.sum(
                self.truncated_gaussian(data_mag[finiteIdx], data_sigma[finiteIdx],
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

class OpticalTransient(BaseEMTransient):
    """An EM transient that can be evaluated across multiple filters

    Parameters
    ----------
    light_curve_model: `nmma.em.LightCurveModelContainer`
        An object which computes the light curve of a transient ignal,
        given a set of parameters
    sample_times: array-like
        Array of times at which the light curve is sampled
    light_curve_data: dict
        Dictionary of light curve data returned from nmma.em.utils.loadEvent
    filters: list, str
        A list of filters to be taken for analysis
        E.g. "u", "g", "r", "i", "z", "y", "J", "H", "K"
    trigger_time: float
        Time of the kilonova trigger in Modified Julian Day
    error_budget: float (default: 1.0)
        Additionally introduced statistical error on the light curve data,
        so as to keep the systematic error in control
    detection_limit: float or dict (default: np.inf)
        Detection limit for the light curve data
    verbose: bool (default: False)
        If True, print additional information during computation
    systematics_file: str, optional
        Path to a YAML file containing systematic error information
    priors: dict, optional
        Dictionary of prior distributions for the model parameters

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
        priors=None,
        **kwargs
    ):  
        
        self.filters = filters
        ##setup light curve data
        self.light_curve_data = utils.process_data(
            light_curve_data, trigger_time, sample_times[0], sample_times[-1])
        
        super().__init__( light_curve_model, sample_times, 
                         detection_limit, error_budget, verbose)
        
        
        #determine_systematic_error_handling
        ## case 1: use systematics_file
        if systematics_file:
            systematics_dict = systematics.load_yaml(systematics_file)
            systematics.validate_only_one_true(systematics_dict)
            time_dep_sys_dict = systematics_dict["config"]["withTime"]
            # case 1a: time-dependent systematics
            if time_dep_sys_dict['value']:
                self.setup_time_systematics(time_dep_sys_dict)
            else:
                # case 1b: no time-dependency and sample with 
                # time-independent error-> this is actually case 2
                ## FIXME would it not be more natural to still have a filter-dependent error, even if it does not vary in time?
                self.compute_em_err = self.em_err_from_parameters

        # case 2: sample over general limit
        elif 'em_syserr' in priors:
            self.compute_em_err = self.em_err_from_parameters
        
        #case 3: preset general limit
        else:
            self.adjust_error_budget(self.error_budget)

    def set_detection_limit(self, detection_limit):
        if detection_limit is None:
            detection_limit = np.inf
        if isinstance(detection_limit, (int, float)):
            self.detection_limit = {filt: detection_limit for filt in self.filters}
        elif isinstance(self.detection_limit, dict):
            self.detection_limit = {filt: detection_limit.get(filt, np.inf) for filt in self.filters}

    def adjust_error_budget(self, error_budget):
        if isinstance(error_budget, (int, float, complex)):
            self.error_budget = {filt:error_budget for filt in self.filters}

        elif isinstance(error_budget, dict):
            for filt in self.filters:
                if filt not in error_budget:
                    raise ValueError(f"filter {filt} missing from error_budget")
            # NOTE We could be more generous and set a default (1?) instead
                    
        self.compute_em_err = self.em_err_from_budget

    def em_err_from_budget(self, filt, _):
        return self.error_budget[filt]

    def setup_time_systematics(self, time_dep_sys_dict):
        #get the time nodes and the filters
        self.systematics_time_nodes = np.round(
            np.linspace(self.sample_times[0], self.sample_times[-1],
                        time_dep_sys_dict["time_nodes"]),
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
    
    def em_err_from_systematics_sampling(self, filt, data_time):
        systematics_filt = self.systematics_filters[filt]
        sampled_filter_systematics = [self.parameters[f"em_syserr_{systematics_filt}{i}"] for i in range(len(self.systematics_time_nodes))]
        return utils.autocomplete_data(data_time, self.systematics_time_nodes, sampled_filter_systematics)
    
    
    ##FIXME Check if this is the right way to handle the error budget
    def em_err_from_parameters(self, *_):
        return self.parameters['em_syserr']


            
    def update_lightcurve_reference(self, _, model_mags):
        lc_data = {}
        t0 = self.parameters["timeshift"]
        d_lum = self.parameters.get("luminosity_distance", 1e-5) ## default 10pc = 1e-5 Mpc
        distance_modulus = distance_modulus_nmma(d_lum)
        for filt, model_mag in model_mags.items():
            usedIdx = np.where(np.isfinite(model_mag))[0]
            if len(usedIdx)<2:
                #no meaningful inter-/extrapolation possible
                lc_data[filt] = (self.sample_times + t0, np.full_like(self.sample_times, np.inf))
            else:
                apparent_magnitude = utils.getFilteredMag(model_mags, filt) + distance_modulus
                lc_data[filt] = (self.sample_times[usedIdx] + t0, apparent_magnitude[usedIdx])
        return lc_data
    
    def band_log_likelihood(self, lc_data):
        minus_chisquare_total = 0.0
        gaussprob_total = 0.0
        for filt in self.filters:
            # decompose the data
            data_time, data_mag, data_sigma  = self.light_curve_data[filt].T
            systematic_em_err = self.compute_em_err(filt, data_time)
            data_sigma = np.sqrt(data_sigma**2 + systematic_em_err**2)
            model_time, model_mags = lc_data[filt]
            # evaluate the light curve magnitude at the data points
            est_mag = utils.autocomplete_data(data_time, model_time, model_mags)
            minus_chisquare, gaussprob = self.chisquare_gaussianlog_from_lc_data( 
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
    light_curve_model: `nmma.em.SimpleBolometricLightCurveModel`
        An object which computes the light curve of a kilonova signal,
        given a set of parameters
    sample_times: array-like
        Array of times at which the light curve is sampled
    light_curve_data: dict
        Dictionary of light curve data returned from nmma.em.utils.loadEvent
    trigger_time: float (default: 0.0)
        Time of the kilonova trigger in Modified Julian Day
    error_budget: float (default: 1.0)
        Additionally introduced statistical error on the light curve data,
        so as to keep the systematic error in control
    detection_limit: float (default: np.inf)
        Detection limit for the light curve data
    verbose: bool (default: False)
        If True, print additional information during computation

    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        a set of model parameters

    """

    def __init__(
        self,
        light_curve_model,
        sample_times,
        light_curve_data,
        trigger_time=0.0,
        error_budget=1.0,
        detection_limit=np.inf,
        verbose=False,
        **kwargs # to catch a few args of an OpticalTransient
    ):
        super().__init__(light_curve_model, sample_times, 
                         detection_limit, error_budget, verbose)
        data_time = light_curve_data['phase'].to_numpy()
        self.data_time = data_time - trigger_time
        self.data_lum = light_curve_data['Lbb'].to_numpy()
        self.data_sigma = light_curve_data['Lbb_unc'].to_numpy()

    def __repr__(self):
        return self.__class__.__name__
    
    def update_lightcurve_reference(self, lbol, _):
        return (self.sample_times + self.parameters["timeshift"], lbol)
    
    def band_log_likelihood(self, lc_data):
        em_err_param = self.parameters.get('em_syserr', self.error_budget)
        
        data_sigma = np.sqrt(self.data_sigma**2 + em_err_param**2)
        est_lum = utils.autocomplete_data(self.data_time, *lc_data)
        minus_chisquare, gaussprob = self.chisquare_gaussianlog_from_lc_data(
            est_lum, self.data_lum, data_sigma, em_err_param, lim=self.detection_limit)
        if isinstance(minus_chisquare, bool):
            return np.nan_to_num(-np.inf)
        else:
            return minus_chisquare + gaussprob


class OpticalLightCurve(EMTransientLikelihood):
    """legacy class for optical kilonova / GRB / kilonova-GRB afterglow likelihood object

    Parameters
    ----------
    light_curve_model: `nmma.em.LightCurveModelContainer`
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