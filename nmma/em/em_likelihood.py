from __future__ import division
import numpy as np
from scipy.stats import norm, truncnorm
from ast import literal_eval
from astropy.time import Time
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)
from ..joint.base import NMMABaseLikelihood, initialisation_args_from_signature_and_namespace
from . import model, utils, systematics



def setup_em_kwargs(priors, data_dump, args,  logger=None):
    #Prerequisites
    ## get lc_data and filters
    light_curve_data= data_dump["light_curve_data"]
    filters = utils.set_filters(args)
    if not filters:
        filters = list(light_curve_data.keys())

    ## setup the light curve model for this transient class and filters
    lc_model = model.identify_model_type(args)
    light_curve_model = model.create_light_curve_model_from_args(lc_model, args, filters)

    try:
        trigger_time = Time(args.trigger_time).mjd
    except ValueError:
        trigger_time = Time(args.trigger_time, format=getattr(args, "time_format", "mjd")).mjd
    except ArithmeticError:
        trigger_time = None

    em_kwargs = initialisation_args_from_signature_and_namespace(
        EMTransientLikelihood, args, ['em_', 'kilonova_']
        )

    # add kwargs manually
    em_likelihood_kwargs = dict(
        light_curve_model=light_curve_model,light_curve_data=light_curve_data,
        priors = priors, filters=filters,
        error_budget=args.em_error_budget,
        trigger_time = trigger_time
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
        Dictionary of light curve data 
    priors: dict, optional
        A dictionary of prior distributions for the model parameters
    filters: list, str, None
        A list of filters to be taken for analysis
        E.g. "u", "g", "r", "i", "z", "y", "J", "H", "K"
    detection_limit: float or dict, default: np.inf
    trigger_time: float, default: None
        Time of the em trigger in Modified Julian Day, by default earliest time in the light curve data
    error_budget: Any (default:1)
        Additionally introduced statistical error on the light curve data,
        so as to keep the systematic error under control. This will only be used if the parameters-dict does not containt a 'em_syserr' sampling parameter.

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
        trigger_time=None,
        error_budget=1.0,
        verbose=False,
        param_conv_func = None, 
        **kwargs
    ):  

        ### FIXME add better criterion to switch modes
        if filters:
            model_type = OpticalTransient
        else:
            model_type=BolometricTransient

        sub_model = model_type(
                light_curve_model, light_curve_data, 
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


    def __init__(self, light_curve_model, detection_limit, error_budget, verbose):
        self.light_curve_model = light_curve_model
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
               
    def sanity_check_passed(self, model_lc):
        if model_lc is None:
            if self.verbose:
                print(f"Model light curve generation failed for {self.parameters}"\
                      "returning -inf log_likelihood")
            return False
        return True
    
    def log_likelihood(self):
        model_lc_data = self.light_curve_model.gen_detector_lc(self.parameters)
        
        if not self.sanity_check_passed(model_lc_data):      
            return np.nan_to_num(-np.inf)
        
        # retrieve usable lightcurve data
        expected_observations = self.update_lightcurve_reference(model_lc_data)

        # compare the estimated light curve and the measured data
        logL_model = self.band_log_likelihood(expected_observations)
        if self.verbose:
            print(self.parameters, logL_model)
        return logL_model

    def update_lightcurve_reference(self, model_lc_data):
        raise NotImplementedError("This method should be implemented in the subclass")
    
    def band_log_likelihood(self, expected_lc):
        raise NotImplementedError("This method should be implemented in the subclass")

    def truncated_gaussian(self, m_det, loc, scale, upper_lim):

        a = -np.inf # no lower bound of truncation
        b = (upper_lim - loc) / scale # upper bound in number of std-deviations
        return truncnorm.logpdf(m_det, a, b, loc=loc, scale=scale)


    def chisquare_gaussianlog_from_lc_data(self, est_mag, data_mag, data_sigma, upperlim_sigma, lim=np.inf):

        # seperate the data into bounds (inf err) and actual measurement
        finiteIdx = np.isfinite(data_sigma)
        infIdx = ~finiteIdx

        # evaluate the chisquare
        if finiteIdx.sum() >= 1:
            minus_chisquare = np.sum(self.truncated_gaussian(data_mag[finiteIdx], 
                loc=est_mag[finiteIdx], scale=data_sigma[finiteIdx], upper_lim=lim))

            ## santiy check:if the chisquare is ill-behaved,
            # we explicitly catch it as Bool in band_log_likelihood
            if np.isnan(minus_chisquare):
                sanity_check_passed = False 
                return sanity_check_passed, -np.inf
        else:
            minus_chisquare = 0.0

        # evaluate the data with infinite error
        gausslogsf=np.zeros(2) ##hack if len(infIdx)==0
        if infIdx.sum() > 0:
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
    light_curve_data: dict
        Dictionary of light curve data 
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
        light_curve_model, light_curve_data,
        filters,
        trigger_time,
        error_budget=1.0,
        detection_limit=np.inf,
        verbose = False,
        systematics_file=None,
        priors=None,
        **kwargs
    ):  
        
        self.observed_filters = filters
        self.model_filter_mapping, self.obs_average_mapping = utils.get_filter_name_mapping(filters)

        super().__init__(light_curve_model, detection_limit, error_budget, verbose)
        
        if priors is not None:
            self.light_curve_model.check_vs_priors(priors)
        
        ##setup light curve data
        self.setup_light_curve_data(light_curve_data, trigger_time)

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

    def setup_light_curve_data(self, light_curve_data, trigger_time):
        """Set up the light curve data for the EM transient

        Parameters
        ----------
        light_curve_data: dict
            Dictionary of light curve data, with keys as filters and values as arrays of time, magnitude, and uncertainty
        trigger_time: float, optional
            Time of the kilonova trigger in Modified Julian Day. If not provided, the minimum time in the data will be used

        """

        lc_times = {}
        lc_mags = {}
        lc_uncertainties = {}

        min_time = np.inf
        for filt, sub_dict in light_curve_data.items():
            lc_mags[filt] = np.array(sub_dict['mag'])
            lc_uncertainties[filt] = np.array(sub_dict['mag_error'])
            lc_times[filt] = np.array(sub_dict['time'])
            min_time = np.minimum(min_time, np.min(sub_dict['time']))

        if trigger_time is None:
            trigger_time = min_time
            if self.verbose:
                print(f"trigger_time is not provided, analysis \
                will use inferred trigger time of {trigger_time}")
        elif trigger_time > min_time:
            raise ValueError(
                f"trigger_time {trigger_time} is later than earliest data time {min_time}. "
                "Please provide a valid trigger time."
            )
        lc_times = {filt: lc_times[filt] - trigger_time for filt in lc_times}

        self.light_curve_times = lc_times
        self.light_curves = lc_mags
        self.light_curve_uncertainties = lc_uncertainties
        self.trigger_time = trigger_time

    def set_detection_limit(self, detection_limit):
        if detection_limit is None:
            detection_limit = np.inf
        self.detection_limit = utils.set_filter_associated_dict(detection_limit, self.observed_filters)

    def adjust_error_budget(self, error_budget):
        if isinstance(error_budget, str):
            error_budget = literal_eval(error_budget)
        self.error_budget   = utils.set_filter_associated_dict(error_budget, self.observed_filters, default_limit=1.)
                                
        self.compute_em_err = self.em_err_from_budget

    def em_err_from_budget(self, filt, _):
        return self.error_budget[filt]

    def setup_time_systematics(self, time_dep_sys_dict):
        #get the time nodes and the filters
        self.systematics_time_nodes = np.round(
            np.linspace(self.light_curve_model.model_times[0], 
                        self.light_curve_model.model_times[-1],
                        time_dep_sys_dict["time_nodes"]),
            decimals=2)
        yaml_filters = list(time_dep_sys_dict["filters"])
        systematics.validate_filters(yaml_filters)

        #iterate over the filters and assign them to a systematics filter group
        systematics_filters = {}
        for filter_group in yaml_filters:
            #this should only be the case if no filters are specified
            if filter_group is None:
                systematics_filters = {filt: 'all' for filt in self.observed_filters}
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
    
    def em_err_from_parameters(self, *_):
        return self.parameters['em_syserr']
            
    def update_lightcurve_reference(self, model_lc_data):
        "Map the output of the light curve model to the expected observations"
        expected_mags = {}
        obs_times, lc_data = model_lc_data
        for filt in self.observed_filters:
            try:
                # observable times and magnitudes according to the model
                obs_mags = lc_data[self.model_filter_mapping[filt]]

                # modelled mags at actual observing times, assume non-detections 
                # if the observed times fall outside the reliably modelled times
                expected_mags[filt] = utils.autocomplete_data(
                    self.light_curve_times[filt], obs_times, obs_mags,extrapolate=np.inf)
            except KeyError:
                # if the model does not provide data for an observed filter, 
                # we can try some known averages
                helper_mags = {}
                for helper_filt in self.obs_average_mapping[filt]:
                    obs_mags = lc_data[self.model_filter_mapping[helper_filt]]
                    helper_mags[helper_filt] = utils.autocomplete_data(
                        self.light_curve_times[filt], obs_times, obs_mags, extrapolate=np.inf)
                expected_mags[filt] = utils.average_mags(helper_mags, filt)
            # if not np.isfinite(expected_mags[filt]).all():
            #     breakpoint()
        return expected_mags

    def band_log_likelihood(self, expected_mags):
        minus_chisquare_total = 0.0
        gaussprob_total = 0.0
        for filt in self.observed_filters:
            systematic_em_err = self.compute_em_err(filt, self.light_curve_times[filt])
            data_sigma = np.sqrt(self.light_curve_uncertainties[filt]**2 + systematic_em_err**2)
            minus_chisquare, gaussprob = self.chisquare_gaussianlog_from_lc_data( 
                expected_mags[filt], self.light_curves[filt] , data_sigma,  systematic_em_err, 
                lim=self.detection_limit[filt]
            )
            if (minus_chisquare is False):
                #this should only be the case if also (self.parameters['timeshift'] <= self.light_curve_times[filt][0] ):
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
    light_curve_data: dict
        Dictionary of light curve data 
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
        light_curve_data,
        trigger_time=None,
        error_budget=1.0,
        detection_limit=np.inf,
        verbose=False,
        **kwargs # to catch a few args of an OpticalTransient
    ):
        super().__init__(light_curve_model, detection_limit, error_budget, verbose)
        data_time = light_curve_data['phase'].to_numpy()
        if trigger_time is None:
            trigger_time = np.min(data_time)
        self.light_curve_times = data_time - trigger_time
        self.light_curves = light_curve_data['Lbb'].to_numpy()
        self.light_curve_uncertainties = light_curve_data['Lbb_unc'].to_numpy()
        self.trigger_time = trigger_time

    def __repr__(self):
        return self.__class__.__name__
    
    def update_lightcurve_reference(self, model_lc_data):
        return utils.autocomplete_data(self.light_curve_times, *model_lc_data)
    
    def sanity_check_passed(self, model_lbol):
        ## FIXME when would the latter ever be the case?
        # Better let lc return None!
        time, model = model_lbol
        if not np.isfinite(model).any() or (model == 0.0).all():
            return False
        return True
    
    def band_log_likelihood(self, expected_lum):
        em_err_param = self.parameters.get('em_syserr', self.error_budget)
        data_sigma = np.sqrt(self.light_curve_uncertainties**2 + em_err_param**2)
        
        minus_chisquare, gaussprob = self.chisquare_gaussianlog_from_lc_data(
            expected_lum, self.light_curves, data_sigma, em_err_param, lim=self.detection_limit)
        if minus_chisquare is False:
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
        Dictionary of light curve data 
    trigger_time: float
        Time of the kilonova trigger in Modified Julian Day
    error_budget: float (default:1)
        Additionally introduced statistical error on the light curve data,
        so as to keep the systematic error in control

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
        verbose=False,
    ):
        
        super().__init__(
                light_curve_model=light_curve_model,
                light_curve_data = light_curve_data,
                trigger_time = trigger_time,
                filters= filters,
                detection_limit = detection_limit,
                error_budget = error_budget,
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
        Dictionary of light curve data 
    error_budget: float (default:1)
        Additionally introduced statistical error on the light curve data,
        so as to keep the systematic error in control

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
        verbose=False
    ):
        super().__init__(
                light_curve_model=light_curve_model,
                light_curve_data = light_curve_data,
                detection_limit = detection_limit,
                error_budget = error_budget,
                verbose = verbose
                )