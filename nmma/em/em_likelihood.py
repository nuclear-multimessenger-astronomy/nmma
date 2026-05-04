import numpy as np
from scipy.stats import norm, truncnorm
from ..core.base import NMMALikelihood, initialisation_args_from_signature_and_namespace
from ..core.conversion import convert_mtot_mni
from ..core.utils import read_trigger_time
from . import model, utils, systematics
from .lightcurve_handling import post_process_bestfit as lch_bestfit
from .plotting_utils import bolometric_lc_plot




def setup_em_kwargs(priors, data_dump, args,  logger=None):
    #Prerequisites
    ## get lc_data and filters
    light_curve_data= data_dump["light_curve_data"]
    filters = data_dump['filters']

    ## setup the light curve model for this transient class and filters
    lc_model = model.identify_model_type(args)
    light_curve_model = model.create_light_curve_model_from_args(lc_model, args, filters)
    trigger_time = read_trigger_time(None, args)
    light_curve_data = utils.setup_filtered_lc_data(light_curve_data, trigger_time)
    light_curve_data = utils.check_model_time_consistency(light_curve_data, light_curve_model, priors, args.injection)
    sys_handler = systematics.FilterSystematicsHandler(filters, 
        data_dump['systematics_dict'], error_budget=args.em_error_budget,
        light_curve_times=light_curve_data[0])

    em_kwargs = initialisation_args_from_signature_and_namespace(
        EMTransientLikelihood, args, ['em_', 'kilonova_']
        )

    # add kwargs manually
    em_likelihood_kwargs = dict(
        light_curve_model=light_curve_model,light_curve_data=light_curve_data,
        priors = priors, filters=filters,
        systematics_handler = sys_handler
    )
    return em_kwargs | em_likelihood_kwargs


class EMTransientLikelihood(NMMALikelihood):
    """A generic EM transient likelihood object

    Parameters
    ----------
    light_curve_model: `nmma.em.SVDLightCurveModel`
        And object which computes the light curve of a kilonova-like signal,
        given a set of parameters
    light_curve_data: dict
        Dictionary of light curve data 
    systematics_handler: nmma.em.systematics.SystematicsHandler
        An object to handle modelling systematics in various ways.
    priors: dict, optional
        A dictionary of prior distributions for the model parameters
    filters: list, str, None
        A list of filters to be taken for analysis
        E.g. "u", "g", "r", "i", "z", "y", "J", "H", "K"
    detection_limit: float or dict, default: np.inf
    verbose: bool (default: False)
        If True, print additional information during computation
    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        a set of  model parameters

    """

    def __init__(self, 
        light_curve_model,
        light_curve_data,
        systematics_handler,
        priors,
        filters=None,
        detection_limit=np.inf,
        verbose=False,
        **kwargs
    ):  
                
        basic_transient_args = (light_curve_model, light_curve_data, systematics_handler,
            priors, detection_limit, verbose)
        
        if filters:
            sub_model = MultiFilterTransient(filters, *basic_transient_args)
        else:
            sub_model = BasicEMTransient(*basic_transient_args)

        super().__init__(sub_model, priors, **kwargs)

    def setup_submodel_conversion(self):
        lc_model = self.sub_model.light_curve_model
    
        # parameter conversion as used in EM-only sector
        model_list = lc_model.model if isinstance(lc_model, model.CombinedLightCurveModelContainer) else [lc_model.model]
        if any(model_name in ['AnBa2022_linear', 'AnBa2022_log'] for model_name in model_list):
            self.conv_functions.append(convert_mtot_mni)
        # elif to be extended...

        self.conv_functions.append(self.sub_model.light_curve_model.parameter_conversion)

    def sanity_checks(self):
        return self.sub_model.light_curve_model.good_parameters
    
    def __repr__(self):
        return f"{self.__class__.__name__} based on {self.sub_model.__repr__()}"
    
    def final_diagnostics(self, bestfit_params, args, result=None):
        """Plot the best-fit light curve against the data

        Parameters
        ----------
        bestfit_params: dict
            Dictionary of best-fit parameters

        Returns
        -------
        fig: matplotlib.figure.Figure
            The figure object containing the plot

        """
        return self.sub_model.final_diagnostics(bestfit_params, args, result)

    def posterior_conversion(self, posterior_samples):
        if 'log10_mej_dyn' in posterior_samples and 'log10_mej_wind' in posterior_samples:
            posterior_samples['log10_mej'] = np.log10(10**(posterior_samples['log10_mej_wind'])
                + 10**(posterior_samples['log10_mej_dyn']) )
        if 'thetaWing' in posterior_samples and 'thetaCore' in posterior_samples:
            posterior_samples['alphaWing'] = posterior_samples['thetaWing'] / posterior_samples['thetaCore']
        elif 'alphaWing' in posterior_samples and 'thetaCore' in posterior_samples:
            posterior_samples['thetaWing'] = posterior_samples['alphaWing'] * posterior_samples['thetaCore']
        return posterior_samples
    
       

class BasicEMTransient:
    """A basic bolometric EM transient object

    Parameters
    ----------
    light_curve_model: `nmma.em.SVDLightCurveModel`
        An object which computes the light curve of a kilonova-like signal,
        given a set of parameters
    light_curve_data: dict
        Dictionary of light curve data
    systematics_handler: nmma.em.systematics.SystematicsHandler
        An object to handle modelling systematics in various ways.
    priors: dict, optional
        A dictionary of bilby-style priors
    detection_limit: float (default: np.inf)
        Detection limit for the light curve data
    verbose: bool (default: False)
        If True, print additional information during computation

    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A likelihood object, able to compute the likelihood of the data given
        a set of  model parameters

    """


    def __init__(self, light_curve_model, light_curve_data, systematics_handler, priors, 
                 detection_limit, verbose):

        self.light_curve_model = light_curve_model

        self.light_curve_model.check_vs_priors(priors)

        (self.light_curve_times, self.light_curves, 
         self.light_curve_uncertainties, self.trigger_time) = light_curve_data
        
        systematics_handler.reset(self.light_curve_model.model_times, priors)
        self.systematics_handler = systematics_handler

        self.verbose = verbose
        self.set_detection_limit(detection_limit)

    def set_detection_limit(self, detection_limit):
        self.detection_limit = detection_limit 

    def __repr__(self):
        return f"{self.__class__.__name__} (light_curve_model={self.light_curve_model})"   
    
    def log_likelihood(self, parameters):
        obs_times, model_lc = self.light_curve_model.gen_detector_lc(parameters)
        
        # sanity check: did the model return a valid light curve?
        if not self.sanity_check(model_lc):
            if self.verbose:
                print(f"Model light curve generation failed for {parameters}"\
                      "returning -inf log_likelihood")
            return np.nan_to_num(-np.inf)
        
        # retrieve usable lightcurve data
        expected_observations = self.update_lightcurve_reference(obs_times, model_lc)

        # compare the estimated light curve and the measured data
        obs_error = self.systematics_handler(parameters)
        logL_model = self.band_log_likelihood(expected_observations, obs_error)
        if self.verbose:
            print(parameters, logL_model)
        return logL_model

    def sanity_check(self, model_lc):
        if not np.isfinite(model_lc).any():
            return False
        return True

    def update_lightcurve_reference(self, obs_times, model_lc):
        return utils.autocomplete_data(self.light_curve_times, obs_times, model_lc)
    
    def band_log_likelihood(self, expected_lc, obs_error):
        data_sigma = np.sqrt(self.light_curve_uncertainties**2 + obs_error**2)

        minus_chisquare, gaussprob = self.chisquare_gaussianlog_from_lc_data(
            expected_lc, self.light_curves, data_sigma, obs_error, lim=self.detection_limit)
        if minus_chisquare is False:
            return np.nan_to_num(-np.inf)
        else:
            return minus_chisquare + gaussprob

    def chisquare_gaussianlog_from_lc_data(self, est_mag, data_mag, data_sigma, upperlim_sigma, lim=np.inf):

        # seperate the data into bounds (inf err) and actual measurement
        finiteIdx = np.isfinite(data_sigma)
        infIdx = ~finiteIdx

        # evaluate the chisquare
        if finiteIdx.sum() >= 1:
            minus_chisquare = np.sum(self.truncated_gaussian(data_mag[finiteIdx], 
                loc=est_mag[finiteIdx], scale=data_sigma[finiteIdx], upper_lim=lim))

            ## sanity check: if the chisquare is ill-behaved,
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
                    data_mag[infIdx], est_mag[infIdx], upperlim_sigma[infIdx]
                )
        return minus_chisquare, np.sum(gausslogsf)
    
    def truncated_gaussian(self, m_det, loc, scale, upper_lim):

        a = -np.inf # no lower bound of truncation
        b = (upper_lim - loc) / scale # upper bound in number of std-deviations
        return truncnorm.logpdf(m_det, a, b, loc=loc, scale=scale)
    
    def final_diagnostics(self, bestfit_params, args, result=None):
        obs_times, obs_lc = self.light_curve_model.gen_detector_lc(bestfit_params)
        if result is None:
            save_path = f'{args.outdir}/{args.label}_bol_lightcurve.png'
        save_path = f'{result.outdir}/{result.label}_bol_lightcurve.png'
        return bolometric_lc_plot(self, obs_times, obs_lc, save_path = save_path)


class MultiFilterTransient(BasicEMTransient):
    """An EM transient that can be evaluated across multiple filters

    Parameters
    ----------
    filters: list, str
        A list of filters to be taken for analysis
        E.g. "u", "g", "r", "i", "z", "y", "J", "H", "K"
    light_curve_model: `nmma.em.LightCurveModelContainer`
        An object which computes the light curve of a transient signal,
        given a set of parameters
    light_curve_data: dict
        Dictionary of light curve data 
    systematics_handler: nmma.em.systematics.FilterSystematicsHandler
        An object to handle filter-dependent modelling systematics in various ways.
    priors: dict, optional
        Dictionary of prior distributions for the model parameters
    detection_limit: float or dict (default: np.inf)
        Detection limit for the light curve data
    verbose: bool (default: False)
        If True, print additional information during computation

    """

    def __init__( self, filters,
        light_curve_model, light_curve_data, systematics_handler, priors,
        detection_limit, verbose
    ):  
        
        self.observed_filters = filters
        self.model_filter_mapping, self.obs_average_mapping = utils.get_filter_name_mapping(filters)

        super().__init__(light_curve_model, light_curve_data, systematics_handler, priors, 
                         detection_limit, verbose)
        

    def set_detection_limit(self, detection_limit):
        self.detection_limit = utils.set_filter_associated_dict(detection_limit, self.observed_filters)

    def sanity_check(self, model_lc):
        if not model_lc:
            return False
        # this may happen if parameter conversion provides improper values, e.g. no E0 as EoS conversion entails a black hole
        if any([np.isinf(mag).all() for mag in model_lc.values()]):
            return False
        return True
    
    def update_lightcurve_reference(self, obs_times, lc_data):
        "Map the output of the light curve model to the expected observations"
        expected_mags = {}
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
                
        return expected_mags
    
    def band_log_likelihood(self, expected_mags, obs_error):
        minus_chisquare_total = 0.0
        gaussprob_total = 0.0
        for filt, err in obs_error.items():
            data_sigma = np.sqrt(self.light_curve_uncertainties[filt]**2 + err**2)
            minus_chisquare, gaussprob = self.chisquare_gaussianlog_from_lc_data( 
                expected_mags[filt], self.light_curves[filt] , data_sigma,  err, 
                lim=self.detection_limit[filt]
            )
            if (minus_chisquare is False):
                #this should only be the case if also (parameters['timeshift'] <= self.light_curve_times[filt][0] ):
                return np.nan_to_num(-np.inf)
            else:
                minus_chisquare_total+=minus_chisquare
                gaussprob_total += gaussprob
        return minus_chisquare_total + gaussprob_total
   
    def final_diagnostics(self, bestfit_params, args, result=None):
        return lch_bestfit(self, bestfit_params, args, result)