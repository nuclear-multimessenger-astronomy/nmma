import numpy as np
from scipy.stats import norm, truncnorm

from ..core.base import NMMALikelihood, initialisation_args_from_signature_and_namespace
from ..core.conversion import convert_mtot_mni
from ..core.utils import read_trigger_time
from . import model, systematics, utils
from .lightcurve_handling import post_process_bestfit as lch_bestfit
from .plotting_utils import bolometric_lc_plot


def setup_em_kwargs(priors, data_dump, args, logger=None):
    """Gather everything an EM likelihood needs in order to be built.

    The data dump carries the observations and the systematics settings, but
    a likelihood also needs a light curve model and data aligned on the
    trigger time. This collects all of it, together with the options the user
    passed on the command line, into a single dictionary.

    Parameters
    ----------
    priors: bilby.core.prior.PriorDict
        Priors on the model parameters.
    data_dump: dict
        Holds the light curve data, the filters and the systematics settings.
    args: argparse.Namespace
        Parsed command-line arguments.
    logger: logging.Logger, optional
        Not used here, but kept so that every sector shares the same
        signature.

    Returns
    -------
    dict
        Keyword arguments, ready to be passed to `EMTransientLikelihood`.
    """

    # Prerequisites
    # get lc_data and filters
    light_curve_data = data_dump["light_curve_data"]
    filters = data_dump["filters"]

    # setup the light curve model for this transient class and filters
    light_curve_model = model.create_light_curve_model_from_args(args, filters)
    trigger_time = read_trigger_time(None, args)
    light_curve_data = utils.setup_filtered_lc_data(light_curve_data, trigger_time)
    # FIXME weizmann: to be activated separately, after the NMMA
    # documentation work.
    # light_curve_data = utils.check_model_time_consistency(
    #     light_curve_data, light_curve_model, priors, args.injection or None
    # )
    light_curve_data = utils.check_model_time_consistency(
        light_curve_data,
        light_curve_model,
        priors,
        args.injection,
        args.allow_data_cuts,
    )
    sys_handler = systematics.FilterSystematicsHandler(
        filters,
        data_dump["systematics_dict"],
        error_budget=args.em_error_budget,
        light_curve_times=light_curve_data[0],
    )

    em_kwargs = initialisation_args_from_signature_and_namespace(
        EMTransientLikelihood, args, ["em_", "kilonova_"]
    )

    # add kwargs manually
    em_likelihood_kwargs = dict(
        light_curve_model=light_curve_model,
        light_curve_data=light_curve_data,
        priors=priors,
        filters=filters,
        systematics_handler=sys_handler,
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

    def __init__(
        self,
        light_curve_model,
        light_curve_data,
        systematics_handler,
        priors,
        filters=None,
        detection_limit=np.inf,
        verbose=False,
        **kwargs,
    ):
        basic_transient_args = (
            light_curve_model,
            light_curve_data,
            systematics_handler,
            priors,
            detection_limit,
            verbose,
        )

        if filters:
            sub_model = MultiFilterTransient(filters, *basic_transient_args)
        else:
            sub_model = BasicEMTransient(*basic_transient_args)

        super().__init__(sub_model, priors, **kwargs)

    def setup_submodel_conversion(self):
        """Register the parameter conversions this light curve model needs.

        Sampled parameters are not always the ones a model expects, so each
        model brings its own conversion. The AnBa2022 models need one more
        step beforehand, turning the total mass into a nickel mass.
        """

        lc_model = self.sub_model.light_curve_model

        # parameter conversion as used in EM-only sector
        model_list = (
            lc_model.model
            if isinstance(lc_model, model.CombinedLightCurveModelContainer)
            else [lc_model.model]
        )
        if any(
            model_name in ["AnBa2022_linear", "AnBa2022_log"]
            for model_name in model_list
        ):
            self.conv_functions.append(convert_mtot_mni)
        # elif to be extended...

        self.conv_functions.append(
            self.sub_model.light_curve_model.parameter_conversion
        )

    def sanity_checks(self):
        """Tell whether the last set of parameters was usable.

        Returns
        -------
        bool
            False when the light curve model rejected the parameters it was
            last given.
        """

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
        """Add the derived quantities readers usually want to the posterior.

        Sampling gives the dynamical and wind ejecta masses separately, but
        the total ejecta mass is what most analyses quote, so it is added
        here. The same goes for the two interchangeable ways of describing
        the jet wing, thetaWing and alphaWing: whichever was sampled, the
        other one is filled in.

        Parameters
        ----------
        posterior_samples: pandas.DataFrame
            Posterior samples. Modified in place.

        Returns
        -------
        pandas.DataFrame
            The same samples, with the derived columns added.
        """

        if (
            "log10_mej_dyn" in posterior_samples
            and "log10_mej_wind" in posterior_samples
        ):
            posterior_samples["log10_mej"] = np.log10(
                10 ** (posterior_samples["log10_mej_wind"])
                + 10 ** (posterior_samples["log10_mej_dyn"])
            )
        if "thetaWing" in posterior_samples and "thetaCore" in posterior_samples:
            posterior_samples["alphaWing"] = (
                posterior_samples["thetaWing"] / posterior_samples["thetaCore"]
            )
        elif "alphaWing" in posterior_samples and "thetaCore" in posterior_samples:
            posterior_samples["thetaWing"] = (
                posterior_samples["alphaWing"] * posterior_samples["thetaCore"]
            )
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

    def __init__(
        self,
        light_curve_model,
        light_curve_data,
        systematics_handler,
        priors,
        detection_limit,
        verbose,
    ):
        self.light_curve_model = light_curve_model

        self.light_curve_model.check_vs_priors(priors)

        (
            self.light_curve_times,
            self.light_curves,
            self.light_curve_uncertainties,
            self.trigger_time,
        ) = light_curve_data

        systematics_handler.reset(self.light_curve_model.model_times, priors)
        self.systematics_handler = systematics_handler

        self.verbose = verbose
        self.set_detection_limit(detection_limit)

    def set_detection_limit(self, detection_limit):
        """Store the magnitude beyond which nothing could have been seen.

        Parameters
        ----------
        detection_limit: float
            Limiting magnitude, shared by every measurement.
        """
        self.detection_limit = detection_limit

    def __repr__(self):
        return f"{self.__class__.__name__} (light_curve_model={self.light_curve_model})"

    def log_likelihood(self, parameters):
        """Evaluate how well one set of parameters explains the data.

        The model light curve is generated, resampled onto the times that
        were actually observed, and compared to the measurements. Some
        parameter combinations are simply unphysical and yield no usable
        light curve; those are rejected rather than raising.

        Parameters
        ----------
        parameters: dict
            Model parameters to evaluate.

        Returns
        -------
        float
            The log-likelihood, or a very large negative value when the
            parameters were rejected.

        Notes
        -----
        Rejected parameters return the most negative finite float64
        instead of -inf, which is what np.nan_to_num substitutes. The same
        idiom is used by the core, resampling and maximum-mass likelihoods.
        """

        obs_times, model_lc = self.light_curve_model.gen_detector_lc(parameters)

        # sanity check: did the model return a valid light curve?
        if not self.sanity_check(model_lc):
            if self.verbose:
                print(
                    f"Model light curve generation failed for {parameters}"
                    "returning -inf log_likelihood"
                )
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
        """Check that the model returned something worth comparing to data.

        Parameters
        ----------
        model_lc: numpy.ndarray
            Magnitudes the model produced.

        Returns
        -------
        bool
            False when not a single magnitude is finite.
        """
        if not np.isfinite(model_lc).any():
            return False
        return True

    def update_lightcurve_reference(self, obs_times, model_lc):
        """Read the model light curve at the times that were observed.

        A model is evaluated on its own time grid, which rarely matches the
        observing epochs, so it has to be interpolated onto them before any
        comparison can be made.

        Parameters
        ----------
        obs_times: numpy.ndarray
            Times on which the model was evaluated.
        model_lc: numpy.ndarray
            Magnitudes at those times.

        Returns
        -------
        numpy.ndarray
            Model magnitudes, one per observation.
        """
        return utils.autocomplete_data(self.light_curve_times, obs_times, model_lc)

    def band_log_likelihood(self, expected_lc, obs_error):
        """Compare the whole light curve to the data, in one band.

        The systematic uncertainty is added in quadrature to the measurement
        errors before the comparison.

        Parameters
        ----------
        expected_lc: numpy.ndarray
            Model magnitudes at the observing epochs.
        obs_error: numpy.ndarray
            Systematic uncertainty budget.

        Returns
        -------
        float
            Log-likelihood of the band, or the most negative finite float64
            if the comparison turned out ill-behaved.
        """
        data_sigma = np.sqrt(self.light_curve_uncertainties**2 + obs_error**2)

        minus_chisquare, gaussprob = self.chisquare_gaussianlog_from_lc_data(
            expected_lc,
            self.light_curves,
            data_sigma,
            obs_error,
            lim=self.detection_limit,
        )
        if minus_chisquare is False:
            return np.nan_to_num(-np.inf)
        else:
            return minus_chisquare + gaussprob

    def chisquare_gaussianlog_from_lc_data(
        self, est_mag, data_mag, data_sigma, upperlim_sigma, lim=np.inf
    ):
        """Score detections and non-detections, which say different things.

        A detection constrains the magnitude directly and is scored with a
        truncated Gaussian. A non-detection only tells us the source stayed
        fainter than the limit, so it is scored with a survival function
        instead. The two are recognised by their uncertainty: non-detections
        carry an infinite one.

        Parameters
        ----------
        est_mag: numpy.ndarray
            Magnitudes the model predicts.
        data_mag: numpy.ndarray
            Measured magnitudes.
        data_sigma: numpy.ndarray
            Uncertainty on each measurement, infinite for non-detections.
        upperlim_sigma: numpy.ndarray
            Uncertainty to use for the non-detections.
        lim: float, optional
            Detection limit, used as the truncation bound.

        Returns
        -------
        minus_chisquare: float or bool
            Contribution of the detections, or False when it came out as
            NaN. The caller tests for that False to reject the parameters.
        gausslogsf: float
            Contribution of the non-detections.
        """

        # seperate the data into bounds (inf err) and actual measurement
        finiteIdx = np.isfinite(data_sigma)
        infIdx = ~finiteIdx

        # evaluate the chisquare
        if finiteIdx.sum() >= 1:
            minus_chisquare = np.sum(
                self.truncated_gaussian(
                    data_mag[finiteIdx],
                    loc=est_mag[finiteIdx],
                    scale=data_sigma[finiteIdx],
                    upper_lim=lim,
                )
            )

            # sanity check: if the chisquare is ill-behaved,
            # we explicitly catch it as Bool in band_log_likelihood
            if np.isnan(minus_chisquare):
                sanity_check_passed = False
                return sanity_check_passed, -np.inf
        else:
            minus_chisquare = 0.0

        # evaluate the data with infinite error, i.e. upper limits,
        # as Gaussian survival function
        gausslogsf = np.zeros(2)  # hack if len(infIdx)==0
        if infIdx.sum() > 0:
            gausslogsf = norm.logsf(
                data_mag[infIdx], est_mag[infIdx], upperlim_sigma[infIdx]
            )
        return minus_chisquare, np.sum(gausslogsf)

    def truncated_gaussian(self, m_det, loc, scale, upper_lim):
        """Compare a measurement to the model, knowing it was detected.

        A detection carries more information than its magnitude alone: it
        also tells us the source was brighter than the detection limit. The
        Gaussian is therefore truncated at that limit, so that the
        probability is normalised over the magnitudes that could have been
        seen at all.

        Parameters
        ----------
        m_det: numpy.ndarray
            Measured magnitudes.
        loc: numpy.ndarray
            Magnitudes the model predicts.
        scale: numpy.ndarray
            Total uncertainty on each measurement.
        upper_lim: float
            Detection limit, used as the truncation bound.

        Returns
        -------
        numpy.ndarray
            Log-probability of each measurement.
        """

        a = -np.inf  # no lower bound of truncation
        b = (upper_lim - loc) / scale  # upper bound in number of std-deviations
        return truncnorm.logpdf(m_det, a, b, loc=loc, scale=scale)

    def final_diagnostics(self, bestfit_params, args, result=None):
        """Plot the best-fit bolometric light curve over the data.

        Parameters
        ----------
        bestfit_params: dict
            Best-fit parameters to draw.
        args: argparse.Namespace
            Parsed command-line arguments, used for the output path.
        result: bilby.core.result.Result, optional
            Sampling result. When given, its own output directory and label
            name the figure.

        Returns
        -------
        matplotlib.figure.Figure
            The figure that was saved.
        """
        obs_times, obs_lc = self.light_curve_model.gen_detector_lc(bestfit_params)
        # FIXME weizmann: to be activated separately, after the NMMA
        # documentation work.
        # if result is None:
        #     save_path = f"{args.outdir}/{args.label}_bol_lightcurve.png"
        # else:
        #     save_path = f"{result.outdir}/{result.label}_bol_lightcurve.png"
        if result is None:
            save_path = f"{args.outdir}/{args.label}_bol_lightcurve.png"
        save_path = f"{result.outdir}/{result.label}_bol_lightcurve.png"
        return bolometric_lc_plot(self, obs_times, obs_lc, save_path=save_path)


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

    def __init__(
        self,
        filters,
        light_curve_model,
        light_curve_data,
        systematics_handler,
        priors,
        detection_limit,
        verbose,
    ):
        self.observed_filters = filters
        (
            self.model_filter_mapping,
            self.obs_average_mapping,
        ) = utils.get_filter_name_mapping(filters)

        super().__init__(
            light_curve_model,
            light_curve_data,
            systematics_handler,
            priors,
            detection_limit,
            verbose,
        )

    def set_detection_limit(self, detection_limit):
        """Give every filter its own detection limit.

        Filters do not go equally deep, so a single number is spread over
        them all, while a mapping is taken as it is.

        Parameters
        ----------
        detection_limit: float or dict
            One limit for every filter, or one per filter.
        """
        self.detection_limit = utils.set_filter_associated_dict(
            detection_limit, self.observed_filters
        )

    def sanity_check(self, model_lc):
        """Check that the model produced usable magnitudes in every filter.

        Parameters
        ----------
        model_lc: dict
            Magnitudes per filter.

        Returns
        -------
        bool
            False when the model returned nothing, or when one filter is
            infinite throughout.
        """
        if not model_lc:
            return False
        # this may happen if parameter conversion provides improper values, e.g. no E0 as EoS conversion entails a black hole
        if any([np.isinf(mag).all() for mag in model_lc.values()]):
            return False
        return True

    def update_lightcurve_reference(self, obs_times, lc_data):
        """Read the model light curve at the epochs observed in each filter.

        The model does not necessarily cover every filter that was observed.
        When one is missing, it is reconstructed by averaging the neighbouring
        bands the model does provide. Epochs falling outside the range the
        model can be trusted on are treated as non-detections.

        Parameters
        ----------
        obs_times: numpy.ndarray
            Times on which the model was evaluated.
        lc_data: dict
            Model magnitudes, keyed by the model's own filter names.

        Returns
        -------
        dict
            Model magnitudes per observed filter, at the observing epochs.
        """
        expected_mags = {}
        for filt in self.observed_filters:
            try:
                # observable times and magnitudes according to the model
                obs_mags = lc_data[self.model_filter_mapping[filt]]

                # modelled mags at actual observing times, assume non-detections
                # if the observed times fall outside the reliably modelled times
                expected_mags[filt] = utils.autocomplete_data(
                    self.light_curve_times[filt],
                    obs_times,
                    obs_mags,
                    extrapolate=np.inf,
                )
            except KeyError:
                # if the model does not provide data for an observed filter,
                # we can try some known averages
                helper_mags = {}
                for helper_filt in self.obs_average_mapping[filt]:
                    obs_mags = lc_data[self.model_filter_mapping[helper_filt]]
                    helper_mags[helper_filt] = utils.autocomplete_data(
                        self.light_curve_times[filt],
                        obs_times,
                        obs_mags,
                        extrapolate=np.inf,
                    )
                expected_mags[filt] = utils.average_mags(helper_mags, filt)

        return expected_mags

    def band_log_likelihood(self, expected_mags, obs_error):
        """Add up the contribution of every filter.

        Each filter carries its own uncertainty budget and its own detection
        limit, so they are scored separately and summed.

        Parameters
        ----------
        expected_mags: dict
            Model magnitudes per filter, at the observing epochs.
        obs_error: dict
            Systematic uncertainty budget per filter.

        Returns
        -------
        float
            Total log-likelihood, or the most negative finite float64 as
            soon as one filter comes out ill-behaved.
        """
        minus_chisquare_total = 0.0
        gaussprob_total = 0.0
        for filt, err in obs_error.items():
            data_sigma = np.sqrt(self.light_curve_uncertainties[filt] ** 2 + err**2)
            minus_chisquare, gaussprob = self.chisquare_gaussianlog_from_lc_data(
                expected_mags[filt],
                self.light_curves[filt],
                data_sigma,
                err,
                lim=self.detection_limit[filt],
            )
            if minus_chisquare is False:
                # this should only be the case if also (parameters['timeshift'] <= self.light_curve_times[filt][0] ):
                return np.nan_to_num(-np.inf)
            else:
                minus_chisquare_total += minus_chisquare
                gaussprob_total += gaussprob
        return minus_chisquare_total + gaussprob_total

    def final_diagnostics(self, bestfit_params, args, result=None):
        """Plot the best-fit light curves, one panel per filter.

        Parameters
        ----------
        bestfit_params: dict
            Best-fit parameters to draw.
        args: argparse.Namespace
            Parsed command-line arguments, used for the output path.
        result: bilby.core.result.Result, optional
            Sampling result, used to name the figure when available.

        Returns
        -------
        matplotlib.figure.Figure
            The figure that was saved.
        """
        return lch_bestfit(self, bestfit_params, args, result)
