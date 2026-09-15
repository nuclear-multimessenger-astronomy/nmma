import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm
from ..core.base import NMMALikelihood, initialisation_args_from_signature_and_namespace
from ..core.conversion import convert_mtot_mni, observation_angle_conversion
from ..core.utils import read_trigger_time
from . import model, utils, systematics
from .lightcurve_handling import post_process_bestfit as lch_bestfit
from .plotting_utils import bolometric_lc_plot


def setup_em_kwargs(priors, data_dump, args, logger=None):
    # Prerequisites
    ## get lc_data and filters
    light_curve_data = data_dump["light_curve_data"]
    filters = data_dump["filters"]

    ## setup the light curve model for this transient class and filters
    light_curve_model = model.create_light_curve_model_from_args(args, filters)
    trigger_time = read_trigger_time(None, args)
    light_curve_data = utils.setup_filtered_lc_data(light_curve_data, trigger_time)
    light_curve_data = utils.check_model_time_consistency(
        light_curve_data, light_curve_model, priors, args.injection
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
        self.detection_limit = detection_limit

    def __repr__(self):
        return f"{self.__class__.__name__} (light_curve_model={self.light_curve_model})"

    def log_likelihood(self, parameters):
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
        if not np.isfinite(model_lc).any():
            return False
        return True

    def update_lightcurve_reference(self, obs_times, model_lc):
        return utils.autocomplete_data(self.light_curve_times, obs_times, model_lc)

    def band_log_likelihood(self, expected_lc, obs_error):
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

            ## sanity check: if the chisquare is ill-behaved,
            # we explicitly catch it as Bool in band_log_likelihood
            if np.isnan(minus_chisquare):
                sanity_check_passed = False
                return sanity_check_passed, -np.inf
        else:
            minus_chisquare = 0.0

        # evaluate the data with infinite error, i.e. upper limits,
        # as Gaussian survival function
        gausslogsf = np.zeros(2)  ##hack if len(infIdx)==0
        if infIdx.sum() > 0:
            gausslogsf = norm.logsf(
                data_mag[infIdx], est_mag[infIdx], upperlim_sigma[infIdx]
            )
        return minus_chisquare, np.sum(gausslogsf)

    def truncated_gaussian(self, m_det, loc, scale, upper_lim):

        a = -np.inf  # no lower bound of truncation
        b = (upper_lim - loc) / scale  # upper bound in number of std-deviations
        return truncnorm.logpdf(m_det, a, b, loc=loc, scale=scale)

    def final_diagnostics(self, bestfit_params, args, result=None):
        obs_times, obs_lc = self.light_curve_model.gen_detector_lc(bestfit_params)
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
        self.model_filter_mapping, self.obs_average_mapping = (
            utils.get_filter_name_mapping(filters)
        )

        super().__init__(
            light_curve_model,
            light_curve_data,
            systematics_handler,
            priors,
            detection_limit,
            verbose,
        )

    def set_detection_limit(self, detection_limit):
        self.detection_limit = utils.set_filter_associated_dict(
            detection_limit, self.observed_filters
        )

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
        return lch_bestfit(self, bestfit_params, args, result)


class MultiFilterNondetectionTransient:
    """A collection of multi-filter non-detection (upper limit) data at many sky positions

    Loads a csv of pointings/non-detections, such as a GWTreasureMap-style survey
    report, and groups the rows by sky position (ra, dec). Every row is by
    definition a non-detection (a "depth", i.e. a limiting magnitude, rather than
    a measurement), so for each position the per-filter observation times and
    limiting magnitudes are stored in the standard nmma light curve dict format
    {filter: {"time": ..., "mag": ..., "mag_error": ...}}, with ``mag_error`` set to
    ``np.inf`` throughout - the same convention used to mark the ``infIdx``
    (non-detection) entries in `BasicEMTransient.chisquare_gaussianlog_from_lc_data`.

    A stored position can be queried by (ra, dec) to retrieve its non-detection
    data, or compared directly against a candidate light curve to compute the
    associated upper-limit log-likelihood contribution.

    Parameters
    ----------
    filename: str
        Path to a csv file of non-detections/pointings.
    ra_column, dec_column: str (default: "ra", "dec")
        Names of the columns holding the sky position of each pointing.
    time_column: str (default: "t_from_T0")
        Name of the column holding the observation time (relative to trigger time).
    filter_column: str (default: "filters")
        Name of the column holding the filter name.
    depth_column: str (default: "depth")
        Name of the column holding the limiting magnitude (upper limit).

    """

    def __init__(
        self,
        filename,
        ra_column="ra",
        dec_column="dec",
        time_column="t_from_T0",
        filter_column="filters",
        depth_column="depth",
    ):
        self.filename = filename
        self.ra_column = ra_column
        self.dec_column = dec_column
        self.time_column = time_column
        self.filter_column = filter_column
        self.depth_column = depth_column

        data = pd.read_csv(filename)
        required_columns = (
            ra_column,
            dec_column,
            time_column,
            filter_column,
            depth_column,
        )
        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            raise ValueError(f"Missing expected column(s) {missing} in {filename}")

        self.observed_filters = sorted(data[filter_column].unique().tolist())

        self.data = {}
        for position, group in data.groupby([ra_column, dec_column], sort=False):
            ra, dec = position
            position_data = {}
            for filt, sub_data in group.groupby(filter_column):
                order = np.argsort(sub_data[time_column].to_numpy())
                position_data[filt] = {
                    "time": sub_data[time_column].to_numpy()[order],
                    "mag": sub_data[depth_column].to_numpy()[order],
                    "mag_error": np.full(len(sub_data), np.inf),
                }
            self.data[(float(ra), float(dec))] = position_data

        self.positions = list(self.data.keys())
        self._positions_array = np.array(self.positions, dtype=float).reshape(-1, 2)

    def __repr__(self):
        return (
            f"{self.__class__.__name__} ({len(self.positions)} positions, "
            f"filters={self.observed_filters})"
        )

    # FIXME: this is where the bodies are buried for this function
    def _resolve_position(self, ra, dec, position_tolerance: float=1e-3):
        """
        Stupid function for now, but should be replaced with the pixel/MOC lookup idea. 
        
        position_tolerance is very loose for testing purposes
        """
        ra, dec = float(ra), float(dec)
        diffs = np.abs(self._positions_array - np.array([ra, dec]))
        matches = np.flatnonzero(np.all(diffs <= position_tolerance, axis=1))
        if len(matches) == 0:
            raise KeyError(
                f"No non-detection data stored for position (ra={ra}, dec={dec})."
            )
        # if more than one stored position falls within tolerance, take the closest
        idx = matches[np.argmin(diffs[matches].sum(axis=1))]
        return self.positions[idx]

    def query(self, ra, dec):
        """Return the stored non-detection dataset for a sky position

        Parameters
        ----------
        ra, dec: float
            Sky position to query. Must match a stored position exactly.

        Returns
        -------
        dict
            {filter: {"time": array, "mag": array, "mag_error": array}}, the upper
            limits (magnitudes) and their observation times for each filter observed
            at that position. ``mag_error`` is ``np.inf`` throughout, marking every
            entry as a non-detection.

        """
        position = self._resolve_position(ra, dec)
        return self.data[position]

    def log_likelihood_upper_limits(self, ra, dec, obs_times, model_lc, sigma):
        """Compute the non-detection (upper limit) log-likelihood contribution of a
        candidate light curve against the stored data for a sky position, using the
        same Gaussian survival function treatment as the ``infIdx`` branch of
        `BasicEMTransient.chisquare_gaussianlog_from_lc_data`.

        Parameters
        ----------
        ra, dec: float
            Sky position whose non-detection data to compare against.
        obs_times: array or dict
            Times at which `model_lc` is evaluated. If a dict, it must contain one
            array per filter (matching `model_lc`); otherwise the same times are
            assumed for every filter.
        model_lc: dict
            Model light curve, filter -> array of model magnitudes evaluated at
            `obs_times`, e.g. as returned by a `LightCurveModel.gen_detector_lc` call.
        sigma: float or dict
            The 1-sigma uncertainty to assume for the model magnitude when
            evaluating the survival function (e.g. a systematic error budget),
            either a single value applied to all filters or a dict per filter.

        Returns
        -------
        float
            The summed log-likelihood over all filters and non-detections at that
            position.

        """
        position_data = self.query(ra, dec)
        sigma_dict = utils.set_filter_associated_dict(
            sigma, list(position_data.keys()), default_limit=0.0
        )

        logL = 0.0
        for filt, filt_data in position_data.items():
            if filt not in model_lc:
                continue

            filt_obs_times = (
                obs_times[filt] if isinstance(obs_times, dict) else obs_times
            )

            est_mag = utils.autocomplete_data(
                filt_data["time"],
                filt_obs_times,
                model_lc[filt],
                extrapolate=np.inf,
            )

            logL += np.sum(norm.logsf(filt_data["mag"], est_mag, sigma_dict[filt]))

        return logL


class NondetectionKilonovaSubModel:
    """Generates a kilonova lightcurve from the (EOS + ejecta) converted
    parameters and scores it against stored non-detections (a
    `MultiFilterNondetectionTransient`) at the sampled sky position.

    Intended as a sub-model to be wrapped in `nmma.core.base.NMMALikelihood`
    and combined with other messengers via
    `nmma.joint.joint_likelihood.MultiMessengerLikelihood`.

    Parameters
    ----------
    lc_model: fiesta.models.FiestaModel
        Any fiesta model (e.g. a Bu2019/Bu2026 kilonova `FiestaKN` surrogate, an
        `FiestaGRB` afterglow model, ...) exposing `.parameter_names`,
        `.parameter_distributions`, and `.predict(parameters)`.
    nondetections: MultiFilterNondetectionTransient
        The stored non-detection (upper limit) data to score the generated
        light curve against.
    filter_map: dict
        Mapping from `nondetections`' (generic) filter names to `lc_model`'s
        filter names, e.g. {"g": "ztfg", "r": "ztfr"}.
    sigma: float or dict
        The 1-sigma uncertainty to assume for the model magnitude, passed
        through to `MultiFilterNondetectionTransient.log_likelihood_upper_limits`.

    """

    def __init__(self, lc_model, nondetections, filter_map, sigma):
        self.lc_model = lc_model
        self.nondetections = nondetections
        self.filter_map = filter_map
        self.sigma = sigma

    def __repr__(self):
        return f"{self.__class__.__name__} (filters={list(self.filter_map)})"

    def noise_log_likelihood(self):
        return 0.0

    def log_likelihood(self, parameters):
        parameters = observation_angle_conversion(parameters)

        bounds = self.lc_model.parameter_distributions
        lc_parameters = {}
        for key in self.lc_model.parameter_names:
            if key not in parameters:
                # mirror nmma.em.model.LightCurveModelContainer.parameter_conversion:
                # fall back to a log10/delog10 counterpart if that's what was sampled
                if key.startswith("log10_") and key[len("log10_") :] in parameters:
                    parameters[key] = np.log10(parameters[key[len("log10_") :]])
                elif "log10_" + key in parameters:
                    parameters[key] = 10 ** parameters["log10_" + key]

            value = parameters[key]
            # The surrogate is only trained within a finite domain (see its
            # *_metadata.pkl); clip to it so the model always returns a
            # (labeled) curve instead of NaNs.
            if key in bounds:
                lo, hi = bounds[key][:2]
                value = float(np.clip(value, lo, hi))
            lc_parameters[key] = value

        # not part of parameter_names, but required by predict()
        lc_parameters["luminosity_distance"] = parameters["luminosity_distance"]
        lc_parameters["redshift"] = parameters["redshift"]

        times, mag = self.lc_model.predict(lc_parameters)
        self.last_times, self.last_mag = times, mag  # stashed for inspection/plotting

        model_lc = {
            generic: np.asarray(mag[ztf_name])
            for generic, ztf_name in self.filter_map.items()
            if ztf_name in mag
        }

        ra_deg = float(np.degrees(parameters["ra"]))
        dec_deg = float(np.degrees(parameters["dec"]))
        return self.nondetections.log_likelihood_upper_limits(
            ra_deg, dec_deg, times, model_lc, sigma=self.sigma
        )
