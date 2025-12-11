from __future__ import division

import numpy as np
import scipy.stats
from scipy.interpolate import interp1d
from scipy.stats import truncnorm
from pathlib import Path
import yaml


from bilby.core.likelihood import Likelihood
from . import utils, systematics


def truncated_gaussian(m_det, m_err, m_est, lim):

    a, b = (-np.inf - m_est) / m_err, (lim - m_est) / m_err
    logpdf = truncnorm.logpdf(m_det, a, b, loc=m_est, scale=m_err)

    return logpdf


class OpticalLightCurve(Likelihood):
    """A optical kilonova / GRB / kilonova-GRB afterglow likelihood object
    see line 1221 gwemlightcurves/sampler/loglike.py

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
        systematics_file,
        detection_limit=None,
        error_budget=1.0,
        tmin=0.0,
        tmax=14.0,
        verbose=False,
    ):
        self.light_curve_model = light_curve_model
        super(OpticalLightCurve, self).__init__(dict())
        self.filters = filters
        self.trigger_time = trigger_time
        self.tmin = tmin
        self.tmax = tmax
        if isinstance(error_budget, (int, float, complex)) and not isinstance(
            error_budget, bool
        ):
            self.error_budget = dict(zip(filters, [error_budget] * len(filters)))
        elif isinstance(error_budget, dict):
            for filt in self.filters:
                if filt not in error_budget:
                    raise ValueError(f"filter {filt} missing from error_budget")
            self.error_budget = error_budget

        processedData = utils.dataProcess(
            light_curve_data, self.filters, self.trigger_time, self.tmin, self.tmax
        )
        self.light_curve_data = processedData
        self.systematics_file = systematics_file

        self.detection_limit = {}
        if detection_limit:
            for filt in self.filters:
                if filt in detection_limit:
                    self.detection_limit[filt] = detection_limit[filt]
                else:
                    self.detection_limit[filt] = np.inf
        else:
            for filt in self.filters:
                self.detection_limit[filt] = np.inf

        self.sample_times = self.light_curve_model.sample_times
        self.verbose = verbose

    def __repr__(self):
        return self.__class__.__name__ + "(light_curve_model={},\n\tfilters={}".format(
            self.light_curve_model, self.filters
        )

    def noise_log_likelihood(self):
        return 0.0

    def log_likelihood(self):
        lbol, mag_abs = self.light_curve_model.generate_lightcurve(
            self.sample_times, self.parameters
        )

        # sanity checking
        if len(np.isfinite(lbol)) == 0:
            return np.nan_to_num(-np.inf)
        if np.sum(lbol) == 0.0:
            return np.nan_to_num(-np.inf)

        # create light curve templates
        mag_app_interp = {}
        for filt in self.filters:
            mag_abs_filt = utils.getFilteredMag(mag_abs, filt)
            if self.parameters["luminosity_distance"] > 0.0:
                mag_app_filt = mag_abs_filt + 5.0 * np.log10(
                    self.parameters["luminosity_distance"] * 1e6 / 10.0
                )
            else:
                mag_app_filt = mag_abs_filt
            usedIdx = np.where(np.isfinite(mag_app_filt))[0]
            sample_times_used = self.sample_times[usedIdx]
            mag_app_used = mag_app_filt[usedIdx]
            t0 = self.parameters["timeshift"]
            if len(mag_app_used) > 0:
                mag_app_interp[filt] = interp1d(
                    sample_times_used + t0,
                    mag_app_used,
                    fill_value="extrapolate",
                    bounds_error=False,
                )
            else:
                mag_app_interp[filt] = interp1d(
                    [-99.0, -99.9],
                    [np.nan, np.nan],
                    fill_value=np.nan,
                    bounds_error=False,
                )

        if self.systematics_file is not None:
            yaml_dict = yaml.safe_load(Path(self.systematics_file).read_text())
            systematics.validate_only_one_true(yaml_dict)

            if yaml_dict["config"]["withTime"]["value"]:
                n = yaml_dict["config"]["withTime"]["time_nodes"]
                time_array = np.round(np.linspace(self.tmin, self.tmax, n), 2)
                yaml_filters = yaml_dict["config"]["withTime"]["filters"]
                systematics.validate_filters(yaml_filters)

                for filter_group in yaml_filters:
                    if isinstance(filter_group, list):
                        filt = "___".join(filter_group)
                    elif filter_group is None:
                        filt = "all"
                    else:
                        filt = filter_group

                    globals()[f"sys_err_{filt}_array"] = np.array([])

                    for i in range(1, n + 1):
                        value = self.parameters.get(f"sys_err_{filt}{i}")
                        globals()[f"sys_err_{filt}_array"] = np.append(
                            globals()[f"sys_err_{filt}_array"], value
                        )

                for filter_group in yaml_dict["config"]["withTime"]["filters"]:
                    if isinstance(filter_group, list):
                        filt = "___".join(filter_group)
                    elif filter_group is None:
                        filt = "all"
                    else:
                        filt = filter_group
                    globals()[f"sys_err_{filt}_interped"] = interp1d(
                        time_array,
                        globals()[f"sys_err_{filt}_array"],
                        fill_value="extrapolate",
                        bounds_error=False,
                    )

        # compare the estimated light curve and the measured data
        minus_chisquare_total = 0.0
        gaussprob_total = 0.0
        for filt in self.filters:
            # decompose the data
            data_time = self.light_curve_data[filt][:, 0]
            data_mag = self.light_curve_data[filt][:, 1]
            data_sigma = self.light_curve_data[filt][:, 2]

            if self.systematics_file is not None:
                if yaml_dict["config"]["withTime"]["value"]:
                    yaml_filters = yaml_dict["config"]["withTime"]["filters"]

                    filter_group_finite_idx_match = False

                    if yaml_filters is not None and yaml_filters != []:
                        for yaml_filt in yaml_filters:
                            if yaml_filt is not None and filt in yaml_filt:
                                if isinstance(yaml_filt, str):
                                    filters_to_use = yaml_filt
                                elif isinstance(yaml_filt, list):
                                    filters_to_use = "___".join(yaml_filt)
                                filter_group_finite_idx_match = True
                                break
                    if not filter_group_finite_idx_match:
                        filters_to_use = "all"
                    data_sigma = np.sqrt(
                        data_sigma**2
                        + (globals()[f"sys_err_{filters_to_use}_interped"](data_time))
                        ** 2
                    )

                elif yaml_dict["config"]["withoutTime"]["value"]:
                    data_sigma = np.sqrt(
                        data_sigma**2 + self.parameters["sys_err"] ** 2
                    )

            # include the error budget into calculation
            elif "sys_err" in self.parameters:
                data_sigma = np.sqrt(data_sigma**2 + self.parameters["sys_err"] ** 2)
            else:
                data_sigma = np.sqrt(data_sigma**2 + self.error_budget[filt] ** 2)

            # evaluate the light curve magnitude at the data points
            mag_est = mag_app_interp[filt](data_time)

            # seperate the data into bounds (inf err) and actual measurement
            infIdx = np.where(~np.isfinite(data_sigma))[0]
            finiteIdx = np.where(np.isfinite(data_sigma))[0]

            # evaluate the chisuquare
            if len(finiteIdx) >= 1:
                minus_chisquare = np.sum(
                    truncated_gaussian(
                        data_mag[finiteIdx],
                        data_sigma[finiteIdx],
                        mag_est[finiteIdx],
                        self.detection_limit[filt],
                    )
                )
            else:
                minus_chisquare = 0.0

            if np.isnan(minus_chisquare):
                return np.nan_to_num(-np.inf)

            minus_chisquare_total += minus_chisquare

            # evaluate the data with infinite error
            if len(infIdx) > 0:
                if self.systematics_file is not None:
                    if yaml_dict["config"]["withTime"]["value"]:
                        yaml_filters = yaml_dict["config"]["withTime"]["filters"]

                        filter_group_infinite_idx_match = False

                        if yaml_filters is not None and yaml_filters != []:
                            for yaml_filt in yaml_filters:
                                if yaml_filt is not None and filt in yaml_filt:
                                    if isinstance(yaml_filt, str):
                                        upperlim_sigma = globals()[
                                            f"sys_err_{yaml_filt}_interped"
                                        ](data_time)[infIdx]
                                    elif isinstance(yaml_filt, list):
                                        filters_to_use = "___".join(yaml_filt)
                                        upperlim_sigma = globals()[
                                            f"sys_err_{filters_to_use}_interped"
                                        ](data_time)[infIdx]
                                    filter_group_infinite_idx_match = True
                                    break
                        if not filter_group_infinite_idx_match:
                            filters_to_use = "all"
                            upperlim_sigma = globals()[
                                f"sys_err_{filters_to_use}_interped"
                            ](data_time)[infIdx]

                        gausslogsf = scipy.stats.norm.logsf(
                            data_mag[infIdx], mag_est[infIdx], upperlim_sigma
                        )

                    elif yaml_dict["config"]["withoutTime"]["value"]:
                        upperlim_sigma = self.parameters["sys_err"]
                        gausslogsf = scipy.stats.norm.logsf(
                            data_mag[infIdx], mag_est[infIdx], upperlim_sigma
                        )
                elif "sys_err" in self.parameters:
                    upperlim_sigma = self.parameters["sys_err"]
                    gausslogsf = scipy.stats.norm.logsf(
                        data_mag[infIdx], mag_est[infIdx], upperlim_sigma
                    )
                else:
                    gausslogsf = scipy.stats.norm.logsf(
                        data_mag[infIdx], mag_est[infIdx], self.error_budget[filt]
                    )
                gaussprob_total += np.sum(gausslogsf)

        log_prob = minus_chisquare_total + gaussprob_total

        if self.verbose:
            print(self.parameters, log_prob)

        return log_prob


class BolometricLightCurve(Likelihood):
    """A bolometric likelihood object

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
        verbose=False,
    ):
        self.light_curve_model = light_curve_model
        super(BolometricLightCurve, self).__init__(dict())
        self.tmin = tmin
        self.tmax = tmax
        self.error_budget = error_budget

        self.light_curve_data = light_curve_data
        self.sample_times = self.light_curve_model.sample_times
        self.verbose = verbose

    def __repr__(self):
        return self.__class__.__name__ + "(light_curve_model={},\n\tfilters={}".format(
            self.light_curve_model, self.filters
        )

    def noise_log_likelihood(self):
        return 0.0

    def log_likelihood(self):
        lbol = self.light_curve_model.generate_lightcurve(
            self.sample_times, self.parameters
        )

        # sanity checking
        if len(np.isfinite(lbol)) == 0:
            return np.nan_to_num(-np.inf)
        if np.sum(lbol) == 0.0:
            return np.nan_to_num(-np.inf)

        # build the interpolation
        idx = np.where(np.isfinite(lbol))[0]
        t0 = self.parameters["timeshift"]
        lbol_interp = interp1d(
            self.sample_times[idx] + t0,
            lbol[idx],
            fill_value="extrapolate",
            bounds_error=False,
        )

        # compare the estimated light curve and the measured data
        minus_chisquare_total = 0.0
        gaussprob_total = 0.0

        data_time = self.light_curve_data["phase"].to_numpy()
        data_lbol = self.light_curve_data["Lbb"].to_numpy()
        data_sigma = self.light_curve_data["Lbb_unc"].to_numpy()

        # include the error budget into calculation
        if "em_syserr" in self.parameters:
            data_sigma = np.sqrt(data_sigma**2 + self.parameters["em_syserr"] ** 2)
        else:
            data_sigma = np.sqrt(data_sigma**2 + self.error_budget**2)

        # evaluate the light curve bolometric luminosity at the data points
        lbol_est = lbol_interp(data_time)

        # seperate the data into bounds (inf err) and actual measurement
        infIdx = np.where(~np.isfinite(data_sigma))[0]
        finiteIdx = np.where(np.isfinite(data_sigma))[0]

        # evaluate the chisuquare
        if len(finiteIdx) >= 1:
            minus_chisquare = np.sum(
                truncated_gaussian(
                    data_lbol[finiteIdx],
                    data_sigma[finiteIdx],
                    lbol_est[finiteIdx],
                    np.inf,
                )
            )
        else:
            minus_chisquare = 0.0

        if np.isnan(minus_chisquare):
            return np.nan_to_num(-np.inf)

        minus_chisquare_total += minus_chisquare

        # evaluate the data with infinite error
        if len(infIdx) > 0:
            if "em_syserr" in self.parameters:
                upperlim_sigma = self.parameters["em_syserr"]
                gausslogsf = scipy.stats.norm.logsf(
                    data_lbol[infIdx], lbol_est[infIdx], upperlim_sigma
                )
            else:
                gausslogsf = scipy.stats.norm.logsf(
                    data_lbol[infIdx], lbol_est[infIdx], self.error_budget
                )
            gaussprob_total += np.sum(gausslogsf)

        log_prob = minus_chisquare_total + gaussprob_total

        if self.verbose:
            print(self.parameters, log_prob)

        return log_prob
