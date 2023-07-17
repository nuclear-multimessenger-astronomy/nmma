from __future__ import division

import numpy as np
import scipy.stats
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.stats import truncnorm, chi2
from scipy.interpolate import Akima1DInterpolator
from scipy.interpolate import PchipInterpolator

from bilby.core.likelihood import Likelihood
from . import utils
from bilby.core.prior import Prior


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
        detection_limit=None,
        error_budget=1.0,
        tmin=0.0,
        tmax=14.0,
        verbose=False,
        likelihood_type="gaussian",
    ):
        self.likelihood_type = likelihood_type
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
            t0 = self.parameters["KNtimeshift"]
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

        # compare the estimated light curve and the measured data

        _temp_total = 0.0  # temp sums
        minus_chisquare_total = 0.0  # gaussian likelihood
        chisquare_prob = 0.0  # chi2 likelihood
        prob_total = 0.0

        if "em_syserr_t_0" in self.parameters:
            x = np.array([])
            y = np.array([])
            uy = np.array([])
            Ky = np.array([])
            for k, v in self.parameters.items():
                if k.startswith("em_syserr_t_"):
                    x = np.append(x, v)
                if k.startswith("em_syserr_val_all"):
                    y = np.append(y, v)
                if k.startswith("em_syserr_val_u"):
                    uy = np.append(uy, v)
                if k.startswith("em_syserr_val_K"):
                    Ky = np.append(Ky, v)

            # change the interpoaltion kind
            em_syserr_all = interp1d(x, y, fill_value="extrapolate", bounds_error=False)
            em_syserr_u = interp1d(x, uy, fill_value="extrapolate", bounds_error=False)
            em_syserr_K = interp1d(x, Ky, fill_value="extrapolate", bounds_error=False)

            # em_syserr_all = PchipInterpolator(x, y, extrapolate=True)
            # em_syserr_u= PchipInterpolator(x, uy, extrapolate=True)
            # em_syserr_K = PchipInterpolator(x, Ky, extrapolate=True)

        for filt in self.filters:
            # decompose the data
            data_time = self.light_curve_data[filt][:, 0]
            data_mag = self.light_curve_data[filt][:, 1]
            data_sigma = self.light_curve_data[filt][:, 2]
            # include the error budget into calculation

            if "em_syserr" in self.parameters:
                data_sigma = np.sqrt(
                    data_sigma**2 + self.parameters["em_syserr"] ** 2
                )

            elif "em_syserr_t_0" in self.parameters:
                # data_sigma = np.sqrt(data_sigma**2 + interp(data_time,x,y) ** 2)

                if filt == "u":
                    data_sigma = np.sqrt(data_sigma**2 + em_syserr_u(data_time) ** 2)
                if filt == "K":
                    data_sigma = np.sqrt(data_sigma**2 + em_syserr_K(data_time) ** 2)
                else:
                    data_sigma = np.sqrt(
                        data_sigma**2 + em_syserr_all(data_time) ** 2
                    )

            else:
                data_sigma = np.sqrt(data_sigma**2 + self.error_budget[filt] ** 2)

            # evaluate the light curve magnitude at the data points
            mag_est = mag_app_interp[filt](data_time)

            # mag_est = (1 + self.parameters["em_syserr"]) * mag_est

            # seperate the data into bounds (inf err) and actual measurement
            infIdx = np.where(~np.isfinite(data_sigma))[0]
            finiteIdx = np.where(np.isfinite(data_sigma))[0]

            if self.likelihood_type == "gaussian":
                # evaluate the chisquare (~1-n^2)
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

                _temp_total = minus_chisquare_total

            elif self.likelihood_type == "chi2":
                chisquarevals = ((data_mag - mag_est) / data_sigma) ** 2
                chisquaresum = np.sum(chisquarevals)

                if np.isnan(chisquaresum):
                    chisquare = np.nan
                    return chisquare

                if not float(len(chisquarevals) - 1) == 0:
                    chisquaresum = (1 / float(len(chisquarevals) - 1)) * chisquaresum

                chisquare = chisquaresum

                if np.isnan(chisquare):
                    prob = -np.inf
                else:
                    prob = chi2.logpdf(chisquare, 1, loc=0, scale=1)

                if np.isnan(prob):
                    prob = -np.inf

                chisquare_prob += prob

                _temp_total = chisquare_prob

            # evaluate the data with infinite error
            if len(infIdx) > 0:
                if "em_syserr" in self.parameters:
                    gausslogsf = scipy.stats.norm.logsf(
                        data_mag[infIdx], mag_est[infIdx], self.parameters["em_syserr"]
                    )

                elif "em_syserr_t_0" in self.parameters:
                    if filt == "u":
                        gausslogsf = scipy.stats.norm.logsf(
                            data_mag[infIdx],
                            mag_est[infIdx],
                            em_syserr_u(data_time[infIdx]),
                        )
                    if filt == "K":
                        gausslogsf = scipy.stats.norm.logsf(
                            data_mag[infIdx],
                            mag_est[infIdx],
                            em_syserr_K(data_time[infIdx]),
                        )
                    else:
                        gausslogsf = scipy.stats.norm.logsf(
                            data_mag[infIdx],
                            mag_est[infIdx],
                            em_syserr_all(data_time[infIdx]),
                        )

                else:
                    gausslogsf = scipy.stats.norm.logsf(
                        data_mag[infIdx], mag_est[infIdx], self.error_budget[filt]
                    )
                prob_total += np.sum(gausslogsf)

        log_prob = _temp_total + prob_total

        # df.write(f'{em_syserr(data_time)}\n')
        # df.close()

        if self.verbose:
            print(self.parameters, log_prob)

        return log_prob
