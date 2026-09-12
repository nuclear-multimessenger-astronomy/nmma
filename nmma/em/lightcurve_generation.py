import copy
from joblib import load

import numpy as np
import pandas as pd
from importlib import resources

import scipy
from scipy.integrate import quad, solve_ivp
from scipy.special import erfc
from scipy.interpolate import CubicSpline

from . import utils
from ..core.utils import read_trigger_time

try:
    import afterglowpy

except ModuleNotFoundError:
    print("afterglowpy not available. Need fiesta or afterglowpy for GRB analyses. ")

try:
    from wrapt_timeout_decorator import timeout
except ImportError:
    print("Install wrapt_timeout_decorator if you want to timeout simulations.")

    def timeout(*args, **kwargs):
        def inner(func):
            return func

        return inner


# some frequently used constants:
from ..core.constants import msun_cgs, c_cgs, h, kb, sigSB, arad, D

seconds_a_day = 86400.0
abs_mag_dist_factor = D**2


# =======================================================
def dummy_add(nu):
    """Add nothing to a blackbody spectrum; the default extra component.

    Parameters
    ----------
    nu: float or numpy.ndarray
        Frequency, ignored.

    Returns
    -------
    float
        Always zero.
    """

    return 0.0


def bb_flux_from_inv_temp(nu, inv_temp, R_photo, dist_squared=abs_mag_dist_factor):
    """Planck flux density of a sphere, at the given frequencies.

    Parameters
    ----------
    nu: float or numpy.ndarray
        Rest-frame frequencies, in Hz.
    inv_temp: float or numpy.ndarray
        Inverse temperature, in 1/K. Taken inverted because the photosphere
        can cool to zero, where the temperature itself diverges.
    R_photo: float or numpy.ndarray
        Photospheric radius, in cm.
    dist_squared: float, optional
        Squared distance to the source, in cm2. Defaults to the 10 pc of the
        absolute magnitude scale.

    Returns
    -------
    float or numpy.ndarray
        Flux density in cgs units.
    """

    exponent = np.clip(h * nu * inv_temp / kb, None, 700)  # to avoid overflow in exp
    bb_factor = 2.0 * h / c_cgs**2
    return bb_factor * nu**3 / np.expm1(exponent) * R_photo * R_photo / dist_squared


def mag_dict_for_blackbody(filters, inv_temp, R_photo, nu_host, add=dummy_add):
    """Turn a blackbody into a magnitude per filter.

    Parameters
    ----------
    filters: list of str
        Filters to evaluate.
    inv_temp: float or numpy.ndarray
        Inverse photospheric temperature, in 1/K.
    R_photo: float or numpy.ndarray
        Photospheric radius, in cm.
    nu_host: numpy.ndarray
        Rest-frame frequency of each filter, in Hz.
    add: callable, optional
        Extra flux to add at each frequency, for models that sit a power law
        on top of the blackbody.

    Returns
    -------
    dict
        AB magnitude per filter.
    """

    mag = {}
    # nu_host = nu_obs * (1 + redshift)
    for idx, filt in enumerate(filters):
        nu_of_filt = nu_host[idx]
        F = bb_flux_from_inv_temp(nu_of_filt, inv_temp, R_photo)
        F += add(nu_of_filt)
        # F *= 1 + redshift # correction factor for shifted flux density bin
        mag[filt] = utils.flux_to_ABmag(F)

    return mag


# =======================================================
# LC MODELS
# =======================================================
# Arnett model convenience functions
def arnett_lc_get_int_A_non_vec(x, y):
    """Integral A of the Arnett solution, for a single time.

    Wrapped by ``arnett_lc_get_int_A`` to accept arrays.

    Parameters
    ----------
    x: float
        Time in units of the diffusion timescale.
    y: float
        Diffusion timescale in units of twice the nickel lifetime.

    Returns
    -------
    float
        Value of the integral.
    """

    def arnett_func(z):
        return 2 * z * np.exp(-2 * z * y + z**2)

    r = quad(arnett_func, 0, x)
    return r[0]


arnett_lc_get_int_A = np.vectorize(arnett_lc_get_int_A_non_vec, excluded=["y"])


def arnett_lc_get_int_B_non_vec(x, y, s):
    """Integral B of the Arnett solution, for a single time.

    Wrapped by ``arnett_lc_get_int_B`` to accept arrays. This is the cobalt
    term, which integral A lacks.

    Parameters
    ----------
    x: float
        Time in units of the diffusion timescale.
    y: float
        Diffusion timescale in units of twice the nickel lifetime.
    s: float
        Diffusion timescale in units of the combined nickel-cobalt scale.

    Returns
    -------
    float
        Value of the integral.
    """

    def arnett_func(z):
        return 2 * z * np.exp(-2 * z * y + 2 * z * s + z**2)

    r = quad(arnett_func, 0, x)
    return r[0]


arnett_lc_get_int_B = np.vectorize(arnett_lc_get_int_B_non_vec, excluded=["y", "s"])

# arnett_constants
epsilon_ni = 3.9e10  # erg / s / g
epsilon_co = 6.78e9  # erg / s / g
tau_ni = 8.8  # days
tau_co = 111.3  # days
y_scale = 2 * tau_ni
s_scale = (2 * tau_co * tau_ni) / (tau_co - tau_ni)


def arnett_lc(t_day, param_dict):
    """bolometric light curve functions from Arnett model of supernovae
    -----------
    Parameters:
    t_day: array-like
        Time in days
    param_dict: dict
        Dictionary containing the parameters for the Arnett model

    Returns:
    Ls: array-like
        Bolometric light curve

    """
    tau_m = param_dict["tau_m"]
    Mni = 10 ** param_dict["log10_mni"] * msun_cgs

    y = tau_m / y_scale
    s = tau_m / s_scale
    x = t_day / tau_m

    int_A = arnett_lc_get_int_A(x, y)
    int_B = arnett_lc_get_int_B(x, y, s)

    lbol = (
        Mni
        * np.exp(-(x**2))
        * ((epsilon_ni - epsilon_co) * int_A + epsilon_co * int_B)
    )

    return lbol


def arnett_modified_lc(t_day, param_dict):
    """time delayed bolometric light curve functions from Arnett model
    -----------
    Parameters:
    t_day: array-like
        Time in days
    param_dict: dict
        Dictionary containing the parameters for the Arnett model

    Returns:
    Ls: array-like
        Bolometric light curve

    """
    Lbol_arnett = arnett_lc(t_day, param_dict)
    return Lbol_arnett * (1.0 - np.exp(-((param_dict["t_0"] / t_day) ** 2)))


# kilonova from SVD model
def calc_svd_lbol(sample_times, param_list, svd_lbol_model, lbol_ncoeff=None):
    """Evaluate a surrogate model for the bolometric luminosity.

    Parameters
    ----------
    sample_times: numpy.ndarray
        Times to evaluate, in days.
    param_list: list
        Model parameters, in the order the surrogate was trained on.
    svd_lbol_model: dict
        Trained surrogate for the bolometric luminosity.
    lbol_ncoeff: int, optional
        Keep only this many SVD coefficients. Defaults to all of them.

    Returns
    -------
    numpy.ndarray
        Bolometric luminosity at the requested times.
    """

    tt_interp, lbol_back = eval_svd_model(svd_lbol_model, lbol_ncoeff, param_list)
    lbol = 10 ** utils.autocomplete_data(sample_times, tt_interp, lbol_back)
    return np.squeeze(lbol)  # * (1. + z) FIXME: shouldn't this be (1 + z)**2


def calc_svd_lc(
    sample_times,
    param_list,
    svd_mag_model,
    mag_ncoeff: int = None,
    filters: list = None,
):
    """Evaluate a surrogate model for the magnitude in each filter.

    Filters the surrogate was not trained on are returned as non-detections,
    which is what lets a kilonova model be combined with radio or X-ray data
    from a GRB.

    Parameters
    ----------
    sample_times: numpy.ndarray
        Times to evaluate, in days.
    param_list: list
        Model parameters, in the order the surrogate was trained on.
    svd_mag_model: dict
        Trained surrogate, keyed by filter.
    mag_ncoeff: int, optional
        Keep only this many SVD coefficients. Defaults to all of them.
    filters: list of str, optional
        Filters to evaluate. Defaults to those the surrogate covers.

    Returns
    -------
    dict
        AB magnitude per filter. Times outside the range the surrogate was
        trained on are returned as non-detections.
    """

    if filters is None:
        filters = list(svd_mag_model.keys())

    # add null output for other filters, especially radio and X-ray filters when using with GRB data
    mAB = {
        filt: np.full_like(sample_times, np.inf)
        for filt in filters
        if filt not in svd_mag_model
    }

    for filt in filters:
        if filt in mAB:
            continue
        tt_interp, mag_back = eval_svd_model(
            svd_mag_model[filt], mag_ncoeff, param_list
        )

        # FIXME quick-fix to not trust lightcurve after outside training time range
        mAB[filt] = utils.autocomplete_data(
            sample_times, tt_interp, mag_back, extrapolate=np.inf
        )
    return mAB


def eval_svd_model(svd_model, ass_ncoeff, param_list):
    """Project one parameter set back into a light curve.

    The surrogate stores a handful of SVD coefficients per light curve, and
    either a neural network or Gaussian processes that predict them. This
    predicts the coefficients, then rebuilds the curve from them.

    Parameters
    ----------
    svd_model: dict
        Trained surrogate: basis, normalisation bounds, time grid and the
        predictor itself.
    ass_ncoeff: int or None
        Keep only this many coefficients. Defaults to all the model has.
    param_list: list
        Model parameters, in the order the surrogate was trained on.

    Returns
    -------
    tuple
        The surrogate's own time grid, and the light curve on it.

    Raises
    ------
    ValueError
        If the surrogate has neither a network nor Gaussian processes.
    """
    if ass_ncoeff:
        n_coeff = min(ass_ncoeff, svd_model["n_coeff"])
    else:
        n_coeff = svd_model["n_coeff"]
    VA = svd_model["VA"]
    param_mins = svd_model["param_mins"]
    param_maxs = svd_model["param_maxs"]
    mins = svd_model["mins"]
    maxs = svd_model["maxs"]
    tt_interp = svd_model["tt"]

    param_list_postprocess = np.array(param_list)
    param_list_postprocess = (param_list_postprocess - param_mins) / (
        param_maxs - param_mins
    )
    try:
        model = svd_model["model"]
        # NOTE: This is much(!) faster for small batch sizes than model.predict. Since we mostly call for single params, we should avoid .predict!!!
        cAproj = model(np.atleast_2d(param_list_postprocess)).numpy().T.flatten()
    except KeyError:
        cAproj = np.zeros((n_coeff,))
        gps = svd_model["gps"]
        if gps is None:
            raise ValueError("Gaussian process model unavailable.")
        for i in range(n_coeff):
            gp = gps[i]
            y_pred, sigma2_pred = gp.predict(
                np.atleast_2d(param_list_postprocess), return_std=True
            )
            cAproj[i] = y_pred.item()

    svd_back = np.dot(VA[:, :n_coeff], cAproj)
    svd_back *= maxs - mins
    svd_back += mins
    return tt_interp, svd_back


# grb afterglow
@timeout(60)
def fluxDensity(t, nu, **params):
    """Call afterglowpy, giving up after a minute.

    Some corners of the afterglow parameter space are very slow to
    integrate, which a sampler would otherwise stall on.

    Parameters
    ----------
    t: numpy.ndarray
        Times, in seconds.
    nu: numpy.ndarray
        Frequencies, in Hz.
    **params
        Afterglow parameters, passed straight through.

    Returns
    -------
    numpy.ndarray
        Flux density in mJy.

    Raises
    ------
    TimeoutError
        If afterglowpy takes more than 60 seconds.
    """

    return afterglowpy.fluxDensity(t, nu, **params)


def flux_density_on_time_array(default_time, obs_frequencies, param_dict):
    """Evaluate the afterglow on the full time-frequency grid at once.

    Parameters
    ----------
    default_time: numpy.ndarray
        Times to evaluate, in seconds.
    obs_frequencies: numpy.ndarray
        Observed frequencies, in Hz.
    param_dict: dict
        Afterglow parameters, passed straight to afterglowpy.

    Returns
    -------
    numpy.ndarray
        Flux density in mJy, with one row per time and one column per
        frequency.
    """

    times = np.tile(default_time, (len(obs_frequencies), 1)).T
    nus = np.tile(obs_frequencies, (len(default_time), 1))
    return fluxDensity(times, nus, **param_dict)


def flux_density_on_E0_array(default_time, obs_frequencies, param_dict):
    """Evaluate the afterglow while the jet is still being energised.

    Instead of one fixed energy, the blast energy grows as a power law
    between the start and the end of the injection, then stays at its final
    value. Each time therefore needs its own afterglowpy call.

    Parameters
    ----------
    default_time: numpy.ndarray
        Times to evaluate, in seconds.
    obs_frequencies: numpy.ndarray
        Observed frequencies, in Hz.
    param_dict: dict
        Afterglow parameters, plus the injection ones: log10_Eend, t_start,
        injection_duration and energy_exponential.

    Returns
    -------
    numpy.ndarray
        Flux density in mJy, with one row per time.
    """

    # fetch parameters
    log10_Eend = param_dict["log10_Eend"]
    t_start = param_dict["t_start"]
    t_end = param_dict["injection_duration"]
    energy_exponential = param_dict["energy_exponential"]
    # populate the E0 along the sample_times
    log10_Estart = log10_Eend + energy_exponential * np.log10(t_start / t_end)
    log10_E0 = np.full_like(default_time, log10_Estart)
    # now adjust the log10_E0
    log10_E0[default_time >= t_end] = log10_Eend
    mask = (default_time > t_start) * (default_time < t_end)
    time_scale = np.log10(default_time / t_end)
    log10_E0[mask] = log10_Eend + energy_exponential * time_scale[mask]
    E0 = 10**log10_E0

    def helper(i):
        return fluxDensity(default_time[i], obs_frequencies, E0=E0[i], **param_dict)

    vec_func = np.vectorize(helper, otypes=[np.ndarray])
    mJys = vec_func(np.arange(len(default_time)))
    return np.stack(mJys)


def afterglowpy_lc(sample_times, param_dict, filters, obs_frequencies, flux_func):
    """Compute a GRB afterglow light curve with afterglowpy.

    The afterglow is evaluated on a geometric time grid of its own, then
    interpolated onto the requested times.

    Parameters
    ----------
    sample_times: numpy.ndarray
        Times to return, in days.
    param_dict: dict
        Afterglow parameters.
    filters: list of str
        Filters to evaluate.
    obs_frequencies: numpy.ndarray
        Observed frequency of each filter, in Hz.
    flux_func: callable
        Either the fixed-energy or the energy-injection evaluator.

    Returns
    -------
    dict
        AB magnitude per filter, or an empty dict when afterglowpy timed out
        or returned a non-positive flux.
    """

    tStart = max(10 ** (-5), np.amin(sample_times))
    tEnd = np.amax(sample_times) + 1
    tnode = min(len(sample_times), 201)
    default_time = np.geomspace(tStart, tEnd, num=tnode) * seconds_a_day

    # output flux density is in milliJansky
    try:
        mJys = flux_func(default_time, obs_frequencies, param_dict)
    except TimeoutError:
        return {}

    if np.any(mJys <= 0.0):
        return {}

    mag = {}
    for filt_idx, filt in enumerate(filters):
        mag_d = utils.flux_to_ABmag(mJys[:, filt_idx], unit="mJy")
        mag[filt] = utils.autocomplete_data(
            sample_times, default_time / seconds_a_day, mag_d
        )

    return mag


# hostmodel lightcurve
def host_lc(sample_times, parameters, filters, host_mag):
    """Add a fading afterglow on top of a constant host galaxy.

    Follows arXiv:2303.12849: a power law in time per filter, on a constant
    host flux.

    Parameters
    ----------
    sample_times: numpy.ndarray
        Times to evaluate, in days.
    parameters: dict
        Holds alpha_AG, and a_AG and f_nu for each filter.
    filters: list of str
        Filters to evaluate.
    host_mag: list of float
        Host magnitude per filter, kept out of the fit.

    Returns
    -------
    dict
        AB magnitude per filter.
    """

    # Based on arxiv:2303.12849
    mag = {}
    alpha = parameters["alpha_AG"]
    for i, filt in enumerate(filters):
        # assumed to be in unit of muJy
        a_AG = parameters[f"a_AG_{filt}"]
        f_nu_filt = parameters[f"f_nu_{filt}"]
        flux_per_filt = a_AG * np.power(sample_times, -alpha) + f_nu_filt
        mag[filt] = utils.flux_to_ABmag(flux_per_filt, residual_mag=host_mag[i])
    return mag


# supernova model
def sn_lc(sample_times_stretched, sn_model, filters, lambdas):
    """Read magnitudes off an sncosmo supernova model.

    Filters sncosmo does not know are evaluated from the spectrum at their
    effective wavelength instead. Filters falling outside the wavelength
    range the model covers are returned as non-detections.

    Parameters
    ----------
    sample_times_stretched: numpy.ndarray
        Times to evaluate, already stretched into the model frame.
    sn_model: sncosmo.Model
        The supernova model.
    filters: list of str
        Filters to evaluate.
    lambdas: list of float
        Effective wavelength of each filter, in metres.

    Returns
    -------
    dict
        AB magnitude per filter.
    """

    mag = {}
    for filt, lambda_ in zip(filters, lambdas):
        try:
            mag[filt] = sn_model.bandmag(filt, "ab", sample_times_stretched)
        except ValueError:
            lambda_AA = 1e10 * lambda_
            if lambda_AA < sn_model.minwave() or lambda_AA > sn_model.maxwave():
                mag[filt] = np.full_like(sample_times_stretched, np.inf)
                continue
            # NOTE: workaround  for potential bug in sncosmo: buffer error if lambdaa as float
            flux_AA = sn_model.flux(sample_times_stretched, [lambda_AA]).flatten()
            # see https://en.wikipedia.org/wiki/AB_magnitude
            mag[filt] = utils.flux_to_ABmag(
                flux_AA * 3.34e4 * lambda_AA**2, unit="Jy"
            )
    return mag


# shock-cooling lightcurve
def sc_bol_lc(sample_times, param_dict, compute_Rs):
    """Bolometric light curve of a shock cooling envelope.

    Two regimes are stitched at the diffusion time: a power-law decline
    while the envelope is still optically thick, an exponential one after.

    Parameters
    ----------
    sample_times: numpy.ndarray
        Times to evaluate, in days.
    param_dict: dict
        Holds log10_Menv, log10_Renv and log10_Ee.
    compute_Rs: bool
        If True, also return the photospheric radius, which the filter
        evaluation needs.

    Returns
    -------
    numpy.ndarray or tuple
        The luminosity, or the luminosity and the photospheric radius.
    """

    t = sample_times * seconds_a_day

    # fetch parameter values
    Me = 10 ** param_dict["log10_Menv"] * msun_cgs
    Renv = 10 ** param_dict["log10_Renv"]
    Ee = 10 ** param_dict["log10_Ee"]

    n = 10
    delta = 1.1
    K = (n - 3) * (3 - delta) / (4 * np.pi * (n - delta))  # K = 0.119
    kappa = 0.2
    vt = np.sqrt(((n - 5) * (5 - delta) / ((n - 3) * (3 - delta))) * (2 * Ee / Me))
    td = np.sqrt((3 * kappa * K * Me) / ((n - 1) * vt * c_cgs))

    # evalute the model, lbol first
    prefactor = np.pi * (n - 1) / (3 * (n - 5)) * c_cgs * Renv * vt * vt / kappa
    L_early = prefactor * np.power(td / t, 4 / (n - 2))
    L_late = prefactor * np.exp(-0.5 * (t * t / td / td - 1))
    lbol = np.zeros_like(t)
    # stiching the two regime
    lbol[t < td] = L_early[t < td]
    lbol[t >= td] = L_late[t >= td]

    if not compute_Rs:
        return lbol

    # else setup for evalution in filters
    tph = np.sqrt(3 * kappa * K * Me / (2 * (n - 1) * vt * vt))
    Rs = np.power(tph / t, 2 / (n - 1)) * vt * t
    late_base = 1 + (delta - 1) / (n - 1) * ((t / tph) ** 2 - 1)
    R_late = np.power(late_base, -1 / (delta - 1)) * vt * t
    Rs[t >= tph] = R_late[t >= tph]
    return lbol, Rs


def sc_lc(lbol, Rs, nu_host, filters):
    """Turn a shock cooling luminosity into magnitudes per filter.

    The effective temperature follows from the luminosity and the
    photospheric radius, assuming the envelope radiates as a blackbody.

    Parameters
    ----------
    lbol: numpy.ndarray
        Bolometric luminosity, in erg/s.
    Rs: numpy.ndarray
        Photospheric radius, in cm.
    nu_host: numpy.ndarray
        Rest-frame frequency of each filter, in Hz.
    filters: list of str
        Filters to evaluate.

    Returns
    -------
    dict
        AB magnitude per filter.
    """

    sigmaT4 = lbol / (4 * np.pi * Rs * Rs)
    T = np.power(sigmaT4 / sigSB, 0.25)
    T[T == 0.0] = np.nan
    one_over_T = 1.0 / T
    one_over_T[~np.isfinite(one_over_T)] = np.inf

    result = mag_dict_for_blackbody(filters, one_over_T, Rs, nu_host)

    return result


# semi-analytical models for kilonovae


def heating_rate_Korobkin_Rosswog(t, eth=0.5):
    """Computes the nuclear specific heating rate over time.

    This implementation is based on a model from Korobkin et al. 2012
    (DOI: 10.1111/j.1365-2966.2012.21859.x), derived from nucleosynthesis
    simulations in compact binary merger ejecta. The model uses these
    parameters: eps0 = 2e18, t0 = 1.3, sig = 0.11, alpha = 1.3.

    Args:
        t: float or numpy.ndarray
           Time (in s) in rest-frame to evaluate the light curve. Can be an array
           for multiple time points.
        eth: float or numpy.ndarray, default=0.5
           Efficiency parameter representing the fraction of nuclear power
           retained in the matter, as defined by Korobkin et al. 2012.

    Returns:
        float or numpy.ndarray: Nuclear specific heating rate in erg/g/s
        (units implied but not explicitly used).
    """
    # Define model constants
    eps0 = 2e18  # erg/g/s
    t0 = 1.3  # s
    sig = 0.11  # s
    alpha = 1.3  # dimensionless
    # Calculate the time evolution term
    time_term = 0.5 - 1.0 / np.pi * np.arctan((t - t0) / sig)
    # Return the heating rate
    return 2 * eps0 * eth * np.power(time_term, alpha)


def metzger_lc(sample_times, param_dict, nu_host, filters):
    """Kilonova light curve from the Metzger semi-analytical model.

    The ejecta are split into velocity shells, each heated by r-process
    decay and cooled by expansion and radiation. The shells are evolved
    together, and the photosphere is read off where the optical depth
    reaches unity.

    Parameters
    ----------
    sample_times: numpy.ndarray
        Times to evaluate, in days.
    param_dict: dict
        Holds log10_mej, log10_vej, beta and log10_kappa_r.
    nu_host: numpy.ndarray
        Rest-frame frequency of each filter, in Hz.
    filters: list of str
        Filters to evaluate.

    Returns
    -------
    dict
        AB magnitude per filter.
    """

    # fetch parameters
    M0 = 10 ** param_dict["log10_mej"] * msun_cgs  # total ejecta mass
    v0 = 10 ** param_dict["log10_vej"] * c_cgs  # minimum escape velocity
    beta = param_dict["beta"]
    kappa_r = 10 ** param_dict["log10_kappa_r"]
    # z = param_dict["redshift"]

    # convert time from day to second
    t = sample_times * seconds_a_day  # / (1 + z)
    tprec = len(t)

    if np.any(t == 0):
        raise ValueError("For Me2017, start later than t=0")

    # define additional parameters
    E0 = 0.5 * M0 * v0 * v0  # initial thermal energy of bulk
    Mn = 1e-8 * msun_cgs  # mass cut for free neutrons
    Ye = 0.1  # electron fraction
    Xn0max = 1 - 2 * Ye  # initial neutron mass fraction in outermost layers
    mprec = 300
    # define mass / velocity array of the outer ejecta, comprised half of the mass
    m = np.geomspace(1e-8, M0 / msun_cgs, mprec)
    vm = v0 * np.power(m * msun_cgs / M0, -1.0 / beta)
    vm[vm > c_cgs] = c_cgs

    # define thermalization efficiency from Barnes+16, eq. 34
    def thermalization_efficiency(time, ca, cb, cd):
        timescale_factor = 2 * cb * time**cd
        eff_therm = (
            np.exp(-ca * time) + np.log(1.0 + timescale_factor) / timescale_factor
        )
        return 0.36 * eff_therm

    eth = thermalization_efficiency(sample_times, ca=0.56, cb=0.17, cd=0.74)
    # eth2= thermalization_efficiency(t_day, ca= 8.2, cb= 1.2, cd=1.52)
    # eth3= thermalization_efficiency(t_day, ca= 1.3, cb= 0.2, cd= 1.1)

    # define radioactive heating rates
    Xn0 = Xn0max * 2 * np.arctan(Mn / m / msun_cgs) / np.pi  # neutron mass fraction
    Xr = 1.0 - Xn0  # r-process fraction

    # define arrays in mass layer and time
    tarray = np.tile(t, (mprec, 1))
    Xn0array = np.tile(Xn0, (tprec, 1)).T
    Xrarray = np.tile(Xr, (tprec, 1)).T
    etharray = np.tile(eth, (mprec, 1))

    Xn = Xn0array * np.exp(-tarray / 900.0)
    edotn = 3.2e14 * Xn
    edotr = 2.1e10 * etharray * ((tarray / seconds_a_day) ** (-1.3))
    edot = edotn + edotr
    kappan = 0.4 * (1.0 - Xn - Xrarray)
    kappar = kappa_r * Xrarray
    kappa = kappan + kappar

    # define total r-process heating of inner layer
    Lr = M0 * heating_rate_Korobkin_Rosswog(t, eth=eth)
    Lr = Lr / 1e20
    Lr = Lr / 1e20

    # *** define arrays by mass layer/time arrays ***
    ene = np.zeros((mprec, tprec))
    lum = np.zeros((mprec, tprec))
    tdiff = np.zeros((mprec, tprec))
    tau = np.zeros((mprec, tprec))
    # properties of photosphere
    R_photo = np.zeros((tprec,))
    vphoto = np.zeros((tprec,))
    mphoto = np.zeros((tprec,))
    kappaphoto = np.zeros((tprec,))
    Lsd = np.zeros((tprec,))

    # *** define arrays for total ejecta (1 zone = deepest layer) ***
    # thermal energy
    E = np.zeros((tprec,))
    # kinetic energy
    Ek = np.zeros((tprec,))
    # velocity
    v = np.zeros((tprec,))
    R = np.zeros((tprec,))
    taues = np.zeros((tprec,))
    Lrad = np.zeros((tprec,))
    temp = np.zeros((tprec,))
    # setting initial conditions
    E[0] = E0 / 1e20
    E[0] = E[0] / 1e20
    Ek[0] = E0 / 1e20
    Ek[0] = Ek[0] / 1e20
    v[0] = v0
    R[0] = t[0] * v[0]

    dt = t[1:] - t[:-1]
    dm = m[1:] - m[:-1]

    for j in range(tprec - 1):
        # one zone calculation

        if E[j] < 0.0:
            E[j] = np.abs(E[j])
        temp[j] = 1e10 * (3 * E[j] / (arad * 4 * np.pi * R[j] ** (3))) ** (0.25)
        if temp[j] > 4000.0:
            kappaoz = kappa_r
        if temp[j] < 4000.0:
            kappaoz = kappa_r * (temp[j] / 4000.0) ** (5.5)
        kappaoz = kappa_r
        LPdV = E[j] * v[j] / R[j]
        tdiff0 = 3 * kappaoz * M0 / (4 * np.pi * c_cgs * v[j] * t[j])
        tlc0 = R[j] / c_cgs
        tdiff0 = tdiff0 + tlc0
        Lrad[j] = E[j] / tdiff0
        Ek[j + 1] = Ek[j] + LPdV * dt[j]
        v[j + 1] = 1e20 * (2 * Ek[j] / M0) ** (0.5)
        E[j + 1] = (Lr[j] + Lsd[j] - LPdV - Lrad[j]) * dt[j] + E[j]
        R[j + 1] = v[j + 1] * dt[j] + R[j]
        taues[j + 1] = M0 * 0.4 / (4 * R[j + 1] ** 2)

        templayer = (
            3 * ene[:-1, j] * dm * msun_cgs / (arad * 4 * np.pi * (t[j] * vm[:-1]) ** 3)
        )

        if np.isnan(templayer).any():
            templayer = np.nan_to_num(templayer)
            templayer = abs(templayer) ** 0.25
        else:
            templayer = abs(templayer) ** (0.25)

        kappa_correction = np.ones(templayer.shape)
        kappa_correction[templayer > 4000.0] = 1.0
        kappa_correction[templayer < 4000.0] = templayer[
            templayer < 4000.0
        ] / 4000.0 ** (5.5)
        kappa_correction[:] = 1

        tdiff[:-1, j] = 0.08 * kappa[:-1, j] * m[:-1] * msun_cgs
        tdiff[:-1, j] *= 3 * kappa_correction / (vm[:-1] * c_cgs * t[j] * beta)
        tau[:-1, j] = (
            m[:-1] * msun_cgs * kappa[:-1, j] / (4 * np.pi * (t[j] * vm[:-1]) ** 2)
        )
        lum[:-1, j] = ene[:-1, j] / (tdiff[:-1, j] + t[j] * (vm[:-1] / c_cgs))
        ene[:-1, j + 1] = ene[:-1, j] + dt[j] * (
            edot[:-1, j] - (ene[:-1, j] / t[j]) - lum[:-1, j]
        )
        lum[:-1, j] = lum[:-1, j] * dm * msun_cgs

        tau[mprec - 1, j] = tau[mprec - 2, j]
        # photosphere
        pig = np.argmin(np.abs(tau[:, j] - 1))
        vphoto[j] = vm[pig]
        R_photo[j] = vphoto[j] * t[j]
        mphoto[j] = m[pig]
        kappaphoto[j] = kappa[pig, j]

    Ltotm = np.sum(lum, axis=0)
    Ltotm = Ltotm / 1e20
    Ltotm = Ltotm / 1e20

    Ltot = np.abs(Ltotm)
    # lbol = Ltotm * 1e40

    Tobs = 1e10 * (Ltot / (4 * np.pi * R_photo**2 * sigSB)) ** (0.25)
    Tobs = utils.autocomplete_data(sample_times, sample_times, Tobs)

    Tobs[Tobs <= 0.0] = np.nan
    one_over_T = 1.0 / Tobs
    one_over_T[~np.isfinite(one_over_T)] = np.inf
    return mag_dict_for_blackbody(filters, one_over_T, R_photo, nu_host)


def eff_metzger_lc(sample_times, param_dict, nu_host, filters):
    """Kilonova light curve from the Metzger model, evaluated cheaply.

    Same physics as the full version, restructured for speed so that a
    sampler can afford to call it.

    Parameters
    ----------
    sample_times: numpy.ndarray
        Times to evaluate, in days.
    param_dict: dict
        Holds log10_mej, log10_vej, beta and log10_kappa_r.
    nu_host: numpy.ndarray
        Rest-frame frequency of each filter, in Hz.
    filters: list of str
        Filters to evaluate.

    Returns
    -------
    dict
        AB magnitude per filter.
    """

    # fetch parameters
    M0 = 10 ** param_dict["log10_mej"] * msun_cgs  # total ejecta mass
    v0 = 10 ** param_dict["log10_vej"] * c_cgs  # minimum escape velocity
    beta = param_dict["beta"]
    kappa_r = 10 ** param_dict["log10_kappa_r"]

    # convert time from day to second
    t = sample_times * seconds_a_day
    tprec = len(t)

    if np.any(t == 0):
        raise ValueError("For Me2017, start later than t=0")

    # define additional parameters
    Mn = 1e-8  # mass cut for free neutrons
    Ye = 0.1  # electron fraction
    Xn0max = 1 - 2 * Ye  # initial neutron mass fraction in outermost layers
    mprec = 300
    # define mass / velocity array of the outer ejecta, comprised half of the mass
    m = np.geomspace(1e-8, M0 / msun_cgs, mprec)
    vm = v0 * np.power(m * msun_cgs / M0, -1.0 / beta)
    vm[vm > c_cgs] = c_cgs

    # define thermalization efficiency from Barnes+16, eq. 34
    def thermalization_efficiency(time, ca, cb, cd):
        timescale_factor = 2 * cb * time**cd
        eff_therm = (
            np.exp(-ca * time) + np.log(1.0 + timescale_factor) / timescale_factor
        )
        return 0.36 * eff_therm

    eth = thermalization_efficiency(sample_times, ca=0.56, cb=0.17, cd=0.74)

    # define radioactive heating rates
    Xn0 = Xn0max * 2 * np.arctan(Mn / m) / np.pi  # neutron mass fraction
    Xr = 1.0 - Xn0  # r-process fraction

    # define arrays in mass layer and time
    tarray = np.tile(t, (mprec, 1))
    Xn0array = np.tile(Xn0, (tprec, 1)).T
    Xrarray = np.tile(Xr, (tprec, 1)).T
    etharray = np.tile(eth, (mprec, 1))

    Xn = Xn0array * np.exp(-tarray / 900.0)
    edotn = 3.2e14 * Xn
    edotr = 2.1e10 * etharray * ((tarray / seconds_a_day) ** (-1.3))
    edot = edotn + edotr
    kappan = 0.4 * (1.0 - Xn - Xrarray)
    kappa = kappan + kappa_r * Xrarray

    # define specific heating rates and opacity of each mass layer

    # *** define arrays by mass layer/time arrays ***
    ene = np.zeros(mprec - 1)
    lum = np.zeros((mprec - 1, tprec))
    # properties of photosphere
    R_photo = np.zeros((tprec,))

    dt = t[1:] - t[:-1]
    dm = m[1:] - m[:-1]

    for j in range(tprec - 1):
        tdiff = 0.08 * kappa[:-1, j] * m[:-1] * msun_cgs * 3
        tdiff /= vm[:-1] * c_cgs * t[j] * beta
        tau = m[:-1] * msun_cgs * kappa[:-1, j] / (4 * np.pi * (t[j] * vm[:-1]) ** 2)
        lum_j = ene / (tdiff + t[j] * (vm[:-1] / c_cgs))
        lum[:, j] = lum_j * dm * msun_cgs

        ene += dt[j] * (edot[:-1, j] - (ene / t[j]) - lum_j)
        # photosphere
        pig = np.argmin(np.abs(tau - 1))
        R_photo[j] = vm[pig] * t[j]

    Ltotm = np.sum(lum, axis=0) / 1e20 / 1e20
    Ltot = np.abs(Ltotm)
    # lbol = Ltotm * 1e40

    Tobs = 1e10 * (Ltot / (4 * np.pi * R_photo**2 * sigSB)) ** (0.25)
    Tobs = utils.autocomplete_data(sample_times, sample_times, Tobs)

    Tobs[Tobs <= 0.0] = np.nan
    one_over_T = 1.0 / Tobs
    one_over_T[~np.isfinite(one_over_T)] = np.inf
    return mag_dict_for_blackbody(filters, one_over_T, R_photo, nu_host)


def HoNa_lc(sample_times, param_dict, nu_host, filters):
    """Kilonova light curve from the Hotokezaka and Nakar model.

    Parameters
    ----------
    sample_times: numpy.ndarray
        Times to evaluate, in days.
    param_dict: dict
        Holds log10_mej, the velocity range, the opacities, and optionally
        the density slope n.
    nu_host: numpy.ndarray
        Rest-frame frequency of each filter, in Hz.
    filters: list of str
        Filters to evaluate.

    Returns
    -------
    dict
        AB magnitude per filter.
    """

    # calculate the temperature and luminosity to feed into the blackbody radiation calculation
    conv_params = setup_HoNa_params(sample_times, param_dict)
    inv_temp, R_photo = temp_photosphere_HoNa(*conv_params, param_dict.get("n", 4.5))
    return mag_dict_for_blackbody(filters, inv_temp, R_photo, nu_host)


def setup_HoNa_params(sample_times, param_dict):
    """Unpack the Hotokezaka and Nakar parameters into physical units.

    Parameters
    ----------
    sample_times: numpy.ndarray
        Times in days. Converted to seconds in place.
    param_dict: dict
        Holds log10_mej, vej_min, vej_max, vej_frac and the two opacities.

    Returns
    -------
    tuple
        Times in seconds, ejecta mass in grams, the three-point velocity
        grid in units of c, and the two opacities in cm2/g.

    Notes
    -----
    sample_times is scaled in place, so the caller's array is modified.
    """

    sample_times *= seconds_a_day
    mej = 10 ** param_dict["log10_mej"] * msun_cgs
    vej_max = param_dict["vej_max"]
    vej_min = param_dict["vej_min"]
    vej_range = vej_max - vej_min
    vej = param_dict["vej_frac"] * vej_range + vej_min
    velocities = np.array([vej_min, vej, vej_max])  # in units of c

    # in cm**2 / g
    opacities = np.array(
        [
            10 ** param_dict["log10_kappa_low_vej"],
            10 ** param_dict["log10_kappa_high_vej"],
        ]
    )
    return sample_times, mej, velocities, opacities


# the following functions are for the semi-analytic model using Hotokezaka & Nakar heating rate
def luminosity_HoNa(E, t, td, be):
    """Luminosity escaping one ejecta shell.

    Parameters
    ----------
    E: numpy.ndarray
        Energy stored in the shell, in erg.
    t: numpy.ndarray
        Time, in seconds.
    td: numpy.ndarray
        Diffusion timescale of the shell, in seconds.
    be: numpy.ndarray
        Shell velocity, in units of c.

    Returns
    -------
    numpy.ndarray
        Escaping luminosity, in erg/s.
    """

    # Calculate diffusion time ratio
    t_dif = td / t
    # Determine escape time
    tesc = np.minimum(t, t_dif) + be * t
    # Calculate maximum y value
    ymax = np.sqrt(0.5 * t_dif / t)
    # Return luminosity using complementary error function
    return erfc(ymax) * E / tesc


def dEdt_HoNa(t, E, dM, td, be):
    """Energy budget of the ejecta shells, for the ODE solver.

    Each shell gains radioactive heat, loses energy to expansion, and loses
    what escapes as light.

    Parameters
    ----------
    t: float
        Time, in seconds.
    E: numpy.ndarray
        Energy currently stored in each shell, in erg.
    dM: numpy.ndarray
        Mass of each shell, in grams.
    td: numpy.ndarray
        Diffusion timescale of each shell, in seconds.
    be: numpy.ndarray
        Velocity of each shell, in units of c.

    Returns
    -------
    numpy.ndarray
        Rate of change of the stored energy, in erg/s.
    """

    # Calculate heating contribution
    heat = dM * heating_rate_Korobkin_Rosswog(t)
    # Calculate luminosity
    L = luminosity_HoNa(E, t, td, be)
    dEdt = -E / t - L + heat
    return dEdt


def temp_photosphere_HoNa(t, mej, velocities, opacities, n):
    """Evolve the ejecta shells and read off the photosphere.

    The ejecta are divided into a hundred velocity shells whose stored
    energy is integrated in time. The photosphere is then located where the
    accumulated optical depth reaches unity, and its temperature follows
    from the total luminosity.

    Parameters
    ----------
    t: numpy.ndarray
        Times to evaluate, in seconds.
    mej: float
        Ejecta mass, in grams.
    velocities: numpy.ndarray
        Velocity grid, in units of c.
    opacities: numpy.ndarray
        Opacity of each velocity range, in cm2/g.
    n: float
        Power-law index of the density profile.

    Returns
    -------
    tuple
        Inverse photospheric temperature in 1/K, and photospheric radius
        in cm.
    """
    # Prepare velocity grid
    be_0 = velocities[0]
    be_max = velocities[-1]
    n_shells = 100
    # Use inverse log spacing for velocity steps - simplified with direct calculations
    # Note: this is not equal to np.geomspace(be_0, be_max, n_shells)!
    bes = be_max + be_0 - np.geomspace(be_0, be_max, n_shells)
    bes = np.flipud(bes)[:-1]  # Flip and remove last element in one operation
    dbe = np.diff(np.append(bes, be_max))  # Calculate diff by appending be_max

    i = np.searchsorted(velocities, bes)

    # Calculate power factors once for reuse
    bej_power = (velocities / be_0) ** (1 - n)
    bes_power = (bes / be_0) ** (1 - n)

    # Vectorized calculation of tau_accum
    tau_accum = -np.cumsum((opacities * np.diff(bej_power))[::-1])[::-1]
    tau_accum = np.append(tau_accum, 0)
    # Vectorized calculation of taus
    taus = tau_accum[i] + opacities[i - 1] * (bes_power - bej_power[i])

    vej_0 = velocities[0] * c_cgs
    rho_0 = mej * (n - 3) / (4 * np.pi * vej_0**3) / (1 - (be_max / be_0) ** (3 - n))
    taus *= vej_0 * rho_0 / (n - 1)

    # Mass and time delay calculations
    bes_power_2n = (bes / be_0) ** (2 - n)  # Calculate power once
    dMs = 4.0 * np.pi * vej_0**3 * rho_0 * bes_power_2n * dbe / be_0
    tds = taus * bes

    # Prepare arrays for solve_ivp - use broadcasting directly
    bes_col = bes[:, np.newaxis]
    tds_col = tds[:, np.newaxis]
    dMs_col = dMs[:, np.newaxis]

    # Evolve in time
    t0 = 5e-2 * seconds_a_day  # initial time to start the integration
    out = solve_ivp(
        dEdt_HoNa,
        (t[0], t[-1]),
        np.zeros_like(bes),
        first_step=t0,
        args=(dMs_col, tds_col, bes_col),
        vectorized=True,
    )

    # Total luminosity calculation
    LL = luminosity_HoNa(out.y, out.t[np.newaxis, :], tds_col, bes_col).sum(0)

    # Log-log space interpolation - preserve only necessary portion
    log_t = np.log(out.t[1:])
    log_LL = np.log(LL[1:])
    log_L_interp = CubicSpline(log_t, log_LL, extrapolate=True)

    # Calculate final results in vectorized operations
    lbol = np.exp(log_L_interp(np.log(t)))
    # Effective radius - use vectorized log operations
    log_taus = np.log(taus[::-1])
    log_bes = np.log(bes[::-1])
    log_t_doubled = 2 * np.log(t)
    be = np.exp(np.interp(log_t_doubled, log_taus, log_bes))
    Rphoto = be * t * c_cgs  # effective radius in cm
    # Effective temperature - use broadcasting for squaring
    sigmaT4 = lbol / (4 * np.pi * Rphoto * Rphoto)
    inv_T = np.power(sigSB / sigmaT4, 0.25)
    # Return results
    return inv_T, Rphoto


def synchrotron_powerlaw(sample_times, param_dict, nu_obs, filters):
    """Light curve of a source that is a power law in time and frequency.

    Parameters
    ----------
    sample_times: numpy.ndarray
        Times to evaluate, in days.
    param_dict: dict
        Holds beta_freq, alpha_time, F_ref and distance_modulus.
    nu_obs: numpy.ndarray
        Observed frequency of each filter, in Hz.
    filters: list of str
        Filters to evaluate.

    Returns
    -------
    dict
        AB magnitude per filter. The distance modulus is subtracted back
        out, the reference flux being defined at the observer.
    """

    beta = param_dict["beta_freq"]  # frequency index
    alpha = param_dict["alpha_time"]  # time index
    F_ref = param_dict["F_ref"]  # in mJy for t=1day and nu=1Hz
    mag = {}
    for idx, filt in enumerate(filters):
        F_pl = F_ref * np.power(nu_obs[idx], -beta) * np.power(sample_times, -alpha)
        # remove the distance modulus for the synchrotron powerlaw
        # as the reference flux is defined at the observer
        mag[filt] = (
            utils.flux_to_ABmag(F_pl, unit="mJy") - param_dict["distance_modulus"]
        )
    return mag


# generic blackbody
def inv_temp_and_photosphere_from_params(param_dict):
    """Derive the photospheric radius of a blackbody from its luminosity.

    Parameters
    ----------
    param_dict: dict
        Holds temperature in K and bb_luminosity in erg/s.

    Returns
    -------
    tuple
        Inverse temperature in 1/K, and photospheric radius in cm.
    """

    # parameter conversion
    inv_temp = 1.0 / param_dict["temperature"]  # blackbody's temperature in K
    R_photo = (
        np.sqrt(
            param_dict["bb_luminosity"]
            / 4
            / np.pi
            / sigSB  # blackboady's total luminosity in erg/s
        )
        * inv_temp
        * inv_temp
    )
    return inv_temp, R_photo


def blackbody_constant_temperature(_, param_dict, nu_host, filters):
    """Light curve of a blackbody that never cools.

    Parameters
    ----------
    _: numpy.ndarray
        Times, unused: the source does not evolve.
    param_dict: dict
        Holds temperature and bb_luminosity.
    nu_host: numpy.ndarray
        Rest-frame frequency of each filter, in Hz.
    filters: list of str
        Filters to evaluate.

    Returns
    -------
    dict
        AB magnitude per filter, constant in time.
    """

    inv_temp, R_photo = inv_temp_and_photosphere_from_params(param_dict)
    return mag_dict_for_blackbody(filters, inv_temp, R_photo, nu_host)


def powerlaw_blackbody_constant_temperature_lc(_, param_dict, nu_host, filters):
    """Light curve of a non-cooling blackbody under a power law.

    The power law is normalised on the g band and added to the blackbody
    flux at every frequency.

    Parameters
    ----------
    _: numpy.ndarray
        Times, unused: the source does not evolve.
    param_dict: dict
        Holds temperature, bb_luminosity, beta and powerlaw_mag.
    nu_host: numpy.ndarray
        Rest-frame frequency of each filter, in Hz.
    filters: list of str
        Filters to evaluate. Must contain g, used as the reference.

    Returns
    -------
    dict
        AB magnitude per filter.
    """

    # calculate the powerlaw prefactor (with the reference filter 'g')
    nu_ref = nu_host[filters.index("g")]  # FIXME, seems like a legacy hack
    powerlaw_prefactor = np.power(nu_ref, param_dict["beta"]) * np.power(
        10, -0.4 * (param_dict["powerlaw_mag"] + 48.6)
    )

    def additive_per_freq(nu):
        return powerlaw_prefactor * np.power(nu, -param_dict["beta"])

    inv_temp, R_photo = inv_temp_and_photosphere_from_params
    return mag_dict_for_blackbody(
        filters, inv_temp, R_photo, nu_host, add=additive_per_freq
    )


# lightcurve data generation
def create_light_curve_data(
    injection_parameters,
    args,
    light_curve_model,
    sample_times=None,
    keep_infinite_data=False,
    rng=None,
):
    """Simulate what a follow-up campaign would have observed.

    Generates the true light curve from the model, samples it down to a
    telescope cadence, adds noise and applies the detection limits.

    Parameters
    ----------
    injection_parameters: dict
        Parameters of the transient to simulate.
    args: argparse.Namespace
        Parsed command-line arguments.
    light_curve_model: nmma.em.model.LightCurveModelContainer
        Model used to generate the curve.
    sample_times: numpy.ndarray, optional
        Times to evaluate. Defaults to the model's own grid.
    keep_infinite_data: bool, optional
        If True, keep the non-detections. They are dropped otherwise.
    rng: numpy.random.Generator, optional
        Source of randomness. Seeded from args when omitted.

    Returns
    -------
    dict
        Photometry per filter.

    Raises
    ------
    ValueError
        If the parameters yield an empty light curve.
    """

    injection_parameters = light_curve_model.parameter_conversion(injection_parameters)
    filters = utils.set_filters(args)
    trigger_time = read_trigger_time(injection_parameters, args)
    if trigger_time is None:
        trigger_time = 0.0
    if rng is None:
        rng = np.random.default_rng(args.generation_seed)
    if getattr(args, "absolute", False):
        # create lightcurve_data
        if sample_times is None:
            sample_times = light_curve_model.model_times
        lc = light_curve_model.generate_lightcurve(sample_times, injection_parameters)
        # if "timeshift" in injection_parameters: # included in gen_detector_lc
        #     trigger_time += injection_parameters["timeshift"]
    else:
        # basic idea: generate lc works on desired times in source_frame,
        # observing times are redshifted and have extra timeshift (interpreted as missed detections)
        sample_times, lc = light_curve_model.gen_detector_lc(
            injection_parameters, sample_times
        )
    if not lc:
        raise ValueError("Injection parameters return empty light curve.")
    # curate data
    true_data = {
        filt: {"time": sample_times + trigger_time, "mag": lc[filt]} for filt in filters
    }
    observable_data = adjust_lc_for_telescopes(
        true_data, args, filters, rng, trigger_time
    )
    observed_data = adjust_lc_for_observations(observable_data, args, filters, rng)

    if not keep_infinite_data:
        for filt, val_dict in observed_data.items():

            keep_idx = np.isfinite(val_dict["mag"]) & np.isfinite(val_dict["mag_error"])
            observed_data[filt] = {key: val[keep_idx] for key, val in val_dict.items()}

    return observed_data


def adjust_lc_for_telescopes(true_data, args, filters, rng, trigger_time):
    """Keep only what a real observing campaign would have caught.

    The true light curve is sampled down to the epochs and filters a given
    cadence would have visited, so that an injection looks like a real
    follow-up campaign rather than a continuous curve.

    Parameters
    ----------
    true_data: dict
        The continuous light curve, per filter.
    args: argparse.Namespace
        Parsed command-line arguments, read for the observing strategy.
    filters: list of str
        Filters to process.
    rng: numpy.random.Generator
        Source of the cadence jitter.
    trigger_time: float
        Time of the trigger, in MJD.

    Returns
    -------
    dict
        Light curve restricted to the epochs that were observed.
    """
    strategy = []
    observable_data = {}
    data_original = copy.deepcopy(true_data)
    # use realistic telescope data
    if getattr(args, "rubin_ToO_type", False):
        strategy.extend(rubin_strategy(args.rubin_ToO_type))

    if getattr(args, "ztf_sampling", False):
        # strategy = adjust_data_for_ztf(data, args, filters,
        #             rng, sample_times, trigger_time, passbands_to_keep)
        strategy.extend(ztf_strategy(rng))

    if strategy:
        mjds, filters = [], []
        for obstime, filts in strategy:
            for filt in filts:
                mjds.append(obstime)
                filters.append(filt)
        sim = pd.DataFrame.from_dict({"mjd": mjds, "filter": filters})

        for filt, group in sim.groupby("filter"):
            if filt not in filters:
                continue
            times = group["mjd"].to_numpy() + trigger_time
            filt_data = data_original[filt]
            observable_data[filt] = {
                "time": times,
                "mag": np.interp(
                    times,
                    filt_data["time"],
                    filt_data["mag"],
                    left=np.inf,
                    right=np.inf,
                ),
            }
    else:
        for filt, filt_data in true_data.items():
            observable_data[filt] = filt_data
            time_mask = filt_data["time"] >= trigger_time
            observable_data[filt] = {
                key: val[time_mask] for key, val in filt_data.items()
            }
    return observable_data


def adjust_lc_for_observations(observable_data, args, filters, rng):
    """Add measurement noise and apply the detection limits.

    Gaussian noise is drawn per filter, then every point fainter than its
    detection limit is replaced by the limit itself with an infinite
    uncertainty, which is how NMMA marks a non-detection.

    Parameters
    ----------
    observable_data: dict
        Light curve restricted to the epochs that were observed.
    args: argparse.Namespace
        Parsed command-line arguments, read for the error budget and the
        detection limits.
    filters: list of str
        Filters to process.
    rng: numpy.random.Generator
        Source of the measurement noise.

    Returns
    -------
    dict
        Photometry per filter, with detections and non-detections mixed.
    """
    dmag = utils.set_filter_associated_dict(args.injection_error_budget, filters, 0.1)
    detection_limit = utils.create_detection_limit(args, filters)
    observations = {}
    for filt, filt_data in observable_data.items():
        det_lim = detection_limit.get(filt, np.inf)
        error = rng.normal(scale=dmag[filt], size=len(filt_data["mag"]))
        obs_mags = filt_data["mag"] + error
        observations[filt] = {
            "time": filt_data["time"],
            "mag": np.full_like(filt_data["mag"], det_lim),  # default non-detection
            "mag_error": np.full_like(filt_data["mag"], np.inf),
        }
        det_mask = obs_mags < det_lim
        observations[filt]["mag"][det_mask] = obs_mags[det_mask]
        observations[filt]["mag_error"][det_mask] = dmag[filt]

    return observations


def ztf_strategy(rng):
    """Epochs and filters of an ad hoc ZTF follow-up campaign.

    Loosely follows arXiv:2203.17135: a latency of a few hours, then visits
    that thin out over the first week, each jittered by about an hour.

    Parameters
    ----------
    rng: numpy.random.Generator
        Source of the latency and the jitter.

    Returns
    -------
    generator
        Pairs of observing time in days and filters visited then.
    """

    # Ad hoc ZTF sampling strategy, vaguely inspired by https://arxiv.org/pdf/2203.17135
    t0 = rng.uniform(1 / 24.0, 12.0 / 24.0)  # initial latency between 1-12 hours
    filts = ["ztfg", "ztfr", "ztfi"]
    add_times = [0.0, 0.2, 0.2, 0.4, 0.4, 1.0, 2.0, 3.0, 5.0, 7.0]
    return ((t0 + dt + rng.normal(scale=1.0 / 24), filts) for dt in add_times)


def rubin_strategy(rubin_ToO):
    """Epochs and filters of a Rubin target-of-opportunity campaign.

    The tier names come from the Rubin 2024 workshop write-up and reflect
    how well the event is localised: the better the skymap, the deeper and
    the more filters are used.

    Parameters
    ----------
    rubin_ToO: str
        Tier to follow: platinum, gold, gold_z, silver or silver_z.

    Returns
    -------
    generator
        Pairs of observing time in days and filters visited then.
    """
    gold_times = [1 / 24.0, 2 / 24.0, 4 / 24.0, 1.0, 2.0, 3.0]
    if rubin_ToO == "platinum":
        # platinum is no official name, means 90% GW skymap <30 sq deg
        # this is the gold strategy for an event similar to GW170817 (close and well localized)
        # Three observation on first night with grizy filters
        # One scan Night 1,2,3 w/ same filters
        filts = ["ps1::g", "ps1::r", "ps1::i", "ps1::z", "ps1::y"]
        return ((time, filts) for time in gold_times)

    elif "gold" in rubin_ToO:
        # gold means 90% GW skymap <100 sq deg
        # use gri or possibly grz if more sensitive to KNe
        init_filts = ["ps1::g", "ps1::r"]
        init_filts.append("ps1::z" if "gold_z" == rubin_ToO else "ps1::i")
        # Three pointings Night 0
        filts = [init_filts] * 3
        # One scan Night 1,2,3 w/ r+i
        follow_up_filts = ["ps1::r", "ps1::i"]
        filts.extend([follow_up_filts] * 3)
        return ((time, filt_list) for time, filt_list in zip(gold_times, filts))

    elif "silver" in rubin_ToO:
        # silver means 90% GW skymap <500 sq deg
        # One scan Night 0 w/ g+i or g+z
        filts = (
            ["ps1::g", "ps1::z"] if rubin_ToO == "silver_z" else ["ps1::g", "ps1::i"]
        )
        # One scan each Night 1,2,3 w/ same filters
        silver_times = [1 / 24.0, 1.0, 2.0, 3.0]
        return ((time, filts) for time in silver_times)

    else:
        raise ValueError(
            "args.rubin_ToO_type should be either platinum, gold, or silver"
        )


# FIXME these binaries need to be reworked for py3.12+ envs, currently not working properly
def adjust_data_for_ztf(data, args, filters, rng, sample_times, trigger_time):
    """Resample a light curve onto a realistic ZTF observing history.

    Rather than an idealised cadence, this draws from the survey's own
    recorded sampling and revisit statistics, so that gaps, weather and
    filter choices look like those of a real ZTF campaign.

    Parameters
    ----------
    data: dict
        The continuous light curve, per filter.
    args: argparse.Namespace
        Parsed command-line arguments.
    filters: list of str
        Filters to process.
    rng: numpy.random.Generator
        Source of the sampling draws.
    sample_times: numpy.ndarray
        Times spanned by the light curve, in days.
    trigger_time: float
        Time of the trigger, in MJD.

    Returns
    -------
    dict
        Light curve restricted to the epochs ZTF would have observed.
    """
    with resources.open_binary(
        __package__ + ".data", "ZTF_revisit_kde_public.joblib"
    ) as f:
        ztfrevisit = load(f)
    with resources.open_binary(__package__ + ".data", "ZTF_sampling_public.pkl") as f:
        ztfsampling = load(f)
    with resources.open_binary(__package__ + ".data", "ZTF_revisit_kde_i.joblib") as f:
        ztfrevisit_i = load(f)
    with resources.open_binary(__package__ + ".data", "lims_public_g.joblib") as f:
        ztflimg = load(f)
    with resources.open_binary(__package__ + ".data", "lims_public_r.joblib") as f:
        ztflimr = load(f)
    with resources.open_binary(__package__ + ".data", "lims_i.joblib") as f:
        ztflimi = load(f)

    ztf_uncertainties = getattr(args, "ztf_uncertainties", False)
    if ztf_uncertainties:
        with resources.open_binary(__package__ + ".data", "ZTF_uncer_params.pkl") as f:
            ztfuncer = load(f)

    ztf_ToO = getattr(args, "ztf_ToO", False)
    if ztf_ToO:
        with resources.open_binary(
            __package__ + ".data", f"sampling_ToO_{ztf_ToO}.pkl"
        ) as f:
            ztftoo = load(f)
        with resources.open_binary(
            __package__ + ".data", f"lims_ToO_{ztf_ToO}_g.joblib"
        ) as f:
            ztftoolimg = load(f)
        with resources.open_binary(
            __package__ + ".data", f"lims_ToO_{ztf_ToO}_r.joblib"
        ) as f:
            ztftoolimr = load(f)

    # Create additional ZTF observations
    filter_map = {1: "ztfg", 2: "ztfr", 3: "ztfi"}

    start = rng.uniform(trigger_time, trigger_time + 2)
    t = start
    # ZTF-II Public
    t_list, bands_list = [], []
    while t < sample_times[-1] + trigger_time:
        sample = ztfsampling.sample().iloc[0]
        t_list.append(t + sample["t"])
        bands_list.append(filter_map[sample["bands"]])
        t += float(ztfrevisit.sample())

    # i-band observations may start later
    start = rng.uniform(trigger_time, trigger_time + 4)
    t = start
    i_times = []
    while t < sample_times[-1] + trigger_time:
        i_times.append(t)
        t += float(ztfrevisit_i.sample())
    t_list.extend(i_times)
    bands_list.extend(["ztfi"] * len(i_times))

    sim = pd.DataFrame({"mjd": t_list, "passband": bands_list})
    sim["ToO"] = False

    # with ToO mode
    if ztf_ToO:
        start = rng.uniform(trigger_time, trigger_time + 1)
        t = start
        t_list, bands_list = [], []
        too_samps = ztftoo.sample(rng.choice([1, 2]))
        for i, too in too_samps.iterrows():
            t_list.append(t + too["t"])
            bands_list.append(filter_map[too["bands"]])
            t += 1

        sim_ToO = pd.DataFrame({"mjd": t_list, "passband": bands_list})
        sim_ToO["ToO"] = True
        sim = pd.concat([sim, sim_ToO])  # join the two dataframes

    sim.sort_values(by=["mjd"], inplace=True)
    sim.reset_index(drop=True, inplace=True)
    sim["mag"] = np.nan  # initialize empty mags
    sim["mag_error"] = np.nan

    observations = {}
    # interpolate the light curve data on ztf observations
    for filt, group in sim.groupby("passband"):
        if filt not in filters:  # skip if we are not observing this filter
            continue

        times = group["mjd"].tolist()
        data_dict = copy.deepcopy(data[filt])
        filt_data = {
            "time": times,
            "mag": np.interp(
                times, data_dict["time"], data_dict["mag"], left=np.inf, right=np.inf
            ),
            "mag_error": np.interp(
                times,
                data_dict["time"],
                data_dict["mag_error"],
                left=np.inf,
                right=np.inf,
            ),
        }

        if ztf_uncertainties:
            sim.loc[group.index, "mag"], sim.loc[group.index, "mag_error"] = (
                filt_data["mag"],
                filt_data["mag_error"],
            )
            mag_err = []
            for idx, row in group.iterrows():
                if filt == "ztfg":
                    lim = (
                        float(ztftoolimg.sample())
                        if row["ToO"]
                        else float(ztflimg.sample())
                    )
                elif filt == "ztfr":
                    lim = (
                        float(ztftoolimr.sample())
                        if row["ToO"]
                        else float(ztflimr.sample())
                    )
                else:
                    lim = float(ztflimi.sample())
                if row["mag"] > lim:
                    sim.loc[row.name, "mag"] = lim
                    sim.loc[row.name, "mag_error"] = np.inf

                if not np.isfinite(sim.loc[row.name, "mag_error"]):
                    mag_err.append(np.inf)
                else:
                    df = pd.DataFrame.from_dict(
                        {"passband": [filt], "mag": [sim.loc[row.name, "mag"]]}
                    )
                    df["passband"] = df["passband"].map(
                        {"ztfg": 1, "ztfr": 2, "ztfi": 3}
                    )  # estimate_mag_err maps filter numbers

                    df["mag_err"] = df.apply(
                        lambda x: (ztfuncer["band"] == x["passband"])
                        & (
                            pd.arrays.IntervalArray(ztfuncer["interval"]).contains(
                                x["mag"]
                            )
                        ),
                        axis=1,
                    ).apply(
                        lambda x: scipy.stats.skewnorm.rvs(
                            ztfuncer[x]["a"], ztfuncer[x]["loc"], ztfuncer[x]["scale"]
                        ),
                        axis=1,
                    )
                    if not df["mag_err"].values:
                        argmin_slice = np.argmin(ztfuncer["interval"])
                        for value in df["mag"].values:
                            if ztfuncer.iloc[argmin_slice]["interval"].left > value:
                                print(
                                    f'WARNING: {value} is outside of the measured uncertainty region with a lower limit of {ztfuncer.iloc[argmin_slice]["interval"].left}'
                                )

                    sim.loc[row.name, "mag_error"] = float(df["mag_error"])
                    mag_err.append(df["mag_error"].tolist()[0])

            filt_data = {
                "time": sim.loc[group.index, "mjd"].tolist(),
                "mag": sim.loc[group.index, "mag"].tolist(),
                "mag_error": mag_err,
            }

        observations[filt] = filt_data

    if getattr(args, "train_stats", False):
        sim["tc"] = trigger_time
        sim.to_csv(args.outdir + "/too.csv", index=False)

    return observations
