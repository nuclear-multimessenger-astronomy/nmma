import copy
from joblib import load

import numpy as np
import pandas as pd
from importlib import resources

from scipy.integrate import quad, solve_ivp
from scipy.special import erfc
from scipy.interpolate import CubicSpline
import sncosmo

from .utils import (
    estimate_mag_err, autocomplete_data, flux_to_ABmag, 
    create_detection_limit, set_filters
)

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


### some frequently used constants:
from nmma.joint.constants import msun_cgs, c_cgs, h, kb, sigSB, arad, D 
seconds_a_day = 86400.0  
abs_mag_dist_factor = D**2

def bb_flux_from_inv_temp(nu, inv_temp, R_photo, dist_squared = abs_mag_dist_factor):
    exp = np.exp(h * nu * inv_temp / kb)
    bb_factor = 2.* h/ c_cgs**2
    return bb_factor * nu**3 /(exp-1) * R_photo * R_photo / dist_squared

def mag_dict_for_blackbody(filters, inv_temp, R_photo, nu_host, add = lambda x: 0.):
    mag = {}
    # nu_host = nu_obs * (1 + redshift)
    for idx, filt in enumerate(filters):
        nu_of_filt = nu_host[idx]
        F = bb_flux_from_inv_temp(nu_of_filt, inv_temp, R_photo)
        F += add(nu_of_filt)
        # F *= 1 + redshift ## correction factor for shifted flux density bin
        mag[filt] = flux_to_ABmag(F)

    return mag


#################################################################
######################### LC MODELS #############################
#################################################################

## Arnett model convenience functions
def arnett_lc_get_int_A_non_vec(x, y):
    r = quad(lambda z: 2 * z * np.exp(-2 * z * y + z**2), 0, x)
    return r[0]


arnett_lc_get_int_A = np.vectorize(arnett_lc_get_int_A_non_vec, excluded=["y"])


def arnett_lc_get_int_B_non_vec(x, y, s):
    r = quad(lambda z: 2 * z * np.exp(-2 * z * y + 2 * z * s + z**2), 0, x)
    return r[0]


arnett_lc_get_int_B =  np.vectorize(arnett_lc_get_int_B_non_vec, excluded=["y", "s"])

# arnett_constants
epsilon_ni = 3.9e10  # erg / s / g
epsilon_co = 6.78e9  # erg / s / g
tau_ni = 8.8   # days
tau_co = 111.3   # days
y_scale = 2* tau_ni
s_scale =  (2 * tau_co * tau_ni) / (tau_co - tau_ni)

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
    Mni = 10**param_dict["log10_mni"] * msun_cgs

    y = tau_m / y_scale
    s = tau_m / s_scale
    x = t_day / tau_m

    int_A = arnett_lc_get_int_A(x, y)
    int_B = arnett_lc_get_int_B(x, y, s)

    lbol = Mni * np.exp(-x**2) * (
        (epsilon_ni - epsilon_co) * int_A + epsilon_co * int_B
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
    return Lbol_arnett* (1. - np.exp(-(param_dict["t_0"]/ t_day)**2))


## kilonova from SVD model
def calc_svd_lbol(sample_times, param_list, svd_lbol_model, lbol_ncoeff = None):
    tt_interp, lbol_back =eval_svd_model(svd_lbol_model, lbol_ncoeff, param_list)
    lbol = 10**autocomplete_data(sample_times, tt_interp, lbol_back)
    return np.squeeze(lbol)  #* (1. + z) FIXME: shouldn't this be (1 + z)**2

def calc_svd_lc(
    sample_times,
    param_list,
    svd_mag_model,
    mag_ncoeff: int = None,
    filters: list = None,
):
    """
    Computes the lightcurve from a surrogate model, given the model parameters.
    Args:
        sample_times_source (dict): A filter-specific Time grid on which to evaluate lightcurve
        param_list (Array): Input parameters for the surrogate model
        svd_mag_model (SVDTrainingModel): Trained surrogate model for mag
        mag_ncoeff (int): Number of coefficients after SVD projection for mag
        filters (Array): List/array of filters at which we want to evaluate the model
    """


    if filters is None:
        filters = list(svd_mag_model.keys())
        
    # add null output for other filters, especially radio and X-ray filters when using with GRB data
    mAB = {filt: np.full_like(sample_times, np.inf) for filt in filters if filt not in svd_mag_model}

    for filt in filters:
        if filt in mAB:
            continue
        tt_interp, mag_back = eval_svd_model(svd_mag_model[filt], mag_ncoeff, param_list)

        ### FIXME quick-fix to not trust lightcurve after outside training time range
        mAB[filt] = autocomplete_data(sample_times, tt_interp, mag_back, extrapolate=np.inf)
    return mAB

def eval_svd_model(svd_model, ass_ncoeff, param_list):
    """Evaluate the SVD model"""
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
    param_list_postprocess = (param_list_postprocess- param_mins) / (param_maxs- param_mins)
    try:
        model = svd_model["model"]
        #NOTE: This is much(!) faster for small batch sizes than model.predict. Since we mostly call for single params, we should avoid .predict!!!
        cAproj = model(np.atleast_2d(param_list_postprocess)).numpy().T.flatten()
    except KeyError:
        cAproj = np.zeros((n_coeff,))
        gps = svd_model["gps"]
        if gps is None:
            raise ValueError(
                f"Gaussian process model unavailable."
            )
        for i in range(n_coeff):
            gp = gps[i]
            y_pred, sigma2_pred = gp.predict(
                np.atleast_2d(param_list_postprocess), return_std=True
            )
            cAproj[i] = y_pred.item()

    
    svd_back = np.dot(VA[:, :n_coeff], cAproj)
    svd_back*= (maxs - mins) 
    svd_back+= mins
    return tt_interp, svd_back


## grb afterglow
@timeout(60)
def fluxDensity(t, nu, **params):
    return afterglowpy.fluxDensity(t, nu, **params)

def flux_density_on_time_array(default_time, obs_frequencies, param_dict):
    times = np.tile(default_time, (len(obs_frequencies), 1)).T
    nus = np.tile(obs_frequencies, (len(default_time), 1))
    return fluxDensity(times, nus, **param_dict)

def flux_density_on_E0_array(default_time, obs_frequencies, param_dict):
    # fetch parameters
    log10_Eend = param_dict['log10_Eend']
    t_start = param_dict['t_start']
    t_end = param_dict['injection_duration']
    energy_exponential = param_dict['energy_exponential']
    # populate the E0 along the sample_times
    log10_Estart = log10_Eend + energy_exponential * np.log10(t_start / t_end)
    log10_E0 = np.full_like(default_time, log10_Estart)
    # now adjust the log10_E0
    log10_E0[default_time >= t_end] = log10_Eend
    mask = (default_time > t_start) * (default_time < t_end)
    time_scale = np.log10(default_time / t_end)
    log10_E0[mask] = log10_Eend + energy_exponential * time_scale[mask]
    E0 = 10 ** log10_E0
    vec_func = np.vectorize(
        lambda i: fluxDensity(
            default_time[i], 
            obs_frequencies, 
            E0=E0[i], 
            **param_dict
        ), 
        otypes=[np.ndarray]
    )
    mJys = vec_func(np.arange(len(default_time)))
    return np.stack(mJys)


def afterglowpy_lc(sample_times, param_dict, filters, obs_frequencies, flux_func):
    tStart = max(10 ** (-5), np.amin(sample_times)) 
    tEnd = (np.amax(sample_times) + 1) 
    tnode = min(len(sample_times), 201)
    default_time = np.geomspace(tStart, tEnd, num=tnode)* seconds_a_day

    # output flux density is in milliJansky
    try:
        mJys = flux_func(default_time, obs_frequencies, param_dict)
    except TimeoutError:
        return {}


    if np.any(mJys <= 0.0):
        return {}

    mag = {}
    for filt_idx, filt in enumerate(filters):
        mag_d = flux_to_ABmag(mJys[:, filt_idx], unit= 'mJy')
        mag[filt] = autocomplete_data(sample_times, default_time / seconds_a_day, mag_d)
    
    return mag


## hostmodel lightcurve
def host_lc(sample_times, parameters, filters, host_mag):
    # Based on arxiv:2303.12849
    mag = {}
    alpha = parameters["alpha_AG"]
    for i, filt in enumerate(filters):
        # assumed to be in unit of muJy
        a_AG = parameters[f"a_AG_{filt}"]
        f_nu_filt = parameters[f"f_nu_{filt}"]
        flux_per_filt = a_AG * np.power(sample_times, -alpha) + f_nu_filt
        mag[filt] = flux_to_ABmag(flux_per_filt, residual_mag=host_mag[i])
    return mag

## supernova model
def sn_lc(
    sample_times_stretched,
    parameters,
    cosmology,
    abs_mag=-19.35,
    regularize_band="bessellv",
    regularize_system="vega",
    model_name="nugent-hyper",
    filters=None,
    lambdas=None,
):
    z= parameters['redshift']
    model = sncosmo.Model(source=model_name)
    if model_name in ["salt2", "salt3"]:
        model.set(
            z=z,
            t0=np.median(sample_times_stretched),
            x0=parameters["x0"],
            x1=parameters["x1"],
            c=parameters["c"],
        )
    else:
        model.set(z=z)

    # regularize the absolute magnitude
    abs_mag -= cosmology.distmod(z).value
    model.set_source_peakabsmag(
        abs_mag, regularize_band, regularize_system, cosmo=cosmology
    )

    mag = {}

    for filt, lambda_A in zip(filters, lambdas):
        # convert back to AA
        lambda_AA = 1e10 * lambda_A
        if lambda_AA < model.minwave() or lambda_AA > model.maxwave():
            mag[filt] = np.full_like(sample_times_stretched,np.inf) 
        else:
            try:
                flux = model.flux(sample_times_stretched, [lambda_AA])[:, 0]
                # see https://en.wikipedia.org/wiki/AB_magnitude
                flux_jy = 3.34e4 * np.power(lambda_AA, 2.0) * flux
                mag[filt] = flux_to_ABmag(flux_jy, unit='Jy')
            except Exception:
                return {}

    return mag

## shock-cooling lightcurve
def sc_bol_lc(sample_times, param_dict, compute_Rs):

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
    prefactor = np.pi*(n-1) / (3*(n-5)) * c_cgs*Renv*vt*vt/kappa
    L_early = prefactor * np.power( td/t , 4/(n-2) )
    L_late = prefactor * np.exp(-0.5 * (t*t /td/td - 1))
    lbol = np.zeros_like(t)
    # stiching the two regime
    lbol[t < td] = L_early[t < td]
    lbol[t >= td] = L_late[t >= td]

    if not compute_Rs:
        return lbol
    
    # else setup for evalution in filters
    tph = np.sqrt(3*kappa*K*Me / (2*(n-1) *vt*vt))
    R_early = np.power(tph / t, 2 / (n - 1)) * vt * t
    R_late = (np.power(1+ (delta-1)/(n-1) * ((t/td)**2 - 1) , -1/(delta+1))
        * vt * t )
    Rs = np.zeros_like(t)
    Rs[t < td] = R_early[t < td]
    Rs[t >= td] = R_late[t >= td]
    return lbol, Rs

def sc_lc(lbol, Rs, nu_host, filters): 

    sigmaT4 = lbol / (4 * np.pi * Rs * Rs)
    T = np.power(sigmaT4 / sigSB, 0.25)
    T[T == 0.0] = np.nan
    one_over_T = 1.0 / T
    one_over_T[~np.isfinite(one_over_T)] = np.inf

    return mag_dict_for_blackbody(filters, one_over_T, Rs, nu_host)

## semi-analytical models for kilonovae

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
    t0 = 1.3     # s
    sig = 0.11   # s
    alpha = 1.3  # dimensionless
    # Calculate the time evolution term
    time_term = 0.5 - 1.0 / np.pi * np.arctan((t-t0) / sig)
    # Return the heating rate
    return 2 * eps0 * eth * np.power(time_term, alpha)

def metzger_lc(sample_times, param_dict, nu_host, filters):
    # fetch parameters
    M0 = 10 ** param_dict["log10_mej"] * msun_cgs  # total ejecta mass
    v0 = 10 ** param_dict["log10_vej"] * c_cgs  # minimum escape velocity
    beta = param_dict["beta"]
    kappa_r = 10 ** param_dict["log10_kappa_r"]
    # z = param_dict["redshift"]

    # convert time from day to second
    t = sample_times * seconds_a_day #/ (1 + z)
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
        timescale_factor = 2*cb * time**cd
        eff_therm = np.exp(-ca*time) + np.log(1.0 + timescale_factor) / timescale_factor
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

        tdiff[:-1, j] = (0.08 * kappa[:-1, j]
            * m[:-1] * msun_cgs
            * 3 * kappa_correction
            / (vm[:-1] * c_cgs * t[j] * beta)
        )
        tau[:-1, j] = (
            m[:-1] * msun_cgs * kappa[:-1, j] / (4 * np.pi * (t[j] * vm[:-1]) ** 2)
        )
        lum[:-1, j] = ene[:-1, j] / (tdiff[:-1, j] + t[j] * (vm[:-1] / c_cgs))
        ene[:-1, j + 1] = ene[:-1, j] + dt[j]*(
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
    Tobs = autocomplete_data(sample_times, sample_times, Tobs)

    Tobs[Tobs <= 0.0] = np.nan
    one_over_T = 1.0 / Tobs
    one_over_T[~np.isfinite(one_over_T)] = np.inf
    return mag_dict_for_blackbody(filters, one_over_T, R_photo, nu_host)

def eff_metzger_lc(sample_times, param_dict, nu_host, filters):
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
        timescale_factor = 2*cb * time**cd
        eff_therm = np.exp(-ca*time) + np.log(1.0 + timescale_factor) / timescale_factor
        return 0.36 * eff_therm
    
    eth = thermalization_efficiency(sample_times, ca=0.56, cb=0.17, cd=0.74)

    # define radioactive heating rates
    Xn0 = Xn0max * 2 * np.arctan(Mn / m ) / np.pi  # neutron mass fraction
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
    ene = np.zeros(mprec-1)
    lum = np.zeros((mprec-1, tprec))
    # properties of photosphere
    R_photo = np.zeros((tprec,))


    dt = t[1:] - t[:-1]
    dm = m[1:] - m[:-1]

    for j in range(tprec - 1):
        tdiff = (0.08* kappa[:-1, j] * m[:-1] * msun_cgs * 3 
            / (vm[:-1] * c_cgs * t[j] * beta) )
        tau = (
            m[:-1] * msun_cgs * kappa[:-1, j] / (4 * np.pi * (t[j] * vm[:-1]) ** 2)
        )
        lum_j= ene / (tdiff+ t[j] * (vm[:-1] / c_cgs))
        lum[:, j] = lum_j * dm * msun_cgs

        ene += dt[j]*(edot[:-1, j] - (ene  / t[j]) - lum_j ) 
        # photosphere
        pig = np.argmin(np.abs(tau - 1))
        R_photo[j] = vm[pig] * t[j]

    Ltotm = np.sum(lum, axis=0) / 1e20 / 1e20
    Ltot = np.abs(Ltotm)
    # lbol = Ltotm * 1e40

    Tobs = 1e10 * (Ltot / (4 * np.pi * R_photo**2 * sigSB)) ** (0.25)
    Tobs = autocomplete_data(sample_times, sample_times, Tobs)

    Tobs[Tobs <= 0.0] = np.nan
    one_over_T = 1.0 / Tobs
    one_over_T[~np.isfinite(one_over_T)] = np.inf
    return mag_dict_for_blackbody(filters, one_over_T, R_photo, nu_host)

def HoNa_lc(sample_times, param_dict, nu_host, filters):
    t0 = 1e-3 
    sample_times = sample_times [sample_times > t0] 
    sample_times*= seconds_a_day # remove t=0, as it is not physical
    t0*= seconds_a_day
    mej = 10**param_dict["log10_Mej"] * msun_cgs
    vej_max = param_dict["vej_max"]   
    vej_min = param_dict["vej_min"]
    vej_range = vej_max - vej_min
    vej = param_dict["vej_frac"] * vej_range + vej_min  
    velocities = np.array([vej_min, vej, vej_max]) # in units of c
    
    # in cm**2 / g
    opacities = np.array([10**param_dict["log10_kappa_low_vej"],  
                          10**param_dict["log10_kappa_high_vej"]] )

    # calculate the temperature and luminosity to feed into the blackbody radiation calculation
    inv_temp, R_photo = temp_photosphere_HoNa(
        sample_times, mej, velocities, opacities, param_dict.get("n", 4.5) )
    # param_dict["z"] = 0.011188892
    # param_dict["Ebv"] = 0

    return mag_dict_for_blackbody(filters, inv_temp, R_photo, nu_host)

# the following functions are for the semi-analytic model using Hotokezaka & Nakar heating rate
def luminosity_HoNa(E, t, td, be):
    # Calculate diffusion time ratio
    t_dif = td / t
    # Determine escape time
    tesc = np.minimum(t, t_dif) + be * t
    # Calculate maximum y value
    ymax = np.sqrt(0.5 * t_dif / t)
    # Return luminosity using complementary error function
    return erfc(ymax) * E / tesc

def dEdt_HoNa(t, E, dM, td, be):
    # Calculate heating contribution
    heat = dM * heating_rate_Korobkin_Rosswog(t)
    # Calculate luminosity
    L = luminosity_HoNa(E, t, td, be)
    dEdt = -E / t - L + heat
    return dEdt

def temp_photosphere_HoNa(t, mej, velocities, opacities, n, t0):
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
    bej_power = (velocities / be_0)**(1 - n)
    bes_power = (bes / be_0)**(1 - n)
    
    # Vectorized calculation of tau_accum
    tau_accum = -np.cumsum((opacities * np.diff(bej_power))[::-1])[::-1]
    tau_accum = np.append(tau_accum, 0)    
    # Vectorized calculation of taus
    taus = tau_accum[i] + opacities[i - 1] * (bes_power - bej_power[i])

    vej_0 = velocities[0] * c_cgs    
    rho_0 = mej * (n - 3) / (4 * np.pi * vej_0**3) / (1 - (be_max/be_0)**(3 - n))
    taus *= vej_0 * rho_0 / (n - 1)
    
    # Mass and time delay calculations
    bes_power_2n = (bes / be_0)**(2 - n)  # Calculate power once
    dMs = 4. * np.pi * vej_0**3 * rho_0 * bes_power_2n * dbe / be_0
    tds = taus * bes
    
    # Prepare arrays for solve_ivp - use broadcasting directly
    bes_col = bes[:, np.newaxis]
    tds_col = tds[:, np.newaxis]
    dMs_col = dMs[:, np.newaxis]
    
    # Evolve in time
    out = solve_ivp(dEdt_HoNa, (t0, t.max()), np.zeros(len(bes)), first_step=t0,
        args=(dMs_col, tds_col, bes_col), vectorized=True)
    
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
    beta = param_dict["beta_freq"]  # frequency index
    alpha = param_dict["alpha_time"]  # time index
    F_ref = param_dict["F_ref"]  # in mJy for t=1day and nu=1Hz
    mag = {}
    for idx, filt in enumerate(filters):
        F_pl = F_ref * np.power(nu_obs[idx], -beta) * np.power(sample_times, -alpha)
        # remove the distance modulus for the synchrotron powerlaw
        # as the reference flux is defined at the observer
        mag[filt] = flux_to_ABmag(F_pl, unit='mJy') - param_dict["distance_modulus"]
    return mag

## generic blackbody
def inv_temp_and_photosphere_from_params(param_dict):
    # parameter conversion
    inv_temp = 1.0 / param_dict["temperature"]  # blackbody's temperature in K
    R_photo = np.sqrt(
        param_dict["bb_luminosity"] / 4 / np.pi / sigSB # blackboady's total luminosity in erg/s
    ) * inv_temp * inv_temp
    return inv_temp, R_photo

def blackbody_constant_temperature(_, param_dict, nu_host, filters):
    inv_temp, R_photo = inv_temp_and_photosphere_from_params(param_dict)
    return mag_dict_for_blackbody(filters, inv_temp, R_photo, nu_host)



def powerlaw_blackbody_constant_temperature_lc(_, param_dict, nu_host, filters):

    # calculate the powerlaw prefactor (with the reference filter 'g')
    nu_ref = nu_host[filters.index("g")]    # FIXME, seems like a legacy hack
    powerlaw_prefactor = np.power(nu_ref, param_dict["beta"]) * np.power(
        10, -0.4 * (param_dict["powerlaw_mag"] + 48.6)
    )
    def additive_per_freq(nu):
        return powerlaw_prefactor * np.power(nu, -param_dict["beta"])
    
    inv_temp, R_photo = inv_temp_and_photosphere_from_params
    return mag_dict_for_blackbody(filters, inv_temp, R_photo,
                    nu_host, add= additive_per_freq)


def fill_lightcurve_data(times, filt_data):
    lc = np.interp(times, filt_data['time'], filt_data['mag'], left=np.inf,right=np.inf)
    lcerr = np.interp(times, filt_data['time'], filt_data['mag_error'], left=np.inf,right=np.inf)
    return { "time": times, "mag": lc, "mag_error": lcerr }


#### lightcurve data generation
def create_light_curve_data(
    injection_parameters,
    args,
    light_curve_model,
    sample_times=None,
    keep_infinite_data=False,
    rng= None
):
    
    filters = set_filters(args)

    detection_limit = create_detection_limit(args, filters)
    trigger_time = injection_parameters.get("trigger_time", 0.)
    if rng is None:
        rng = np.random.default_rng(args.generation_seed)
    dmag = args.injection_error_budget

    ## use extra data
    ztf_sampling = getattr(args, "ztf_sampling", False)
    rubin_ToO = getattr(args, "rubin_ToO_type", False)
    if getattr(args, 'absolute', False):
        # create lightcurve_data
        if sample_times is None:
            sample_times = light_curve_model.model_times
        lc = light_curve_model.generate_lightcurve(sample_times, injection_parameters)
        # if "timeshift" in injection_parameters: ## included in gen_detector_lc
        #     trigger_time += injection_parameters["timeshift"]
    else:
        # basic idea: generate lc works on desired times in source_frame, 
        # observing times are redshifted and have extra timeshift (interpreted as missed detections)
        sample_times, lc = light_curve_model.gen_detector_lc(injection_parameters, sample_times)
    if not lc:
        raise ValueError("Injection parameters return empty light curve.")

    # curate data
    data = {}

    for filt, mag_per_filt in lc.items():
        det_lim = detection_limit.get(filt,np.inf)

        data[filt] = {'time': sample_times + trigger_time, 'mag': np.full_like(sample_times, det_lim), 'mag_error': np.full_like(sample_times, np.inf)}  # defaults

        det_mask = (mag_per_filt < det_lim) * (sample_times >= 0.)
        errors = rng.normal(scale=dmag, size=np.sum(det_mask))
        data[filt]['mag'][det_mask] = mag_per_filt[det_mask] + errors
        data[filt]['mag_error'][det_mask] = dmag

    ## edit data for ztf, rubin
    data_original = copy.deepcopy(data)
    passbands_to_keep = []

    if ztf_sampling:
        data, passbands_to_keep = adjust_data_for_ztf(data, args, filters, 
                    rng, sample_times, trigger_time, passbands_to_keep)

    if rubin_ToO:
        data, passbands_to_keep = adjust_data_for_rubin(
            data, rubin_ToO, trigger_time, data_original, passbands_to_keep)

    if ztf_sampling or rubin_ToO:
        passbands_to_lose = set(data.keys()) - set(passbands_to_keep)
        for filt in passbands_to_lose:
            del data[filt]



    if not keep_infinite_data:
        for filt, val_dict in data.items():
            # NOTE: old version treated this as an "or", but was likely a bug
            keep_idx = np.isfinite(val_dict['mag']) & np.isfinite(val_dict['mag_error'])
            data[filt] = {key: val[keep_idx] for key, val in val_dict.items()}

    return data

def adjust_data_for_ztf(data, args, filters, rng, sample_times, trigger_time,
                        passbands_to_keep):
    """
    Adjust the light curve data for ZTF observations.
    """

    with resources.open_binary(
        __package__ + ".data", "ZTF_revisit_kde_public.joblib"
    ) as f:
        ztfrevisit = load(f)
    with resources.open_binary(
        __package__ + ".data", "ZTF_sampling_public.pkl"
    ) as f:
        ztfsampling = load(f)
    with resources.open_binary(
        __package__ + ".data", "ZTF_revisit_kde_i.joblib"
    ) as f:
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
        sim = pd.concat([sim, sim_ToO]) # join the two dataframes

    sim.sort_values(by=["mjd"], inplace=True)
    sim.reset_index(drop=True, inplace=True)
    sim["mag"] = np.nan         # initialize empty mags
    sim["mag_error"] = np.nan


    # interpolate the light curve data on ztf observations
    for filt, group in sim.groupby("passband"):
        if filt not in filters: # skip if we are not observing this filter
            continue
        else:
            # if we are observing this filter, we need to keep it
            passbands_to_keep.append(filt)
        
        filt_data = fill_lightcurve_data( # do interpolation
            group["mjd"].tolist(),            # on additional ztf times
            copy.deepcopy(data[filt])) # for the original data

        if ztf_uncertainties:
            sim.loc[group.index, "mag"] , sim.loc[group.index, "mag_error"]  = filt_data["mag"], filt_data["mag_error"]
            mag_err = []
            for idx, row in group.iterrows():
                if filt == "ztfg":
                    lim = float(ztftoolimg.sample()) if row["ToO"] else float(ztflimg.sample())
                elif filt == "ztfr":
                    lim = float(ztftoolimr.sample()) if row["ToO"] else float(ztflimr.sample())
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
                    df = estimate_mag_err(ztfuncer, df)
                    sim.loc[row.name, "mag_error"] = float(df["mag_error"])
                    mag_err.append(df["mag_error"].tolist()[0])

            filt_data = {'time':sim.loc[group.index, "mjd"].tolist(),
                         'mag': sim.loc[group.index, "mag"].tolist(),
                         'mag_error': mag_err }
            
        data[filt] = filt_data

    
    if getattr(args, "train_stats", False):
        sim["tc"] = trigger_time
        sim.to_csv(args.outdir + "/too.csv", index=False)

    return data, passbands_to_keep

def adjust_data_for_rubin(
    data, rubin_ToO, trigger_time, data_original, passbands_to_keep,
):
    """
    Adjust the light curve data for Rubin observations.
    """
    gold_times = [1 / 24.0, 2 / 24.0, 4 / 24.0, 1.0, 2.0, 3.0]
    if rubin_ToO == "platinum":
        # platinum is no official name, means 90% GW skymap <30 sq deg
        # this is the gold strategy for an event similar to GW170817 (close and well localized)
        # Three observation on first night with grizy filters
        # One scan Night 1,2,3 w/ same filters
        filts = ["ps1::g", "ps1::r", "ps1::i", "ps1::z", "ps1::y"]
        strategy= ((time, filts) for time in gold_times)

    elif "gold" in rubin_ToO:
        # gold means 90% GW skymap <100 sq deg
        # use gri or possibly grz if more sensitive to KNe
        init_filts = ["ps1::g", "ps1::r"]
        init_filts.append("ps1::z" if "gold_z" == rubin_ToO else "ps1::i")
        # Three pointings Night 0 
        filts = [init_filts]*3
        # One scan Night 1,2,3 w/ r+i
        follow_up_filts = ["ps1::r", "ps1::i"] 
        filts.extend([follow_up_filts]*3)
        strategy = ((time, filt_list) for time, filt_list in zip(gold_times, filts))

    elif "silver" in rubin_ToO:
        # silver means 90% GW skymap <500 sq deg
        # One scan Night 0 w/ g+i or g+z
        filts = ["ps1::g", "ps1::z"] if rubin_ToO == "silver_z" else ["ps1::g", "ps1::i"]
        # One scan each Night 1,2,3 w/ same filters
        silver_times = [1 / 24.0, 1.0, 2.0, 3.0]
        strategy = ((time, filts) for time in silver_times)

    else:
        raise ValueError("args.rubin_ToO_type should be either platinum, gold, or silver")
    # took type names from Rubin 2024 Workshop write-up

    mjds, passbands = [], []
    for (obstime, filts) in strategy:
        for filt in filts:
            mjds.append(trigger_time + obstime)
            passbands.append(filt)
    sim = pd.DataFrame.from_dict({"mjd": mjds, "passband": passbands})

    for filt, group in sim.groupby("passband"):
        data_per_filt = copy.deepcopy(data_original[filt])
        times = group["mjd"].tolist()
        data[filt] = fill_lightcurve_data(times, data_per_filt)
        passbands_to_keep.append(filt)
    return data, passbands_to_keep