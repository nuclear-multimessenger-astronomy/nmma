import copy
from joblib import load
from ast import literal_eval
import numpy as np
import pandas as pd
from importlib import resources

from scipy.integrate import quad
from  astropy import constants as const
import sncosmo

from .model import SVDLightCurveModel, KilonovaGRBLightCurveModel
from .utils import estimate_mag_err, autocomplete_data, flux_to_ABmag, setup_sample_times, transform_to_app_mag_dict
from nmma.em.training import SVDTrainingModel

try:
    import afterglowpy

    AFTERGLOWPY_INSTALLED = True
except ImportError:
    print("Install afterglowpy if you want to simulate afterglows.")
    AFTERGLOWPY_INSTALLED = False

try:
    from wrapt_timeout_decorator import timeout
except ImportError:
    print("Install wrapt_timeout_decorator if you want timeout simulations.")

    def timeout(*args, **kwargs):
        def inner(func):
            return func

        return inner


### some frequently used constants:

# convert time from day to second
seconds_a_day = 86400.0  # in seconds
msun_cgs = const.M_sun.cgs.value
c_cgs = const.c.cgs.value
h = const.h.cgs.value
kb = const.k_B.cgs.value
sigSB = const.sigma_sb.cgs.value
arad = 4 * sigSB / c_cgs
Mpc = const.pc.cgs.value * 1e6
D = 10 * const.pc.cgs.value  # ref distance for absolute magnitude
abs_mag_dist_factor = D**2
dummy_lum = 1 # 1e43   ### dummy value returned for conformity that should never be used further

def bb_flux_from_inv_temp(nu, inv_temp, R_photo, dist_squared = abs_mag_dist_factor):
    exp = np.exp(h * nu * inv_temp / kb)
    bb_factor = 2.* h/ c_cgs**2
    return bb_factor * nu**3 /(exp-1) * R_photo * R_photo / dist_squared

#################################################################
######################### LC MODELS #############################
#################################################################

## Arnett model convenience functions
def arnett_lc_get_int_A_non_vec(x, y):
    r = quad(lambda z: 2 * z * np.exp(-2 * z * y + z**2), 0, x)
    int_A = r[0]
    return int_A


arnett_lc_get_int_A = np.vectorize(arnett_lc_get_int_A_non_vec, excluded=["y"])


def arnett_lc_get_int_B_non_vec(x, y, s):
    r = quad(lambda z: 2 * z * np.exp(-2 * z * y + 2 * z * s + z**2), 0, x)
    int_B = r[0]
    return int_B


arnett_lc_get_int_B =  np.vectorize(arnett_lc_get_int_B_non_vec, excluded=["y", "s"])


def arnett_lc(t_day, param_dict):
    """bolometric light curve functions from Arnett model
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
    epsilon_ni = 3.9e10  # erg / s / g
    epsilon_co = 6.78e9  # erg / s / g
    tau_ni = 8.8 * seconds_a_day  # s
    tau_co = 111.3 * seconds_a_day  # s

    ts = t_day * seconds_a_day
    tau_m = param_dict["tau_m"] * seconds_a_day
    Mni = 10**param_dict["log10_mni"] * msun_cgs

    x = ts / tau_m
    y = tau_m / (2 * tau_ni)
    s = tau_m * (tau_co - tau_ni) / (2 * tau_co * tau_ni)

    int_A = arnett_lc_get_int_A(x, y)
    int_B = arnett_lc_get_int_B(x, y, s)

    Lbol = Mni * np.exp(-x**2) * (
        (epsilon_ni - epsilon_co) * int_A + epsilon_co * int_B)

    return Lbol


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
def calc_lc(
    tt: np.array,
    param_list: np.array,
    svd_mag_model: SVDTrainingModel = None,
    svd_lbol_model: SVDTrainingModel = None,
    mag_ncoeff: int = None,
    lbol_ncoeff: int = None,
    interpolation_type: str = "sklearn_gp",
    filters: list = None,
) -> "tuple[np.array, np.array]":
    """
    Computes the lightcurve from a surrogate model, given the model parameters.
    Args:
        tt (Array): Time grid on which to evaluate lightcurve
        param_list (Array): Input parameters for the surrogate model
        svd_mag_model (SVDTrainingModel): Trained surrogate model for mag
        svd_lbol_model (SVDTrainingModel): Trained surrogate model for lbol
        mag_ncoeff (int): Number of coefficients after SVD projection for mag
        lbol_ncoeff (int): Number of coefficients after SVD projection for lbol
        interpolation_type (str): String denoting which interpolation type is used for the surrogate model
        filters (Array): List/array of filters at which we want to evaluate the model
    """

    mAB = {}

    if filters is None:
        filters = list(svd_mag_model.keys())
    else:
        # add null output for radio and X-ray filters
        for filt in filters:
            if filt.startswith(("radio", "X-ray")):
                mAB[filt] = np.inf * np.ones(len(tt))

    for filt in filters:
        if filt in mAB:
            continue
        tt_interp, mag_back = eval_svd_model(svd_mag_model[filt], mag_ncoeff, param_list, interpolation_type)

        mag_back[tt_interp>=20]= np.inf ### FIXME quick-fix to not trust lightcurve after 20 days 
        mAB[filt] = autocomplete_data(tt, tt_interp, mag_back )

    if svd_lbol_model is not None:
        tt_interp, lbol_back =eval_svd_model(svd_lbol_model, lbol_ncoeff, param_list, interpolation_type)
        lbol = 10**autocomplete_data(tt, tt_interp, lbol_back)
    else:
        lbol = np.full_like(tt, np.nan)

    return np.squeeze(lbol), mAB

def eval_svd_model(svd_model, ass_ncoeff, param_list, interpolation_type):
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

    if interpolation_type == "tensorflow":
        model = svd_model["model"]
        cAproj = model.predict(np.atleast_2d(param_list_postprocess)).T.flatten()
    else:
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
            cAproj[i] = y_pred

    
    svd_back = np.dot(VA[:, :n_coeff], cAproj)
    svd_back*= (maxs - mins) + mins
    return tt_interp, svd_back


## grb afterglow
@timeout(60)
def fluxDensity(t, nu, **params):
    if AFTERGLOWPY_INSTALLED:
        mJy = afterglowpy.fluxDensity(t, nu, **params)
    else:
        raise ValueError("afterglowpy required for GRB afterglow")
    return mJy


def grb_lc(t_day, param_dict, filters, obs_frequencies):
    tStart = max(10 ** (-5), np.amin(t_day)) * seconds_a_day
    tEnd = (np.amax(t_day) + 1) * seconds_a_day
    tnode = min(len(t_day), 201)
    default_time = np.logspace(np.log10(tStart), np.log10(tEnd), base=10.0, num=tnode)

    times = np.empty((len(default_time), len(filters)))
    nus = np.empty((len(default_time), len(filters)))

    times[:, :] = default_time[:, None]
    for nu_idx, nu_0 in enumerate(obs_frequencies):
        nus[:, nu_idx] = nu_0

    mag = {}
    lbol = np.full_like(t_day, dummy_lum)
    # output flux density is in milliJansky
    try:
        mJys = fluxDensity(times, nus, **param_dict)
    except TimeoutError:
        return t_day, lbol, mag

    if np.any(mJys <= 0.0):
        return t_day, lbol, mag

    for filt_idx, filt in enumerate(filters):
        mag_d = flux_to_ABmag(mJys[:, filt_idx], unit= 'mJy')
        mag[filt] = autocomplete_data(t_day, default_time / seconds_a_day, mag_d)

    return lbol, mag

## hostmodel lightcurve
def host_lc(sample_times, parameters, filters, host_mag):
    # Based on arxiv:2303.12849
    mag = {}
    lbol = np.full_like(sample_times, dummy_lum)  # random

    alpha = parameters["alpha_AG"]
    for i, filt in enumerate(filters):
        # assumed to be in unit of muJy
        a_AG = parameters[f"a_AG_{filt}"]
        f_nu_filt = parameters[f"f_nu_{filt}"]
        flux_per_filt = a_AG * np.power(sample_times, -alpha) + f_nu_filt
        mag[filt] = flux_to_ABmag(flux_per_filt, residual_mag=host_mag[i])
    return lbol, mag

## supernova model
def sn_lc(
    tt,
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
            t0=np.median(tt),
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
    lbol = np.full_like(tt, dummy_lum)

    for filt, lambda_A in zip(filters, lambdas):
        # convert back to AA
        lambda_AA = 1e10 * lambda_A
        if lambda_AA < model.minwave() or lambda_AA > model.maxwave():
            mag[filt] = np.inf * np.ones(tt.shape)
        else:
            try:
                flux = model.flux(tt, [lambda_AA])[:, 0]
                # see https://en.wikipedia.org/wiki/AB_magnitude
                flux_jy = 3.34e4 * np.power(lambda_AA, 2.0) * flux
                mag[filt] = flux_to_ABmag(flux_jy, unit='Jy')
            except Exception:
                mag[filt] = np.ones(tt.shape) * np.nan
                lbol = np.zeros(tt.shape)

    return lbol, mag

## shock-cooling lightcurve
def sc_lc(t_day, param_dict,nus, filters):

    t = t_day * seconds_a_day

    # fetch parameter values
    Me = 10 ** param_dict["log10_Menv"] * msun_cgs
    Renv = 10 ** param_dict["log10_Renv"]
    Ee = 10 ** param_dict["log10_Ee"]
    z = param_dict["redshift"]

    nu_host = nus * (1 + z)
    t /= 1 + z


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
    lbol = np.zeros(len(t))
    # stiching the two regime
    lbol[t < td] = L_early[t < td]
    lbol[t >= td] = L_late[t >= td]

    # evalute the mAB per filter
    tph = np.sqrt(3 * kappa * K * Me / (2 * (n - 1) * vt * vt))
    R_early = np.power(tph / t, 2 / (n - 1)) * vt * t
    R_late = (
        np.power((delta - 1) / (n - 1) * ((t / td) ** 2 - 1) + 1, -1 / (delta + 1))
        * vt
        * t
    )
    Rs = np.zeros(len(t))
    Rs[t < td] = R_early[t < td]
    Rs[t >= td] = R_late[t >= td]

    sigmaT4 = lbol / (4 * np.pi * Rs * Rs)
    T = np.power(sigmaT4 / sigSB, 0.25)
    T[T == 0.0] = np.nan
    one_over_T = 1.0 / T
    one_over_T[~np.isfinite(one_over_T)] = np.inf

    mag = {}
    for idx, filt in enumerate(filters):
        nu_of_filt = nu_host[idx]
        F = bb_flux_from_inv_temp(nu_of_filt, one_over_T, Rs)
        F *= 1 + z
        mag[filt] = flux_to_ABmag(F)

    return lbol, mag

# def inv_temp_from_rad_bb_and_lum(lbol, r_bb, time):
#     inv_t_obs = (4 *sigSB* np.pi * r_bb* r_bb/lbol)**0.25
#     inv_t_obs = autocomplete_data(time, time, inv_t_obs)

## metzger model for kilonovae
def metzger_lc(t_day, param_dict, nu_obs, filters):

    # convert time from day to second
    t = t_day * seconds_a_day
    nu_host = nu_obs * (1 + z)
    t /= 1 + z
    tprec = len(t)

    if len(np.where(t == 0)[0]) > 0:
        raise ValueError("For Me2017, start later than t=0")
    # fetch parameters
    M0 = 10 ** param_dict["log10_mej"] * msun_cgs  # total ejecta mass
    v0 = 10 ** param_dict["log10_vej"] * c_cgs  # minimum escape velocity
    beta = param_dict["beta"]
    kappa_r = 10 ** param_dict["log10_kappa_r"]
    z = param_dict["redshift"]

    # define additional parameters
    E0 = 0.5 * M0 * v0 * v0  # initial thermal energy of bulk
    Mn = 1e-8 * msun_cgs  # mass cut for free neutrons
    Ye = 0.1  # electron fraction
    Xn0max = 1 - 2 * Ye  # initial neutron mass fraction in outermost layers

    # define mass / velocity array of the outer ejecta, comprised half of the mass
    mmin = np.log(1e-8)
    mmax = np.log(M0 / msun_cgs)
    mprec = 300
    m = np.arange(mprec) * (mmax - mmin) / (mprec - 1) + mmin
    m = np.exp(m)

    vm = v0 * np.power(m * msun_cgs / M0, -1.0 / beta)
    vm[vm > c_cgs] = c_cgs

    # define thermalization efficiency from Barnes+16, eq. 34
    def thermalization_efficiency(time, ca, cb, cd):
        timescale_factor = 2*cb * time**cd
        eff_therm = np.exp(-ca*time) + np.log(1.0 + timescale_factor) / timescale_factor
        return 0.36 * eff_therm
    
    eth = thermalization_efficiency(t_day, ca=0.56, cb=0.17, cd=0.74)
    # eth2= thermalization_efficiency(t_day, ca= 8.2, cb= 1.2, cd=1.52)
    # eth3= thermalization_efficiency(t_day, ca= 1.3, cb= 0.2, cd= 1.1)

    # define radioactive heating rates
    Xn0 = Xn0max * 2 * np.arctan(Mn / m / msun_cgs) / np.pi  # neutron mass fraction
    Xr = 1.0 - Xn0  # r-process fraction

    # define arrays in mass layer and time
    Xn = np.zeros((mprec, tprec))
    edotn = np.zeros((mprec, tprec))
    edotr = np.zeros((mprec, tprec))
    edot = np.zeros((mprec, tprec))
    kappa = np.zeros((mprec, tprec))
    kappan = np.zeros((mprec, tprec))
    kappar = np.zeros((mprec, tprec))

    # define specific heating rates and opacity of each mass layer
    t0 = 1.3
    sig = 0.11

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
    Lr = M0 * 4e18 * (0.5 - (1.0 / np.pi) * np.arctan((t - t0) / sig)) ** (1.3) * eth
    Lr = Lr / 1e20
    Lr = Lr / 1e20

    # *** define arrays by mass layer/time arrays ***
    ene = np.zeros((mprec, tprec))
    lum = np.zeros((mprec, tprec))
    tdiff = np.zeros((mprec, tprec))
    tau = np.zeros((mprec, tprec))
    # properties of photosphere
    Rphoto = np.zeros((tprec,))
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
        Rphoto[j] = vphoto[j] * t[j]
        mphoto[j] = m[pig]
        kappaphoto[j] = kappa[pig, j]

    Ltotm = np.sum(lum, axis=0)
    Ltotm = Ltotm / 1e20
    Ltotm = Ltotm / 1e20

    Ltot = np.abs(Ltotm)
    lbol = Ltotm * 1e40

    Tobs = 1e10 * (Ltot / (4 * np.pi * Rphoto**2 * sigSB)) ** (0.25)
    Tobs = autocomplete_data(t_day, t_day, Tobs)

    Tobs[Tobs <= 0.0] = np.nan
    one_over_T = 1.0 / Tobs
    one_over_T[~np.isfinite(one_over_T)] = np.inf

    mag = {}
    for idx, filt in enumerate(filters):
        nu_of_filt = nu_host[idx]
        F = bb_flux_from_inv_temp(nu_of_filt, one_over_T, Rphoto)
        F *= 1 + z
        mag[filt] = flux_to_ABmag(F)

    return lbol, mag

## generic blackbody
def blackbody_constant_temperature(t_day, param_dict, nu_obs, filters=None):

    # fetch parameters
    bb_luminosity = param_dict[
        "bb_luminosity"
    ]  # blackboady's total luminosity in erg/s
    temperature = param_dict["temperature"]  # blackbody's temperature in K
    z = param_dict["redshift"]

    # parameter conversion
    one_over_T = 1.0 / temperature
    bb_radius = np.sqrt(bb_luminosity / 4 / np.pi / sigSB) * one_over_T * one_over_T

    # convert time from day to second
    t = t_day * seconds_a_day
    nu_host = nu_obs * (1 + z)
    t /= 1 + z

    mag = {}
    for idx, filt in enumerate(filters):
        nu_of_filt = nu_host[idx]
        F = bb_flux_from_inv_temp(nu_of_filt, one_over_T, bb_radius)
        F *= 1 + z
        mag[filt] = flux_to_ABmag(F)

    lbol = np.full_like(t_day, dummy_lum)  # some dummy value


    return lbol, mag


def synchrotron_powerlaw(t_day, param_dict, nu_obs, filters):
    beta = param_dict["beta_freq"]  # frequency index
    alpha = param_dict["alpha_time"]  # time index
    F_ref = param_dict["F_ref"]  # in mJy for t=1day and nu=1Hz

    mag = {}
    for idx, filt in enumerate(filters):
        F_pl = F_ref * np.power(nu_obs[idx], -beta) * np.power(t_day, -alpha)
        mag[filt] = flux_to_ABmag(F_pl, unit='mJy')

    lbol = np.full_like(t_day, dummy_lum)  # some dummy value
    return lbol, mag


def powerlaw_blackbody_constant_temperature_lc(t_day, param_dict, nu_obs, filters):
    # fetch parameters
    bb_luminosity = param_dict["bb_luminosity"]  # blackboady's total luminosity
    temperature = param_dict["temperature"]  # for the blackbody radiation
    beta = param_dict["beta"]  # for the power-law
    powerlaw_mag = param_dict["powerlaw_mag"]
    powerlaw_filt_ref = "g"
    z = param_dict["redshift"]

    # parameter conversion
    one_over_T = 1.0 / temperature
    bb_radius = np.sqrt(bb_luminosity / 4 / np.pi / sigSB) * one_over_T * one_over_T

    # calculate the powerlaw prefactor (with the reference filter)
    nu_ref = nu_obs[filters.index(powerlaw_filt_ref)]
    powerlaw_prefactor = np.power(nu_ref, beta) * np.power(
        10, -0.4 * (powerlaw_mag + 48.6)
    )

    # convert time from day to second
    t = t_day * seconds_a_day
    nu_host = nu_obs * (1 + z)
    t /= 1 + z

    mag = {}
    for idx, filt in enumerate(filters):
        nu_of_filt = nu_host[idx]
        F = bb_flux_from_inv_temp(nu_of_filt, one_over_T, bb_radius)
        F+= powerlaw_prefactor * np.power(nu_of_filt, -beta)
        F*= 1 + z
        mag[filt] = flux_to_ABmag(F)

    lbol = np.full_like(t_day, dummy_lum)  # some dummy value
    return lbol, mag


def fill_lightcurve_data(times, data_per_filt):
    time_data, lc_data, err_data = data_per_filt.T
    lc = np.interp(times, time_data, lc_data, left=np.inf,right=np.inf)
    lcerr = np.interp(times, time_data, err_data, left=np.inf,right=np.inf)
    return np.vstack([times, lc, lcerr]).T


#### lightcurve data generation
def create_light_curve_data(
    injection_parameters,
    args,
    light_curve_model=None,
    sample_times=None,
    doAbsolute=False,
    keep_infinite_data=False,
):

    kilonova_kwargs = dict(
        model=getattr(args, 'em_injection_model', args.model),
        svd_path=getattr(args, 'injection_svd_path', args.svd_path),
        mag_ncoeff = getattr(args, 'injection_svd_mag_ncoeff', args.svd_mag_ncoeff),   
        lbol_ncoeff = getattr(args, 'injection_svd_lbol_ncoeff', args.svd_lbol_ncoeff),
    )

    if args.filters:
        filters = args.filters.split(",")
        bands = {i + 1: b for i, b in enumerate(filters)}
        inv_bands = {v: k for k, v in bands.items()}
        detection_limit = create_detection_limit(args, filters)
    else:
        filters = None
        bands = {}
        inv_bands = {}
        detection_limit = {}


    ## load extra data
    rubin_ToO = getattr(args, "rubin_ToO", False)
    photometry_augmentation = getattr(args, "photometry_augmentation", False)
    ztf_sampling = getattr(args, "ztf_sampling", False)
    if ztf_sampling :
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

    tc = injection_parameters["kilonova_trigger_time"]

    if "timeshift" in injection_parameters:
        tc = tc + injection_parameters["timeshift"]


    seed = args.generation_seed

    np.random.seed(seed)

    #create light_curve_model
    if light_curve_model is None:
        if args.with_grb_injection:
            light_curve_model = KilonovaGRBLightCurveModel(
                kilonova_kwargs=kilonova_kwargs,
                GRB_resolution=np.inf,
            )

        else:
            light_curve_model = SVDLightCurveModel(
                interpolation_type=args.interpolation_type, 
                **kilonova_kwargs
            )

    # create lightcurve_data
    if sample_times is None:
        sample_times = setup_sample_times(args)

    _, mag = light_curve_model.generate_lightcurve(
        sample_times, injection_parameters
    )
    if not doAbsolute:
        mag = transform_to_app_mag_dict(mag, injection_parameters)
    dmag = args.em_transient_error
    if not mag:
        raise ValueError("Injection parameters return empty light curve.")


    # curate data
    data = {}
    if ztf_sampling or rubin_ToO or photometry_augmentation:
        passbands_to_keep = []

    for filt, mag_per_filt in mag.items():

        # identify detection_limit
        if filt in detection_limit:
            det_lim = detection_limit[filt]
        elif (
            photometry_augmentation
            and filt in args.photometry_augmentation_filters.split(",")
        ):
            det_lim = 30.0
        else:
            det_lim = np.inf

        ## set data
        data_per_filt = np.zeros([len(sample_times), 3])
        for i, sample_time in enumerate(sample_times):
            if mag_per_filt[i] < det_lim: ## a detection can be made
                noise_scale = noise_level = dmag  
                if ztf_uncertainties and filt in ["g", "r", "i", "ztfg", "ztfr", "ztfi"]:
                    try:
                        df = pd.DataFrame.from_dict(
                            {"passband": [inv_bands[filt]],
                                "mag": [mag_per_filt[i]]}
                            )
                        df = estimate_mag_err(ztfuncer, df)
                        noise_level = df["mag_err"].values[0]
                        ##FIXME : Should this not rather be np.sqrt(dmag**2 + df["mag_err"].values[0] ** 2), ie. should not noise_level == noise_scale?
                        noise_scale =np.sqrt(dmag**2 + noise_level ** 2)
                    except:
                        pass
                noise = np.random.normal(scale=noise_scale)
                detection_mag =  mag_per_filt[i] + noise
            else:
                detection_mag = det_lim     #non-detection
                noise_level = np.inf        #default
            data_per_filt[i] = [sample_time+ tc , detection_mag, noise_level]
        data[filt] = data_per_filt

    ## edit data for ztf, rubin, augmentation
    data_original = copy.deepcopy(data)
    if ztf_sampling:
        sim = pd.DataFrame()
        start = np.random.uniform(tc, tc + 2)
        t = start
        # ZTF-II Public
        while t < sample_times[-1] + tc:
            sample = ztfsampling.sample()
            sim = pd.concat(
                [
                    sim,
                    pd.DataFrame(
                        np.array(
                            [t + sample["t"].values[0], sample["bands"].values[0]]
                        ).T
                    ),
                ]
            )
            t += float(ztfrevisit.sample())
        # i-band
        start = np.random.uniform(tc, tc + 4)
        t = start
        while t < sample_times[-1] + tc:
            sim = pd.concat([sim, pd.DataFrame([[t, 3]])])
            t += float(ztfrevisit_i.sample())
        sim["ToO"] = False
        # toO
        if ztf_ToO:
            sim_ToO = pd.DataFrame()
            start = np.random.uniform(tc, tc + 1)
            t = start
            too_samps = ztftoo.sample(np.random.choice([1, 2]))
            for i, too in too_samps.iterrows():
                sim_ToO = pd.concat(
                    [sim_ToO, pd.DataFrame(np.array([t + too["t"], too["bands"]]).T)]
                )
                t += 1
            sim_ToO["ToO"] = True
            sim = pd.concat([sim, sim_ToO])

        sim = (
            sim.rename(columns={0: "mjd", 1: "passband"})
            .sort_values(by=["mjd"])
            .reset_index(drop=True)
        )
        sim["passband"] = sim["passband"].map({1: "ztfg", 2: "ztfr", 3: "ztfi"})
        sim["mag"] = np.nan
        sim["mag_err"] = np.nan

        for filt, group in sim.groupby("passband"):
            if filt not in filters:
                continue
            data_per_filt = copy.deepcopy(data_original[filt])
            times = group["mjd"].tolist()
            data_per_filt = fill_lightcurve_data(times, data_per_filt)

            if ztf_uncertainties and filt in ["ztfg", "ztfr", "ztfi"]:
                _, sim.loc[group.index, "mag"] , sim.loc[group.index, "mag_err"]  = data_per_filt.T
                mag_err = []
                for idx, row in group.iterrows():
                    if filt == "ztfg" and row["ToO"] is False:
                        limg = float(ztflimg.sample())
                        if row["mag"] > limg:
                            sim.loc[row.name, "mag"] = limg
                            sim.loc[row.name, "mag_err"] = np.inf
                    elif filt == "ztfg" and row["ToO"] is True:
                        toolimg = float(ztftoolimg.sample())
                        if row["mag"] > toolimg:
                            sim.loc[row.name, "mag"] = toolimg
                            sim.loc[row.name, "mag_err"] = np.inf
                    elif filt == "ztfr" and row["ToO"] is False:
                        limr = float(ztflimr.sample())
                        if row["mag"] > limr:
                            sim.loc[row.name, "mag"] = limr
                            sim.loc[row.name, "mag_err"] = np.inf
                    elif filt == "ztfr" and row["ToO"] is True:
                        toolimr = float(ztftoolimr.sample())
                        if row["mag"] > toolimr:
                            sim.loc[row.name, "mag"] = toolimr
                            sim.loc[row.name, "mag_err"] = np.inf
                    else:
                        limi = float(ztflimi.sample())
                        if row["mag"] > limi:
                            sim.loc[row.name, "mag"] = limi
                            sim.loc[row.name, "mag_err"] = np.inf

                    if not np.isfinite(sim.loc[row.name, "mag_err"]):
                        mag_err.append(np.inf)
                    else:
                        df = pd.DataFrame.from_dict(
                            {"passband": [filt], "mag": [sim.loc[row.name, "mag"]]}
                        )
                        df["passband"] = df["passband"].map(
                            {"ztfg": 1, "ztfr": 2, "ztfi": 3}
                        )  # estimate_mag_err maps filter numbers
                        df = estimate_mag_err(ztfuncer, df)
                        sim.loc[row.name, "mag_err"] = float(df["mag_err"])
                        mag_err.append(df["mag_err"].tolist()[0])

                data_per_filt = np.vstack(
                    [
                        sim.loc[group.index, "mjd"].tolist(),
                        sim.loc[group.index, "mag"].tolist(),
                        mag_err,
                    ]
                ).T
            data[filt] = data_per_filt
            passbands_to_keep.append(filt)

        if getattr(args, "train_stats", False):
            sim["tc"] = tc
            sim.to_csv(args.outdir + "/too.csv", index=False)

    if rubin_ToO:
        start = sample_times[0] + tc
        if args.rubin_ToO_type == "platinum":
            # platinum means 90% GW skymap <30 sq deg
            # I made this name up, this is the gold strategy for an event similar to GW170817 (close and well localized)
            # Three observations Night0 with grizy filters
            # One scan Night 1,2,3 w/ same filters
            strategy = [
                [1 / 24.0, ["ps1__g", "ps1__r", "ps1__i", "ps1__z", "ps1__y"]],
                [2 / 24.0, ["ps1__g", "ps1__r", "ps1__i", "ps1__z", "ps1__y"]],
                [4 / 24.0, ["ps1__g", "ps1__r", "ps1__i", "ps1__z", "ps1__y"]],
                [1.0, ["ps1__g", "ps1__r", "ps1__i", "ps1__z", "ps1__y"]],
                [2.0, ["ps1__g", "ps1__r", "ps1__i", "ps1__z", "ps1__y"]],
                [3.0, ["ps1__g", "ps1__r", "ps1__i", "ps1__z", "ps1__y"]],
            ]
        elif args.rubin_ToO_type == "gold":
            # gold means 90% GW skymap <100 sq deg
            # Three pointings Night 0 with gri (possibly grz if more sensitive to KNe)
            # One scan Night 1,2,3 w/ r+i
            strategy = [
                [1 / 24.0, ["ps1__g", "ps1__r", "ps1__i"]],
                [2 / 24.0, ["ps1__g", "ps1__r", "ps1__i"]],
                [4 / 24.0, ["ps1__g", "ps1__r", "ps1__i"]],
                [1.0, ["ps1__r", "ps1__i"]],
                [2.0, ["ps1__r", "ps1__i"]],
                [3.0, ["ps1__r", "ps1__i"]],
            ]
        elif args.rubin_ToO_type == "gold_z":
            # gold means 90% GW skymap <100 sq deg
            # Three pointings Night 0 with gri (possibly grz if more sensitive to KNe)
            # One scan Night 1,2,3 w/ r+i
            strategy = [
                [1 / 24.0, ["ps1__g", "ps1__r", "ps1__z"]],
                [2 / 24.0, ["ps1__g", "ps1__r", "ps1__z"]],
                [4 / 24.0, ["ps1__g", "ps1__r", "ps1__z"]],
                [1.0, ["ps1__r", "ps1__i"]],
                [2.0, ["ps1__r", "ps1__i"]],
                [3.0, ["ps1__r", "ps1__i"]],
            ]
        elif args.rubin_ToO_type == "silver":
            # silver means 90% GW skymap <500 sq deg
            # One scan Night 0 w/ g+i or g+z
            # One scan each Night 1,2,3 w/ same filters
            strategy = [
                [1 / 24.0, ["ps1__g", "ps1__i"]],
                [1.0, ["ps1__g", "ps1__i"]],
                [2.0, ["ps1__g", "ps1__i"]],
                [3.0, ["ps1__g", "ps1__i"]],
            ]
        elif args.rubin_ToO_type == "silver_z":
            # silver means 90% GW skymap <500 sq deg
            # One scan Night 0 w/ g+i or g+z
            # One scan each Night 1,2,3 w/ same filters
            strategy = [
                [1 / 24.0, ["ps1__g", "ps1__z"]],
                [1.0, ["ps1__g", "ps1__z"]],
                [2.0, ["ps1__g", "ps1__z"]],
                [3.0, ["ps1__g", "ps1__z"]],
            ]
        else:
            raise ValueError("args.rubin_ToO_type should be either platinum, gold, or silver")
        # took type names from Rubin 2024 Workshop write-up

        mjds, passbands = [], []
        sim = pd.DataFrame()
        for (obstime, filts) in strategy:
            for filt in filts:
                mjds.append(tc + obstime)
                passbands.append(filt)
        sim = pd.DataFrame.from_dict({"mjd": mjds, "passband": passbands})

        for filt, group in sim.groupby("passband"):
            data_per_filt = copy.deepcopy(data_original[filt])
            times = group["mjd"].tolist()
            data_per_filt = fill_lightcurve_data(times, data_per_filt)
            data[filt] = data_per_filt
            passbands_to_keep.append(filt)

    if photometry_augmentation:
        np.random.seed(args.photometry_augmentation_seed)
        if args.photometry_augmentation_filters is None:
            filts = np.random.choice(
                list(data.keys()),
                size=args.photometry_augmentation_N_points,
                replace=True,
            )
        else:
            filts = args.photometry_augmentation_filters.split(",")

        if args.photometry_augmentation_times is None:
            tt = np.random.uniform(
                sample_times[0] + tc, sample_times[-1] + tc, size=args.photometry_augmentation_N_points
            )
        else:
            tt = tc + np.array(
                [float(x) for x in args.photometry_augmentation_times.split(",")]
            )

        for filt in list(set(filts)):
            data_per_filt = copy.deepcopy(data_original[filt])
            idx = np.where(filt == filts)[0]
            times = tt[idx]

            if len(times) == 0:
                continue
            data_per_filt = fill_lightcurve_data(times, data_per_filt)
            if filt not in data:
                data[filt] = data_per_filt
            else:
                data[filt] = np.vstack([data[filt], data_per_filt])
                data[filt] = data[filt][data[filt][:, 0].argsort()]
            passbands_to_keep.append(filt)

    if ztf_sampling or rubin_ToO or photometry_augmentation:
        passbands_to_lose = set(list(data.keys())) - set(passbands_to_keep)
        for filt in passbands_to_lose:
            del data[filt]

    if not keep_infinite_data:
        filters_to_check = list(data.keys())
        for filt in filters_to_check:
            idx = np.union1d(
                np.where(np.isfinite(data[filt][:, 1]))[0],
                np.where(np.isfinite(data[filt][:, 2]))[0],
            )
            data[filt] = data[filt][idx, :]

    return data



def create_detection_limit(args, filters):
    if getattr(args, 'detection_limit', None) is None:
        detection_limit = {x: np.inf for x in filters}
    else:
        try:
            detection_limit = literal_eval(args.detection_limit)
        except:
            detection_limit = {
                x: float(y)
                for x, y in zip(
                    filters,
                    args.injection_detection_limit.split(","),
                )
            }
    return detection_limit