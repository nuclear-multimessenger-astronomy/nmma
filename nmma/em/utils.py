import copy

import numpy as np
import pandas as pd
from scipy.interpolate import interpolate as interp
import scipy.signal
import scipy.constants
import scipy.stats

import afterglowpy
import sncosmo
import dust_extinction.shapes as dustShp

from astropy.time import Time
from astropy.cosmology import Planck18, z_at_value
import astropy.units
import astropy.constants

from wrapt_timeout_decorator import timeout

import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

filts = [
    "u",
    "g",
    "r",
    "i",
    "z",
    "y",
    "J",
    "H",
    "K",
    "radio-3GHz",
    "radio-1.25GHz",
    "radio-5.5GHz",
    "radio-6GHz",
    "X-ray-1keV",
    "X-ray-5keV",
]
# these lambdas are in meter
lambdas_optical = 1e-10 * np.array(
    [3561.8, 4866.46, 6214.6, 7687.0, 7127.0, 7544.6, 8679.5, 9633.3, 12350.0]
)
lambdas_radio = scipy.constants.c / np.array([1.25e9, 3e9, 5.5e9, 6e9])
lambdas_Xray = scipy.constants.c / (
    np.array([1e3, 5e3]) * scipy.constants.eV / scipy.constants.h
)

lambdas = np.concatenate([lambdas_optical, lambdas_radio, lambdas_Xray])

nu = scipy.constants.c / lambdas

filt_to_nu_dict = dict(zip(filts, nu))


def extinctionFactorP92SMC(nu, Ebv, z, cutoff_hi=2e16):

    # Return the extinction factor (e ^ -0.4 * Ax) for the
    # Pei 1992 SMC model

    # Get model wavelength range
    ext_range_nu_lo = dustShp.P92.x_range[0] * 1e4 * afterglowpy.c
    ext_range_nu_hi = min(cutoff_hi, dustShp.P92.x_range[1] * 1e4 * afterglowpy.c)

    # host-frame frequencies
    nu_host = nu * (1 + z)

    # mask for frequencies that dust model is applicable for
    opt = (nu_host >= ext_range_nu_lo) & (nu_host <= ext_range_nu_hi)

    # host-frame wavelengths
    lam_host = (afterglowpy.c / nu_host[opt]) * astropy.units.cm

    # amplitudes have to be converted from B reference to V
    abav = dustShp.P92.AbAv

    # coefficients from Pei 1992
    extModel = dustShp.P92(
        BKG_amp=185.0 * abav,
        BKG_lambda=0.042,
        BKG_b=90.0,
        BKG_n=2.0,
        FUV_amp=27 * abav,
        FUV_lambda=0.08,
        FUV_b=5.5,
        FUV_n=4.0,
        NUV_amp=0.005 * abav,
        NUV_lambda=0.22,
        NUV_b=-1.95,
        NUV_n=2.0,
        SIL1_amp=0.010 * abav,
        SIL1_lambda=9.7,
        SIL1_b=-1.95,
        SIL1_n=2.0,
        SIL2_amp=0.012 * abav,
        SIL2_lambda=18.0,
        SIL2_b=-1.80,
        SIL2_n=2.0,
        FIR_amp=0.030 * abav,
        FIR_lambda=25.0,
        FIR_b=0.0,
        FIR_n=2.0,
    )

    Ax_o_Av = extModel(lam_host)
    Av = 2.93 * Ebv  # Rv = 2.93

    ext = np.ones(nu.shape)
    ext[opt] = np.power(10.0, -0.4 * Ax_o_Av * Av)

    return ext


def getRedShift(parameters):

    if "redshift" in parameters:
        z = parameters["redshift"]
    else:
        if parameters["luminosity_distance"] > 0:
            z = z_at_value(
                Planck18.luminosity_distance,
                parameters["luminosity_distance"] * astropy.units.Mpc,
                zmin=0.0,
                zmax=2.0,
            )
            if hasattr(z, "value"):
                z = z.value
        else:
            z = 0.0
    return z


def getFilteredMag(mag, filt):
    unprocessed_filt = [
        "u",
        "g",
        "r",
        "i",
        "z",
        "y",
        "J",
        "H",
        "K",
        "X-ray-1keV",
        "X-ray-5keV",
        "radio-5.5GHz",
        "radio-1.25GHz",
        "radio-6GHz",
        "radio-3GHz",
    ]
    # These average between filters is equivalent to
    # the geometric mean of the flux. These averages
    # are kind of justifiable because the spectral
    # commonly goes as F_\nu \propto \nu^\alpha,
    # where \nu is the frequency.
    if filt in unprocessed_filt:
        return mag[filt]
    elif filt == "w":
        return (mag["g"] + mag["r"] + mag["i"]) / 3.0
    elif filt in ["U", "UVW2", "UVW1", "UVM2"]:
        return mag["u"]
    elif filt == "B":
        return mag["g"]
    elif filt in ["c", "V", "F606W"]:
        return (mag["g"] + mag["r"]) / 2.0
    elif filt == "o":
        return (mag["r"] + mag["i"]) / 2.0
    elif filt == "R":
        return mag["z"]
    elif filt in ["I", "F814W"]:
        return (mag["z"] + mag["y"]) / 2.0
    elif filt == "F160W":
        return mag["H"]
    else:
        print("Unknown filter\n")
        return 0


def dataProcess(raw_data, filters, triggerTime, tmin, tmax):
    processedData = copy.deepcopy(raw_data)
    for filt in filters:
        if filt not in processedData:
            continue
        time = processedData[filt][:, 0]
        mag = processedData[filt][:, 1]
        dmag = processedData[filt][:, 2]
        # shift the time by the triggerTime
        time -= triggerTime

        # filter the out of range data
        idx = np.where((time > tmin) * (time < tmax))[0]

        data = np.vstack((time[idx], mag[idx], dmag[idx])).T
        processedData[filt] = data

    return processedData


def loadEvent(filename):
    lines = [line.rstrip("\n") for line in open(filename)]
    lines = filter(None, lines)

    data = {}
    for line in lines:
        lineSplit = line.split(" ")
        lineSplit = list(filter(None, lineSplit))
        mjd = Time(lineSplit[0], format="isot").mjd
        filt = lineSplit[1]
        mag = float(lineSplit[2])
        dmag = float(lineSplit[3])

        if filt not in data:
            data[filt] = np.empty((0, 3), float)
        data[filt] = np.append(data[filt], np.array([[mjd, mag, dmag]]), axis=0)

    return data


def loadEventSpec(filename):

    data_out = np.loadtxt(filename)
    spec = {}

    spec["lambda"] = data_out[:, 0]  # Angstroms
    spec["data"] = np.abs(data_out[:, 1])  # ergs/s/cm2./Angs
    spec["error"] = np.zeros(spec["data"].shape)  # ergs/s/cm2./Angs
    spec["error"][:-1] = np.abs(np.diff(spec["data"]))
    spec["error"][-1] = spec["error"][-2]
    idx = np.where(spec["error"] <= 0.5 * spec["data"])[0]
    spec["error"][idx] = 0.5 * spec["data"][idx]

    return spec


def read_spectroscopy_files(
    files, wavelength_min=3000.0, wavelength_max=10000.0, smooth=False
):

    data = {}
    for filename in files:
        name = (
            filename.replace("_spec", "")
            .replace(".spec", "")
            .replace(".txt", "")
            .replace(".dat", "")
            .split("/")[-1]
        )
        df = pd.read_csv(filename, names=["wavelength", "time", "fnu"])
        df_group = df.groupby("time")

        t_d = []
        lambda_d = []
        spec_d = []
        for ii, (tt, group) in enumerate(df_group):
            t_d.append(tt)
            if ii == 0:
                lambda_d = group["wavelength"].to_numpy()
                jj = np.where(
                    (lambda_d >= wavelength_min) & ((lambda_d <= wavelength_max))
                )[0]
                lambda_d = lambda_d[jj]
            spec = group["fnu"].to_numpy()[jj]
            if smooth:
                spec = scipy.signal.medfilt(spec, kernel_size=9)
            spec_d.append(spec)

        data[name] = {}
        data[name]["t"] = np.array(t_d)
        data[name]["lambda"] = np.array(lambda_d)
        data[name]["fnu"] = np.array(spec_d)

    return data


def read_photometry_files(files, filters=None, tt=np.linspace(0, 14, 100)):

    data = {}
    for filename in files:
        name = (
            filename.replace(".csv", "")
            .replace(".txt", "")
            .replace(".dat", "")
            .split("/")[-1]
        )

        # ZTF rest style file
        if "forced.csv" in filename or "alerts.csv" in filename:
            df = pd.read_csv(filename)
            idx = np.where(df["mag_unc"] != 99.0)[0]
            if len(idx) < 2:
                print(f"{name} does not have enough detections to interpolate.")
                continue

            jd_min = df["jd"].iloc[idx[0]]

            data[name] = {}
            data[name]["t"] = tt
            for filt in ["u", "g", "r", "i", "z", "y", "J", "H", "K"]:
                data[name][filt] = np.nan * tt

            for filt, group in df.groupby("filter"):
                idx = np.where(group["mag_unc"] != 99.0)[0]
                if len(idx) < 2:
                    continue
                lc = interp.interp1d(
                    group["jd"].iloc[idx] - jd_min,
                    group["mag"].iloc[idx],
                    fill_value=np.nan,
                    bounds_error=False,
                    assume_sorted=True,
                )
                data[name][filt] = lc(tt)
        else:
            mag_d = np.loadtxt(filename)

            data[name] = {}
            data[name]["t"] = mag_d[:, 0]
            data[name]["u"] = mag_d[:, 1]
            data[name]["g"] = mag_d[:, 2]
            data[name]["r"] = mag_d[:, 3]
            data[name]["i"] = mag_d[:, 4]
            data[name]["z"] = mag_d[:, 5]
            data[name]["y"] = mag_d[:, 6]
            data[name]["J"] = mag_d[:, 7]
            data[name]["H"] = mag_d[:, 8]
            data[name]["K"] = mag_d[:, 9]

        if filters is not None:
            filters_to_remove = set(list(data[name].keys())) - set(filters + ["t"])
            for filt in filters_to_remove:
                del data[name][filt]

    return data


def calc_lc(
    tt,
    param_list,
    svd_mag_model=None,
    svd_lbol_model=None,
    mag_ncoeff=None,
    lbol_ncoeff=None,
    interpolation_type="sklearn_gp",
    filters=["u", "g", "r", "i", "z", "y", "J", "H", "K"],
):

    mAB = {}
    for jj, filt in enumerate(filters):
        if mag_ncoeff:
            n_coeff = min(mag_ncoeff, svd_mag_model[filt]["n_coeff"])
        else:
            n_coeff = svd_mag_model[filt]["n_coeff"]
        # param_array = svd_mag_model[filt]["param_array"]
        # cAmat = svd_mag_model[filt]["cAmat"]
        VA = svd_mag_model[filt]["VA"]
        param_mins = svd_mag_model[filt]["param_mins"]
        param_maxs = svd_mag_model[filt]["param_maxs"]
        mins = svd_mag_model[filt]["mins"]
        maxs = svd_mag_model[filt]["maxs"]
        tt_interp = svd_mag_model[filt]["tt"]

        param_list_postprocess = np.array(param_list)
        for i in range(len(param_mins)):
            param_list_postprocess[i] = (param_list_postprocess[i] - param_mins[i]) / (
                param_maxs[i] - param_mins[i]
            )

        if interpolation_type == "tensorflow":
            model = svd_mag_model[filt]["model"]
            cAproj = model(np.atleast_2d(param_list_postprocess)).numpy().T.flatten()
            cAstd = np.ones((n_coeff,))
        elif interpolation_type == "api_gp":
            seed = 32
            random_state = np.random.RandomState(seed)

            gps = svd_mag_model[filt]["gps"]
            cAproj = np.zeros((n_coeff,))
            cAstd = np.zeros((n_coeff,))
            for i in range(n_coeff):
                gp = gps[i]
                y_pred = gp.mean(np.atleast_2d(param_list_postprocess))
                y_samples_test = gp.rvs(
                    100,
                    np.atleast_2d(param_list_postprocess),
                    random_state=random_state,
                )
                y_90_lo_test, y_90_hi_test = np.percentile(
                    y_samples_test, [5, 95], axis=1
                )
                cAproj[i] = y_pred
                cAstd[i] = y_90_hi_test - y_90_lo_test
        else:
            gps = svd_mag_model[filt]["gps"]
            cAproj = np.zeros((n_coeff,))
            cAstd = np.zeros((n_coeff,))
            for i in range(n_coeff):
                gp = gps[i]
                y_pred, sigma2_pred = gp.predict(
                    np.atleast_2d(param_list_postprocess), return_std=True
                )
                cAproj[i] = y_pred
                cAstd[i] = sigma2_pred

        # coverrors = np.dot(VA[:, :n_coeff], np.dot(np.power(np.diag(cAstd[:n_coeff]), 2), VA[:, :n_coeff].T))
        # errors = np.diag(coverrors)

        mag_back = np.dot(VA[:, :n_coeff], cAproj)
        mag_back = mag_back * (maxs - mins) + mins
        # mag_back = scipy.signal.medfilt(mag_back, kernel_size=3)

        ii = np.where((~np.isnan(mag_back)) * (tt_interp < 20.0))[0]
        if len(ii) < 2:
            maginterp = np.nan * np.ones(tt.shape)
        else:
            f = interp.interp1d(tt_interp[ii], mag_back[ii], fill_value="extrapolate")
            maginterp = f(tt)
        mAB[filt] = maginterp

    if svd_lbol_model is not None:
        if lbol_ncoeff:
            n_coeff = min(lbol_ncoeff, svd_lbol_model["n_coeff"])
        else:
            n_coeff = svd_lbol_model["n_coeff"]
        # param_array = svd_lbol_model["param_array"]
        # cAmat = svd_lbol_model["cAmat"]
        VA = svd_lbol_model["VA"]
        param_mins = svd_lbol_model["param_mins"]
        param_maxs = svd_lbol_model["param_maxs"]
        mins = svd_lbol_model["mins"]
        maxs = svd_lbol_model["maxs"]
        gps = svd_lbol_model["gps"]
        tt_interp = svd_lbol_model["tt"]

        param_list_postprocess = np.array(param_list)
        for i in range(len(param_mins)):
            param_list_postprocess[i] = (param_list_postprocess[i] - param_mins[i]) / (
                param_maxs[i] - param_mins[i]
            )

        if interpolation_type == "tensorflow":
            model = svd_lbol_model["model"]
            cAproj = model.predict(np.atleast_2d(param_list_postprocess)).T.flatten()
            cAstd = np.ones((n_coeff,))
        else:
            cAproj = np.zeros((n_coeff,))
            for i in range(n_coeff):
                gp = gps[i]
                y_pred, sigma2_pred = gp.predict(
                    np.atleast_2d(param_list_postprocess), return_std=True
                )
                cAproj[i] = y_pred

        lbol_back = np.dot(VA[:, :n_coeff], cAproj)
        lbol_back = lbol_back * (maxs - mins) + mins
        # lbol_back = scipy.signal.medfilt(lbol_back, kernel_size=3)

        ii = np.where(~np.isnan(lbol_back))[0]
        if len(ii) < 2:
            lbolinterp = np.nan * np.ones(tt.shape)
        else:
            f = interp.interp1d(tt_interp[ii], lbol_back[ii], fill_value="extrapolate")
            lbolinterp = 10 ** f(tt)
        lbol = lbolinterp
    else:
        lbol = np.inf * np.ones(len(tt))

    # fill radio and X-ray with null light curves
    mAB["radio-5.5GHz"] = np.inf * np.ones(len(tt))
    mAB["radio-1.25GHz"] = np.inf * np.ones(len(tt))
    mAB["radio-3GHz"] = np.inf * np.ones(len(tt))
    mAB["radio-6GHz"] = np.inf * np.ones(len(tt))
    mAB["X-ray-1keV"] = np.inf * np.ones(len(tt))
    mAB["X-ray-5keV"] = np.inf * np.ones(len(tt))

    return np.squeeze(tt), np.squeeze(lbol), mAB


def calc_spectra(tt, lambdaini, lambdamax, dlambda, param_list, svd_spec_model=None):

    # lambdas = np.arange(lambdaini, lambdamax+dlambda, dlambda)
    lambdas = np.arange(lambdaini, lambdamax, dlambda)

    spec = np.zeros((len(lambdas), len(tt)))
    for jj, lambda_d in enumerate(lambdas):
        n_coeff = svd_spec_model[lambda_d]["n_coeff"]
        # param_array = svd_spec_model[lambda_d]["param_array"]
        # cAmat = svd_spec_model[lambda_d]["cAmat"]
        # cAstd = svd_spec_model[lambda_d]["cAstd"]
        VA = svd_spec_model[lambda_d]["VA"]
        param_mins = svd_spec_model[lambda_d]["param_mins"]
        param_maxs = svd_spec_model[lambda_d]["param_maxs"]
        mins = svd_spec_model[lambda_d]["mins"]
        maxs = svd_spec_model[lambda_d]["maxs"]
        gps = svd_spec_model[lambda_d]["gps"]
        tt_interp = svd_spec_model[lambda_d]["tt"]

        param_list_postprocess = np.array(param_list)
        for i in range(len(param_mins)):
            param_list_postprocess[i] = (param_list_postprocess[i] - param_mins[i]) / (
                param_maxs[i] - param_mins[i]
            )

        cAproj = np.zeros((n_coeff,))
        for i in range(n_coeff):
            gp = gps[i]
            y_pred, sigma2_pred = gp.predict(
                np.atleast_2d(param_list_postprocess), return_std=True
            )
            cAproj[i] = y_pred

        spectra_back = np.dot(VA[:, :n_coeff], cAproj)
        spectra_back = spectra_back * (maxs - mins) + mins
        # spectra_back = scipy.signal.medfilt(spectra_back, kernel_size=3)

        N = 3  # Filter order
        Wn = 0.1  # Cutoff frequency
        B, A = scipy.signal.butter(N, Wn, output="ba")
        # spectra_back = scipy.signal.filtfilt(B, A, spectra_back)

        ii = np.where(~np.isnan(spectra_back))[0]
        if len(ii) < 2:
            specinterp = np.nan * np.ones(tt.shape)
        else:
            f = interp.interp1d(
                tt_interp[ii], spectra_back[ii], fill_value="extrapolate"
            )
            specinterp = 10 ** f(tt)
        spec[jj, :] = specinterp

    for jj, t in enumerate(tt):
        spectra_back = np.log10(spec[:, jj])
        spectra_back[~np.isfinite(spectra_back)] = -99.0
        if t < 7.0:
            spectra_back[1:-1] = scipy.signal.medfilt(spectra_back, kernel_size=5)[1:-1]
        else:
            spectra_back[1:-1] = scipy.signal.medfilt(spectra_back, kernel_size=5)[1:-1]
        ii = np.where((spectra_back != 0) & ~np.isnan(spectra_back))[0]
        if len(ii) < 2:
            specinterp = np.nan * np.ones(lambdas.shape)
        else:
            f = interp.interp1d(lambdas[ii], spectra_back[ii], fill_value="extrapolate")
            specinterp = 10 ** f(lambdas)
        spec[:, jj] = specinterp

    return np.squeeze(tt), np.squeeze(lambdas), spec


@timeout(60)
def fluxDensity(t, nu, **params):
    mJy = afterglowpy.fluxDensity(t, nu, **params)
    return mJy


def grb_lc(t_day, Ebv, param_dict):

    day = 86400.0  # in seconds
    tStart = (np.amin(t_day)) * day
    tEnd = (np.amax(t_day) + 1) * day
    tnode = min(len(t_day), 201)
    default_time = np.logspace(np.log10(tStart), np.log10(tEnd), base=10.0, num=tnode)

    filts = [
        "u",
        "g",
        "r",
        "i",
        "z",
        "y",
        "J",
        "H",
        "K",
        "radio-3GHz",
        "radio-1.25GHz",
        "radio-5.5GHz",
        "radio-6GHz",
        "X-ray-1keV",
        "X-ray-5keV",
    ]
    lambdas_optical = 1e-10 * np.array(
        [3561.8, 4866.46, 6214.6, 7687.0, 7127.0, 7544.6, 8679.5, 9633.3, 12350.0]
    )
    lambdas_radio = scipy.constants.c / np.array([1.25e9, 3e9, 5.5e9, 6e9])
    lambdas_Xray = scipy.constants.c / (
        np.array([1e3, 5e3]) * scipy.constants.eV / scipy.constants.h
    )

    lambdas = np.concatenate([lambdas_optical, lambdas_radio, lambdas_Xray])

    nu_0s = scipy.constants.c / lambdas

    if Ebv != 0.0:
        ext = extinctionFactorP92SMC(nu_0s, Ebv, param_dict["z"])
    else:
        ext = np.ones(len(nu_0s))

    times = np.empty((len(default_time), len(filts)))
    nus = np.empty((len(default_time), len(filts)))

    times[:, :] = default_time[:, None]
    for nu_idx, nu_0 in enumerate(nu_0s):
        nus[:, nu_idx] = nu_0

    # output flux density is in milliJansky
    try:
        mJys = fluxDensity(times, nus, **param_dict)
    except TimeoutError:
        return t_day, np.zeros(t_day.shape), {}

    Jys = 1e-3 * mJys

    if np.any(Jys <= 0.0):
        return t_day, np.zeros(t_day.shape), {}

    mag = {}
    lbol = 1e43 * np.ones(t_day.shape)

    for filt_idx, filt in enumerate(filts):

        Jy = Jys[:, filt_idx] * ext[filt_idx]

        # see https://en.wikipedia.org/wiki/AB_magnitude
        mag_d = -48.6 + -1 * np.log10(Jy / 1e23) * 2.5

        ii = np.where(np.isfinite(mag_d))[0]
        if len(ii) >= 2:
            f = interp.interp1d(
                default_time[ii] / day, mag_d[ii], fill_value="extrapolate"
            )
            maginterp = f(t_day)
        else:
            maginterp = np.nan * np.ones(t_day.shape)
            lbol = np.zeros(t_day.shape)

        mag[filt] = maginterp

    return t_day, lbol, mag


def sn_lc(
    tt,
    z,
    Ebv,
    abs_mag=-19.0,
    regularize_band="sdssu",
    model_name="nugent-hyper",
    parameters={},
):

    filts = [
        "u",
        "g",
        "r",
        "i",
        "z",
        "y",
        "J",
        "H",
        "K",
        "radio-1.25GHz",
        "radio-5.5GHz",
        "radio-6GHz",
        "radio-3GHz",
        "X-ray-1keV",
        "X-ray-5keV",
    ]
    # wavelength in Angstrom
    lambdas_optical = np.array(
        [3561.8, 4866.46, 6214.6, 6389.4, 7127.0, 7544.6, 8679.5, 9633.3, 12350.0]
    )
    lambdas_radio = scipy.constants.c / np.array([1.25e9, 5.5e9, 6e9, 3e9]) / 1e-10
    lambdas_Xray = (
        scipy.constants.c
        / (np.array([1e3, 5e3]) * scipy.constants.eV / scipy.constants.h)
        / 1e-10
    )

    lambdas = np.concatenate([lambdas_optical, lambdas_radio, lambdas_Xray])

    nus = scipy.constants.c / (1e-10 * lambdas)

    model = sncosmo.Model(source=model_name)
    if model_name == "nugent-hyper":
        model.set(z=z)
    elif model_name == "salt2":
        model.set(
            z=z,
            t0=np.median(tt),
            x0=parameters["x0"],
            x1=parameters["x1"],
            c=parameters["c"],
        )

    # regularize the absolute magnitude
    abs_mag -= Planck18.distmod(z).value
    model.set_source_peakabsmag(abs_mag, regularize_band, "ab", cosmo=Planck18)

    if Ebv != 0.0:
        ext = extinctionFactorP92SMC(nus, Ebv, z)
    else:
        ext = np.ones(len(nus))

    mag = {}
    lbol = 1e43 * np.ones(tt.shape)

    for filt_idx, (filt, lambda_A) in enumerate(zip(filts, lambdas)):

        if lambda_A < model.minwave() or lambda_A > model.maxwave():
            mag[filt] = np.inf * np.ones(tt.shape)

        else:
            try:
                # the output is in ergs / s / cm^2 / Angstrom
                flux = model.flux(tt, [lambda_A]) * ext[filt_idx]
                # see https://en.wikipedia.org/wiki/AB_magnitude
                flux_jy = 3.34e4 * np.power(lambda_A, 2.0) * flux
                mag_per_filt = -2.5 * np.log10(flux_jy) + 8.9
                mag[filt] = mag_per_filt[:, 0]
            except Exception:
                mag[filt] = np.ones(tt.shape) * np.nan
                lbol = np.zeros(tt.shape)

    return tt, lbol, mag


def sc_lc(t_day, param_dict):

    day = 86400.0  # in seconds
    t = t_day * day

    # fetch parameter values
    Me = 10 ** param_dict["log10_Menv"] * astropy.constants.M_sun.cgs.value
    Renv = 10 ** param_dict["log10_Renv"]
    Ee = 10 ** param_dict["log10_Ee"]
    Ebv = param_dict["Ebv"]
    z = param_dict["z"]

    filts = [
        "u",
        "g",
        "r",
        "i",
        "z",
        "y",
        "J",
        "H",
        "K",
        "radio-3GHz",
        "radio-1.25GHz",
        "radio-5.5GHz",
        "radio-6GHz",
        "X-ray-1keV",
        "X-ray-5keV",
    ]
    lambdas_optical = 1e-10 * np.array(
        [3561.8, 4866.46, 6214.6, 7687.0, 7127.0, 7544.6, 8679.5, 9633.3, 12350.0]
    )
    lambdas_radio = scipy.constants.c / np.array([1.25e9, 3e9, 5.5e9, 6e9])
    lambdas_Xray = scipy.constants.c / (
        np.array([1e3, 5e3]) * scipy.constants.eV / scipy.constants.h
    )

    lambdas = np.concatenate([lambdas_optical, lambdas_radio, lambdas_Xray])

    nu_obs = scipy.constants.c / lambdas
    nu_host = nu_obs * (1 + z)
    t /= 1 + z

    if Ebv != 0.0:
        ext = extinctionFactorP92SMC(nu_obs, Ebv, param_dict["z"])
    else:
        ext = np.ones(len(nu_obs))

    # define relevant constants
    c = astropy.constants.c.cgs.value
    h = astropy.constants.h.cgs.value
    kb = astropy.constants.k_B.cgs.value
    sb = astropy.constants.sigma_sb.cgs.value
    D = 10 * astropy.constants.pc.cgs.value
    n = 10
    delta = 1.1
    K = (n - 3) * (3 - delta) / (4 * np.pi * (n - delta))  # K = 0.119
    kappa = 0.2
    vt = np.sqrt(((n - 5) * (5 - delta) / ((n - 3) * (3 - delta))) * (2 * Ee / Me))
    td = np.sqrt((3 * kappa * K * Me) / ((n - 1) * vt * c))

    # evalute the model, lbol first
    prefactor = np.pi * (n - 1) / (3 * (n - 5)) * c * Renv * vt * vt / kappa
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
    T = np.power(sigmaT4 / sb, 0.25)
    T[T == 0.0] = np.nan
    one_over_T = 1.0 / T
    one_over_T[~np.isfinite(one_over_T)] = np.inf

    mag = {}
    for idx, filt in enumerate(filts):
        nu_of_filt = nu_host[idx]
        ext_per_filt = ext[idx]
        exp = np.exp(-h * nu_of_filt * one_over_T / kb)
        F = (
            (2.0 * (h * nu_of_filt) * (nu_of_filt / c) ** 2)
            * exp
            / (1 - exp)
            * Rs
            * Rs
            / D
            / D
        )
        F *= ext_per_filt
        F *= 1 + z
        mAB = np.ones(len(F))
        mAB *= np.inf
        mAB[F > 0] = -2.5 * np.log10(F[F > 0]) - 48.6
        mag[filt] = mAB

        # make sure there are at least two valid data point for interpolation
        if len(np.where(np.isfinite(mAB))[0]) < 2:
            mag[filt] = np.ones(t_day.shape) * np.nan

    return t_day, lbol, mag


def metzger_lc(t_day, param_dict):

    # convert time from day to second
    day = 86400.0  # in seconds
    t = t_day * day
    tprec = len(t)

    if len(np.where(t == 0)[0]) > 0:
        raise ValueError("For Me2017, start later than t=0")

    # define constants
    c = astropy.constants.c.cgs.value
    h = astropy.constants.h.cgs.value
    kb = astropy.constants.k_B.cgs.value
    Msun = astropy.constants.M_sun.cgs.value
    sigSB = astropy.constants.sigma_sb.cgs.value
    arad = 4 * sigSB / c
    Mpc = astropy.constants.pc.cgs.value * 1e6

    # fetch parameters
    M0 = 10 ** param_dict["log10_Mej"] * Msun  # total ejecta mass
    v0 = 10 ** param_dict["log10_vej"] * c  # minimum escape velocity
    beta = param_dict["beta"]
    kappa_r = 10 ** param_dict["log10_kappa_r"]
    z = param_dict["z"]
    Ebv = param_dict["Ebv"]
    D = 1e-5 * Mpc  # 10pc

    # define additional parameters
    E0 = 0.5 * M0 * v0 * v0  # initial thermal energy of bulk
    Mn = 1e-8 * Msun  # mass cut for free neutrons
    Ye = 0.1  # electron fraction
    Xn0max = 1 - 2 * Ye  # initial neutron mass fraction in outermost layers

    # define mass / velocity array of the outer ejecta, comprised half of the mass
    mmin = np.log(1e-8)
    mmax = np.log(M0 / Msun)
    mprec = 300
    m = np.arange(mprec) * (mmax - mmin) / (mprec - 1) + mmin
    m = np.exp(m)

    vm = v0 * np.power(m * Msun / M0, -1.0 / beta)
    vm[vm > c] = c

    # define thermalization efficiency rom Barnes+16
    ca3 = 1.3
    cb3 = 0.2
    cd3 = 1.1

    ca2 = 8.2
    cb2 = 1.2
    cd2 = 1.52

    ca = 0.56
    cb = 0.17
    cd = 0.74
    eth = np.exp(-ca * t_day) + np.log(1.0 + 2 * cb * (t_day ** (cd))) / (
        2 * cb * t_day ** (cd)
    )
    eth *= 0.36
    eth2 = np.exp(-ca2 * t_day) + np.log(1.0 + 2 * cb2 * (t_day ** (cd2))) / (
        2 * cb2 * t_day ** (cd2)
    )
    eth2 *= 0.36
    eth3 = np.exp(-ca3 * t_day) + np.log(1.0 + 2 * cb3 * (t_day ** (cd3))) / (
        2 * cb3 * t_day ** (cd3)
    )
    eth3 *= 0.36

    # define radioactive heating rates
    Xn0 = Xn0max * 2 * np.arctan(Mn / m / Msun) / np.pi  # neutron mass fraction
    Xr = 1.0 - Xn0  # r-process fraction

    # setting up wavelength and filters
    filts = [
        "u",
        "g",
        "r",
        "i",
        "z",
        "y",
        "J",
        "H",
        "K",
        "radio-3GHz",
        "radio-1.25GHz",
        "radio-5.5GHz",
        "radio-6GHz",
        "X-ray-1keV",
        "X-ray-5keV",
    ]
    lambdas_optical = 1e-10 * np.array(
        [3561.8, 4866.46, 6214.6, 7687.0, 7127.0, 7544.6, 8679.5, 9633.3, 12350.0]
    )
    lambdas_radio = scipy.constants.c / np.array([1.25e9, 3e9, 5.5e9, 6e9])
    lambdas_Xray = scipy.constants.c / (
        np.array([1e3, 5e3]) * scipy.constants.eV / scipy.constants.h
    )

    lambdas = np.concatenate([lambdas_optical, lambdas_radio, lambdas_Xray])

    nu_obs = scipy.constants.c / lambdas
    nu_host = nu_obs * (1 + z)
    t /= 1 + z

    if Ebv != 0.0:
        ext = extinctionFactorP92SMC(nu_obs, Ebv, param_dict["z"])
    else:
        ext = np.ones(len(nu_obs))

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
    edotr = (
        4.0e18
        * Xrarray
        * (0.5 - (1.0 / np.pi) * np.arctan((tarray - t0) / sig)) ** (1.3)
        * etharray
    )
    edotr = 2.1e10 * etharray * ((tarray / day) ** (-1.3))
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
        temp[j] = 1e10 * (3 * E[j] / (arad * 4 * np.pi * R[j] ** (3))) ** (0.25)
        if temp[j] > 4000.0:
            kappaoz = kappa_r
        if temp[j] < 4000.0:
            kappaoz = kappa_r * (temp[j] / 4000.0) ** (5.5)
        kappaoz = kappa_r
        LPdV = E[j] * v[j] / R[j]
        tdiff0 = 3 * kappaoz * M0 / (4 * np.pi * c * v[j] * t[j])
        tlc0 = R[j] / c
        tdiff0 = tdiff0 + tlc0
        Lrad[j] = E[j] / tdiff0
        Ek[j + 1] = Ek[j] + LPdV * dt[j]
        v[j + 1] = 1e20 * (2 * Ek[j] / M0) ** (0.5)
        E[j + 1] = (Lr[j] + Lsd[j] - LPdV - Lrad[j]) * dt[j] + E[j]
        R[j + 1] = v[j + 1] * dt[j] + R[j]
        taues[j + 1] = M0 * 0.4 / (4 * R[j + 1] ** 2)

        templayer = (
            3 * ene[:-1, j] * dm * Msun / (arad * 4 * np.pi * (t[j] * vm[:-1]) ** 3)
        ) ** (0.25)
        kappa_correction = np.ones(templayer.shape)
        kappa_correction[templayer > 4000.0] = 1.0
        kappa_correction[templayer < 4000.0] = templayer[
            templayer < 4000.0
        ] / 4000.0 ** (5.5)
        kappa_correction[:] = 1

        tdiff[:-1, j] = (
            0.08
            * kappa[:-1, j]
            * m[:-1]
            * Msun
            * 3
            * kappa_correction
            / (vm[:-1] * c * t[j] * beta)
        )
        tau[:-1, j] = (
            m[:-1] * Msun * kappa[:-1, j] / (4 * np.pi * (t[j] * vm[:-1]) ** 2)
        )
        lum[:-1, j] = ene[:-1, j] / (tdiff[:-1, j] + t[j] * (vm[:-1] / c))
        ene[:-1, j + 1] = (edot[:-1, j] - (ene[:-1, j] / t[j]) - lum[:-1, j]) * dt[
            j
        ] + ene[:-1, j]
        lum[:-1, j] = lum[:-1, j] * dm * Msun

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

    Ltot = Ltotm
    lbol = Ltotm * 1e40
    Tobs = 1e10 * (Ltot / (4 * np.pi * Rphoto ** 2 * sigSB)) ** (0.25)

    ii = np.where(~np.isnan(Tobs) & (Tobs > 0))[0]
    f = interp.interp1d(t_day[ii], Tobs[ii], fill_value="extrapolate")
    Tobs = f(t_day)

    Tobs[Tobs == 0.0] = np.nan
    one_over_T = 1.0 / Tobs
    one_over_T[~np.isfinite(one_over_T)] = np.inf

    mag = {}
    for idx, filt in enumerate(filts):
        nu_of_filt = nu_host[idx]
        ext_per_filt = ext[idx]
        exp = np.exp(-h * nu_of_filt * one_over_T / kb)
        F = (
            (2.0 * (h * nu_of_filt) * (nu_of_filt / c) ** 2)
            * exp
            / (1 - exp)
            * Rphoto
            * Rphoto
            / D
            / D
        )
        F *= ext_per_filt
        F *= 1 + z
        mAB = np.ones(len(F))
        mAB *= np.inf
        mAB[F > 0] = -2.5 * np.log10(F[F > 0]) - 48.6
        mag[filt] = mAB

    return t_day, lbol, mag


def powerlaw_blackbody_constant_temperature_lc(t_day, param_dict):

    # prevent the output message flooded by these warning messages
    old = np.seterr()
    np.seterr(invalid="ignore")
    np.seterr(divide="ignore")

    # convert time from day to second
    day = 86400.0  # in seconds
    t = t_day * day

    # define constants
    c = astropy.constants.c.cgs.value
    h = astropy.constants.h.cgs.value
    kb = astropy.constants.k_B.cgs.value
    sigSB = astropy.constants.sigma_sb.cgs.value
    Mpc = astropy.constants.pc.cgs.value * 1e6

    # fetch parameters
    bb_luminosity = param_dict["bb_luminosity"]  # blackboady's total luminosity
    temperature = param_dict["temperature"]  # for the blackbody radiation
    beta = param_dict["beta"]  # for the power-law
    powerlaw_mag = param_dict["powerlaw_mag"]
    powerlaw_filt_ref = "g"
    z = param_dict["z"]
    Ebv = param_dict["Ebv"]
    D = 1e-5 * Mpc  # 10pc

    # parameter conversion
    one_over_T = 1.0 / temperature
    bb_radius = np.sqrt(bb_luminosity / 4 / np.pi / sigSB) * one_over_T * one_over_T
    # calculate the powerlaw prefactor (with the reference filter)
    nu_ref = filt_to_nu_dict[powerlaw_filt_ref]
    powerlaw_prefactor = np.power(nu_ref, beta) * np.power(
        10, -0.4 * (powerlaw_mag + 48.6)
    )

    # setting up wavelength and filters
    filts = [
        "u",
        "g",
        "r",
        "i",
        "z",
        "y",
        "J",
        "H",
        "K",
        "radio-3GHz",
        "radio-1.25GHz",
        "radio-5.5GHz",
        "radio-6GHz",
        "X-ray-1keV",
        "X-ray-5keV",
    ]
    # these lambdas are in meter
    lambdas_optical = 1e-10 * np.array(
        [3561.8, 4866.46, 6214.6, 7687.0, 7127.0, 7544.6, 8679.5, 9633.3, 12350.0]
    )
    lambdas_radio = scipy.constants.c / np.array([1.25e9, 3e9, 5.5e9, 6e9])
    lambdas_Xray = scipy.constants.c / (
        np.array([1e3, 5e3]) * scipy.constants.eV / scipy.constants.h
    )

    lambdas = np.concatenate([lambdas_optical, lambdas_radio, lambdas_Xray])

    nu_obs = scipy.constants.c / lambdas
    nu_host = nu_obs * (1 + z)
    t /= 1 + z

    if Ebv != 0.0:
        ext = extinctionFactorP92SMC(nu_obs, Ebv, param_dict["z"])
    else:
        ext = np.ones(len(nu_obs))

    mag = {}
    for idx, filt in enumerate(filts):
        nu_of_filt = nu_host[idx]
        ext_per_filt = ext[idx]
        exp = np.exp(-h * nu_of_filt * one_over_T / kb)
        F_bb = (
            (2.0 * (h * nu_of_filt) * (nu_of_filt / c) ** 2)
            * exp
            / (1 - exp)
            * bb_radius
            * bb_radius
            / D
            / D
        )
        F_pl = powerlaw_prefactor * np.power(nu_of_filt, -beta)

        F = F_bb + F_pl  # adding the two contributions

        F *= ext_per_filt
        F *= 1 + z
        mAB = np.ones(len(t_day))
        mAB *= -2.5 * np.log10(F) - 48.6
        mag[filt] = mAB

    lbol = 1e43 * np.ones(t_day.shape)  # some dummy value

    np.seterr(**old)

    return t_day, lbol, mag


def estimate_mag_err(uncer_params, df):
    df["mag_err"] = df.apply(
        lambda x: (uncer_params["band"] == x["passband"])
        & (pd.arrays.IntervalArray(uncer_params["interval"]).contains(x["mag"])),
        axis=1,
    ).apply(
        lambda x: scipy.stats.skewnorm.rvs(
            uncer_params[x]["a"], uncer_params[x]["loc"], uncer_params[x]["scale"]
        ),
        axis=1,
    )
    return df
