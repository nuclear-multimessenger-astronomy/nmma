import copy
import numpy as np
import os
import pandas as pd
import scipy.interpolate as interp
import scipy.signal
import scipy.constants
import scipy.stats

import sncosmo
from sncosmo.bandpasses import _BANDPASSES, _BANDPASS_INTERPOLATORS

import dust_extinction.shapes as dustShp

import astropy.units
import astropy.constants

import matplotlib
import matplotlib.pyplot as plt


import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)





# Define default filters variable for import in other parts of the code
DEFAULT_FILTERS = [
    "bessellux",
    "bessellb",
    "bessellv",
    "bessellr",
    "besselli",
    "sdssu",
    "ps1::g",
    "ps1::r",
    "ps1::i",
    "ps1::z",
    "ps1::y",
    "uvot::b",
    "uvot::u",
    "uvot::uvm2",
    "uvot::uvw1",
    "uvot::uvw2",
    "uvot::v",
    "uvot::white",
    "atlasc",
    "atlaso",
    "2massj",
    "2massh",
    "2massks",
    "ztfg",
    "ztfr",
    "ztfi",
    "ultrasat",
]

def setup_sample_times(args):
    n_step = int((args.tmax - args.tmin) / args.dt) +1
    if getattr(args, 'log_space_time', False):
        return np.geomspace(args.tmin, args.tmax, n_step)
    else:
        return np.linspace(args.tmin, args.tmax, n_step)

def transform_to_app_mag_dict(mag_dict, params):
    if params["luminosity_distance"] > 0:
        for k in mag_dict.keys():
            mag_dict[k] += 5.0 * np.log10(
            params["luminosity_distance"] * 1e6 / 10.0)
    return mag_dict


def extinctionFactorP92SMC(nu, Ebv, z, cutoff_hi=2e16):

    # Return the extinction factor (e ^ -0.4 * Ax) for the
    # Pei 1992 SMC model

    # Get model wavelength range
    ext_range_nu_lo = dustShp.P92.x_range[0] * 1e4 * c_cgs
    ext_range_nu_hi = min(cutoff_hi, dustShp.P92.x_range[1] * 1e4 * c_cgs)

    # host-frame frequencies
    nu_host = nu * (1 + z)

    # mask for frequencies that dust model is applicable for
    opt = (nu_host >= ext_range_nu_lo) & (nu_host <= ext_range_nu_hi)

    # host-frame wavelengths
    lam_host = (c_cgs / nu_host[opt]) * astropy.units.cm

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

def get_all_bandpass_metadata():
    """
    Retrieves and combines the metadata for all registered bandpasses and interpolators.

    Returns:
        list: Combined list of metadata dictionaries from bandpasses and interpolators for sncosmo.
    """

    bandpass_metadata = _BANDPASSES.get_loaders_metadata()
    interpolator_metadata = _BANDPASS_INTERPOLATORS.get_loaders_metadata()

    combined_metadata = bandpass_metadata + interpolator_metadata

    return combined_metadata


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
        "sdss__u",
        "sdss__g",
        "sdss__r",
        "sdss__i",
        "sdss__z",
        "swope2__y",
        "swope2__J",
        "swope2__H",
    ]
    sncosmo_filts = [val["name"] for val in get_all_bandpass_metadata()]
    sncosmo_maps = {
        name.replace(":", "_"): name.replace(":", "_") for name in sncosmo_filts
    }
    sncosmo_maps.update({name: name.replace(":", "_") for name in sncosmo_filts})

    # These average between filters is equivalent to
    # the geometric mean of the flux. These averages
    # are kind of justifiable because the spectral
    # commonly goes as F_\nu \propto \nu^\alpha,
    # where \nu is the frequency.
    if filt in unprocessed_filt or filt.startswith(("radio", "X-ray")):
        return mag[filt]
    elif filt in sncosmo_maps:
        return mag[sncosmo_maps[filt]]
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
        raise ValueError(f"Unknown filter: {filt}")


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


def interpolate_nans(data_dict: dict) -> dict:
    """
    Interpolates the NaN values in a photometric data.

    Args:
        data_dict (dict): Dictionary containing photometric data. The keys correspond to the filenames
        of the data. The values are dictionaries of which the keys correspond to time (t) or the filters considered.
        The corresponding values are the time grid (in days) and the magnitudes of the different filters.
    """

    # Iterate over all the data files
    for name, sub_dict in data_dict.items():
        # For each file, iterate over the time or the filters present
        for key, val in sub_dict.items():
            # Skip over the time grid and filters which have no NaN values
            if key == "t":
                time_array = val
                continue
            if not any(np.isnan(val)):
                continue

            interpolated_data = autocomplete_data(time_array, time_array, val)
            ##interpolated_data contains only nans, if there were not at least 2 usable data points
            if np.isnan(interpolated_data[0]):
                continue
            
            data_dict[name][key] = interpolated_data
                

    return data_dict

def autocomplete_data(interp_points, ref_points, ref_data, extrapolate='linear', ref_value=np.nan):
    """
    Interpolates and extrapolates reference data to a 1-D array of arguments. This can be wide off!
    This basically extends np.interp to ignore nans and provide simple extrapolations. 
    """
    
    ii = np.where(np.isfinite(ref_data))[0]
    if len(ii) < 2:
        return np.full_like(interp_points, ref_value)
    
    fin_ref= np.asarray(ref_points)[ii]
    fin_data=np.asarray(ref_data)[ii]
    interp_points=np.atleast_1d(interp_points)

    if isinstance(extrapolate, (float , int)):
        interp_data = np.interp(interp_points, fin_ref, fin_data,
                                 left=extrapolate, right=extrapolate)
      
    elif isinstance(extrapolate, str):
        if extrapolate=='spline':
            spline = interp.UnivariateSpline(fin_ref, fin_data, s=ref_value)
            interp_data = spline(interp_points)

        if extrapolate=='linear':
            interp_data = np.interp(interp_points, fin_ref, fin_data)
            x0, x1, xm, xn = fin_ref[[0,1,-2,-1]]
            y0, y1, ym, yn = fin_data[[0,1,-2,-1]]
            lower_extrap_args= np.argwhere(interp_points<x0)
            interp_data[lower_extrap_args] = y0 + (y1-y0)/(x1-x0)*(interp_points[lower_extrap_args]-x0)
            upper_extrap_args = np.argwhere(interp_points>xn)
            interp_data[upper_extrap_args] = yn + (yn-ym)/(xn-xm)*(interp_points[upper_extrap_args]-xn)
        ## TODO Allow more sophisticated treatment of extrapolation
    
    else:
        interp_data = np.interp(interp_points, fin_ref, fin_data,
                                 left=extrapolate[0], right=extrapolate[-1])   
    return interp_data


def get_default_filts_lambdas(filters=None):

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
        "U",
        "B",
        "V",
        "R",
        "I",
        "radio-1.25GHz",
        "radio-3GHz",
        "radio-5.5GHz",
        "radio-6GHz",
        "X-ray-1keV",
        "X-ray-5keV",
    ]
    lambdas_sloan = 1e-10 * np.array(
        [3561.8, 4866.46, 6214.6, 7687.0, 7127.0, 7544.6, 8679.5, 9633.3, 12350.0]
    )
    lambdas_bessel = 1e-10 * np.array([3605.07, 4413.08, 5512.12, 6585.91, 8059.88])
    lambdas_radio = scipy.constants.c / np.array([1.25e9, 3e9, 5.5e9, 6e9])
    lambdas_Xray = scipy.constants.c / (
        np.array([1e3, 5e3]) * scipy.constants.eV / scipy.constants.h
    )

    lambdas = np.concatenate(
        [lambdas_sloan, lambdas_bessel, lambdas_radio, lambdas_Xray]
    )

    bandpasses = []
    for val in get_all_bandpass_metadata():
        if val["name"] in [
            "ultrasat",
            "megacampsf::u",
            "megacampsf::g",
            "megacampsf::r",
            "megacampsf::i",
            "megacampsf::z",
            "megacampsf::y",
        ]:
            bandpass = sncosmo.get_bandpass(val["name"], 3)
            bandpass.name = bandpass.name.split()[0]
        else:
            bandpass = sncosmo.get_bandpass(val["name"])

        bandpasses.append(bandpass)

    filts = filts + [band.name for band in bandpasses]
    lambdas = np.concatenate([lambdas, [1e-10 * band.wave_eff for band in bandpasses]])

    if filters is not None:
        filts_slice = []
        lambdas_slice = []

        for filt in filters:
            if filt.startswith("radio") and filt not in filts:
                # for additional radio filters that not in the list
                # calculate the lambdas based on the filter name
                # split the filter name
                freq_string = filt.replace("radio-", "")
                freq_unit = freq_string[-3:]
                freq_val = float(freq_string.replace(freq_unit, ""))
                # make use of the astropy.units to be more flexible
                freq = astropy.units.Quantity(freq_val, unit=freq_unit)
                freq = freq.to("Hz").value
                # adding to the list
                filts_slice.append(filt)
                lambdas_slice.append(scipy.constants.c / freq)
            elif filt.startswith("X-ray-") and filt not in filts:
                # for additional X-ray filters that not in the list
                # calculate the lambdas based on the filter name
                # split the filter name
                energy_string = filt.replace("X-ray-", "")
                energy_unit = energy_string[-3:]
                energy_val = float(energy_string.replace(energy_unit, ""))
                # make use of the astropy.units to be more flexible
                energy = astropy.units.Quantity(energy_val, unit=energy_unit)
                freq = energy.to("eV").value * scipy.constants.eV / scipy.constants.h
                # adding to the list
                filts_slice.append(filt)
                lambdas_slice.append(scipy.constants.c / freq)
            else:
                try:
                    ii = filts.index(filt)
                    filts_slice.append(filts[ii])
                    lambdas_slice.append(lambdas[ii])
                except ValueError:
                    ii = filts.index(filt.replace("_", ":"))
                    filts_slice.append(filts[ii].replace(":", "_"))
                    lambdas_slice.append(lambdas[ii])

        filts = filts_slice
        lambdas = np.array(lambdas_slice)

    return filts, lambdas


def flux_to_ABmag(flux, unit='cgs', residual_mag = None):
    """ see https://en.wikipedia.org/wiki/AB_magnitude """
    if unit=='cgs':
        residual_magnitude =  - 48.6
    elif unit == 'Jy':
        residual_magnitude = 8.9
    elif unit == 'mJy':
        residual_magnitude = 16.4
    if residual_mag:
        residual_magnitude = residual_mag

    suff_flux = np.argwhere(flux> 0)
    if len(suff_flux)< 2:
        return np.full_like(flux, np.nan)
    mAB = np.full_like(flux, np.inf)

    mAB[suff_flux] = -2.5 * np.log10(flux [suff_flux]) + residual_magnitude
    return mAB

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
    if not df["mag_err"].values:
        argmin_slice = np.argmin(uncer_params["interval"])
        for value in df["mag"].values:
            if uncer_params.iloc[argmin_slice]["interval"].left > value:
                print(
                    f'WARNING: {value} is outside of the measured uncertainty region with a lower limit of {uncer_params.iloc[argmin_slice]["interval"].left}'
                )

    return df



# The following LANL File readers are taken from Eve Chase's cocteau package


class SpectraOverTime(object):
    """
    A collection of spectra at successive timesteps
    Written by Eve Chase.
    """

    def __init__(self, timesteps=np.array([]), spectra=np.array([]), num_angles=1):
        """
        Parameters:
        -----------
        timesteps: array
            array of timesteps in days
        spectra: array
            array of Spectrum objects, each at corresponding timestep
        num_angles: int
            number of angular bins. This assumes each
            bin spans equal solid angle.
        """
        # FIXME: this seems like the wrong datastructure here
        self.timesteps = timesteps
        self.spectra = spectra
        self.num_angles = num_angles


class Spectrum(object):
    """
    Spectrum as a function of wavelength
    Written by Eve Chase.
    """

    def __init__(self, timestep=None, wavelengths=None, flux_density=None):
        """
        Parameters:
        -----------
        timestep: float
            time in days corresponding to spectrum
        wavelengths: array
            input wavelengths in cm
        flux_density: array
            flux at R=10pc in units of erg / s / cm^3
        """
        # FIXME: assert that wavelengths are sorted
        self.timestep = timestep
        self.wavelength_arr = wavelengths.cgs
        self.flux_density_arr = flux_density.cgs

    def interpolate(self):
        """
        Interpolate a functional form of the flux density
        array as a function of wavelength
        Returns
        -------
        spectrum_func: scipy.interpolate.interpolate.interp1d
            functional representation of spectrum
        """

        # Values must be in cgs for interpolation
        return interp.interp1d(
            self.wavelength_arr.cgs,
            self.flux_density_arr.cgs,
            bounds_error=False,
            fill_value=0,
        )

    def plot(self, ax=None, **kwargs):
        """
        Plot spectrum in format similar to Even et al. (2019)
        Returns:
        --------
        ax: Axes object
            contains figure information
        """

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(
            self.wavelength_arr.cgs.value * 1e4,
            np.log10(
                self.flux_density_arr.cgs.value
                * (4 * np.pi * (10 * 3.08567758e18) ** 2)
            ),
            **kwargs,
        )

        ax.set_ylabel(r"$\log_{10}$ dL\d$\lambda$  (erg s$^-1 \AA^{-1}$) + const. ")
        ax.set_xlabel("Wavelength (Microns)")
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_tick_params(which="minor", bottom=False)

        return ax


def read_LANL_spectra(
    self,
    filename,
    time_units=astropy.units.day,
    wl_units=astropy.units.cm,
    angles=[0],
    remove_zero=False,
    fd_units=(
        astropy.units.erg
        / astropy.units.s
        / astropy.units.cm**2
        / astropy.units.angstrom
    ),
):
    """
    Read in spectra at multiple timesteps
    for Even et al. (2019) and subsequent
    paper data format. Written by Eve Chase.
    Parameters
    ----------
    filename: string
        path to spectrum file
    Returns
    -------
    spectra: dictionary
        - time in days as keys
        - each time contains a dictionary with
        a wavelength array in cm and a flux density
        array in erg / s / cm^3
    """

    assert os.path.isfile(filename)

    # Check that units are appropriate
    wl_units.to(astropy.units.angstrom)
    fd_units.to(
        astropy.units.erg
        / astropy.units.s
        / astropy.units.cm**2
        / astropy.units.angstrom
    )

    # Determine time steps in file
    nrows, timesteps_in_file = self.parse_file(filename, key="time")

    if len(timesteps_in_file) == 0:
        raise IOError("File not read. Check file type.")

    # Set up properties to collect spectra
    timesteps = np.array(list(timesteps_in_file.keys())) * time_units
    spectra_arr = np.zeros(len(timesteps), dtype=object)

    col_names = ["wavelength_low", "wavelength_high"]
    spectra = {}
    num_angles = len(angles)
    for angle in angles:
        col_names.append(f"spec_angle{angle}")
        spectra[angle] = SpectraOverTime(timesteps=timesteps, num_angles=num_angles)

    col_idx = np.concatenate([np.array([0, 1]), np.asarray(angles) + 2])

    # Read in the spectrum at a given timestep
    for i, time in enumerate(timesteps):
        rows_to_skip = np.arange(timesteps_in_file[time.value])

        spectrum_df = pd.read_csv(
            filename,
            skiprows=rows_to_skip,
            names=col_names,
            usecols=col_idx,
            nrows=nrows,
            delim_whitespace=True,
            dtype="float",
        )
        # Store each angular bin separately
        for angle in angles:
            col_name = f"spec_angle{angle}"
            spectrum_copy = spectrum_df.copy()

            # Remove all points where the spectrum is zero
            if remove_zero:
                spectrum_copy = spectrum_copy.drop(
                    spectrum_copy[spectrum_copy[col_name] == 0].index
                )

            # Compute average wavelength in bin
            wavelengths = (
                0.5
                * (
                    spectrum_copy["wavelength_low"] + spectrum_copy["wavelength_high"]
                ).values
                * wl_units
            )

            # Make Spectrum object
            flux_density_arr = spectrum_copy[col_name].values * fd_units

            spectra[angle].spectra = np.append(
                spectra[angle].spectra,
                Spectrum(time, wavelengths, flux_density_arr),
            )

    assert timesteps.size > 0
    assert spectra_arr.size > 0

    return spectra


def get_knprops_from_LANLfilename(filename):
    """
    Read the standard LANL filename format.
    Typically this looks something like this:
    'Run_TP_dyn_all_lanth_wind2_all_md0.1_vd0.3_mw0.001_vw0.05_mags_2020-01-04.dat'
    Written by Eve Chase.
    Parameters
    ----------
    filename: str
        string representation of filename
    """

    wind = None
    morph = None
    md = None
    vd = None
    mw = None
    vw = None
    KNTheta = None
    num_comp = 2

    # Reduce filename to last part
    filename = filename.split("/")[-1]

    for info in filename.split("_"):
        # Record morphology
        if morph is None:
            if "TS" in info:
                morph = 0
            elif "TP" in info:
                morph = 1
            elif "ST" in info:
                morph = 2
            elif "SS" in info:
                morph = 3
            elif "SP" in info:
                morph = 4
            elif "PS" in info:
                morph = 5
            elif "H" in info:
                morph = 6
                num_comp = 1
            elif "P" in info:
                morph = 7
                num_comp = 1
            elif "R" in info and "Run" not in info:
                morph = 8
                num_comp = 1
            elif "S" in info:
                morph = 9
                num_comp = 1
            elif "T" in info:
                morph = 10
                num_comp = 1

        # Record velocity and mass for two component models
        if num_comp == 2:
            # Record wind
            if "wind" in info:
                wind = int(info[-1])

            # Record dynamical ejecta mass
            elif "md" in info:
                md = float(info[2:])
                if "." not in info:
                    if "1" in info:
                        md /= 100
                    else:
                        md /= 1000

            # Record dynamical ejecta velocity
            elif "vd" in info:
                vd = float(info[2:])
                if "." not in info:
                    if "5" in info:
                        vd /= 100
                    else:
                        vd /= 10

            # Record wind ejecta mass
            elif "mw" in info:
                mw = float(info[2:])
                if "." not in info:
                    if "1" in info:
                        mw /= 100
                    else:
                        mw /= 1000

            # Record wind ejecta velocity
            elif "vw" in info:
                vw = float(info[2:])
                if "." not in info:
                    if "5" in info:
                        vw /= 100
                    else:
                        vw /= 10

            elif "theta" in info:
                KNTheta = float(info[5:])

        # Record velocity and ejecta mass for single component models
        elif num_comp == 1:
            if "m" in info and "v" in info:
                mass, vel = info.split("v")
                md = float(mass[2:])
                vd = float(vel)

            # Record mass
            elif "m" == info[0] and info != "mags":
                md = float(info[2:])
                # Recast masses
                if md in [1, 5]:
                    md /= 100
                elif md in [2]:
                    md /= 1000

            # Record velocity
            elif "v" == info[0]:
                vd = float(info[1:]) / 100

            # Record composition
            elif "Ye" == info[:2]:
                wind = float(info[2:]) / 100

            elif "theta" in info:
                KNTheta = float(info[5:])

    param_values = {
        "morphology": morph,
        "Ye_wind": wind,
        "mej_dyn": md,
        "vej_dyn": vd,
        "mej_wind": mw,
        "vej_wind": vw,
        "KNtheta": KNTheta,
    }
    knprops = {}
    for prop in [
        # "morphology",
        # "Ye_wind",
        "mej_dyn",
        "vej_dyn",
        "mej_wind",
        "vej_wind",
        "KNtheta",
    ]:
        prop_value = param_values[prop]
        if prop_value is not None:
            knprops[prop] = prop_value

    return knprops


def parse_LANLfile(filename, key="band"):
    """
    Tool for reading data from the Wollaeger et al. (2018)
    and subsequent paper data format.
    Used to determine the number of rows for a given passband
    filter or timestep.
    Written by Eve Chase.
    Parameters
    ----------
    filename: string
        path to magnitude file
    key: string
        key to search for in file. Options: 'band', 'time'
    Returns
    -------
    nrows: int
        Number of rows between successive appearances of key
    keys_in_file: dictionary
        Dictionary where keys are each occurance of the selected
        keyword (i.e. each bandname or each timestep) and values
        are the line number to start searching for that value in.
    """

    assert key in ["time", "band", "bolometric"]

    keys_in_file = {}

    # Find out how many rows are in each band
    with open(filename, "r") as datafile:
        # Read each line until the key appears
        count = 1
        key_count = 0  # Line number where key appears
        line = datafile.readline()
        if key in ["band", "bolometric"]:
            line = datafile.readline()
        while line:
            if key in line:
                key_count = count
                if key in ["band", "bolometric"]:
                    keys_in_file[line.split()[1]] = count
                    if key == "bolometric":
                        key = "band"
                elif key == "time":
                    keys_in_file[float(line.split()[-1])] = count
            line = datafile.readline()

            count += 1
    # if key == 'band':
    #    nrows = count - key_count - 4
    # elif key == 'time':
    nrows = count - key_count - 3

    return nrows, keys_in_file

