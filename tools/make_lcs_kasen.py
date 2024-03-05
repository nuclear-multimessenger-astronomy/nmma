#!/usr/bin/env python3

import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import savgol_filter
import scipy.interpolate as interp
import sncosmo
import argparse
import h5py
from astropy.cosmology import Planck18 as cosmo
from nmma.em.utils import DEFAULT_FILTERS


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filters",
        type=str,
        default=DEFAULT_FILTERS,
        help="comma-separated list of filters for photometric lcs; must be from the bandpasses listed here: \
                        https://sncosmo.readthedocs.io/en/stable/bandpass-list.html",
    )
    parser.add_argument(
        "--modeldir",
        type=str,
        help="directory where .hdf5 files are located",
        default="model/",
    )
    parser.add_argument(
        "--lcdir",
        type=str,
        help="output directory for generated lightcurves",
        default="lcs/",
    )
    parser.add_argument(
        "--doLbol",
        help="extract bolometric lightcurves",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--doAB",
        help="extract photometric lightcurves",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--doSmoothing",
        help="Employ Savitzky-Golay filter for smoothing",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--dMpc",
        type=float,
        help="distance in Mpc, default is 10 pc to get lightcurves in Absolute Mag",
        default=1e-5,
    )
    parser.add_argument(
        "--z",
        type=float,
        help="redshift, if provided it dominates over dMpc",
        default=None,
    )

    return parser.parse_args()


args = parse()

if isinstance(args.filters, str):
    filters = args.filters.replace(" ", "")
    filters = filters.split(",")
else:
    filters = args.filters

lcdir = args.lcdir
if not os.path.isdir(lcdir):
    os.makedirs(lcdir)
if not args.doAB and not args.doLbol:
    raise SystemExit(
        "ERROR: Neither --doAB nor --doLbol are enabled. Please enable at least one."
    )

files = [f for f in os.listdir(args.modeldir) if "h5" in f]
numfiles = len(files)

# Use redshift or dMpc if z is not provided
if args.z is None:
    dMpc = args.dMpc
    D_cm = dMpc * 1e6 * 3.0857e18
    H0 = cosmo.H0.value
    CLIGHT = 2.99792458e5
    CLIGHT_cm_s = 1e5 * CLIGHT
    ztest = np.arange(0.0001, 1, 0.00001)
    Dtest = np.array(cosmo.luminosity_distance(ztest).to("Mpc").value)
    z = ztest[np.argmin(abs(dMpc - Dtest))]
else:
    z = args.z
    dMpc = cosmo.luminosity_distance(z).to("Mpc").value

for kk, filename in enumerate(files):
    if kk % 10 == 0:
        print(f"{kk * 100 / numfiles:.2f}% done")

    with h5py.File(os.path.join(args.modeldir, filename), "r") as f:
        nu = np.array(f["nu"], dtype="d")
        time = np.array(f["time"])
        Lnu = np.array(f["Lnu"], dtype="d")

    # smooth over missing data
    Lnu[Lnu == 0.0] = 1e20
    Lnu = 10 ** gaussian_filter(np.log10(Lnu), 3.0)

    nuS = np.tile(nu, (len(time), 1))

    Llam = Lnu * nuS**2.0 / CLIGHT_cm_s / 1e8  # ergs/s/Angstrom
    Llam = Llam / (4 * np.pi * D_cm**2)  # ergs / s / cm^2 / Angstrom

    ph = np.array(time) / (60 * 60 * 24)  # get time in days
    wave = CLIGHT_cm_s / nu * 1e8  # AA

    # flip axes to make wavelength increasing
    wave = np.flip(wave)
    Llam = np.flip(Llam, axis=1)

    # extract photometric lightcurves
    if args.doAB:
        if args.z is not None:
            lc = open(
                os.path.join(lcdir, f"{filename[:-5]}_z{z}.dat"),
                "w",
            )
        else:
            lc = open(
                os.path.join(
                    lcdir,
                    f"{filename[:-5]}_dMpc{int(dMpc)}.dat",
                ),
                "w",
            )

        lc.write(f'# t[days] {" ".join(filters)} \n')
        m_tot = []
        for filt in filters:
            source = sncosmo.TimeSeriesSource(ph, wave, Llam)

            if filt == "ultrasat":
                bandpass = sncosmo.get_bandpass(filt, 5.0)
            else:
                bandpass = sncosmo.get_bandpass(filt)

            m = source.bandmag(bandpass, "ab", ph)

            # apply smoothing filter
            if args.doSmoothing:
                ii = np.where(~np.isnan(m))[0]
                if len(ii) > 1:
                    f = interp.interp1d(ph[ii], m[ii], fill_value="extrapolate")
                    m = f(ph)
                m = savgol_filter(m, window_length=17, polyorder=3)

            m_tot.append(m)

        for i, t in enumerate(ph):
            if t < 0:
                continue
            lc.write(f"{t:.3f} ")
            for ifilt in range(len(filters)):
                lc.write(f"{m_tot[ifilt][i]:.3f} ")
            lc.write("\n")
        lc.close()

    # extract bolometric lightcurves
    if args.doLbol:
        if args.z is not None:
            Lbol_f = open(
                os.path.join(lcdir, f"{filename[:-5]}_z{z}_Lbol.dat"),
                "w",
            )
        else:
            Lbol_f = open(
                os.path.join(
                    lcdir,
                    f"{filename[:-5]}_dMpc{int(dMpc)}_Lbol.dat",
                ),
                "w",
            )
        Lbol_f.write("# t[days] Lbol[erg/s] \n")

        Lbol = np.trapz(
            Lnu * np.flipud(nu) ** 2.0 / CLIGHT_cm_s / 1e8 * (4 * np.pi * D_cm**2),
            x=wave,
        )

        if args.doSmoothing:
            ii = np.where(np.isfinite(np.log10(Lbol)))[0]
            f = interp.interp1d(ph[ii], np.log10(Lbol[ii]), fill_value="extrapolate")
            Lbol = 10 ** f(ph)
            Lbol = savgol_filter(Lbol, window_length=17, polyorder=3)

        for i, t in enumerate(ph):
            if t < 0:
                continue
            Lbol_f.write(f"{t:.3f} {Lbol[i]:.5e} \n")

        Lbol_f.close()
