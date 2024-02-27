#!/usr/bin/env python3

import os
import numpy as np
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
    # parser.add_argument('--doSmoothing',help='Employ Savitzky-Golay filter for smoothing',action="store_true",default=False)
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

files = [f for f in os.listdir(args.modeldir) if "hdf5" in f]
numfiles = len(files)

# Use redshift or dMpc if z is not provided
if args.z is None:
    dMpc = args.dMpc
    D_cm = dMpc * 1e6 * 3.0857e18
    H0 = cosmo.H0.value
    CLIGHT = 2.99792458e5
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
        data = f["observables"]
        stokes = np.array(data["stokes"])
        ph = np.array(data["time"]) / (60 * 60 * 24)  # get time in days
        wave = np.array(data["wave"]) * (1 + z)  # wavelength spec w/ redshift
        Lbol = np.array(data["lbol"])

    Istokes = stokes[:, :, :, 0]  # get I stokes parameter
    Nobs = stokes.shape[0]  # num observing angles
    Ntime = stokes.shape[1]  # num sampled times
    Nwave = stokes.shape[2]  # num samped wavelengths

    cos = np.linspace(0, 1, Nobs)
    theta = np.arccos(cos) * 180 / np.pi

    for obs in range(0, Nobs):
        fl = Istokes[obs] * (1.0 / dMpc) ** 2 / (1 + z)

        # extract photometric lightcurves
        if args.doAB:
            if args.z is not None:
                lc = open(
                    os.path.join(
                        lcdir, f"{filename[:-5]}_theta{theta[obs]:.2f}_z{z}.dat"
                    ),
                    "w",
                )
            else:
                lc = open(
                    os.path.join(
                        lcdir,
                        f"{filename[:-5]}_theta{theta[obs]:.2f}_dMpc{int(dMpc)}.dat",
                    ),
                    "w",
                )

            lc.write(f'# t[days] {" ".join(filters)} \n')
            m_tot = []
            for filt in filters:
                source = sncosmo.TimeSeriesSource(ph, wave, fl)

                if filt == "ultrasat":
                    bandpass = sncosmo.get_bandpass(filt, 5.0)
                else:
                    bandpass = sncosmo.get_bandpass(filt)

                m = source.bandmag(bandpass, "ab", ph)
                m_tot.append(m)

            for i, t in enumerate(ph):
                lc.write(f"{t:.3f} ")
                for ifilt in range(len(filters)):
                    lc.write(f"{m_tot[ifilt][i]:.3f} ")
                lc.write("\n")
            lc.close()

        # extract bolometric lightcurves
        if args.doLbol:
            if args.z is not None:
                Lbol_f = open(
                    os.path.join(
                        lcdir, f"{filename[:-5]}_theta{theta[obs]:.2f}_z{z}_Lbol.dat"
                    ),
                    "w",
                )
            else:
                Lbol_f = open(
                    os.path.join(
                        lcdir,
                        f"{filename[:-5]}_theta{theta[obs]:.2f}_dMpc{int(dMpc)}_Lbol.dat",
                    ),
                    "w",
                )
            Lbol_f.write("# t[days] Lbol[erg/s] \n")

            for i, t in enumerate(ph):
                Lbol_f.write(f"{t:.3f} {Lbol[obs][i]:.5e} \n")

            Lbol_f.close()
