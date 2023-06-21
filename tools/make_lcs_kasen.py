#!/usr/bin/env python3

import os
import numpy as np
import sncosmo
import argparse
import h5py
from astropy.cosmology import Planck18 as cosmo


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filters",
        type=str,
        default="bessellux,bessellb,bessellv,bessellr,besselli,sdssu,ps1::g,ps1::r,ps1::i,ps1::z,ps1::y,uvot::b,uvot::u,uvot::uvm2,uvot::uvw1,uvot::uvw2,uvot::v,uvot::white,atlasc,atlaso,2massj,2massh,2massks,ztfg,ztfr,ztfi",
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

filters = args.filters.split(",")
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
    z = H0 * dMpc / CLIGHT
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

    ph = np.array(time) / (60 * 60 * 24)  # get time in days
    wave = np.flipud(CLIGHT_cm_s / nu * 1e8)  # AA

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
            Llam = Lnu * np.flipud(nu) ** 2.0 / CLIGHT_cm_s / 1e8  # ergs/s/Angstrom
            Llam = Llam / (4 * np.pi * D_cm**2)  # ergs / s / cm^2 / Angstrom
            source = sncosmo.TimeSeriesSource(ph, wave, Llam)
            m = source.bandmag(filt, "ab", ph)
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

        for i, t in enumerate(ph):
            if t < 0:
                continue
            Lbol_f.write(f"{t:.3f} {Lbol[i]:.5e} \n")

        Lbol_f.close()
