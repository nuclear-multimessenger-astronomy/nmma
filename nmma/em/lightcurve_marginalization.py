import glob
import os

import bilby
import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy import time
from astropy.cosmology import LambdaCDM
from gwpy.table import Table
from ligo.skymap import bayestar, distance
from ligo.skymap.io import read_sky_map

from nmma.em.injection import create_light_curve_data
from nmma.em.model import (
    GRBLightCurveModel,
    KilonovaGRBLightCurveModel,
    SupernovaGRBLightCurveModel,
    SVDLightCurveModel,
)

from nmma.joint.conversion import (MultimessengerConversion, EOS2Parameters, mass_ratio_to_eta,
                                   chirp_mass_and_eta_to_component_masses)

from ..eos.eos_processing import load_tabulated_macro_eos_set_to_dict
from .em_parsing import lc_marginalisation_parser, parsing_and_logging
from .utils import setup_sample_times

np.random.seed(0)


def ms2mc(m1, m2):
    eta = m1 * m2 / ((m1 + m2) * (m1 + m2))
    mchirp = ((m1 * m2) ** (3.0 / 5.0)) * ((m1 + m2) ** (-1.0 / 5.0))
    q = m2 / m1

    return (mchirp, eta, q)



def main(args=None):
    args = parsing_and_logging(lc_marginalisation_parser, args)

    # initialize light curve model
    sample_times = setup_sample_times(args)

    if args.joint_light_curve:

        assert args.model != "TrPi2018", "TrPi2018 is not a kilonova / supernova model"

        if args.model != "nugent-hyper" or args.model != "salt2":

            kilonova_kwargs = dict(
                model=args.model,
                svd_path=args.svd_path,
                mag_ncoeff=args.svd_mag_ncoeff,
                lbol_ncoeff=args.svd_lbol_ncoeff,
                parameter_conversion=None,
            )

            light_curve_model = KilonovaGRBLightCurveModel(
                sample_times=sample_times,
                kilonova_kwargs=kilonova_kwargs,
                GRB_resolution=args.grb_resolution,
                jetType=args.jet_type,
            )

        else:

            light_curve_model = SupernovaGRBLightCurveModel(
                sample_times=sample_times,
                GRB_resolution=args.grb_resolution,
                SNmodel=args.model,
                jetType=args.jet_type,
            )

    else:
        if args.model == "TrPi2018":
            light_curve_model = GRBLightCurveModel(
                sample_times=sample_times,
                resolution=args.grb_resolution,
                jetType=args.jet_type,
            )

        else:
            light_curve_kwargs = dict(
                model=args.model,
                sample_times=sample_times,
                svd_path=args.svd_path,
                mag_ncoeff=args.svd_mag_ncoeff,
                lbol_ncoeff=args.svd_lbol_ncoeff,
            )
            light_curve_model = SVDLightCurveModel(  # noqa: F841
                **light_curve_kwargs, interpolation_type=args.interpolation_type
            )

    args.kilonova_tmin = args.tmin
    args.kilonova_tmax = args.tmax
    args.kilonova_tstep = args.dt

    args.kilonova_injection_model = args.model
    args.kilonova_injection_svd = args.svd_path
    args.injection_svd_mag_ncoeff = args.svd_mag_ncoeff
    args.injection_svd_lbol_ncoeff = args.svd_lbol_ncoeff
    args.kilonova_error = 0

    EOS_data, weights, Neos = load_tabulated_macro_eos_set_to_dict(args.eos_dir, args.eos_weights)

    if args.template_file is not None:
        try:
            names = ["SNRdiff", "erf", "weight", "m1", "m2", "a1", "a2", "dist"]
            data_out = Table.read(args.template_file, names=names, format="ascii")
        except Exception:
            names = ["SNRdiff", "erf", "weight", "m1", "m2", "dist"]
            data_out = Table.read(args.template_file, names=names, format="ascii")

            data_out["a1"] = 0.0
            data_out["a2"] = 0.0

        data_out["theta_jn"], data_out["tilt1"], data_out["tilt2"] = 0.0, 0.0, 0.0

        data_out["mchirp"], data_out["eta"], data_out["q"] = ms2mc(
            data_out["m1"], data_out["m2"]
        )

        data_out["chi_eff"] = (
            data_out["m1"] * data_out["a1"] + data_out["m2"] * data_out["a2"]
        ) / (data_out["m1"] + data_out["m2"])
    elif args.hdf5_file is not None:
        f = h5py.File(args.hdf5_file, "r")
        posterior = f["lalinference"]["lalinference_mcmc"]["posterior_samples"][()]

        data_out = Table(posterior)
        data_out["eta"] = mass_ratio_to_eta(data_out["q"])
        data_out["mchirp"] = data_out["mc"]
        data_out["m1"], data_out["m2"] = chirp_mass_and_eta_to_component_masses(data_out["mchirp"], data_out["eta"])
        data_out["weight"] = 1.0 / len(data_out["m1"])

        data_out["chi_eff"] = (
            data_out["m1"] * data_out["a1"] + data_out["m2"] * data_out["a2"]
        ) / (data_out["m1"] + data_out["m2"])

        args.gps = np.median(data_out["t0"])

    elif args.coinc_file is not None:
        data_out = Table.read(
            args.coinc_file, format="ligolw", tablename="sngl_inspiral"
        )
        data_out["m1"], data_out["m2"] = data_out["mass1"], data_out["mass2"]
        data_out["mchirp"], data_out["eta"], data_out["q"] = ms2mc(
            data_out["m1"], data_out["m2"]
        )
        data_out["weight"] = 1.0 / len(data_out["m1"])
        data_out["theta_jn"], data_out["tilt1"], data_out["tilt2"] = 0.0, 0.0, 0.0
        data_out["a1"], data_out["a2"] = data_out["spin1z"], data_out["spin2z"]

        data_out["chi_eff"] = (
            data_out["m1"] * data_out["a1"] + data_out["m2"] * data_out["a2"]
        ) / (data_out["m1"] + data_out["m2"])

        skymap = read_sky_map(args.skymap, moc=True, distances=True)
        order = hp.nside2order(512)
        skymap = bayestar.rasterize(skymap, order)
        dist_mean, dist_std = distance.parameters_to_marginal_moments(
            skymap["PROB"], skymap["DISTMU"], skymap["DISTSIGMA"]
        )
        data_out["dist"] = dist_mean + np.random.randn(len(data_out["m1"])) * dist_std

    else:
        print("Needs template_file, hdf5_file, or coinc_file")
        exit(1)

    idxs = np.random.choice(np.arange(len(weights)), args.Nmarg, p=weights)
    idys = np.random.choice(
        np.arange(len(data_out["m1"])),
        args.Nmarg,
        p=data_out["weight"] / np.sum(data_out["weight"]),
    )

    mag_ds, matter = {}, {}
    for ii in range(args.Nmarg):

        outdir = os.path.join(args.outdir, "%d" % ii)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        injection_outfile = os.path.join(outdir, "lc.dat")
        matter_outfile = os.path.join(outdir, "matter.dat")
        if os.path.isfile(injection_outfile) and os.path.isfile(matter_outfile):
            mag_ds[ii] = np.loadtxt(injection_outfile)
            matter[ii] = np.loadtxt(matter_outfile)
            continue

        idx, idy = int(idxs[ii]), int(idys[ii])
        mchirp = data_out["mchirp"][idy]
        m1, m2 = data_out["m1"][idy], data_out["m2"][idy]
        dist = data_out["dist"][idy]
        a1, a2 = data_out["a1"][idy], data_out["a2"][idy]
        theta_jn = data_out["theta_jn"][idy]
        tilt1, tilt2 = data_out["tilt1"][idy], data_out["tilt2"][idy]

        mMax, rMax, lam1, lam2, r1, r2, R_14, R_16 = EOS2Parameters(
            EOS_data[idx]["M"],
            EOS_data[idx]["R"],
            EOS_data[idx]["Lambda"],
            m1,
            m2,
        )

        params = {
            "luminosity_distance": dist,
            "chirp_mass": mchirp,
            "ratio_epsilon": 1e-20,
            "theta_jn": theta_jn,
            "a_1": a1,
            "a_2": a2,
            "mass_1": m1,
            "mass_2": m2,
            "EOS": idx,
            "cos_tilt_1": np.cos(tilt1),
            "cos_tilt_2": np.cos(tilt2),
            "KNphi": 30,
        }

        if (m1 < mMax) and (m2 < mMax):
            binary_type = "BNS"

            alpha_min, alpha_max = 1e-2, 2e-2
            log10zeta_min, log10zeta_max = -3, 0
            alpha = np.random.uniform(alpha_min, alpha_max)
            zeta = 10 ** np.random.uniform(log10zeta_min, log10zeta_max)
        elif (m1 > mMax) and (m2 < mMax):
            binary_type = "NSBH"

            log10_alpha_min, log10_alpha_max = -3, -1
            log10zeta_min, log10zeta_max = -3, 0
            log10_alpha = np.random.uniform(log10_alpha_min, log10_alpha_max)
            zeta = 10 ** np.random.uniform(log10zeta_min, log10zeta_max)
        else:
            binary_type = "BBH"

        conversion = MultimessengerConversion(args.eos_dir, Neos, binary_type)
        parameter_conversion = conversion.convert_to_multimessenger_parameters

        if binary_type == "BNS":
            params = {
                **params,
                "alpha": alpha,
                "ratio_zeta": zeta,
            }
        elif binary_type == "NSBH":
            params = {
                **params,
                "log10_alpha": log10_alpha,
                "ratio_zeta": zeta,
            }
        else:
            fid = open(matter_outfile, "w")
            fid.write("0 0 0\n")
            fid.close()

            fid = open(injection_outfile, "w")
            fid.write("# t[days] ")
            fid.write(str(" ".join(args.filters.split(","))))
            fid.write("\n")
            for ii, tt in enumerate(sample_times):
                fid.write("%.5f " % sample_times[ii])
                for filt in args.filters.split(","):
                    fid.write("%.3f " % 99.9)
                fid.write("\n")
            fid.close()

        complete_parameters, _ = parameter_conversion(params)

        fid = open(matter_outfile, "w")
        fid.write(
            "1 %.5f %.5f\n"
            % (
                complete_parameters["log10_mej_dyn"],
                complete_parameters["log10_mej_wind"],
            )
        )
        fid.close()

        tc_gps = time.Time(args.gps, format="gps")
        trigger_time = tc_gps.mjd

        complete_parameters["kilonova_trigger_time"] = trigger_time

        data = create_light_curve_data(
            complete_parameters, args, doAbsolute=args.absolute
        )
        print("Injection generated")

        fid = open(injection_outfile, "w")
        fid.write("# t[days] ")
        fid.write(str(" ".join(args.filters.split(","))))
        fid.write("\n")
        for ii, tt in enumerate(sample_times):
            fid.write("%.5f " % sample_times[ii])
            for filt in data.keys():
                if args.filters:
                    if filt not in args.filters.split(","):
                        continue
                fid.write("%.3f " % data[filt][ii, 1])
            fid.write("\n")
        fid.close()

        mag_ds[ii] = np.loadtxt(injection_outfile)
        matter[ii] = np.loadtxt(matter_outfile)

    if args.plot:
        NS, dyn, wind = [], [], []
        for jj, key in enumerate(list(matter.keys())):
            NS.append(matter[key][0])
            dyn.append(matter[key][1])
            wind.append(matter[key][2])

        print("Fraction of samples with NS: %.5f" % (np.sum(NS) / len(NS)))

        bins = np.linspace(-3, 0, 25)
        dyn_hist, bin_edges = np.histogram(dyn, bins=bins, density=True)
        wind_hist, bin_edges = np.histogram(wind, bins=bins, density=True)
        bins = (bin_edges[1:] + bin_edges[:-1]) / 2.0

        plotName = os.path.join(args.outdir, "matter.pdf")
        fig = plt.figure(figsize=(10, 6))
        plt.step(bins, dyn_hist, "k--", label="Dynamical")
        plt.step(bins, wind_hist, "b-", label="Wind")
        plt.xlabel(r"log10(Ejecta Mass / $M_\odot$)")
        plt.ylabel("Probability Density Function")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plotName, bbox_inches="tight")
        plt.close()

        from .plotting_utils import lc_plot
        filters = args.filters.split(",")
        plotpath= os.path.join(args.outdir, "lc.pdf")
        plot_dict = {filt: np.vstack([lc_data[:, i+1] for lc_data in mag_ds.values()]) for i, filt in enumerate(filters)}
        if args.absolute:
            ylim = [-12, -18]
        else:
            ylim = [24, 15]
        lc_plot(filters, plot_dict, sample_times, plotpath= plotpath, ylim = ylim, n_yticks=4, ylabel_kwargs = dict(fontsize=30, rotation=0, labelpad=14))


if __name__ == "__main__":
    main()
