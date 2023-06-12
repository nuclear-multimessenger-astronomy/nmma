import os
import glob
import argparse
import numpy as np
import healpy as hp

import bilby
from astropy.cosmology import LambdaCDM
import h5py
from gwpy.table import Table
from astropy import time
from scipy.interpolate import interpolate as interp
import matplotlib.pyplot as plt

from ligo.skymap.io import read_sky_map
from ligo.skymap import distance, bayestar

from nmma.joint.conversion import MultimessengerConversion, EOS2Parameters
from nmma.em.model import SVDLightCurveModel, GRBLightCurveModel
from nmma.em.model import SupernovaGRBLightCurveModel, KilonovaGRBLightCurveModel
from nmma.em.injection import create_light_curve_data


np.random.seed(0)
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)


def q2eta(q):
    return q / (1 + q) ** 2


def ms2mc(m1, m2):
    eta = m1 * m2 / ((m1 + m2) * (m1 + m2))
    mchirp = ((m1 * m2) ** (3.0 / 5.0)) * ((m1 + m2) ** (-1.0 / 5.0))
    q = m2 / m1

    return (mchirp, eta, q)


def mc2ms(mc, eta):
    """
    Utility function for converting mchirp,eta to component masses. The
    masses are defined so that m1>m2. The rvalue is a tuple (m1,m2).
    """
    root = np.sqrt(0.25 - eta)
    fraction = (0.5 + root) / (0.5 - root)
    invfraction = 1 / fraction

    m2 = mc * np.power((1 + fraction), 0.2) / np.power(fraction, 0.6)

    m1 = mc * np.power(1 + invfraction, 0.2) / np.power(invfraction, 0.6)
    return (m1, m2)


def main():

    parser = argparse.ArgumentParser(
        description="Summary analysis for nmma injection file"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the kilonova model to be used"
    )
    parser.add_argument(
        "--template-file", type=str, help="The template file to be used"
    )
    parser.add_argument("--hdf5-file", type=str, help="The hdf5 file to be used")
    parser.add_argument("--coinc-file", type=str, help="The coinc xml file to be used")
    parser.add_argument("-g", "--gps", type=int, default=1187008882)
    parser.add_argument(
        "-s",
        "--skymap",
        type=str,
    )
    parser.add_argument(
        "--eos-dir",
        type=str,
        required=True,
        help="EOS file directory in (radius [km], mass [solar mass], lambda)",
    )
    parser.add_argument("-e", "--gw170817-eos", type=str)
    parser.add_argument("-o", "--outdir", type=str, default="outdir")
    parser.add_argument("-n", "--Nmarg", type=int, default=100)
    parser.add_argument("--label", type=str, required=True, help="Label for the run")
    parser.add_argument(
        "--svd-path",
        type=str,
        help="Path to the SVD directory, with {model}_mag.pkl and {model}_lbol.pkl",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=0.0,
        help="Days to be started analysing from the trigger time (default: 0)",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=14.0,
        help="Days to be stoped analysing from the trigger time (default: 14)",
    )
    parser.add_argument(
        "--dt", type=float, default=0.1, help="Time step in day (default: 0.1)"
    )
    parser.add_argument(
        "--svd-mag-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for mag evaluation (default: 10)",
    )
    parser.add_argument(
        "--svd-lbol-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for lbol evaluation (default: 10)",
    )
    parser.add_argument(
        "--filters",
        type=str,
        help="A comma seperated list of filters to use (e.g. g,r,i). If none is provided, will use all the filters available",
        default="u,g,r,i,z,y,J,H,K",
    )
    parser.add_argument(
        "--gptype", type=str, help="SVD interpolation scheme.", default="sklearn"
    )
    parser.add_argument(
        "--generation-seed",
        metavar="seed",
        type=int,
        default=42,
        help="Injection generation seed (default: 42)",
    )
    parser.add_argument(
        "--joint-light-curve",
        help="Flag for using both kilonova and GRB afterglow light curve",
        action="store_true",
    )
    parser.add_argument(
        "--with-grb-injection",
        help="If the injection has grb included",
        action="store_true",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False, help="add best fit plot"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="print out log likelihoods",
    )
    parser.add_argument(
        "--injection-detection-limit",
        metavar="mAB",
        type=str,
        default=None,
        help="The highest mAB to be presented in the injection data set, any mAB higher than this will become a non-detection limit. Should be comma delimited list same size as injection set.",
    )

    parser.add_argument(
        "--absolute", action="store_true", default=False, help="Absolute Magnitude"
    )

    args = parser.parse_args()

    bilby.core.utils.setup_logger(outdir=args.outdir, label=args.label)
    bilby.core.utils.check_directory_exists_and_if_not_mkdir(args.outdir)

    # initialize light curve model
    sample_times = np.arange(args.tmin, args.tmax + args.dt, args.dt)

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
                **light_curve_kwargs, gptype=args.gptype
            )

    args.kilonova_tmin = args.tmin
    args.kilonova_tmax = args.tmax
    args.kilonova_tstep = args.dt

    args.kilonova_injection_model = args.model
    args.kilonova_injection_svd = args.svd_path
    args.injection_svd_mag_ncoeff = args.svd_mag_ncoeff
    args.injection_svd_lbol_ncoeff = args.svd_lbol_ncoeff
    args.kilonova_error = 0

    filenames = sorted(glob.glob(os.path.join(args.eos_dir, "*.dat")))
    global Neos
    Neos = len(filenames)
    EOS_GW170817 = np.loadtxt(args.gw170817_eos)

    global EOS_data
    EOS_data, weights = {}, []
    for EOSIdx in range(0, Neos):
        data = np.loadtxt("{0}/{1}.dat".format(args.eos_dir, EOSIdx + 1))
        EOS_data[EOSIdx] = {}
        EOS_data[EOSIdx]["R"] = np.array(data[:, 0])
        EOS_data[EOSIdx]["M"] = np.array(data[:, 1])
        EOS_data[EOSIdx]["Lambda"] = np.array(data[:, 2])
        EOS_data[EOSIdx]["weight"] = EOS_GW170817[EOSIdx]

        EOS_data[EOSIdx]["interp_mass_lambda"] = interp.interp1d(
            EOS_data[EOSIdx]["M"], EOS_data[EOSIdx]["Lambda"]
        )
        EOS_data[EOSIdx]["interp_mass_radius"] = interp.interp1d(
            EOS_data[EOSIdx]["M"], EOS_data[EOSIdx]["R"]
        )

        weights.append(EOS_data[EOSIdx]["weight"])

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
        data_out["eta"] = q2eta(data_out["q"])
        data_out["mchirp"] = data_out["mc"]
        data_out["m1"], data_out["m2"] = mc2ms(data_out["mchirp"], data_out["eta"])
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

        mMax, rMax, lam1, lam2, r1, r2 = EOS2Parameters(
            EOS_data[idx]["interp_mass_radius"],
            EOS_data[idx]["interp_mass_lambda"],
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

        plotName = os.path.join(args.outdir, "lc.pdf")
        fig = plt.figure(figsize=(16, 18))
        filts = args.filters.split(",")
        ncols = 1
        nrows = int(np.ceil(len(filts) / ncols))
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.6, hspace=0.5)

        for ii, filt in enumerate(filts):
            loc_x, loc_y = np.divmod(ii, nrows)
            loc_x, loc_y = int(loc_x), int(loc_y)
            ax = fig.add_subplot(gs[loc_y, loc_x])

            data_out = []
            for jj, key in enumerate(list(mag_ds.keys())):
                data_out.append(mag_ds[key][:, ii + 1])
            data_out = np.vstack(data_out)

            bins = np.linspace(-20, 25, 101)

            def return_hist(x):
                hist, bin_edges = np.histogram(x, bins=bins)
                return hist

            hist = np.apply_along_axis(lambda x: return_hist(x), -1, data_out.T)
            bins = (bins[1:] + bins[:-1]) / 2.0

            X, Y = np.meshgrid(sample_times, bins)
            hist = hist.astype(np.float64)
            hist[hist == 0.0] = np.nan

            ax.pcolormesh(X, Y, hist.T, shading="auto", cmap="viridis", alpha=0.7)

            # plot 10th, 50th, 90th percentiles
            ax.plot(
                sample_times,
                np.nanpercentile(data_out, 50, axis=0),
                c="k",
                linestyle="--",
            )
            ax.plot(sample_times, np.nanpercentile(data_out, 90, axis=0), "k--")
            ax.plot(sample_times, np.nanpercentile(data_out, 10, axis=0), "k--")

            ax.set_xlim([0, 14])
            ax.set_ylabel(filt, fontsize=30, rotation=0, labelpad=14)

            if ii == len(filts) - 1:
                ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

            if args.absolute:
                ax.set_ylim([-12, -18])
                ax.set_yticks([-18, -16, -14, -12])
            else:
                ax.set_ylim([24, 15])
                ax.set_yticks([24, 21, 18, 15])

            ax.tick_params(axis="x", labelsize=30)
            ax.tick_params(axis="y", labelsize=30)
            ax.grid(which="both", alpha=0.5)

        fig.text(0.45, 0.05, "Time [days]", fontsize=30)
        fig.text(
            0.01,
            0.5,
            "Absolute Magnitude",
            va="center",
            rotation="vertical",
            fontsize=30,
        )

        plt.tight_layout()
        plt.savefig(plotName, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
