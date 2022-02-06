import os
import glob
import pickle
import argparse
import copy
import json
import random
import pandas as pd
import numpy as np

import pymultinest
from ligo.skymap.io import read_sky_map
from ligo.skymap.postprocess import crossmatch

from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde
from astropy.cosmology import LambdaCDM

from astropy.coordinates import Distance, SkyCoord
import astropy.units as u

import corner
import scipy.stats as ss
from scipy.interpolate import interpolate as interp

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

import lal

from nmma.joint.conversion import MultimessengerConversion, EOS2Parameters

np.random.seed(0)
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)


def q2eta(q):
    return q / (1 + q) ** 2


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


def ms2mc(m1, m2):
    eta = m1 * m2 / ((m1 + m2) * (m1 + m2))
    mchirp = ((m1 * m2) ** (3.0 / 5.0)) * ((m1 + m2) ** (-1.0 / 5.0))
    q = m2 / m1

    return (mchirp, eta, q)


def greedy_kde_areas_2d(pts):

    pts = np.random.permutation(pts)

    mu = np.mean(pts, axis=0)
    cov = np.cov(pts, rowvar=0)

    L = np.linalg.cholesky(cov)

    pts = np.linalg.solve(L, (pts - mu).T).T

    Npts = pts.shape[0]
    kde_pts = pts[: int(Npts / 2), :]

    kde = ss.gaussian_kde(kde_pts.T)

    kdedir = {}
    kdedir["kde"] = kde
    kdedir["mu"] = mu
    kdedir["L"] = L

    return kdedir


def kde_eval(kdedir, truth):

    kde = kdedir["kde"]
    mu = kdedir["mu"]
    L = kdedir["L"]

    truth = np.linalg.solve(L, truth - mu)
    td = kde(truth)

    return td


def greedy_kde_areas_1d(pts):

    pts = np.random.permutation(pts)
    mu = np.mean(pts, axis=0)

    Npts = pts.shape[0]
    kde_pts = pts[: int(Npts / 2)]

    kde = ss.gaussian_kde(kde_pts.T)

    kdedir = {}
    kdedir["kde"] = kde
    kdedir["mu"] = mu

    return kdedir


def kde_eval_single(kdedir, truth):

    kde = kdedir["kde"]
    td = kde(truth)

    return td


def prior_H0(cube, ndim, nparams):
    cube[0] = cube[0] * 200.0
    cube[1] = cube[1] * 2000.0


def loglike_H0(cube, ndim, nparams):
    H0 = cube[0]
    d = cube[1]

    c = 299792458.0 * 1e-3
    vr_mean, vr_std = z * c, zerr * c  # noqa: F821
    pvr = (1 / np.sqrt(2 * np.pi * vr_std ** 2)) * np.exp(
        (-1 / 2.0) * ((vr_mean - H0 * d) / vr_std) ** 2
    )
    prob_dist = kde_eval_single(kdedir_dist, [d])[0]  # noqa: F821
    # print(H0, d, vp, np.log(pvr), np.log(pvp), np.log(prob_dist))

    prob = np.log(pvr) + np.log(prob_dist)

    if np.isnan(prob):
        prob = -np.inf

    return prob


def prior_EOS_BNS(cube, ndim, nparams):
    cube[0] = cube[0] * 1.0 + 1.0
    cube[1] = cube[1] * 5000
    cube[2] = cube[2] * 2 * 1e-2 - 1e-2
    cube[3] = cube[3] * 3 - 3


def loglike_EOS_BNS(cube, ndim, nparams):

    q = cube[0]
    eos = np.floor(cube[1])
    alpha = cube[2]
    zeta = 10 ** cube[3]

    params = copy.deepcopy(default_parameters)
    eta = q2eta(q)
    (m1, m2) = mc2ms(params["chirp_mass"], eta)

    params = {
        **params,
        "mass_1": m1,
        "mass_2": m2,
        "EOS": eos,
        "alpha": alpha,
        "ratio_zeta": zeta,
    }
    complete_parameters, _ = parameter_conversion(params)

    vals = np.array(
        [complete_parameters["log10_mej_dyn"], complete_parameters["log10_mej_wind"]]
    ).T
    kdeeval = kde_eval(kdedir, vals)[0]
    prob = np.log(kdeeval)

    return prob


def prior_EOS_NSBH(cube, ndim, nparams):
    cube[0] = cube[0] * 10.0 + 1.0
    cube[1] = cube[1] * 5000
    cube[2] = cube[2] * 2 - 3
    cube[3] = cube[3] * 3 - 3


def loglike_EOS_NSBH(cube, ndim, nparams):

    q = cube[0]
    eos = np.floor(cube[1])
    log10_alpha = cube[2]
    zeta = 10 ** cube[3]

    params = copy.deepcopy(default_parameters)
    eta = q2eta(q)
    (m1, m2) = mc2ms(params["chirp_mass"], eta)

    params = {
        **params,
        "mass_1": m1,
        "mass_2": m2,
        "EOS": eos,
        "log10_alpha": log10_alpha,
        "ratio_zeta": zeta,
    }
    complete_parameters, _ = parameter_conversion(params)

    if not np.isfinite(complete_parameters["log10_mej_dyn"]):
        return -np.inf
    if not np.isfinite(complete_parameters["log10_mej_wind"]):
        return -np.inf

    vals = np.array(
        [complete_parameters["log10_mej_dyn"], complete_parameters["log10_mej_wind"]]
    ).T
    kdeeval = kde_eval(kdedir, vals)[0]
    prob = np.log(kdeeval)

    return prob


def prior_R14(cube, ndim, nparams):
    cube[0] = cube[0] * 5000


def loglike_R14(cube, ndim, nparams):

    eos = np.floor(cube[0])
    if np.isnan(eos):
        return -np.inf

    R14 = EOS_data[eos]["R14"]
    weight = EOS_data[eos]["weight"]

    kdeeval = kde_eval_single(kdedir_R14_GW, [R14])[0]
    prob = np.log(kdeeval) + np.log(weight)

    return prob


def get_EOS_weight(EOS):

    EOS = np.floor(EOS)
    Nsamples = len(EOS)
    EOSIdxSamples = EOS.astype(int) + 1
    uniqueEOSIdx = np.unique(EOSIdxSamples)
    counts = {}
    for idx in uniqueEOSIdx:
        counts[idx] = len(np.where(EOSIdxSamples == idx)[0])

    weight = []
    for i in range(0, Neos):
        if i in counts:
            weight.append(counts[i] / float(Nsamples))
        else:
            weight.append(0.0)
    weight = np.array(weight)
    weight /= np.sum(weight)

    weight_sort = np.argsort(weight)
    weight_cumsum = np.cumsum(weight[weight_sort])
    fivepercentIdx = weight_sort[np.where(weight_cumsum >= 0.05)[0]]

    return weight, fivepercentIdx


def compute_constraint(bins, prob):
    prob_cumsum = np.cumsum(prob)
    idx = np.argmin(np.abs(prob_cumsum - 0.16))
    H0_16 = bins[idx]
    idx = np.argmin(np.abs(prob_cumsum - 0.50))
    H0_50 = bins[idx]
    idx = np.argmin(np.abs(prob_cumsum - 0.84))
    H0_84 = bins[idx]

    return H0_16, H0_50, H0_84


def main():

    parser = argparse.ArgumentParser(
        description="Summary analysis for nmma injection file"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the kilonova model to be used"
    )
    parser.add_argument(
        "--injection-file",
        type=str,
        required=True,
        help="The bilby injection json file to be used",
    )
    parser.add_argument(
        "--eos-file",
        type=str,
        required=True,
        help="EOS file in (radius [km], mass [solar mass], lambda)",
    )
    parser.add_argument(
        "--eos-dir",
        type=str,
        required=True,
        help="EOS file directory in (radius [km], mass [solar mass], lambda)",
    )
    parser.add_argument(
        "--skymap-dir",
        type=str,
        required=True,
        help="skymap file directory with Bayestar skymaps",
    )
    parser.add_argument(
        "--binary-type", type=str, required=True, help="Either BNS or NSBH"
    )
    parser.add_argument("-i", "--indices-file", type=str)
    parser.add_argument("-g", "--gw170817-h0", type=str)
    parser.add_argument("-e", "--gw170817-eos", type=str)
    parser.add_argument("-d", "--detections-file", type=str)
    parser.add_argument("-o", "--outdir", type=str, default="outdir")
    parser.add_argument(
        "--H0", action="store_true", default=False, help="do H0 analysis"
    )
    parser.add_argument(
        "--eos", action="store_true", default=False, help="do EOS analysis"
    )
    parser.add_argument(
        "--ejecta", action="store_true", default=False, help="do ejecta analysis"
    )

    args = parser.parse_args()

    # load the injection json file
    if args.injection_file:
        if args.injection_file.endswith(".json"):
            with open(args.injection_file, "rb") as f:
                injection_data = json.load(f)
                datadict = injection_data["injections"]["content"]
                dataframe_from_inj = pd.DataFrame.from_dict(datadict)
        else:
            print("Only json supported.")
            exit(1)

    if len(dataframe_from_inj) > 0:
        args.n_injection = len(dataframe_from_inj)

    indices = np.loadtxt(args.indices_file)
    if args.detections_file:
        detections = np.loadtxt(args.detections_file)

    c = 299792458.0 * 1e-3

    radius, mass, Lambda = np.loadtxt(args.eos_file, unpack=True, usecols=[0, 1, 2])
    interp_mass_lambda = interp.interp1d(mass, Lambda)
    interp_mass_radius = interp.interp1d(mass, radius)
    R14_true = interp_mass_radius(1.4)

    n_live_points = 1000
    evidence_tolerance = 0.5
    title_fontsize = 26
    label_fontsize = 30

    filenames = sorted(glob.glob(os.path.join(args.eos_dir, "*.dat")))
    global Neos
    Neos = len(filenames)
    conversion = MultimessengerConversion(args.eos_dir, Neos, args.binary_type)

    EOS_GW170817 = np.loadtxt(args.gw170817_eos)

    global EOS_data
    EOS_data, R14, R14_weights = {}, [], []
    for EOSIdx in range(0, Neos):
        data = np.loadtxt("{0}/{1}.dat".format(args.eos_dir, EOSIdx + 1))
        EOS_data[EOSIdx] = {}
        EOS_data[EOSIdx]["R"] = np.array(data[:, 0])
        EOS_data[EOSIdx]["M"] = np.array(data[:, 1])
        EOS_data[EOSIdx]["R14"] = interp.interp1d(
            EOS_data[EOSIdx]["M"], EOS_data[EOSIdx]["R"]
        )(1.4)
        EOS_data[EOSIdx]["weight"] = EOS_GW170817[EOSIdx]

        R14.append(EOS_data[EOSIdx]["R14"])
        R14_weights.append(EOS_data[EOSIdx]["weight"])

    R14_GW170817 = np.random.choice(R14, 10000, p=R14_weights)

    global parameter_conversion
    parameter_conversion = conversion.convert_to_multimessenger_parameters

    color1, color2, color3 = "cornflowerblue", "coral", "palegreen"

    pcklFile = "%s/summary.pkl" % (args.outdir)
    if not os.path.isfile(pcklFile):
        data_out = {}
        for index, row in dataframe_from_inj.iterrows():
            print("Running object %d/%d" % (index, len(dataframe_from_inj)))

            # if index > 100: continue

            # if log10_mej_wind < -2:
            #    print('Small ejecta in %d... continuing' % index)
            #    continue

            outdir = os.path.join(args.outdir, str(index))
            posterior_file = os.path.join(
                outdir, "injection_" + args.model + "_posterior_samples.dat"
            )
            if not os.path.isfile(posterior_file):
                continue
            posterior_samples = pd.read_csv(posterior_file, header=0, delimiter=" ")

            if args.H0:
                skymap_file = os.path.join(args.skymap_dir, "%d.fits" % indices[index])
                skymap = read_sky_map(skymap_file, moc=True, distances=True)

                ra, dec = row["ra"] * 360.0 / (2 * np.pi), row["dec"] * 360.0 / (
                    2 * np.pi
                )

                test_distances = np.linspace(1.400, 600) * u.Mpc
                results = crossmatch(
                    skymap, SkyCoord(ra * u.deg, dec * u.deg, test_distances)
                )
                probdensity_vol = results.probdensity_vol

            lc_file = os.path.join(outdir, "lc.csv")
            lc_data = np.genfromtxt(
                lc_file, dtype=None, delimiter=",", skip_header=1, encoding=None
            )
            check_mags = [np.isclose(row[2], 99.0) for row in lc_data]
            if all(check_mags):
                print("No detections in %d... continuing" % index)
                continue

            dist = Distance(value=row["luminosity_distance"], unit=u.Mpc)

            data_out[index] = {}
            data_out[index]["dist"] = row["luminosity_distance"]
            data_out[index]["z"] = dist.compute_z(cosmology=cosmo)
            data_out[index]["zerr"] = 0.001
            data_out[index]["distance"] = posterior_samples[
                "luminosity_distance"
            ].to_numpy()

            data_out[index]["log10_mej_wind_true"] = row["log10_mej_wind"]
            data_out[index]["log10_mej_dyn_true"] = row["log10_mej_dyn"]
            data_out[index]["log10_mej_wind"] = posterior_samples["log10_mej_wind"]
            data_out[index]["log10_mej_dyn"] = posterior_samples["log10_mej_dyn"]

            if args.eos:

                global kdedir
                pts = np.vstack(
                    (
                        posterior_samples["log10_mej_dyn"],
                        posterior_samples["log10_mej_wind"],
                    )
                ).T
                kdedir = greedy_kde_areas_2d(pts)

                EOSDir = os.path.join(outdir, "EOS")
                if not os.path.isdir(EOSDir):
                    os.makedirs(EOSDir)

                mMax, rMax, lam1, lam2, r1, r2 = EOS2Parameters(
                    interp_mass_radius,
                    interp_mass_lambda,
                    row["mass_1_source"],
                    row["mass_2_source"],
                )

                global default_parameters
                (mchirp, eta, q) = ms2mc(row["mass_1_source"], row["mass_2_source"])

                default_parameters = {
                    "luminosity_distance": row["luminosity_distance"],
                    "chirp_mass": mchirp,
                    "ratio_epsilon": 1e-20,
                    "theta_jn": row["theta_jn"],
                    "redshift": data_out[index]["z"],
                    "a_1": row["a_1"],
                    "a_2": row["a_2"],
                }

                if args.binary_type == "NSBH":
                    default_parameters["cos_tilt_1"] = row["cos_tilt_1"]
                    default_parameters["cos_tilt_2"] = row["cos_tilt_2"]

                parameters = ["q", "EOS", "alpha", "zeta"]
                labels = [r"$q$", r"EOS", r"$\log_{10} \alpha$", r"$\zeta$"]
                n_params = len(parameters)

                multifile = "%s/2-post_equal_weights.dat" % EOSDir
                if not os.path.isfile(multifile):
                    if args.binary_type == "BNS":
                        pymultinest.run(
                            loglike_EOS_BNS,
                            prior_EOS_BNS,
                            n_params,
                            importance_nested_sampling=False,
                            resume=True,
                            verbose=True,
                            sampling_efficiency="parameter",
                            n_live_points=n_live_points,
                            outputfiles_basename="%s/2-" % EOSDir,
                            evidence_tolerance=evidence_tolerance,
                            multimodal=False,
                        )
                    elif args.binary_type == "NSBH":
                        pymultinest.run(
                            loglike_EOS_NSBH,
                            prior_EOS_NSBH,
                            n_params,
                            importance_nested_sampling=False,
                            resume=True,
                            verbose=True,
                            sampling_efficiency="parameter",
                            n_live_points=n_live_points,
                            outputfiles_basename="%s/2-" % EOSDir,
                            evidence_tolerance=evidence_tolerance,
                            multimodal=False,
                        )

                mass_1_source = row["mass_1_source"]
                mass_2_source = row["mass_2_source"]

                mass_ratio = mass_2_source / mass_1_source
                data_out[index]["mass_ratio"] = mass_ratio

                lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(
                    row["lambda_1"], row["lambda_2"], mass_1_source, mass_2_source
                )
                R14_approx = (
                    mchirp
                    * np.power(lambda_tilde / 0.0042, 1.0 / 6.0)
                    * lal.MRSUN_SI
                    / 1e3
                )
                R14_diff = R14_approx - R14_true

                data_out[index]["lambda_tilde"] = lambda_tilde

                data = np.loadtxt(multifile)

                data_out[index]["log10_alpha_true"] = row["log10_alpha"]
                data_out[index]["ratio_zeta_true"] = row["ratio_zeta"]
                data_out[index]["log10_alpha"] = data[:, 2]
                data_out[index]["ratio_zeta"] = data[:, 3]

                m_TOV, R14_EM = np.zeros(len(data)), np.zeros(len(data))
                lambda_tilde_EM = np.zeros(len(data))
                for ii, (q, eos, alpha, zeta, logl) in enumerate(data):
                    zeta = 10 ** zeta
                    eos = np.floor(eos)

                    params = copy.deepcopy(default_parameters)
                    eta = q2eta(q)
                    (m1, m2) = mc2ms(params["chirp_mass"], eta)

                    if args.binary_type == "BNS":
                        params = {
                            **params,
                            "mass_1": m1,
                            "mass_2": m2,
                            "EOS": eos,
                            "alpha": alpha,
                            "ratio_zeta": zeta,
                        }
                    elif args.binary_type == "NSBH":
                        params = {
                            **params,
                            "mass_1": m1,
                            "mass_2": m2,
                            "EOS": eos,
                            "log10_alpha": alpha,
                            "ratio_zeta": zeta,
                        }
                    complete_parameters, _ = parameter_conversion(params)
                    m_TOV[ii] = complete_parameters["TOV_mass"]
                    R14_EM[ii] = complete_parameters["R_16"] + R14_diff

                    lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(
                        complete_parameters["lambda_1"],
                        complete_parameters["lambda_2"],
                        complete_parameters["mass_1_source"],
                        complete_parameters["mass_2_source"],
                    )
                    lambda_tilde_EM[ii] = lambda_tilde
                data_out[index]["lambda_tilde_EM"] = lambda_tilde_EM

                data_out[index]["R14_EM"] = R14_EM
                kdedir_R14_EM = greedy_kde_areas_1d(R14_EM)
                data_out[index]["kdedir_R14_EM"] = kdedir_R14_EM
                data_out[index]["R14"] = R14_true

                q_EM, EOS_EM, alpha, zeta = (
                    data[:, 0],
                    data[:, 1],
                    data[:, 2],
                    data[:, 3],
                )
                q_EM = 1 / q_EM

                data_out[index]["q_EM"] = q_EM

                myclip_a, myclip_b = 0, 5000
                my_mean, my_std = data_out[index]["lambda_tilde"], 200.0
                a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
                lambda_tilde_GW = ss.truncnorm.rvs(
                    a, b, loc=my_mean, scale=my_std, size=1000
                )
                data_out[index]["lambda_tilde_GW"] = lambda_tilde_GW

                R14_GW = (
                    mchirp
                    * np.power(lambda_tilde_GW / 0.0042, 1.0 / 6.0)
                    * lal.MRSUN_SI
                    / 1e3
                )
                data_out[index]["R14_GW"] = R14_GW + R14_diff
                global kdedir_R14_GW
                kdedir_R14_GW = greedy_kde_areas_1d(data_out[index]["R14_GW"])
                data_out[index]["kdedir_R14_GW"] = kdedir_R14_GW

                bins = np.linspace(0, 5000, 201)
                lambda_tilde_EM_hist, bin_edges = np.histogram(
                    data_out[index]["lambda_tilde_EM"], bins=bins, density=True
                )
                bins = (bin_edges[1:] + bin_edges[:-1]) / 2.0
                bins = np.linspace(0, 5000, 201)
                lambda_tilde_GW_hist, bin_edges = np.histogram(
                    data_out[index]["lambda_tilde_GW"], bins=bins, density=True
                )
                bins = (bin_edges[1:] + bin_edges[:-1]) / 2.0

                kdedir_lambda_tilde_EM = greedy_kde_areas_1d(
                    data_out[index]["lambda_tilde_EM"]
                )
                kdedir_lambda_tilde_GW = greedy_kde_areas_1d(
                    data_out[index]["lambda_tilde_GW"]
                )

                data_out[index]["kdedir_lambda_tilde_EM"] = kdedir_lambda_tilde_EM
                data_out[index]["kdedir_lambda_tilde_GW"] = kdedir_lambda_tilde_GW

                max_lambda_tilde = np.max(
                    [np.max(lambda_tilde_EM_hist), np.max(lambda_tilde_GW_hist)]
                )

                plotName = "%s/lambda_tilde.png" % (EOSDir)
                if not os.path.isfile(plotName):
                    fig = plt.figure()
                    ax = plt.gca()
                    ax.set_ylabel("Probability Density Function")
                    ax.set_xlabel("Lambda Tilde")
                    ax.set_xlim([0, 5000])
                    ax.plot(
                        bins,
                        lambda_tilde_EM_hist,
                        color=color1,
                        alpha=1,
                        linestyle="-",
                        linewidth=3,
                        zorder=10,
                    )
                    ax.plot(
                        bins,
                        lambda_tilde_GW_hist,
                        color=color2,
                        alpha=1,
                        linestyle="-",
                        linewidth=3,
                        zorder=0,
                    )
                    ax.plot(
                        [
                            data_out[index]["lambda_tilde"],
                            data_out[index]["lambda_tilde"],
                        ],
                        [0, max_lambda_tilde],
                        color="k",
                        linestyle="--",
                    )
                    plt.savefig(plotName)
                    plt.close()

                labels = [r"$q$", r"$\alpha$", r"$\log_{10} \zeta$", r"$R_{\rm 1.4}$"]
                data = np.vstack((q_EM, alpha, zeta, R14_EM)).T

                if args.binary_type == "BNS":
                    truths = [
                        data_out[index]["mass_ratio"],
                        row["alpha"],
                        np.log10(row["ratio_zeta"]),
                        R14_true,
                    ]
                elif args.binary_type == "NSBH":
                    truths = [
                        data_out[index]["mass_ratio"],
                        row["log10_alpha"],
                        np.log10(row["ratio_zeta"]),
                        R14_true,
                    ]

                plotName = "%s/corner.png" % (EOSDir)
                if not os.path.isfile(plotName):
                    figure = corner.corner(
                        data,
                        labels=labels,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": title_fontsize},
                        label_kwargs={"fontsize": label_fontsize},
                        title_fmt=".3f",
                        smooth=3,
                        truths=truths,
                    )
                    figure.set_size_inches(18.0, 18.0)
                    plt.savefig(plotName)
                    plt.close()

                labels = [r"$R_{\rm 1.4}$"]
                n_params = len(labels)
                multifile = "%s/3-post_equal_weights.dat" % EOSDir
                if not os.path.isfile(multifile):
                    pymultinest.run(
                        loglike_R14,
                        prior_R14,
                        n_params,
                        importance_nested_sampling=False,
                        resume=True,
                        verbose=True,
                        sampling_efficiency="parameter",
                        n_live_points=n_live_points,
                        outputfiles_basename="%s/3-" % EOSDir,
                        evidence_tolerance=evidence_tolerance,
                        multimodal=False,
                    )
                data = np.loadtxt(multifile)
                EOS_GW = data[:, 0]

                myclip_a, myclip_b = 0, 1
                my_mean, my_std = (
                    data_out[index]["mass_ratio"],
                    0.15 * data_out[index]["mass_ratio"],
                )
                a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
                q_GW = ss.truncnorm.rvs(
                    a, b, loc=my_mean, scale=my_std, size=len(EOS_GW)
                )
                data_out[index]["q_GW"] = q_GW

                bins = np.linspace(0.1, 1, 31)
                q_EM_hist, bin_edges = np.histogram(
                    data_out[index]["q_EM"], bins=bins, density=True
                )
                bins = (bin_edges[1:] + bin_edges[:-1]) / 2.0
                bins = np.linspace(0.1, 1, 31)
                q_GW_hist, bin_edges = np.histogram(
                    data_out[index]["q_GW"], bins=bins, density=True
                )
                bins = (bin_edges[1:] + bin_edges[:-1]) / 2.0

                kdedir_q_EM = greedy_kde_areas_1d(data_out[index]["q_EM"])
                kdedir_q_GW = greedy_kde_areas_1d(data_out[index]["q_GW"])

                data_out[index]["kdedir_q_EM"] = kdedir_q_EM
                data_out[index]["kdedir_q_GW"] = kdedir_q_GW

                maxq = np.max([np.max(q_EM_hist), np.max(q_GW_hist)])

                plotName = "%s/mass_ratio.png" % (EOSDir)
                if not os.path.isfile(plotName):
                    fig = plt.figure()
                    ax = plt.gca()
                    ax.set_ylabel("Probability Density Function")
                    ax.set_xlabel("Mass Ratio")
                    ax.set_xlim([0.1, 1])
                    ax.plot(
                        bins,
                        q_EM_hist,
                        color=color1,
                        alpha=1,
                        linestyle="-",
                        linewidth=3,
                        zorder=10,
                    )
                    ax.plot(
                        bins,
                        q_GW_hist,
                        color=color2,
                        alpha=1,
                        linestyle="-",
                        linewidth=3,
                        zorder=0,
                    )
                    ax.plot(
                        [data_out[index]["mass_ratio"], data_out[index]["mass_ratio"]],
                        [0, maxq],
                        color="k",
                        linestyle="--",
                    )
                    plt.savefig(plotName)
                    plt.close()

                weight_GW, ninetyFivepercentIdx_GW = get_EOS_weight(EOS_GW)
                print("%d/%d GW EOS included" % (len(ninetyFivepercentIdx_GW), Neos))
                weight_EM, ninetyFivepercentIdx_EM = get_EOS_weight(EOS_EM)
                print("%d/%d EM EOS included" % (len(ninetyFivepercentIdx_EM), Neos))

                weight = weight_GW + weight_EM
                weight = weight / np.sum(weight)
                weight_sort = np.argsort(weight)
                weight_cumsum = np.cumsum(weight[weight_sort])
                ninetyFivepercentIdx_EMGW = weight_sort[
                    np.where(weight_cumsum >= 0.05)[0]
                ]
                print(
                    "%d/%d EM-GW EOS included" % (len(ninetyFivepercentIdx_EMGW), Neos)
                )

                data_out[index]["EOS_weight_GW"] = weight_GW
                data_out[index]["EOS_weight_EM"] = weight_EM
                data_out[index]["EOS_weight_EMGW"] = weight

                plotName = "%s/MR.png" % (EOSDir)
                if not os.path.isfile(plotName):
                    fig = plt.figure()
                    ax = plt.gca()
                    ax.set_ylabel(r"$M\ [\\rm M_\\odot]$")
                    ax.set_xlabel(r"$R\ [\\rm km]$")
                    ax.set_xlim([7.95, 15])
                    ax.set_ylim([0.8, 3.6])
                    for EOSIdx in range(0, Neos):
                        R, M = EOS_data[EOSIdx]["R"], EOS_data[EOSIdx]["M"]
                        if EOSIdx in ninetyFivepercentIdx_EMGW:
                            ax.plot(
                                R,
                                M,
                                color=color1,
                                alpha=0.20,
                                linestyle="-",
                                linewidth=0.5,
                                zorder=10,
                            )
                        elif EOSIdx in ninetyFivepercentIdx_GW:
                            ax.plot(
                                R,
                                M,
                                color=color2,
                                alpha=0.20,
                                linestyle="-",
                                linewidth=0.5,
                                zorder=10,
                            )
                        elif EOSIdx in ninetyFivepercentIdx_EM:
                            ax.plot(
                                R,
                                M,
                                color=color3,
                                alpha=0.20,
                                linestyle="-",
                                linewidth=0.5,
                                zorder=10,
                            )
                        else:
                            ax.plot(
                                R,
                                M,
                                color="grey",
                                alpha=0.01,
                                linestyle="-",
                                linewidth=0.5,
                                zorder=0,
                            )

                    plt.savefig(plotName)
                    plt.close()

                bins = np.linspace(7, 17, 201)
                R14_EM_hist, bin_edges = np.histogram(
                    data_out[index]["R14_EM"], bins=bins, density=True
                )
                bins = (bin_edges[1:] + bin_edges[:-1]) / 2.0
                bins = np.linspace(7, 17, 201)
                R14_GW_hist, bin_edges = np.histogram(
                    data_out[index]["R14_GW"], bins=bins, density=True
                )
                bins = (bin_edges[1:] + bin_edges[:-1]) / 2.0

                max_R14 = np.max([np.max(R14_EM_hist), np.max(R14_GW_hist)])
                plotName = "%s/R14.png" % (EOSDir)
                if not os.path.isfile(plotName):
                    fig = plt.figure()
                    ax = plt.gca()
                    ax.set_ylabel("Probability Density Function")
                    ax.set_xlabel(r"$R_{1.4} [km]$")
                    ax.set_xlim([7, 17])
                    ax.plot(
                        bins,
                        R14_EM_hist,
                        color=color1,
                        alpha=1,
                        linestyle="-",
                        linewidth=3,
                        zorder=10,
                    )
                    ax.plot(
                        bins,
                        R14_GW_hist,
                        color=color2,
                        alpha=1,
                        linestyle="-",
                        linewidth=3,
                        zorder=0,
                    )
                    ax.plot(
                        [data_out[index]["R14"], data_out[index]["R14"]],
                        [0, max_R14],
                        color="k",
                        linestyle="--",
                    )
                    plt.savefig(plotName)
                    plt.close()

            if args.H0:
                H0Dir = os.path.join(outdir, "H0")
                if not os.path.isdir(H0Dir):
                    os.makedirs(H0Dir)

                z = data_out[index]["z"]
                distances = data_out[index]["distance"]

                vr_mean = z * c
                H0_EM = vr_mean / distances
                kdedir_H0_EM = greedy_kde_areas_1d(H0_EM)

                H0_GW = vr_mean / test_distances

                data_out[index]["H0_EM"] = H0_EM
                data_out[index]["kdedir_H0_EM"] = kdedir_H0_EM

                data_out[index]["H0_GW"] = H0_GW
                data_out[index]["interp_H0_GW"] = interp.interp1d(
                    H0_GW, probdensity_vol, fill_value="extrapolate"
                )

                plotName = "%s/H0_%d.png" % (H0Dir, index)
                if not os.path.isfile(plotName):
                    bins = np.linspace(20, 150, 500)

                    kdedir_H0_EM = data_out[index]["kdedir_H0_EM"]
                    interp_H0_GW = data_out[index]["interp_H0_GW"]

                    prob_EM = kde_eval_single(kdedir_H0_EM, bins)
                    prob_EM = prob_EM / np.sum(prob_EM)
                    prob_GW = interp_H0_GW(bins)
                    prob_GW = prob_GW / np.sum(prob_GW)
                    prob_EMGW = prob_EM * prob_GW
                    prob_EMGW = prob_EMGW / np.sum(prob_EMGW)

                    H0_EM_16, H0_EM_50, H0_EM_84 = compute_constraint(bins, prob_EM)
                    H0_GW_16, H0_GW_50, H0_GW_84 = compute_constraint(bins, prob_GW)
                    H0_EMGW_16, H0_EMGW_50, H0_EMGW_84 = compute_constraint(
                        bins, prob_EMGW
                    )
                    print(
                        "H0 EM: %.1f +%.1f -%.1f"
                        % (H0_EM_50, H0_EM_84 - H0_EM_50, H0_EM_50 - H0_EM_16)
                    )
                    print(
                        "H0 GW: %.1f +%.1f -%.1f"
                        % (H0_GW_50, H0_GW_84 - H0_GW_50, H0_GW_50 - H0_GW_16)
                    )
                    print(
                        "H0 EM-GW: %.1f +%.1f -%.1f"
                        % (H0_EMGW_50, H0_EMGW_84 - H0_EMGW_50, H0_EMGW_50 - H0_EMGW_16)
                    )

                    filename = "%s/H0.dat" % (H0Dir)
                    fid = open(filename, "w")
                    fid.write(
                        "H0 EM: %.1f +%.1f -%.1f\n"
                        % (H0_EM_50, H0_EM_84 - H0_EM_50, H0_EM_50 - H0_EM_16)
                    )
                    fid.write(
                        "H0 GW: %.1f +%.1f -%.1f\n"
                        % (H0_GW_50, H0_GW_84 - H0_GW_50, H0_GW_50 - H0_GW_16)
                    )
                    fid.write(
                        "H0 EM-GW: %.1f +%.1f -%.1f"
                        % (H0_EMGW_50, H0_EMGW_84 - H0_EMGW_50, H0_EMGW_50 - H0_EMGW_16)
                    )
                    fid.close()

                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111)

                    plt.plot(bins, prob_EM, linestyle="--", color="r")
                    plt.plot(bins, prob_GW, linestyle="-", color="r")
                    plt.plot(bins, prob_EMGW, linestyle=":", color="r")

                    maxH0 = np.nanmax(
                        [np.max(prob_GW), np.max(prob_EM), np.max(prob_EMGW)]
                    )
                    plt.plot(
                        [cosmo.H(0).value, cosmo.H(0).value],
                        [0, maxH0],
                        color="k",
                        linestyle="--",
                    )

                    planck_mu, planck_std = 67.74, 0.46
                    shoes_mu, shoes_std = 74.03, 1.42
                    superluminal_mu, superluminal_std = 68.9, 4.6
                    plt.plot(
                        [planck_mu, planck_mu],
                        [0, 1],
                        alpha=0.3,
                        color="g",
                        label="Planck",
                    )
                    rect1 = Rectangle(
                        (planck_mu - planck_std, 0),
                        2 * planck_std,
                        1,
                        alpha=0.8,
                        color="g",
                    )
                    rect2 = Rectangle(
                        (planck_mu - 2 * planck_std, 0),
                        4 * planck_std,
                        1,
                        alpha=0.5,
                        color="g",
                    )
                    plt.plot(
                        [shoes_mu, shoes_mu],
                        [0, 1],
                        alpha=0.3,
                        color="r",
                        label="SHoES",
                    )
                    rect3 = Rectangle(
                        (shoes_mu - shoes_std, 0),
                        2 * shoes_std,
                        1,
                        alpha=0.8,
                        color="r",
                    )
                    rect4 = Rectangle(
                        (shoes_mu - 2 * shoes_std, 0),
                        4 * shoes_std,
                        1,
                        alpha=0.5,
                        color="r",
                    )
                    plt.plot(
                        [superluminal_mu, superluminal_mu],
                        [0, 1],
                        alpha=0.3,
                        color="c",
                        label="Superluminal",
                    )
                    rect5 = Rectangle(
                        (superluminal_mu - superluminal_std, 0),
                        2 * superluminal_std,
                        0.12,
                        alpha=0.3,
                        color="c",
                    )
                    rect6 = Rectangle(
                        (superluminal_mu - 2 * superluminal_std, 0),
                        4 * superluminal_std,
                        0.12,
                        alpha=0.1,
                        color="c",
                    )

                    ax.add_patch(rect1)
                    ax.add_patch(rect2)
                    ax.add_patch(rect3)
                    ax.add_patch(rect4)
                    ax.add_patch(rect5)
                    ax.add_patch(rect6)

                    plt.ylim([0, maxH0])

                    plt.xlabel(r"$H_0$ [km $\mathrm{s}^{-1}$ $\mathrm{Mpc}^{-1}$]")
                    plt.ylabel("Probability Density Function")
                    plt.grid()
                    plt.savefig(plotName)
                    plt.close()

        f = open(pcklFile, "wb")
        pickle.dump(data_out, f)
        f.close()

    f = open(pcklFile, "rb")
    data_out = pickle.load(f)
    f.close()

    if args.detections_file:
        keys = list(data_out.keys())
        for ii, key in enumerate(keys):
            det = detections[int(key)]
            if det == 0:
                print("No detection for this object... removing")
                del data_out[key]
        outdir = os.path.join(args.outdir, "summary-detections")
    else:
        outdir = os.path.join(args.outdir, "summary")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if args.ejecta:
        params_pairs = [
            [
                "log10_mej_wind_true",
                "log10_mej_wind",
                r"${\rm log}_{10} (M_{\rm ej,wind})$",
            ],
            [
                "log10_mej_dyn_true",
                "log10_mej_dyn",
                r"${\rm log}_{10} (M_{\rm ej,dyn})$",
            ],
            ["log10_alpha_true", "log10_alpha", r"$\alpha$"],
            ["ratio_zeta_true", "ratio_zeta", r"$\zeta$"],
        ]
        for params_pair in params_pairs:
            if params_pair[1] in ["log10_mej_wind", "log10_mej_dyn"]:
                dmin, dmax = -3, -1
            else:
                dmin = np.min(dataframe_from_inj[params_pair[1]])
                dmax = np.max(dataframe_from_inj[params_pair[1]])

            plotName = "%s/%s.png" % (outdir, params_pair[1])
            fig = plt.figure(figsize=(24, 18))
            ax = fig.add_subplot(111)
            for ii, key in enumerate(list(data_out.keys())):
                vals = data_out[key][params_pair[1]]
                if params_pair[1] == "ratio_zeta":
                    vals = 10 ** vals
                true = data_out[key][params_pair[0]]

                parts = plt.violinplot(vals, [true], widths=0.01)
                for partname in ("cbars", "cmins", "cmaxes"):
                    vp = parts[partname]
                    vp.set_edgecolor(color2)
                    vp.set_linewidth(1)
                for pc in parts["bodies"]:
                    pc.set_facecolor(color2)
                    pc.set_edgecolor(color2)

            ds = np.linspace(dmin, dmax, 1000)
            plt.plot(ds, ds, "k--")

            plt.ylabel("%s [Measured]" % params_pair[2], fontsize=48)
            plt.xlabel("%s [True]" % params_pair[2], fontsize=48)
            plt.grid()
            plt.yticks(fontsize=48)
            plt.xticks(fontsize=48)
            plt.savefig(plotName)
            plt.close()

            bin_edges = np.linspace(dmin, dmax, 51)
            bins = (bin_edges[1:] + bin_edges[:-1]) / 2.0
            hist = np.zeros((len(bins), len(bins)))

            for ii, key in enumerate(list(data_out.keys())):
                vals = data_out[key][params_pair[1]]
                if params_pair[1] == "ratio_zeta":
                    vals = 10 ** vals
                true = data_out[key][params_pair[0]]
                dhist, edges = np.histogram(vals, bins=bin_edges)
                idx = np.argmin(np.abs(bins - true))
                hist[:, idx] = hist[:, idx] + dhist

            percentiles = np.zeros((len(bins), 3))
            for ii in range(len(bins)):
                hist[:, ii] = hist[:, ii] / np.sum(hist[:, ii])
                cumsum = np.cumsum(hist[:, ii])
                percentiles[ii, 0] = bins[np.argmin(np.abs(cumsum - 0.1))]
                percentiles[ii, 1] = bins[np.argmin(np.abs(cumsum - 0.5))]
                percentiles[ii, 2] = bins[np.argmin(np.abs(cumsum - 0.9))]

            hist = hist.astype(np.float64)
            hist[hist == 0.0] = np.nan

            X, Y = np.meshgrid(bins, bins)
            plotName = "%s/%s_hist.png" % (outdir, params_pair[1])
            fig = plt.figure(figsize=(24, 18))
            ax = fig.add_subplot(111)

            ax.pcolormesh(X, Y, hist, shading="auto", cmap="viridis", alpha=0.7)

            # plot 10th, 50th, 90th percentiles
            ax.plot(bins, percentiles[:, 1], c="w", linestyle="--")
            ax.plot(bins, percentiles[:, 2], "w--")
            ax.plot(bins, percentiles[:, 0], "w--")

            ds = np.linspace(dmin, dmax, 1000)
            plt.plot(ds, ds, "k--")

            plt.ylabel("%s [Measured]" % params_pair[2], fontsize=48)
            plt.xlabel("%s [True]" % params_pair[2], fontsize=48)
            plt.grid()
            plt.yticks(fontsize=48)
            plt.xticks(fontsize=48)
            plt.savefig(plotName)
            plt.close()

    if args.eos:

        bins = np.linspace(9, 17, 500)

        kdedir_R14_GW170817 = greedy_kde_areas_1d(R14_GW170817)
        prob_GW170817 = kde_eval_single(kdedir_R14_GW170817, bins)
        prob_GW170817 = prob_GW170817 / np.sum(prob_GW170817)

        R14_GW170817_16, R14_GW170817_50, R14_GW170817_84 = compute_constraint(
            bins, prob_GW170817
        )
        R14_GW170817_std = (R14_GW170817_84 - R14_GW170817_16) / 2

        ntrials = 100
        keys = list(data_out.keys())
        R14_EM_std = np.zeros((ntrials, len(keys)))
        R14_GW_std = np.zeros((ntrials, len(keys)))
        R14_EMGW_std = np.zeros((ntrials, len(keys)))

        for nn in range(ntrials):
            R14_EM_all = copy.deepcopy(prob_GW170817)
            R14_GW_all = copy.deepcopy(prob_GW170817)
            R14_EMGW_all = copy.deepcopy(prob_GW170817)

            random.shuffle(keys)
            for ii, key in enumerate(keys):
                R14_EM = data_out[key]["R14_EM"]
                R14_GW = data_out[key]["R14_GW"]

                kdedir_R14_EM = data_out[key]["kdedir_R14_EM"]
                kdedir_R14_GW = data_out[key]["kdedir_R14_GW"]

                prob_EM = kde_eval_single(kdedir_R14_EM, bins)
                prob_EM = prob_EM / np.sum(prob_EM)
                prob_GW = kde_eval_single(kdedir_R14_GW, bins)
                prob_GW = prob_GW / np.sum(prob_GW)
                prob_EMGW = prob_EM * prob_GW
                prob_EMGW = prob_EMGW / np.sum(prob_EMGW)

                R14_EM_all = R14_EM_all * prob_EM
                R14_GW_all = R14_GW_all * prob_GW
                R14_EMGW_all = R14_EMGW_all * prob_EMGW

                R14_EM_all_norm = R14_EM_all / np.sum(R14_EM_all)
                R14_GW_all_norm = R14_GW_all / np.sum(R14_GW_all)
                R14_EMGW_all_norm = R14_EMGW_all / np.sum(R14_EMGW_all)
                R14_EM_16, R14_EM_50, R14_EM_84 = compute_constraint(
                    bins, R14_EM_all_norm
                )
                R14_GW_16, R14_GW_50, R14_GW_84 = compute_constraint(
                    bins, R14_GW_all_norm
                )
                R14_EMGW_16, R14_EMGW_50, R14_EMGW_84 = compute_constraint(
                    bins, R14_EMGW_all_norm
                )

                R14_EM_std[nn, ii] = (R14_EM_84 - R14_EM_16) / 2
                R14_GW_std[nn, ii] = (R14_GW_84 - R14_GW_16) / 2
                R14_EMGW_std[nn, ii] = (R14_EMGW_84 - R14_EMGW_16) / 2

        print("R14 EM: %.2f" % (np.median(R14_EM_std[:, -1])))
        print("R14 GW: %.2f" % (np.median(R14_GW_std[:, -1])))
        print("R14 EM-GW: %.2f" % (np.median(R14_EMGW_std[:, -1])))

        plotName = "%s/R14_detections.png" % (outdir)
        fig = plt.figure(figsize=(24, 18))
        ax = fig.add_subplot(111)

        plt.plot(
            [0, ntrials],
            [R14_GW170817_std, R14_GW170817_std],
            "c--",
            label="GW170817",
            linewidth=3,
        )

        # where some data has already been plotted to ax
        handles, labels = ax.get_legend_handles_labels()

        for ii, key in enumerate(list(data_out.keys())):
            parts = plt.violinplot(R14_EM_std[:, ii], [ii + 1], widths=0.6)
            for partname in ("cbars", "cmins", "cmaxes"):
                vp = parts[partname]
                vp.set_edgecolor(color2)
                vp.set_linewidth(1)
            for pc in parts["bodies"]:
                pc.set_facecolor(color2)
                pc.set_edgecolor(color2)
            parts = plt.violinplot(R14_GW_std[:, ii], [ii + 1], widths=0.6)
            for partname in ("cbars", "cmins", "cmaxes"):
                vp = parts[partname]
                vp.set_edgecolor(color1)
                vp.set_linewidth(1)
            for pc in parts["bodies"]:
                pc.set_facecolor(color1)
                pc.set_edgecolor(color1)
            parts = plt.violinplot(R14_EMGW_std[:, ii], [ii + 1], widths=0.6)
            for partname in ("cbars", "cmins", "cmaxes"):
                vp = parts[partname]
                vp.set_edgecolor(color3)
                vp.set_linewidth(1)
            for pc in parts["bodies"]:
                pc.set_facecolor(color3)
                pc.set_edgecolor(color3)
            if ii == 0:
                handles.append(mpatches.Patch(color=color2, label="EM"))
                handles.append(mpatches.Patch(color=color1, label="GW"))
                handles.append(mpatches.Patch(color=color3, label="EM-GW"))

        # plt.xlim([0, 60])
        # plt.ylim([7, 15])
        plt.xlabel("Number of Detections", fontsize=48)
        plt.ylabel(r"1 $\sigma$ Error in $R_{1.4}$ [km]", fontsize=48)
        plt.grid()
        plt.xlim([0, 60])
        plt.ylim([0, 0.8])
        plt.legend(handles=handles, fontsize=48, loc=9, ncol=4)
        plt.yticks(fontsize=48)
        plt.xticks(fontsize=48)
        plt.savefig(plotName)
        plt.close()

        weight_GW = np.zeros(Neos)
        weight_EM = np.zeros(Neos)
        weight_EMGW = np.zeros(Neos)

        for index in data_out.keys():
            weight_GW = weight_GW + data_out[index]["EOS_weight_GW"]
            weight_EM = weight_EM + data_out[index]["EOS_weight_EM"]
            weight_EMGW = weight_EMGW + data_out[index]["EOS_weight_EMGW"]

        weight_GW = weight_GW / np.sum(weight_GW)
        weight_EM = weight_EM / np.sum(weight_EM)
        weight_EMGW = weight_EMGW / np.sum(weight_EMGW)

        weight_sort = np.argsort(weight_GW)
        weight_cumsum = np.cumsum(weight_GW[weight_sort])
        ninetyFivepercentIdx_GW = weight_sort[np.where(weight_cumsum >= 0.05)[0]]
        print("%d/%d GW EOS included" % (len(ninetyFivepercentIdx_GW), Neos))

        weight_sort = np.argsort(weight_EM)
        weight_cumsum = np.cumsum(weight_EM[weight_sort])
        ninetyFivepercentIdx_EM = weight_sort[np.where(weight_cumsum >= 0.05)[0]]
        print("%d/%d EM EOS included" % (len(ninetyFivepercentIdx_EM), Neos))

        weight_sort = np.argsort(weight_EMGW)
        weight_cumsum = np.cumsum(weight_EMGW[weight_sort])
        ninetyFivepercentIdx_EMGW = weight_sort[np.where(weight_cumsum >= 0.05)[0]]
        print("%d/%d EM-GW EOS included" % (len(ninetyFivepercentIdx_EMGW), Neos))

        fig = plt.figure()
        ax = plt.gca()
        ax.set_ylabel(r"$M\ [\\rm M_\\odot]$")
        ax.set_xlabel(r"$R\ [\\rm km]$")
        ax.set_xlim([7.95, 15])
        ax.set_ylim([0.8, 3.6])
        for EOSIdx in range(0, Neos):
            R, M = EOS_data[EOSIdx]["R"], EOS_data[EOSIdx]["M"]
            if EOSIdx in ninetyFivepercentIdx_EMGW:
                ax.plot(
                    R,
                    M,
                    color=color1,
                    alpha=0.20,
                    linestyle="-",
                    linewidth=0.5,
                    zorder=10,
                )
            elif EOSIdx in ninetyFivepercentIdx_GW:
                ax.plot(
                    R,
                    M,
                    color=color2,
                    alpha=0.20,
                    linestyle="-",
                    linewidth=0.5,
                    zorder=10,
                )
            elif EOSIdx in ninetyFivepercentIdx_EM:
                ax.plot(
                    R,
                    M,
                    color=color3,
                    alpha=0.20,
                    linestyle="-",
                    linewidth=0.5,
                    zorder=10,
                )
            else:
                ax.plot(
                    R,
                    M,
                    color="grey",
                    alpha=0.01,
                    linestyle="-",
                    linewidth=0.5,
                    zorder=0,
                )

        plotName = "%s/MR.png" % (outdir)
        plt.savefig(plotName)
        plt.close()

        plotName = "%s/massratio.png" % (outdir)
        fig = plt.figure(figsize=(24, 18))
        ax = fig.add_subplot(111)
        for ii, key in enumerate(list(data_out.keys())):
            q_EM = data_out[key]["q_EM"]
            mass_ratio = data_out[key]["mass_ratio"]
            parts = plt.violinplot(q_EM, [mass_ratio], widths=0.01)
            for partname in ("cbars", "cmins", "cmaxes"):
                vp = parts[partname]
                vp.set_edgecolor(color2)
                vp.set_linewidth(1)
            for pc in parts["bodies"]:
                pc.set_facecolor(color2)
                pc.set_edgecolor(color2)

            q_GW = data_out[key]["q_GW"]
            mass_ratio = data_out[key]["mass_ratio"]
            parts = plt.violinplot(q_GW, [mass_ratio], widths=0.01)
            for partname in ("cbars", "cmins", "cmaxes"):
                vp = parts[partname]
                vp.set_edgecolor(color1)
                vp.set_linewidth(1)
            for pc in parts["bodies"]:
                pc.set_facecolor(color1)
                pc.set_edgecolor(color1)

        plt.plot([0.7, 1.0], [0.7, 1.0], "k--")
        plt.ylabel("Mass Ratio [measured]", fontsize=48)
        plt.xlabel("Mass Ratio [true]", fontsize=48)
        plt.grid()
        plt.yticks(fontsize=48)
        plt.xticks(fontsize=48)
        plt.savefig(plotName)
        plt.close()

        plotName = "%s/R14.png" % (outdir)
        fig = plt.figure(figsize=(24, 18))
        ax = fig.add_subplot(111)
        for ii, key in enumerate(list(data_out.keys())):
            R14_EM = data_out[key]["R14_EM"]
            parts = plt.violinplot(R14_EM, [key - 0.5], widths=1.0)
            for partname in ("cbars", "cmins", "cmaxes"):
                vp = parts[partname]
                vp.set_edgecolor(color2)
                vp.set_linewidth(1)
            for pc in parts["bodies"]:
                pc.set_facecolor(color2)
                pc.set_edgecolor(color2)

            R14_GW = data_out[key]["R14_GW"]
            parts = plt.violinplot(R14_GW, [key - 0.5], widths=1.0)
            for partname in ("cbars", "cmins", "cmaxes"):
                vp = parts[partname]
                vp.set_edgecolor(color1)
                vp.set_linewidth(1)
            for pc in parts["bodies"]:
                pc.set_facecolor(color1)
                pc.set_edgecolor(color1)

        dmin = np.min(dataframe_from_inj["luminosity_distance"])
        dmax = np.max(dataframe_from_inj["luminosity_distance"])
        ds = np.linspace(dmin, dmax, 1000)
        plt.plot([0, len(list(data_out.keys()))], [R14_true, R14_true], "k--")

        plt.ylabel(r"$R_{1.4} [km]$", fontsize=48)
        plt.xlabel("Object Number", fontsize=48)
        plt.grid()
        plt.yticks(fontsize=48)
        plt.xticks(fontsize=48)
        plt.savefig(plotName)
        plt.close()

        bins = np.linspace(9, 17, 500)
        R14_EM_all = np.ones(len(bins))
        R14_GW_all = np.ones(len(bins))
        R14_EMGW_all = np.ones(len(bins))

        colors = cm.rainbow(np.linspace(0, 1, len(list(data_out.keys()))))

        plotName = "%s/R14_hist.png" % (outdir)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        for ii, key in enumerate(list(data_out.keys())):
            kdedir_R14_EM = data_out[key]["kdedir_R14_EM"]
            prob_EM = kde_eval_single(kdedir_R14_EM, bins)
            prob_EM = prob_EM / np.sum(prob_EM)
            plt.plot(bins, prob_EM, linestyle="--", color=colors[ii])
            R14_EM_all = R14_EM_all * prob_EM

            kdedir_R14_GW = data_out[key]["kdedir_R14_GW"]
            prob_GW = kde_eval_single(kdedir_R14_GW, bins)
            prob_GW = prob_GW / np.sum(prob_GW)
            plt.plot(bins, prob_GW, linestyle="-", color=colors[ii])
            R14_GW_all = R14_GW_all * prob_GW

            R14_EMGW_all = R14_EMGW_all * prob_EM * prob_GW

        R14_EM_all = R14_EM_all / np.sum(R14_EM_all)
        plt.plot(bins, R14_EM_all, linestyle="-", color="k")
        R14_GW_all = R14_GW_all / np.sum(R14_GW_all)
        plt.plot(bins, R14_GW_all, linestyle="--", color="k")
        R14_EMGW_all = R14_EMGW_all / np.sum(R14_EMGW_all)
        plt.plot(bins, R14_EMGW_all, linestyle="--", color="k")

        maxR14 = np.max([np.max(R14_EM_all), np.max(R14_GW_all), np.max(R14_EMGW_all)])

        plt.plot([R14_true, R14_true], [0, maxR14], color="k", linestyle="--")

        plt.xlabel(r"$R_{1.4}$ [km]")
        plt.ylabel("Probability Density Function")
        plt.grid()
        plt.savefig(plotName)
        plt.close()

    if args.H0:

        bins = np.linspace(20, 150, 500)

        H0_GW170817 = np.load(args.gw170817_h0)["H0"]
        kdedir_H0_GW170817 = greedy_kde_areas_1d(H0_GW170817)
        prob_GW170817 = kde_eval_single(kdedir_H0_GW170817, bins)
        prob_GW170817 = prob_GW170817 / np.sum(prob_GW170817)

        ntrials = 100
        keys = list(data_out.keys())
        H0_EM_std = np.zeros((ntrials, len(keys)))
        H0_GW_std = np.zeros((ntrials, len(keys)))
        H0_EMGW_std = np.zeros((ntrials, len(keys)))

        for nn in range(ntrials):
            H0_EM_all = copy.deepcopy(prob_GW170817)
            H0_GW_all = copy.deepcopy(prob_GW170817)
            H0_EMGW_all = copy.deepcopy(prob_GW170817)

            random.shuffle(keys)
            for ii, key in enumerate(keys):
                H0_EM = data_out[key]["H0_EM"]
                H0_GW = data_out[key]["H0_GW"]

                kdedir_H0_EM = data_out[key]["kdedir_H0_EM"]
                interp_H0_GW = data_out[key]["interp_H0_GW"]

                prob_EM = kde_eval_single(kdedir_H0_EM, bins)
                prob_EM = prob_EM / np.sum(prob_EM)
                prob_GW = interp_H0_GW(bins)
                prob_GW = prob_GW / np.sum(prob_GW)
                prob_EMGW = prob_EM * prob_GW
                prob_EMGW = prob_EMGW / np.sum(prob_EMGW)

                if not (np.any(np.isnan(prob_EM)) or np.any(np.isnan(prob_GW))):
                    H0_EM_all = H0_EM_all * prob_EM
                    H0_GW_all = H0_GW_all * prob_GW
                    H0_EMGW_all = H0_EMGW_all * prob_EMGW

                H0_EM_all_norm = H0_EM_all / np.sum(H0_EM_all)
                H0_GW_all_norm = H0_GW_all / np.sum(H0_GW_all)
                H0_EMGW_all_norm = H0_EMGW_all / np.sum(H0_EMGW_all)
                H0_EM_16, H0_EM_50, H0_EM_84 = compute_constraint(bins, H0_EM_all_norm)
                H0_GW_16, H0_GW_50, H0_GW_84 = compute_constraint(bins, H0_GW_all_norm)
                H0_EMGW_16, H0_EMGW_50, H0_EMGW_84 = compute_constraint(
                    bins, H0_EMGW_all_norm
                )

                H0_EM_std[nn, ii] = (H0_EM_84 - H0_EM_16) / 2
                H0_GW_std[nn, ii] = (H0_GW_84 - H0_GW_16) / 2
                H0_EMGW_std[nn, ii] = (H0_EMGW_84 - H0_EMGW_16) / 2

        planck_mu, planck_std = 67.74, 0.46
        shoes_mu, shoes_std = 74.03, 1.42
        superluminal_mu, superluminal_std = 68.9, 4.6

        plotName = "%s/H0_detections.png" % (outdir)
        fig = plt.figure(figsize=(24, 18))
        ax = fig.add_subplot(111)

        plt.plot(
            [0, ntrials], [planck_std, planck_std], "g--", label="Planck", linewidth=3
        )
        plt.plot(
            [0, ntrials], [shoes_std, shoes_std], "r--", label="SHOES", linewidth=3
        )
        plt.plot(
            [0, ntrials],
            [superluminal_std, superluminal_std],
            "c--",
            label="GW170817",
            linewidth=3,
        )

        # where some data has already been plotted to ax
        handles, labels = ax.get_legend_handles_labels()

        for ii, key in enumerate(list(data_out.keys())):
            parts = plt.violinplot(H0_EM_std[:, ii], [ii + 1], widths=0.6)
            for partname in ("cbars", "cmins", "cmaxes"):
                vp = parts[partname]
                vp.set_edgecolor(color2)
                vp.set_linewidth(1)
            for pc in parts["bodies"]:
                pc.set_facecolor(color2)
                pc.set_edgecolor(color2)
            parts = plt.violinplot(H0_GW_std[:, ii], [ii + 1], widths=0.6)
            for partname in ("cbars", "cmins", "cmaxes"):
                vp = parts[partname]
                vp.set_edgecolor(color1)
                vp.set_linewidth(1)
            for pc in parts["bodies"]:
                pc.set_facecolor(color1)
                pc.set_edgecolor(color1)
            parts = plt.violinplot(H0_EMGW_std[:, ii], [ii + 1], widths=0.6)
            for partname in ("cbars", "cmins", "cmaxes"):
                vp = parts[partname]
                vp.set_edgecolor(color3)
                vp.set_linewidth(1)
            for pc in parts["bodies"]:
                pc.set_facecolor(color3)
                pc.set_edgecolor(color3)
            if ii == 0:
                handles.append(mpatches.Patch(color=color2, label="EM"))
                handles.append(mpatches.Patch(color=color1, label="GW"))
                handles.append(mpatches.Patch(color=color3, label="EM-GW"))

        plt.xlim([0, 60])
        plt.ylim([0, 6])
        plt.xlabel("Number of Detections", fontsize=48)
        plt.ylabel(
            r"1 $\sigma$ Error in $H_0$ [km $\mathrm{s}^{-1}$ $\mathrm{Mpc}^{-1}$]",
            fontsize=48,
        )
        plt.grid()
        plt.legend(handles=handles, fontsize=48, loc=9, ncol=3)
        plt.yticks(fontsize=48)
        plt.xticks(fontsize=48)
        plt.savefig(plotName)
        plt.close()

        H0_EM_all = copy.deepcopy(prob_GW170817)
        H0_GW_all = copy.deepcopy(prob_GW170817)
        H0_EMGW_all = copy.deepcopy(prob_GW170817)

        colors = cm.rainbow(np.linspace(0, 1, len(list(data_out.keys()))))

        plotName = "%s/H0.png" % (outdir)
        fig = plt.figure(figsize=(24, 18))
        ax = fig.add_subplot(111)
        for ii, key in enumerate(list(data_out.keys())):
            H0_EM = data_out[key]["H0_EM"]
            H0_GW = data_out[key]["H0_GW"]

            kdedir_H0_EM = data_out[key]["kdedir_H0_EM"]
            interp_H0_GW = data_out[key]["interp_H0_GW"]

            prob_EM = kde_eval_single(kdedir_H0_EM, bins)
            prob_EM = prob_EM / np.sum(prob_EM)
            prob_GW = interp_H0_GW(bins)
            prob_GW = prob_GW / np.sum(prob_GW)
            prob_EMGW = prob_EM * prob_GW
            prob_EMGW = prob_EMGW / np.sum(prob_EMGW)

            plt.plot(bins, prob_EM, linestyle="--", color=colors[ii])
            plt.plot(bins, prob_GW, linestyle="-", color=colors[ii])
            plt.plot(bins, prob_EMGW, linestyle=":", color="r")

            if not (np.any(np.isnan(prob_EM)) or np.any(np.isnan(prob_GW))):
                H0_EM_all = H0_EM_all * prob_EM
                H0_GW_all = H0_GW_all * prob_GW
                H0_EMGW_all = H0_EMGW_all * prob_EMGW

        H0_EM_all = H0_EM_all / np.sum(H0_EM_all)
        H0_GW_all = H0_GW_all / np.sum(H0_GW_all)
        H0_EMGW_all = H0_EMGW_all / np.sum(H0_EMGW_all)

        H0_EM_16, H0_EM_50, H0_EM_84 = compute_constraint(bins, H0_EM_all)
        H0_GW_16, H0_GW_50, H0_GW_84 = compute_constraint(bins, H0_GW_all)
        H0_EMGW_16, H0_EMGW_50, H0_EMGW_84 = compute_constraint(bins, H0_EMGW_all)

        print(
            "H0 EM: %.1f +%.1f -%.1f"
            % (H0_EM_50, H0_EM_84 - H0_EM_50, H0_EM_50 - H0_EM_16)
        )
        print(
            "H0 GW: %.1f +%.1f -%.1f"
            % (H0_GW_50, H0_GW_84 - H0_GW_50, H0_GW_50 - H0_GW_16)
        )
        print(
            "H0 EM-GW: %.1f +%.1f -%.1f"
            % (H0_EMGW_50, H0_EMGW_84 - H0_EMGW_50, H0_EMGW_50 - H0_EMGW_16)
        )

        filename = "%s/H0.dat" % (outdir)
        fid = open(filename, "w")
        fid.write(
            "H0 EM: %.1f +%.1f -%.1f\n"
            % (H0_EM_50, H0_EM_84 - H0_EM_50, H0_EM_50 - H0_EM_16)
        )
        fid.write(
            "H0 GW: %.1f +%.1f -%.1f\n"
            % (H0_GW_50, H0_GW_84 - H0_GW_50, H0_GW_50 - H0_GW_16)
        )
        fid.write(
            "H0 EM-GW: %.1f +%.1f -%.1f"
            % (H0_EMGW_50, H0_EMGW_84 - H0_EMGW_50, H0_EMGW_50 - H0_EMGW_16)
        )
        fid.close()

        plt.plot(bins, H0_EM_all, linestyle="--", color="k")
        plt.plot(bins, H0_GW_all, linestyle="-", color="k")
        plt.plot(bins, H0_EMGW_all, linestyle=":", color="k")

        maxH0 = np.max(
            [
                np.max(H0_GW_all),
                np.max(H0_EM_all),
                np.max(H0_EMGW_all),
                np.max(prob_GW170817),
            ]
        )
        # plt.plot([cosmo.H(0).value, cosmo.H(0).value],
        #         [0, maxH0], color='k', linestyle='--', label='Combined')

        plt.plot([planck_mu, planck_mu], [0, 1], alpha=0.3, color="g", label="Planck")
        rect1 = Rectangle(
            (planck_mu - planck_std, 0), 2 * planck_std, 1, alpha=0.8, color="g"
        )
        rect2 = Rectangle(
            (planck_mu - 2 * planck_std, 0), 4 * planck_std, 1, alpha=0.5, color="g"
        )
        plt.plot([shoes_mu, shoes_mu], [0, 1], alpha=0.3, color="r", label="SHoES")
        rect3 = Rectangle(
            (shoes_mu - shoes_std, 0), 2 * shoes_std, 1, alpha=0.8, color="r"
        )
        rect4 = Rectangle(
            (shoes_mu - 2 * shoes_std, 0), 4 * shoes_std, 1, alpha=0.5, color="r"
        )
        plt.plot(
            [superluminal_mu, superluminal_mu],
            [0, 1],
            alpha=0.3,
            color="c",
            label="Superluminal",
        )
        rect5 = Rectangle(
            (superluminal_mu - superluminal_std, 0),
            2 * superluminal_std,
            0.12,
            alpha=0.3,
            color="c",
        )
        rect6 = Rectangle(
            (superluminal_mu - 2 * superluminal_std, 0),
            4 * superluminal_std,
            0.12,
            alpha=0.1,
            color="c",
        )

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        ax.add_patch(rect4)
        ax.add_patch(rect5)
        ax.add_patch(rect6)

        plt.plot(bins, prob_GW170817, linestyle="-", color="b", label="GW170817")

        # plt.ylim([0, maxH0])
        plt.legend(loc=1, fontsize=36)

        plt.xlabel(r"$H_0$ [km $\mathrm{s}^{-1}$ $\mathrm{Mpc}^{-1}$]", fontsize=36)
        plt.ylabel("Probability Density Function", fontsize=36)
        plt.yticks(fontsize=48)
        plt.xticks(fontsize=48)
        plt.grid()
        plt.savefig(plotName)
        plt.close()

        plotName = "%s/distance.png" % (outdir)
        fig = plt.figure(figsize=(24, 18))
        ax = fig.add_subplot(111)
        for key in data_out.keys():
            dist, distance = data_out[key]["dist"], data_out[key]["distance"]
            parts = plt.violinplot(distance, [dist], widths=10.0)
            for partname in ("cbars", "cmins", "cmaxes"):
                vp = parts[partname]
                vp.set_edgecolor(color2)
                vp.set_linewidth(1)
            for pc in parts["bodies"]:
                pc.set_facecolor(color2)
                pc.set_edgecolor(color2)

        dmin = np.min(dataframe_from_inj["luminosity_distance"])
        dmax = np.max(dataframe_from_inj["luminosity_distance"])
        ds = np.linspace(dmin, dmax, 1000)
        plt.plot(ds, ds, "k--")

        plt.xlabel("True Distance [Mpc]", fontsize=48)
        plt.ylabel("Recovered Distance [Mpc]", fontsize=48)
        plt.grid()
        plt.yticks(fontsize=48)
        plt.xticks(fontsize=48)
        plt.xlim([0, 400])
        plt.ylim([0, 400])
        plt.savefig(plotName)
        plt.close()

        plotName = "%s/redshift.png" % (outdir)
        fig = plt.figure(figsize=(24, 18))
        ax = fig.add_subplot(111)
        for key in data_out.keys():
            z, distance = data_out[key]["z"], data_out[key]["distance"]
            parts = plt.violinplot(distance, [z], widths=0.001)
            for partname in ("cbars", "cmins", "cmaxes"):
                vp = parts[partname]
                vp.set_edgecolor(color2)
                vp.set_linewidth(1)
            for pc in parts["bodies"]:
                pc.set_facecolor(color2)
                pc.set_edgecolor(color2)

        redshifts = np.logspace(-2, -1, 1000)
        ds = cosmo.luminosity_distance(redshifts)
        plt.plot(redshifts, ds, "k--")

        ax.set_xscale("log")
        plt.xlabel("Redshift", fontsize=48)
        plt.ylabel("Distance [Mpc]", fontsize=48)
        plt.grid()
        plt.yticks(fontsize=48)
        plt.xticks(fontsize=48)
        plt.savefig(plotName)
        plt.close()


if __name__ == "__main__":
    main()
