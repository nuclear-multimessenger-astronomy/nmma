import os
import numpy as np
import argparse
import json
import pandas as pd

import matplotlib

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import gridspec

font = {"size": 30}
matplotlib.rc("font", **font)


def ms2mc(m1, m2):
    eta = m1 * m2 / ((m1 + m2) * (m1 + m2))
    mchirp = ((m1 * m2) ** (3.0 / 5.0)) * ((m1 + m2) ** (-1.0 / 5.0))
    q = m2 / m1

    return (mchirp, eta, q)


def main():

    parser = argparse.ArgumentParser(
        description="Skymap recovery of kilonovae light curves."
    )
    parser.add_argument(
        "--injection-file",
        type=str,
        required=True,
        help="The bilby injection json file to be used",
    )
    parser.add_argument(
        "--skymap-dir",
        type=str,
        required=True,
        help="skymap file directory with Bayestar skymaps",
    )
    parser.add_argument(
        "--lightcurve-dir",
        type=str,
        required=True,
        help="lightcurve file directory with lightcurves",
    )
    parser.add_argument("-i", "--indices-file", type=str)
    parser.add_argument("-d", "--detections-file", type=str)
    parser.add_argument(
        "--binary-type", type=str, required=True, help="Either BNS or NSBH"
    )
    parser.add_argument(
        "-c", "--configDirectory", help="gwemopt config file directory.", required=True
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Path to the output directory"
    )
    parser.add_argument(
        "--telescope", type=str, default="ZTF", help="telescope to recover"
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
        default=3.0,
        help="Days to be stoped analysing from the trigger time (default: 14)",
    )
    parser.add_argument(
        "--filters",
        type=str,
        help="A comma seperated list of filters to use.",
        default="g,r",
    )
    parser.add_argument(
        "--generation-seed",
        metavar="seed",
        type=int,
        default=42,
        help="Injection generation seed (default: 42)",
    )
    parser.add_argument("--exposuretime", type=int, required=True)

    parser.add_argument(
        "--parallel", action="store_true", default=False, help="parallel the runs"
    )
    parser.add_argument(
        "--number-of-cores", type=int, default=1, help="Number of cores"
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

    commands = []
    lcs = {}
    for index, row in dataframe_from_inj.iterrows():
        outdir = os.path.join(args.outdir, str(index))
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        skymap_file = os.path.join(args.skymap_dir, "%d.fits" % indices[index])
        lc_file = os.path.join(args.lightcurve_dir, "%d.dat" % index)
        lcs[index] = np.loadtxt(lc_file)

        efffile = os.path.join(outdir, f"efficiency_true_{index}.txt")
        if os.path.isfile(efffile):
            continue

        if not os.path.isfile(lc_file):
            continue

        ra, dec = row["ra"] * 360.0 / (2 * np.pi), row["dec"] * 360.0 / (2 * np.pi)
        dist = row["luminosity_distance"]
        try:
            gpstime = row["geocent_time_x"]
        except KeyError:
            gpstime = row["geocent_time"]

        exposuretime = ",".join(
            [str(args.exposuretime) for i in args.filters.split(",")]
        )

        command = f"gwemopt_run --telescopes {args.telescope} --do3D --doTiles --doSchedule --doSkymap --doTrueLocation --true_ra {ra} --true_dec {dec} --true_distance {dist} --doObservability --doObservabilityExit --timeallocationType powerlaw --scheduleType greedy -o {outdir} --gpstime {gpstime} --skymap {skymap_file} --filters {args.filters} --exposuretimes {exposuretime} --doSingleExposure --doAlternatingFilters --doEfficiency --lightcurveFiles {lc_file} --modelType file --configDirectory {args.configDirectory}"
        commands.append(command)

    print("Number of jobs remaining... %d." % len(commands))

    if args.parallel:
        from joblib import Parallel, delayed

        Parallel(n_jobs=args.number_of_cores)(
            delayed(os.system)(command) for command in commands
        )
    else:
        for command in commands:
            os.system(command)

    absmag, effs, probs = [], [], []
    fid = open(args.detections_file, "w")
    for index, row in dataframe_from_inj.iterrows():
        outdir = os.path.join(args.outdir, str(index))
        efffile = os.path.join(outdir, f"efficiency_true_{index}.txt")
        absmag.append(np.min(lcs[index][:, 3]))
        if not os.path.isfile(efffile):
            fid.write("0\n")
            effs.append(0.0)
            probs.append(0.0)
            continue
        data_out = np.loadtxt(efffile)
        fid.write("%d\n" % data_out)

        efffile = os.path.join(outdir, "efficiency.txt")
        if not os.path.isfile(efffile):
            effs.append(0.0)
        else:
            with open(efffile, "r") as file:
                data_out = file.read()
            effs.append(float(data_out.split("\n")[1].split("\t")[4]))

        efffile = os.path.join(outdir, f"efficiency_{index}.txt")
        if not os.path.isfile(efffile):
            probs.append(0.0)
        else:
            data_out = np.loadtxt(efffile, skiprows=1)
            probs.append(data_out[0, 1])
    fid.close()

    detections = np.loadtxt(args.detections_file)
    absmag = np.array(absmag)
    effs = np.array(effs)
    probs = np.array(probs)

    idx = np.where(detections)[0]
    idy = np.setdiff1d(np.arange(len(dataframe_from_inj)), idx)
    dataframe_from_detected = dataframe_from_inj.iloc[idx]
    dataframe_from_missed = dataframe_from_inj.iloc[idy]
    absmag_det = absmag[idx]
    absmag_miss = absmag[idy]
    effs_det = effs[idx]
    effs_miss = effs[idy]
    probs_det = probs[idx]
    probs_miss = probs[idy]

    (mchirp_det, eta_det, q_det) = ms2mc(
        dataframe_from_detected["mass_1_source"],
        dataframe_from_detected["mass_2_source"],
    )
    (mchirp_miss, eta_miss, q_miss) = ms2mc(
        dataframe_from_missed["mass_1_source"], dataframe_from_missed["mass_2_source"]
    )

    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    plotdir = os.path.join(args.outdir, "summary")
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    plt.figure()
    plt.scatter(
        absmag_det,
        dataframe_from_detected["luminosity_distance"],
        cmap=cmap,
        marker="*",
        c=probs_det,
        alpha=effs_det,
        label="Detected",
    )
    plt.scatter(
        absmag_miss,
        dataframe_from_missed["luminosity_distance"],
        cmap=cmap,
        marker="o",
        c=probs_miss,
        alpha=effs_miss,
        label="Missed",
    )
    if args.binary_type == "BNS":
        plt.xlim([-17.5, -14.0])
    elif args.binary_type == "NSBH":
        plt.xlim([-16.5, -13.0])
    plt.gca().invert_xaxis()
    plt.xlabel("Absolute Magnitude")
    plt.ylabel("Distance [Mpc]")
    plt.legend()
    # cbar = plt.colorbar(cmap=cmap, norm=norm)
    # cbar.set_label(r'Detection Efficiency')
    plotName = os.path.join(plotdir, "missed_found.pdf")
    plt.savefig(plotName)
    plt.close()

    fig = plt.figure(figsize=(20, 16))

    gs = gridspec.GridSpec(4, 4)
    ax1 = fig.add_subplot(gs[1:4, 0:3])
    ax2 = fig.add_subplot(gs[0, 0:3])
    ax3 = fig.add_subplot(gs[1:4, 3], sharey=ax1)
    ax4 = fig.add_axes([0.03, 0.17, 0.5, 0.10])
    plt.setp(ax3.get_yticklabels(), visible=False)

    plt.axes(ax1)
    plt.scatter(
        absmag_det,
        1 - probs_det,
        s=150 * np.ones(absmag_det.shape),
        cmap=cmap,
        norm=norm,
        marker="*",
        alpha=effs_det,
        c=effs_det,
    )
    plt.scatter(
        absmag_miss,
        1 - probs_miss,
        s=150 * np.ones(absmag_miss.shape),
        cmap=cmap,
        norm=norm,
        marker="o",
        alpha=effs_miss,
        c=effs_miss,
    )
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Missed",
            markersize=20,
            markerfacecolor="k",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="Found",
            markersize=20,
            markerfacecolor="k",
        ),
    ]
    if args.binary_type == "BNS":
        plt.xlim([-17.5, -14.0])
    elif args.binary_type == "NSBH":
        plt.xlim([-16.5, -13.0])
    plt.ylim([0.001, 1.0])
    ax1.set_yscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("Absolute Magnitude")
    plt.ylabel("1 - 2D Probability")
    plt.legend(handles=legend_elements, loc=4)
    plt.grid()

    plt.axes(ax4)
    plt.axis("off")
    cbar = plt.colorbar(sm, shrink=0.5, orientation="horizontal")
    cbar.set_label(r"Detection Efficiency")

    plt.axes(ax3)
    yedges = np.logspace(-3, 0, 30)
    hist, bin_edges = np.histogram(1 - probs_miss, bins=yedges, density=False)
    bins = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    plt.barh(
        bins[:-1],
        hist[:-1],
        height=np.diff(bins),
        log=True,
        align="center",
        facecolor="w",
        edgecolor="r",
        label="Missed",
    )
    hist, bin_edges = np.histogram(1 - probs_det, bins=yedges, density=False)
    bins = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    plt.barh(
        bins[:-1],
        hist[:-1],
        height=np.diff(bins),
        log=True,
        align="center",
        facecolor="w",
        edgecolor="g",
        label="Detected",
    )
    plt.legend()
    plt.xlabel("Counts")
    # plt.xlim([0.02, 50])
    plt.ylim([0.001, 1.0])
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_yticklabels([])

    plt.axes(ax2)
    if args.binary_type == "BNS":
        xedges = np.linspace(-17.5, -14.0, 30)
    elif args.binary_type == "NSBH":
        xedges = np.linspace(-16.5, -13.0, 30)

    hist, bin_edges = np.histogram(absmag_miss, bins=xedges, density=False)
    bins = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    plt.bar(
        bins[:-1],
        hist[:-1],
        width=np.diff(bins),
        align="center",
        facecolor="w",
        edgecolor="r",
        label="Missed",
    )
    hist, bin_edges = np.histogram(absmag_det, bins=xedges, density=False)
    bins = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    plt.bar(
        bins[:-1],
        hist[:-1],
        width=np.diff(bins),
        align="center",
        facecolor="w",
        edgecolor="g",
        label="Detected",
    )
    plt.legend()
    plt.ylabel("Counts")
    if args.binary_type == "BNS":
        plt.xlim([-17.5, -14.0])
    elif args.binary_type == "NSBH":
        plt.xlim([-16.5, -13.0])
    # plt.xlim([0.02, 50])
    # ax2.set_xscale('log')
    plt.gca().invert_xaxis()
    ax2.set_yscale("log")
    ax2.set_xticklabels([])

    plotName = os.path.join(plotdir, "eff.pdf")
    plt.savefig(plotName, bbox_inches="tight")
    plt.close()
