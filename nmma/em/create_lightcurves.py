import os
import numpy as np
import argparse
import json

from astropy import time

import bilby
import bilby.core

from .model import create_light_curve_model_from_args
from .injection import create_light_curve_data
from .io import read_lightcurve_file
from ..utils.models import refresh_models_list


def get_parser():

    parser = argparse.ArgumentParser(
        description="Inference on kilonova ejecta parameters."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the kilonova model to be used"
    )
    parser.add_argument(
        "--svd-path",
        type=str,
        default="svdmodels",
        help="Path to the SVD directory, with {model}_mag.pkl and {model}_lbol.pkl",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Path to the output directory"
    )
    parser.add_argument("--label", type=str, required=True, help="Label for the run")
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
        "--generation-seed",
        metavar="seed",
        type=int,
        default=42,
        help="Injection generation seed (default: 42)",
    )
    parser.add_argument(
        "--injection", metavar="PATH", type=str, help="Path to the injection json file"
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
        "--grb-resolution",
        type=float,
        default=5,
        help="The upper bound on the ratio between thetaWing and thetaCore (default: 5)",
    )
    parser.add_argument(
        "--jet-type",
        type=int,
        default=0,
        help="Jet type to used used for GRB afterglow light curve (default: 0)",
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
        "--interpolation-type",
        type=str,
        help="SVD interpolation scheme.",
        default="sklearn_gp",
    )
    parser.add_argument(
        "--absolute", action="store_true", default=False, help="Absolute Magnitude"
    )
    parser.add_argument(
        "--ztf-sampling", help="Use realistic ZTF sampling", action="store_true"
    )
    parser.add_argument(
        "--ztf-uncertainties",
        help="Use realistic ZTF uncertainties",
        action="store_true",
    )
    parser.add_argument(
        "--ztf-ToO",
        help="Adds realistic ToO obeservations during the first one or two days. Sampling depends on exposure time specified. Valid values are 180 (<1000sq deg) or 300 (>1000sq deg). Won't work w/o --ztf-sampling",
        type=str,
        choices=["180", "300"],
    )
    parser.add_argument(
        "--rubin-ToO",
        help="Adds ToO obeservations based on the strategy presented in arxiv.org/abs/2111.01945.",
        action="store_true",
    )
    parser.add_argument(
        "--rubin-ToO-type",
        help="Type of ToO observation. Won't work w/o --rubin-ToO",
        type=str,
        choices=["BNS", "NSBH"],
    )
    parser.add_argument(
        "--photometry-augmentation",
        help="Augment photometry to improve parameter recovery",
        action="store_true",
    )
    parser.add_argument(
        "--photometry-augmentation-seed",
        metavar="seed",
        type=int,
        default=0,
        help="Optimal generation seed (default: 0)",
    )
    parser.add_argument(
        "--photometry-augmentation-N-points",
        help="Number of augmented points to include",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--photometry-augmentation-filters",
        type=str,
        help="A comma seperated list of filters to use for augmentation (e.g. g,r,i). If none is provided, will use all the filters available",
    )
    parser.add_argument(
        "--photometry-augmentation-times",
        type=str,
        help="A comma seperated list of times to use for augmentation in days post trigger time (e.g. 0.1,0.3,0.5). If none is provided, will use random times between tmin and tmax",
    )
    parser.add_argument(
        "--train-stats",
        help="Creates a file too.csv to derive statistics",
        action="store_true",
    )
    parser.add_argument(
        "--refresh-models-list",
        type=bool,
        default=False,
        help="Refresh the list of models available on Zenodo",
    )

    return parser


def main(args=None):

    if args is None:
        parser = get_parser()
        args = parser.parse_args()

    refresh = False
    try:
        refresh = args.refresh_model_list
    except AttributeError:
        pass
    if refresh:
        refresh_models_list(
            models_home=args.svd_path if args.svd_path not in [None, ""] else None
        )

    seed = args.generation_seed
    np.random.seed(seed)

    bilby.core.utils.setup_logger(outdir=args.outdir, label=args.label)
    bilby.core.utils.check_directory_exists_and_if_not_mkdir(args.outdir)
    # initialize light curve model
    sample_times = np.arange(args.tmin, args.tmax + args.dt, args.dt)

    if args.filters:
        filters = args.filters.split(",")
    else:
        filters = None

    _, _, light_curve_model = create_light_curve_model_from_args(
        args.model,
        args,
        sample_times,
        filters=filters,
    )

    # read injection file
    with open(args.injection, "r") as f:
        injection_dict = json.load(f, object_hook=bilby.core.utils.decode_bilby_json)

    args.kilonova_tmin = args.tmin
    args.kilonova_tmax = args.tmax
    args.kilonova_tstep = args.dt

    args.kilonova_injection_model = args.model
    args.kilonova_injection_svd = args.svd_path
    args.injection_svd_mag_ncoeff = args.svd_mag_ncoeff
    args.injection_svd_lbol_ncoeff = args.svd_lbol_ncoeff

    # args.injection_detection_limit = np.inf
    args.kilonova_error = 0

    injection_df = injection_dict["injections"]

    # save simulation_id from observing scenarios data
    # we save lighcurve for each event with its initial simulation ID
    # from observing scenarios
    simulation_id = injection_df["simulation_id"]

    mag_ds = {}
    for index, row in injection_df.iterrows():

        if len(injection_df) == 1:
            injection_outfile = os.path.join(args.outdir, "%s.dat" % args.label)
        else:
            injection_outfile = os.path.join(
                args.outdir, f"{args.label}_{simulation_id[index]}.dat"
            )
        if os.path.isfile(injection_outfile):
            try:
                mag_ds[index] = read_lightcurve_file(injection_outfile)
                continue

            except ValueError:
                print(
                    "\n==================================================================="
                )
                print(
                    "The previous run generated light curves with unreadable content.\n"
                )
                print("Please remove all output files in .dat format then retry.")
                print(
                    "===================================================================\n"
                )
                exit()

        injection_parameters = row.to_dict()

        try:
            tc_gps = time.Time(injection_parameters["geocent_time_x"], format="gps")
        except KeyError:
            tc_gps = time.Time(injection_parameters["geocent_time"], format="gps")
        trigger_time = tc_gps.mjd

        injection_parameters["kilonova_trigger_time"] = trigger_time

        data = create_light_curve_data(
            injection_parameters,
            args,
            doAbsolute=args.absolute,
            light_curve_model=light_curve_model,
        )
        print("Injection generated")

        try:
            fid = open(injection_outfile, "w")
            # fid.write('# t[days] u g r i z y J H K\n')
            # fid.write(str(" ".join(('# t[days]'," ".join(args.filters.split(',')),"\n"))))
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

        except IndexError:
            print(
                "\n==================================================================="
            )
            print(
                "Sorry we could not use ZTF or Rubin flags to generate those statistics\n"
            )
            print("Please remove all flags rely on with ZTF or Rubin then retry again")
            print(
                "===================================================================\n"
            )
            exit()

        mag_ds[index] = read_lightcurve_file(injection_outfile)

    if args.plot:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("agg")
        params = {
            "backend": "pdf",
            "axes.labelsize": 42,
            "legend.fontsize": 42,
            "xtick.labelsize": 42,
            "ytick.labelsize": 42,
            "text.usetex": True,
            "font.family": "Times New Roman",
            "figure.figsize": [16, 20],
        }
        matplotlib.rcParams.update(params)

        plotName = os.path.join(
            args.outdir, "injection_" + args.model + "_lightcurves.pdf"
        )

        fig = plt.figure()

        filts = list(set(mag_ds[index].keys()).difference({"t"}))

        ncols = 1
        nrows = int(np.ceil(len(filts) / ncols))
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.6, hspace=0.5)

        for ii, filt in enumerate(filts):
            loc_x, loc_y = np.divmod(ii, nrows)
            loc_x, loc_y = int(loc_x), int(loc_y)
            ax = fig.add_subplot(gs[loc_y, loc_x])

            data_out = []
            for jj, key in enumerate(list(mag_ds.keys())):
                data_out.append(mag_ds[key][filt])
            data_out = np.vstack(data_out)

            bins = np.linspace(-20, 1, 50)

            def return_hist(x):
                hist, bin_edges = np.histogram(x, bins=bins)
                return hist

            hist = np.apply_along_axis(lambda x: return_hist(x), -1, data_out.T)
            bins = (bins[1:] + bins[:-1]) / 2.0

            X, Y = np.meshgrid(sample_times, bins)

            hist = hist.astype(np.float64)
            hist[hist == 0.0] = np.nan

            c = ax.pcolormesh(X, Y, hist.T, shading="auto", cmap="cividis", alpha=0.7)

            # plot 10th, 50th, 90th percentiles
            ax.plot(sample_times, np.nanpercentile(data_out, 50, axis=0), "k--")
            ax.plot(sample_times, np.nanpercentile(data_out, 90, axis=0), "k--")
            ax.plot(sample_times, np.nanpercentile(data_out, 10, axis=0), "k--")

            ax.set_xlim([0, 14])
            ax.set_ylim([-12, -18])
            ax.set_ylabel(filt, fontsize=30, rotation=0, labelpad=14)

            if ii == len(filts) - 1:
                ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_yticks([-18, -16, -14, -12])
            ax.tick_params(axis="x", labelsize=42)
            ax.tick_params(axis="y", labelsize=42)
            ax.grid(which="both", alpha=0.5)

        fig.colorbar(c, ax=ax)
        fig.text(0.4, 0.05, r"Time [days]", fontsize=42)
        fig.text(
            0.01,
            0.5,
            r"Absolute Magnitude",
            va="center",
            rotation="vertical",
            fontsize=42,
        )

        # plt.tight_layout()
        plt.savefig(plotName, bbox_inches="tight")
        plt.close()
