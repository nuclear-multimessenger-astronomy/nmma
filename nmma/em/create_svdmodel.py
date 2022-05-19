import os
import copy
import numpy as np
import argparse
import glob
import inspect

from .training import SVDTrainingModel
from .model import SVDLightCurveModel
from .utils import read_photometry_files, read_spectroscopy_files
from . import model_parameters


def axial_symmetry(training_data):

    modelkeys = list(training_data.keys())
    if any(["KNtheta" not in training_data[key] for key in modelkeys]):
        raise ValueError("unknown symmetry parameter")

    for key in modelkeys:
        training = training_data[key]
        key_new = key + "_flipped"
        training_data[key_new] = copy.deepcopy(training)
        training_data[key_new]["KNtheta"] = -training_data[key_new]["KNtheta"]
        key_new = key + "_flipped_180"
        training_data[key_new] = copy.deepcopy(training)
        training_data[key_new]["KNtheta"] = 180 - training_data[key_new]["KNtheta"]

    return training_data


def main():

    parser = argparse.ArgumentParser(
        description="Inference on kilonova ejecta parameters."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the SVD model to create"
    )
    parser.add_argument(
        "--svd-path",
        type=str,
        help="Path to the SVD directory, with {model}_mag.pkl and {model}_lbol.pkl",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the directory of light curve files",
    )
    parser.add_argument(
        "--interpolation_type",
        type=str,
        required=True,
        help="Type of interpolation to perform",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="photometry",
        help="Data type for interpolation [photometry or spectroscopy]",
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
        "--lmin",
        type=float,
        default=0.0,
        help="Minimum wavelength to analyze (default: 3000)",
    )
    parser.add_argument(
        "--lmax",
        type=float,
        default=10000.0,
        help="Maximum wavelength to analyze (default: 10000)",
    )
    parser.add_argument(
        "--svd-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for SVD evaluation (default: 10)",
    )
    parser.add_argument(
        "--tensorflow-nepochs",
        type=int,
        default=15,
        help="Number of epochs for tensorflow training (default: 15)",
    )
    parser.add_argument(
        "--filters",
        type=str,
        help="A comma seperated list of filters to use (e.g. g,r,i). If none is provided, will use all the filters available",
        default="u,g,r,i,z,y,J,H,K",
    )
    parser.add_argument(
        "--outdir", type=str, default="output", help="Path to the output directory"
    )
    parser.add_argument(
        "--axial-symmetry",
        action="store_true",
        default=False,
        help="add training samples based on the fact that there is axial symmetry",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False, help="enable plotting"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="print out log likelihoods",
    )
    args = parser.parse_args()

    # initialize light curve model
    sample_times = np.arange(args.tmin, args.tmax + args.dt, args.dt)
    filts = args.filters.split(",")

    MODEL_FUNCTIONS = {
        k: v for k, v in model_parameters.__dict__.items() if inspect.isfunction(v)
    }
    if args.model not in list(MODEL_FUNCTIONS.keys()):
        raise ValueError(
            f"{args.model} unknown. Please add to nmma.em.model_parameters"
        )
    model_function = MODEL_FUNCTIONS[args.model]

    filenames = glob.glob(f"{args.data_path}/*.dat") + glob.glob(
        f"{args.data_path}/*.csv"
    )
    if len(filenames) == 0:
        raise ValueError("Need at least one file to interpolate.")

    if args.data_type == "photometry":
        data = read_photometry_files(filenames)
    elif args.data_type == "spectroscopy":
        data = read_spectroscopy_files(
            filenames, wavelength_min=args.lmin, wavelength_max=args.lmax, smooth=True
        )
        keys = list(data.keys())
        filts = data[keys[0]]["lambda"]
    else:
        raise ValueError("data-type should be photometry or spectroscopy")

    training_data, parameters = model_function(data)
    if args.axial_symmetry:
        training_data = axial_symmetry(training_data)

    training_model = SVDTrainingModel(
        args.model,
        training_data,
        parameters,
        sample_times,
        filts,
        n_coeff=args.svd_ncoeff,
        n_epochs=args.tensorflow_nepochs,
        svd_path=args.svd_path,
        interpolation_type=args.interpolation_type,
        data_type=args.data_type,
        plot=args.plot,
        plotdir=args.outdir,
    )

    light_curve_model = SVDLightCurveModel(
        args.model,
        sample_times,
        svd_path=args.svd_path,
        mag_ncoeff=args.svd_ncoeff,
        interpolation_type=args.interpolation_type,
        model_parameters=training_model.model_parameters,
    )
    if args.plot:
        # we can plot an example where we compare the model performance
        # to the grid points

        modelkeys = list(training_data.keys())
        training = training_data[modelkeys[0]]
        parameters = training_model.model_parameters
        data = {param: training[param] for param in parameters}
        data["redshift"] = 0
        if args.data_type == "photometry":
            lbol, mag = light_curve_model.generate_lightcurve(sample_times, data)
        elif args.data_type == "spectroscopy":
            spec = light_curve_model.generate_spectra(
                sample_times, training_model.filters, data
            )

        import matplotlib.pyplot as plt

        plotName = os.path.join(
            args.outdir, "injection_" + args.model + "_lightcurves.png"
        )

        if args.data_type == "photometry":

            fig = plt.figure(figsize=(16, 18))
            ncols = 1
            nrows = int(np.ceil(len(filts) / ncols))
            gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.6, hspace=0.5)

            for ii, filt in enumerate(filts):
                loc_x, loc_y = np.divmod(ii, nrows)
                loc_x, loc_y = int(loc_x), int(loc_y)
                ax = fig.add_subplot(gs[loc_y, loc_x])

                plt.plot(sample_times, training["data"][:, ii], "k--", label="grid")
                plt.plot(sample_times, mag[filt], "b-", label="interpolated")
                ax.set_xlim([0, 14])
                if args.model == "CV":
                    ax.set_ylim([28, 16])
                else:
                    ax.set_ylim([-12, -18])
                ax.set_ylabel(filt, fontsize=30, rotation=0, labelpad=14)

                if ii == 0:
                    ax.legend(fontsize=16)

                if ii == len(filts) - 1:
                    ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
                else:
                    plt.setp(ax.get_xticklabels(), visible=False)

                if args.data_type == "photometry":
                    if args.model == "CV":
                        ax.set_yticks([28, 25, 22, 19, 16])
                    else:
                        ax.set_yticks([-18, -16, -14, -12])

                ax.tick_params(axis="x", labelsize=30)
                ax.tick_params(axis="y", labelsize=30)
                ax.grid(which="both", alpha=0.5)

            fig.text(0.45, 0.05, "Time [days]", fontsize=30)
            ylabel = "Absolute Magnitude"

            fig.text(
                0.01,
                0.5,
                ylabel,
                va="center",
                rotation="vertical",
                fontsize=30,
            )

        elif args.data_type == "spectroscopy":

            fig = plt.figure(figsize=(32, 14))
            ncols = 3
            nrows = 1
            vmin = -10
            vmax = -2
            gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.6, hspace=0.5)

            XX, YY = np.meshgrid(sample_times, filts)
            ZZ = np.log10(training["data"].T)

            ax = fig.add_subplot(gs[0, 0])
            plt.pcolor(XX, YY, ZZ, vmin=vmin, vmax=vmax)
            ax.set_xlabel("Time [days]", fontsize=30)
            ax.set_ylabel("Wavelength [AA]", fontsize=30)
            ax.set_title("Original", fontsize=30)
            ax.tick_params(axis="x", labelsize=30)
            ax.tick_params(axis="y", labelsize=30)
            ax.grid(which="both", alpha=0.5)
            cbar = plt.colorbar()
            cbar.set_label("log10(Flux)", fontsize=30)
            cbar.ax.tick_params(labelsize=30)

            spec_plot = np.array([spec[key] for key in filts]).T
            ZZ = np.log10(spec_plot.T)

            ax = fig.add_subplot(gs[0, 1])
            plt.pcolor(XX, YY, ZZ, vmin=vmin, vmax=vmax)
            ax.set_xlabel("Time [days]", fontsize=30)
            ax.set_ylabel("Wavelength [AA]", fontsize=30)
            ax.set_title("Interpolated", fontsize=30)
            ax.tick_params(axis="x", labelsize=30)
            ax.tick_params(axis="y", labelsize=30)
            ax.grid(which="both", alpha=0.5)
            cbar = plt.colorbar()
            cbar.set_label("log10(Flux)", fontsize=30)
            cbar.ax.tick_params(labelsize=30)

            ZZ = np.log10(
                np.abs(
                    (
                        (np.log10(spec_plot) - np.log10(training["data"]))
                        / np.log10(training["data"])
                    )
                )
            ).T

            vmin = -3
            vmax = 0
            ax = fig.add_subplot(gs[0, 2])
            plt.pcolor(XX, YY, ZZ, vmin=vmin, vmax=vmax)
            ax.set_xlabel("Time [days]", fontsize=30)
            ax.set_ylabel("Wavelength [AA]", fontsize=30)
            ax.set_title("(Original - Interpolated) / Interpolated", fontsize=30)
            ax.tick_params(axis="x", labelsize=30)
            ax.tick_params(axis="y", labelsize=30)
            ax.grid(which="both", alpha=0.5)
            cbar = plt.colorbar()
            cbar.set_label("Relative Difference", fontsize=30)
            cbar.ax.tick_params(labelsize=30)

        plt.tight_layout()
        plt.savefig(plotName, bbox_inches="tight")
        plt.close()
