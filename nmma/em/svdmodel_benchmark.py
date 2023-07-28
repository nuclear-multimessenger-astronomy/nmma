import os
import numpy as np
import argparse
import glob
import inspect
from p_tqdm import p_map

from .model import SVDLightCurveModel
from .io import read_photometry_files
from . import model_parameters

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {"font.size": 16, "text.usetex": True, "font.family": "Times New Roman"}
)


def main():

    parser = argparse.ArgumentParser(
        description="Surrogate model performance benchmark"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the SVD model created"
    )
    parser.add_argument(
        "--svd-path",
        type=str,
        help="Path to the SVD directory, \
              with {model}_mag.pkl, {model}_lbol.pkl or {model_tf.pkl}",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the directory of light curve files",
    )
    parser.add_argument(
        "--data-file-type",
        type=str,
        help="Type of light curve files [bulla, standard, ztf]",
        default="bulla",
    )
    parser.add_argument(
        "--interpolation-type",
        type=str,
        required=True,
        help="Type of interpolation performed",
    )
    parser.add_argument(
        "--data-time-unit",
        type=str,
        default="days",
        help="Time unit of input data (days, hours, minutes, or seconds)",
    )
    parser.add_argument(
        "--svd-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for SVD evaluation (default: 10)",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=0.0,
        help="Days to be started considering from the trigger time (default: 0)",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=14.0,
        help="Days to be stoped considering from the trigger time (default: 14)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Time step in the sample_times (default: 0.1)",
    )
    parser.add_argument(
        "--filters",
        type=str,
        help="A comma seperated list of filters to use (e.g. g,r,i). If none is provided, will use all the filters available",
    )
    parser.add_argument(
        "--ncpus",
        type=int,
        default=4,
        help="Number of CPU to be used (default: 4)",
    )
    parser.add_argument(
        "--outdir", type=str, default="output", help="Path to the output directory"
    )
    parser.add_argument(
        "--ignore-bolometric",
        action="store_true",
        default=True,
        help="ignore bolometric light curve files (ending in _Lbol.file_extension)",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        default=False,
        help="only look for local svdmodels (ignore Zenodo)",
    )
    args = parser.parse_args()

    # make the outdir
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    # get the grid data file path
    file_extensions = ["dat", "csv", "dat.gz", "h5"]
    filenames = []
    for file_extension in file_extensions:
        if not args.ignore_bolometric:
            filenames = filenames + glob.glob(f"{args.data_path}/*.{file_extension}")
        else:
            filenames = filenames + glob.glob(
                f"{args.data_path}/*[!_Lbol].{file_extension}"
            )
    if len(filenames) == 0:
        raise ValueError("Need at least one file to interpolate.")

    # read the grid data
    grid_data = read_photometry_files(filenames, datatype=args.data_file_type)

    # create the SVD training data
    MODEL_FUNCTIONS = {
        k: v for k, v in model_parameters.__dict__.items() if inspect.isfunction(v)
    }
    if args.model not in list(MODEL_FUNCTIONS.keys()):
        raise ValueError(
            f"{args.model} unknown. Please add to nmma.em.model_parameters"
        )
    model_function = MODEL_FUNCTIONS[args.model]
    grid_training_data, parameters = model_function(grid_data)

    # create the SVDlight curve model
    sample_times = np.arange(args.tmin, args.tmax + args.dt, args.dt)
    light_curve_model = SVDLightCurveModel(
        args.model,
        sample_times,
        svd_path=args.svd_path,
        mag_ncoeff=args.svd_ncoeff,
        interpolation_type=args.interpolation_type,
        local_only=args.local_only,
    )

    # get the filts
    if not args.filters:
        first_entry_name = list(grid_training_data.keys())[0]
        first_entry = grid_training_data[first_entry_name]
        filts = first_entry.keys() - set(["t"] + parameters)
        filts = list(filts)
    else:
        filts = args.filters

    print(f"Benchmarking model {args.model} on filter {filts} with {args.ncpus} cpus")

    def chi2_func(grid_entry_name, data_time_unit="days"):
        # fetch the grid data and parameter
        grid_entry = grid_training_data[grid_entry_name]
        grid_t = np.array(grid_entry["t"])

        # Convert input data time values to days
        if data_time_unit in ["days", "day", "d"]:
            time_scale_factor = 1.0
        elif data_time_unit in ["hours", "hour", "hr", "h"]:
            time_scale_factor = 24.0
        elif data_time_unit in ["minutes", "minute", "min", "m"]:
            time_scale_factor = 1440.0
        elif data_time_unit in ["seconds", "second", "sec", "s"]:
            time_scale_factor = 86400.0
        else:
            raise ValueError(
                "data_time_unit must be one of days, hours, minutes, or seconds."
            )
        grid_t = grid_t / time_scale_factor

        used_grid_t = grid_t[(grid_t > args.tmin) * (grid_t < args.tmax)]
        grid_mAB = {}
        for filt in filts:
            time_indices = (grid_t > args.tmin) * (grid_t < args.tmax)
            grid_mAB_per_filt_array = np.array(grid_entry[filt])
            grid_mAB[filt] = grid_mAB_per_filt_array[time_indices]
        # fetch the grid parameters
        parameter_entry = {param: grid_entry[param] for param in parameters}
        parameter_entry["redshift"] = 0.0
        # generate the corresponding light curve with SVD model
        _, estimate_mAB = light_curve_model.generate_lightcurve(
            used_grid_t, parameter_entry
        )
        # calculate chi2
        chi2 = {}
        for filt in grid_mAB.keys():
            chi2[filt] = np.nanmean(np.power(grid_mAB[filt] - estimate_mAB[filt], 2.0))
        return chi2

    grid_entry_names = list(grid_training_data.keys())
    if args.ncpus == 1:
        chi2_dict_array = [
            chi2_func(grid_entry_name, data_time_unit=args.data_time_unit)
            for grid_entry_name in grid_entry_names
        ]
    else:
        chi2_dict_array = p_map(
            chi2_func,
            grid_entry_names,
            data_time_unit=args.data_time_unit,
            num_cpus=args.ncpus,
        )

    chi2_array_by_filt = {}
    for filt in chi2_dict_array[0].keys():
        chi2_array_by_filt[filt] = [dict_entry[filt] for dict_entry in chi2_dict_array]

    # make the plots
    for figidx, filt in enumerate(filts):
        plt.figure(figidx)
        plt.xlabel(r"$\chi^2 / {\rm d.o.f.}$")
        plt.ylabel("Count")
        plt.hist(chi2_array_by_filt[filt], label=filt, bins=51, histtype="step")
        plt.legend()
        plt.savefig(f"{args.outdir}/{filt}.pdf", bbox_inches="tight")
        plt.close()
