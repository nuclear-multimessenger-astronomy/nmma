import os
import numpy as np
import argparse
import glob
import inspect
from p_tqdm import p_map
import json

from .model import SVDLightCurveModel
from .io import read_photometry_files
from . import model_parameters
from .utils import running_in_ci

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = not running_in_ci()
matplotlib.rcParams.update({"font.size": 16, "font.family": "Times New Roman"})


def get_parser():

    parser = argparse.ArgumentParser(
        description="Surrogate model performance benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the SVD model created",
        required=True,
    )
    parser.add_argument(
        "--svd-path",
        type=str,
        help="Path to the SVD directory with {model}.joblib",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the directory of light curve files",
        required=True,
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
        default="tensorflow",
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
        help="A comma-seperated list of filters to use (e.g. ztfg,ztfr,ztfi). If none is provided, will use all the filters available",
    )
    parser.add_argument(
        "--ncpus",
        type=int,
        default=1,
        help="Number of CPU to be used (default: 1)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="benchmark_output",
        help="Path to the output directory",
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
        help="only look for local svdmodels (ignore Gitlab)",
    )

    return parser


def create_benchmark(
    model,
    svd_path,
    data_path,
    data_file_type="bulla",
    interpolation_type="tensorflow",
    data_time_unit="days",
    svd_ncoeff=10,
    tmin=0.0,
    tmax=14.0,
    dt=0.1,
    filters=None,
    ncpus=1,
    outdir="benchmark_output",
    ignore_bolometric=True,
    local_only=False,
):

    # get the grid data file path
    file_extensions = ["dat", "csv", "dat.gz", "h5"]
    filenames = []
    for file_extension in file_extensions:
        if not ignore_bolometric:
            filenames = filenames + glob.glob(f"{data_path}/*.{file_extension}")
        else:
            filenames = filenames + glob.glob(f"{data_path}/*[!_Lbol].{file_extension}")
    if len(filenames) == 0:
        raise ValueError("Need at least one file to interpolate.")

    # read the grid data
    grid_data = read_photometry_files(filenames, datatype=data_file_type)

    # create the SVD training data
    MODEL_FUNCTIONS = {
        k: v for k, v in model_parameters.__dict__.items() if inspect.isfunction(v)
    }
    if model not in list(MODEL_FUNCTIONS.keys()):
        raise ValueError(f"{model} unknown. Please add to nmma.em.model_parameters")
    model_function = MODEL_FUNCTIONS[model]
    grid_training_data, parameters = model_function(grid_data)

    # get the filts
    if not filters:
        first_entry_name = list(grid_training_data.keys())[0]
        first_entry = grid_training_data[first_entry_name]
        filts = first_entry.keys() - set(["t"] + parameters)
        filts = list(filts)
    elif isinstance(filters, str):
        filts = filters.replace(" ", "")  # remove all whitespace
        filts = filts.split(",")
    else:
        # list input from analysis test code
        filts = filters

    if len(filts) == 0:
        raise ValueError("Need at least one valid filter.")

    # create the SVDlight curve model
    sample_times = np.arange(tmin, tmax + dt, dt)
    light_curve_model = SVDLightCurveModel(
        model,
        sample_times,
        svd_path=svd_path,
        mag_ncoeff=svd_ncoeff,
        interpolation_type=interpolation_type,
        filters=filts,
        local_only=local_only,
    )

    print(f"Benchmarking model {model} on filter {filts} with {ncpus} cpus")

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

        used_grid_t = grid_t[(grid_t > tmin) * (grid_t < tmax)]
        grid_mAB = {}
        for filt in filts:
            time_indices = (grid_t > tmin) * (grid_t < tmax)
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
    if ncpus == 1:
        chi2_dict_array = [
            chi2_func(grid_entry_name, data_time_unit=data_time_unit)
            for grid_entry_name in grid_entry_names
        ]
    else:
        chi2_dict_array = p_map(
            chi2_func,
            grid_entry_names,
            data_time_unit,
            num_cpus=ncpus,
        )

    chi2_array_by_filt = {}
    for filt in chi2_dict_array[0].keys():
        chi2_array_by_filt[filt] = [dict_entry[filt] for dict_entry in chi2_dict_array]

    results_dct = {}
    results_dct[model] = {}

    print(
        "Stats below are reduced chi2 distribution percentiles (0, 25, 50, 75, 100) for each filter:"
    )

    model_subscript = ""
    if interpolation_type == "tensorflow":
        model_subscript = "_tf"

    # make the outdir
    outpath = f"{outdir}/{model}{model_subscript}"
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    # make the plots
    for figidx, filt in enumerate(filts):
        plt.figure(figidx)
        plt.xlabel(r"$\chi^2 / {\rm d.o.f.}$")
        plt.ylabel("Count")
        plt.hist(chi2_array_by_filt[filt], label=filt, bins=51, histtype="step")
        plt.legend()
        plt.savefig(f"{outpath}/{filt}.pdf", bbox_inches="tight")
        plt.close()

        percentiles_list = [
            np.round(np.percentile(chi2_array_by_filt[filt], 0), 2),
            np.round(np.percentile(chi2_array_by_filt[filt], 25), 2),
            np.round(np.percentile(chi2_array_by_filt[filt], 50), 2),
            np.round(np.percentile(chi2_array_by_filt[filt], 75), 2),
            np.round(np.percentile(chi2_array_by_filt[filt], 100), 2),
        ]

        print(filt, percentiles_list)

        results_dct[model][filt] = percentiles_list

    with open(f"{outpath}/benchmark_chi2_percentiles_0_25_50_75_100.json", "w") as f:
        # save json file with filter-by-filter details
        json.dump(results_dct, f)

    print("Saved json file containing reduced chi2 percentiles.")


def main():
    parser = get_parser()
    args = parser.parse_args()
    create_benchmark(**vars(args))
