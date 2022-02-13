import numpy as np
import argparse
import glob

from .training import SVDTrainingModel
from .utils import read_files
from . import model_parameters


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
        "--dath-path",
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
        "--svd-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for mag evaluation (default: 10)",
    )
    parser.add_argument(
        "--filters",
        type=str,
        help="A comma seperated list of filters to use (e.g. g,r,i). If none is provided, will use all the filters available",
        default="u,g,r,i,z,y,J,H,K",
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
    args = parser.parse_args()

    # initialize light curve model
    sample_times = np.arange(args.tmin, args.tmax + args.dt, args.dt)
    filts = args.filters.split(",")

    MODEL_FUNCTIONNAMES = [k for k, v in model_parameters.__dict__.items() if v is dict]
    print(MODEL_FUNCTIONNAMES)

    data_path = "output/bulla_2Comp_mv/"
    filenames = glob.glob("%s/*.dat" % data_path)
    data = read_files(filenames)

    SVDTrainingModel(
        args.model,
        data,
        sample_times,
        filts,
        n_coeff=args.svd_ncoeff,
        svd_path=args.svd_path,
        interpolation_type=args.interpolation_type,
    )
