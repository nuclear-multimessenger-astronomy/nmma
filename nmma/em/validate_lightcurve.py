import os
import numpy as np
import argparse
import json

from .utils import DEFAULT_FILTERS
from .io import loadEvent

def get_parser(**kwargs):
    """
    Arguments provided when function is called from command line
    """
    add_help = kwargs.get("add_help", True)

    parser = argparse.ArgumentParser(
        description="Validation that a lightcurve meets a minimum number of observations within a set time.",
        add_help=add_help,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the data file in [time(isot) filter magnitude error] format",
    )
    parser.add_argument(
        "--filters",
        default="",
        type=str,
        help="Comma separated list of filters to validate against. If not provided, all filters in the data will be used.",
    )
    parser.add_argument(
        "--min-obs",
        default=3,
        type=int,
        help="Minimum number of observations required in each filter.",
    )
    parser.add_argument(
        "--cutoff-time",
        default=0,
        type=float,
        help="Cutoff time (relative to the first data point) that the minimum observations must be in. If not provided, the entire lightcurve will be evaluated",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Suppress output",
    )
    return parser

def validate_lightcurve(args):
    """
    Evaluates whether the lightcurve has the requisite user-defined number of observations in the filters provided within the defined time window from the first observation

    Args:
        args (argparse.Namespace): Arguments provided when function is called from command line. See get_parser() for individual arguments.

    Returns:
        bool: True if the lightcurve meets the minimum number of observations in the defined time window, False otherwise.
    """
    data = loadEvent(args.data)

    ## determine filters to consider
    if args.filters:
        filters_to_check = args.filters.replace(" ", "").split(",")
    else:
        filters_to_check = list(data.keys())

    ## determine time window to consider
    min_time = np.min([np.min(array[:,0]) for array in data.values()])
    max_time = min_time + args.cutoff_time if args.cutoff_time > 0 else np.max([np.max(array[:,0]) for array in data.values()])

    ## evaluate lightcurve for each filter
    for filter in filters_to_check:
        if filter not in DEFAULT_FILTERS:
            raise ValueError(f"Filter {filter} not in supported filter list")
        elif filter not in data.keys():
            print(f"{filter} not present in data file, cannot validate") if not args.silent else None
            return False
        filter_data_indices = np.where(data[filter][:,0] <= max_time)[0]
        filter_data = [data[filter][i] for i in filter_data_indices]
        
        ## evaluate the number of detections
        num_observations = sum(1 for value in filter_data if value[1] != np.inf and not np.isnan(value[1]))
        num_detections = sum(1 for value in filter_data if value[2] != np.inf and not np.isnan(value[2]) and value[2] != 99)
        if num_detections < args.min_obs:
            print(f"{filter} filter has {num_detections} detections, less than the required {args.min_obs}") if not args.silent else None
            return False
        else:
            continue
    print(f"Lightcurve has at least {args.min_obs} detections in the filters within the first {max_time-min_time} days") if not args.silent else None

    return True

def main(args=None):
    if args is None:
        parser = get_parser()
        args = parser.parse_args(args)
    validate_lightcurve(args)