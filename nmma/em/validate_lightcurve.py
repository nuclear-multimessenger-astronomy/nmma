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
        default='',
        type=float,
        help="cutoff time from the first data point that the minimum observations must be in. If not provided, the entire lightcurve will be evaluated",
    )
    return parser

