import numpy as np
import pandas as pd
import os, sys, time, glob
import json
import warnings
from tqdm import tqdm
import nflows.utils as torchutils
from IPython.display import clear_output
from time import time
from time import sleep
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from os.path import exists
import argparse

from data_processing import load_embedding_dataset

def main():
    parser = argparse.ArgumentParser(
        description='Load NMMA light curves into training tensors'
    )
    parser.add_argument(
        '--dir_path_var', type=str, required=True, 
        help='Directory containing varied light curves.',
    )
    parser.add_argument(
        '--inj_file_var', type=str, required=True,
        help='Name of varied injection file in dir-path-var.',
    )
    parser.add_argument(
        '--label_var', type=str, required=True,
        help='Label of the varied light curves.',
    )
    parser.add_argument(
        '--dir_path_fix', type=str, required=True, 
        help='Directory containing fixed light curves. Use None to skip.',
    )
    parser.add_argument(
        '--inj_file_fix', type=str, required=True,
        help='Name of fixed injection file in dir-path-fix. Use None to skip.',
    )
    parser.add_argument(
        '--label_fix', type=str, required=True,
        help='Label of the fixed light curves. Use None to skip.',
    )
    parser.add_argument(
        '--detection_limit', type=float, required=True,
        help='Photometric detection limit.',
        default=22.0, 
    )
    parser.add_argument(
        '--filters',
        type=lambda s: [f.strip() for f in s.split(',')],
        help='Comma-separated list of photometric bands, e.g. ztfg,ztfr,ztfi',
    )
    parser.add_argument(
        '--dt', type=float,
        help='Time step in day (default: 0.1)',
        default=0.1, 
    )
    parser.add_argument(
        '--data_filler', type=float,
        help='Value to use for data padding.',
        default=22.0, 
    )
    parser.add_argument(
        '--params',
        type=lambda s: [f.strip() for f in s.split(',')],
        help='Comma-separated list of inference parameters.',
    )
    parser.add_argument(
        '--num_repeats', type=int,
        help='Number of repeated injections.',
        default=1,
    )
    parser.add_argument(
        '--save_lc_data_fix', type=str,
        help='Optionally save fixed lc data in tensor form.',
    )
    parser.add_argument(
        '--save_lc_params_fix', type=str,
        help='Optionally save fixed parameters in tensor form.',
    )
    parser.add_argument(
        '--save_lc_data_var', type=str,
        help='Optionally save varied lc data in tensor form.',
    )
    parser.add_argument(
        '--save_lc_params_var', type=str,
        help='Optionally save varied lc parameters in tensor form.',
    )

    args = parser.parse_args()

    if args.dir_path_fix and args.inj_file_fix and args.label_fix:
        lc_data_var, lc_params_var, lc_data_fix, lc_params_fix = load_embedding_dataset(
            dir_path_var=args.dir_path_var,
            inj_file_var=args.inj_file_var,
            label_var=args.label_var,
            dir_path_fix=args.dir_path_fix,
            inj_file_fix=args.inj_file_fix,
            label_fix=args.label_fix,
            detection_limit=args.detection_limit,
            bands=args.filters,
            step=args.dt,
            data_filler=args.data_filler,
            params=args.params,
            num_repeats=args.num_repeats,
        )
    else:
        lc_data_fix, lc_params_fix = None, None
        lc_data_var, lc_params_var, _, _ = load_embedding_dataset(
            dir_path_var=args.dir_path_var,
            inj_file_var=args.inj_file_var,
            label_var=args.label_var,
            dir_path_fix=None,
            inj_file_fix=None,
            label_fix=None,
            detection_limit=args.detection_limit,
            bands=args.filters,
            step=args.dt,
            data_filler=args.data_filler,
            params=args.params,
            num_repeats=args.num_repeats,
        )

    if args.save_lc_data_fix:
        torch.save(lc_data_fix, args.save_lc_data_fix)
    if args.save_lc_params_fix:
        torch.save(lc_params_fix, args.save_lc_params_fix)
    if args.save_lc_data_var:
        torch.save(lc_data_var, args.save_lc_data_var)
    if args.save_lc_params_var:
        torch.save(lc_params_var, args.save_lc_params_var)

if __name__ == "__main__":
    main()
