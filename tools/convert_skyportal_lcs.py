#!/usr/bin/env python
from astropy.table import Table
import numpy as np
from astropy.time import Time
import pathlib
import argparse

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()


def convert_skyportal_lcs(filepath):
    data_path = BASE_DIR / filepath
    try:
        data = Table.read(data_path, format="ascii.csv")
    except Exception as e:
        raise ValueError(f"input data is not in the expected format {e}")

    try:
        local_data_path = str(data_path.with_suffix(".dat"))
        with open(local_data_path, "w") as f:
            # output the data in the format desired by NMMA:
            # remove rows where mag and magerr are missing, or not float, or negative
            data = data[
                np.isfinite(data["mag"])
                & np.isfinite(data["magerr"])
                & (data["mag"] > 0)
                & (data["magerr"] > 0)
            ]
            for row in data:
                tt = Time(row["mjd"], format="mjd").isot
                filt = row["filter"]
                mag = row["mag"]
                magerr = row["magerr"]
                f.write(f"{tt} {filt} {mag} {magerr}\n")
        print(f"Wrote reformatted lightcurve to {local_data_path}")
    except Exception as e:
        raise ValueError(f"failed to format data {e}")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        type=str,
        required=True,
        help="path/name of lightcurve file (including extension) within base nmma directory",
    )

    return parser


def main(args=None):
    if args is None:
        parser = get_parser()
        args = parser.parse_args()

    convert_skyportal_lcs(**vars(args))
