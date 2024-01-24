#!/usr/bin/env python
import argparse
import pathlib
import h5py
import numpy as np
import os

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()


class Grid:
    def __init__(
        self, gridpath, base_dirname="lcs_grid", base_filename="lcs", random_seed=21
    ):
        # If gridpath is fully-qualified, ignore BASE_DIR
        if gridpath[0] == "/":
            full_gridpath = gridpath
        else:
            full_gridpath = BASE_DIR / gridpath

        self.file = h5py.File(full_gridpath)
        self.keys_init = [x for x in self.file.keys()]

        self.base_dirname = base_dirname
        self.base_filename = base_filename

        self.rng = np.random.default_rng(random_seed)
        self.copy_keys()

    def copy_keys(self):
        self.keys_copy = self.keys_init.copy()

    def shuffle_keys(self):
        self.rng.shuffle(self.keys_copy)

    def downsample(self, factor=10, shuffle=False):
        print(
            f"Shuffling and downsampling grid by a factor of {factor}..."
            if shuffle
            else f"Downsampling grid by a factor of {factor}..."
        )
        self.copy_keys()

        shuffle_tag = ""
        if shuffle:
            shuffle_tag = "shuffled_"
            self.shuffle_keys()

        self.save_dirname = f"{self.base_dirname}_{shuffle_tag}downsampled_{factor}x"
        self.save_filename = (
            f"{self.base_filename}_{shuffle_tag}downsampled_{factor}x.h5"
        )

        self.keys_save = self.keys_copy[::factor]

        self.save()
        print("Done.")

    def fragment(self, factor=10, shuffle=False):
        print(
            f"Shuffling and fragmenting grid by a factor of {factor}..."
            if shuffle
            else f"Fragmenting grid by a factor of {factor}..."
        )
        self.copy_keys()

        shuffle_tag = ""
        if shuffle:
            shuffle_tag = "shuffled_"
            self.shuffle_keys()

        new_dirname = f"{self.base_dirname}_{shuffle_tag}fragmented"
        new_filename = f"{self.base_filename}_{shuffle_tag}fragmented"

        self.keys_save = np.array_split(np.array(self.keys_copy), factor)

        for i in range(factor):
            print(f"{i+1}/{factor}")

            self.save_dirname = f"{new_dirname}_{i+1}_of_{factor}"
            self.save_filename = f"{new_filename}_{i+1}_of_{factor}.h5"

            self.save(index=i)
        print("Done.")

    def save(self, index=None):
        if index is None:
            keys = self.keys_save
        else:
            keys = self.keys_save[index]

        os.makedirs(self.save_dirname, exist_ok=True)
        with h5py.File(f"{self.save_dirname}/{self.save_filename}", "w") as new_file:
            for key in keys:
                original_obj = self.file[key]
                new_file.copy(original_obj, key)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gridpath",
        type=str,
        help="Path to grid files (fully-qualified or within nmma base directory)",
    )
    parser.add_argument(
        "--base-dirname",
        type=str,
        default="lcs_grid",
        help="Base name of directory to save new grid(s)",
    )
    parser.add_argument(
        "--base-filename",
        type=str,
        default="lcs",
        help="Base name of file to save new grid(s)",
    )
    parser.add_argument(
        "--factor",
        type=int,
        help="Integer factor by which to downsample grid",
    )
    parser.add_argument(
        "--do-downsample",
        action="store_true",
        help="If set, downsample grid by --factor and save results",
    )
    parser.add_argument(
        "--do-fragment",
        action="store_true",
        help="If set, fragment grid into --factor chunks and save results",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="If set, shuffle file keys before downsampling/fragmenting",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=21,
        help="Random seed for numpy",
    )

    return parser


def main(args=None):
    if args is None:
        parser = get_parser()
        args = parser.parse_args()

    if not args.gridpath.endswith(".h5"):
        raise ValueError(
            "Resampling currently only supports grid files with a .h5 extension."
        )

    grid = Grid(
        gridpath=args.gridpath,
        random_seed=args.random_seed,
        base_dirname=args.base_dirname,
        base_filename=args.base_filename,
    )

    if args.do_downsample:
        grid.downsample(factor=args.factor, shuffle=args.shuffle)

    if args.do_fragment:
        grid.fragment(factor=args.factor, shuffle=args.shuffle)
