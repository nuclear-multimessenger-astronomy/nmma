import os
from argparse import Namespace

from tools import resample_grid
from tools import convert_skyportal_lcs


def test_resampling():
    workingDir = os.path.dirname(__file__)

    base_dirname = "lcs_grid"
    base_filename = "lcs"
    gridpath = os.path.join(workingDir, "data", "lowmass_collapsar_updated.h5")

    # Start with downsampling by 5x, no shuffle
    args = Namespace(
        gridpath=gridpath,
        random_seed=21,
        base_dirname=base_dirname,
        base_filename=base_filename,
        do_downsample=True,
        do_fragment=False,
        factor=5,
        shuffle=False,
    )
    resample_grid.main(args)

    # Downsample by 5x, shuffle
    args.shuffle = True
    resample_grid.main(args)

    # Fragment by 5x, shuffle
    args.do_downsample = False
    args.do_fragment = True
    resample_grid.main(args)

    # Fragment by 5x, no shuffle
    args.shuffle = False
    resample_grid.main(args)


def test_lc_conversion():
    workingDir = os.path.dirname(__file__)
    filepath = os.path.join(workingDir, "data", "ZTF23aaxeacr_partial.csv")

    args = Namespace(
        filepath=filepath,
    )

    convert_skyportal_lcs.main(args)
