import os
from argparse import Namespace

from tools.resample_grid import Grid
from tools import convert_skyportal_lcs


def test_resampling():
    workingDir = os.path.dirname(__file__)

    base_dirname = "lcs_grid"
    base_filename = "lcs"
    gridpath = os.path.join(workingDir, "data", "lowmass_collapsar_updated.h5")

    grid = Grid(
        gridpath=gridpath,
        random_seed=21,
        base_dirname=base_dirname,
        base_filename=base_filename,
    )

    grid.downsample(factor=5, shuffle=False)
    grid.downsample(factor=5, shuffle=True)

    grid.fragment(factor=5, shuffle=False)
    grid.fragment(factor=5, shuffle=True)


def test_lc_conversion():
    workingDir = os.path.dirname(__file__)
    filepath = os.path.join(workingDir, "data", "ZTF23aaxeacr_partial.csv")

    args = Namespace(
        filepath=filepath,
    )

    convert_skyportal_lcs.main(args)
