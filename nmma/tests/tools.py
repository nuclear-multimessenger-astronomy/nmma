import os

from tools.resample_grid import Grid


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
