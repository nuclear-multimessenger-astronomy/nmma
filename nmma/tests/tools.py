import os
from argparse import Namespace
from ..em.lightcurve_handling import resample_lightcurve_grid
from ..em.io import convert_skyportal_lcs

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
        downsample=True,
        fragment=False,
        factor=5,
        shuffle=False,
        remove = True,
    )
    resample_lightcurve_grid(args)

    # Downsample by 5x, shuffle
    args.shuffle = True
    resample_lightcurve_grid(args)

    # Fragment by 5x, shuffle
    args.do_downsample = False
    args.do_fragment = True
    resample_lightcurve_grid(args)

    # Fragment by 5x, no shuffle
    args.shuffle = False
    resample_lightcurve_grid(args)


def test_lc_conversion():
    workingDir = os.path.dirname(__file__)
    filepath = os.path.join(workingDir, "data", "ZTF23aaxeacr_partial.csv")
    convert_skyportal_lcs(filepath=filepath)
