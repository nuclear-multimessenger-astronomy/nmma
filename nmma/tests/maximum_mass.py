from argparse import Namespace
import os
from pathlib import Path
import pytest
import shutil


from ..post_processing import maximum_mass_constraint

@pytest.fixture(scope="module")
def args():
    workingDir = os.path.dirname(__file__)
    dataDir = os.path.join(workingDir, "data")
    priorDir = Path(__file__).resolve().parent.parent.parent
    priorDir = os.path.join(priorDir, "priors")

    args = Namespace(
        outdir="outdir",
        prior = f"{priorDir}/maximum_mass_resampling.prior",
        joint_posterior = f"{dataDir}/GW+KN+GRB_posterior",
        eos_path_macro = f"{dataDir}/eos_macro",
        eos_path_micro = f"{dataDir}/eos_micro",
        nlive = 32,
        use_M_Kepler = False
    )

    return args

@pytest.fixture(autouse=True)
def cleanup_outdir(args):
    yield
    if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir)

def test_maximum_mass_resampling(args):
    
    maximum_mass_constraint.main(args)
