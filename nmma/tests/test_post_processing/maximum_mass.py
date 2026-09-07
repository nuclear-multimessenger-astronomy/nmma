from argparse import Namespace
from pathlib import Path
import pytest
import shutil


from nmma.post_processing import maximum_mass_constraint

@pytest.fixture(scope="module")
def args():
    working_dir = Path(__file__).parent.parent
    data_dir = working_dir / "data"

    args = Namespace(
        outdir= "outdir",
        prior = data_dir / "maximum_mass_resampling.prior",
        joint_posterior = data_dir / "GW+KN+GRB_posterior.dat",
        eos_path_macro = data_dir / "eos_macro",
        eos_path_micro = data_dir / "eos_micro",
        nlive = 32,
        use_M_Kepler = False
    )

    return args

@pytest.fixture(autouse=True)
def cleanup_outdir(args):
    yield
    if Path(args.outdir).exists():
        shutil.rmtree(args.outdir, ignore_errors=True)

def test_maximum_mass_resampling(args):
    
    maximum_mass_constraint.main(args)
