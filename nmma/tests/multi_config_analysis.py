from argparse import Namespace
import os
import pytest
import shutil

from ..em import multi_config_analysis


WORKING_DIR = os.path.dirname(__file__)


@pytest.fixture(autouse=True)
def cleanup_outdir(args):
    yield
    files = [os.path.join(WORKING_DIR, "outdir_1"), os.path.join(WORKING_DIR, "outdir_2")]
    for file in files:
        
        assert os.path.exists(os.path.join(file, "injection_posterior_samples.dat"))
        if os.path.exists(file):
            shutil.rmtree(file)


@pytest.fixture(scope="module")
def args():
    WORKING_DIR = os.path.dirname(__file__)
    os.environ["WORKING_DIR"] = WORKING_DIR
    config = os.path.join(WORKING_DIR, "data/multi_config_analysis/config.yaml")

    args = Namespace(config=config, process=4, parallel=True)

    return args


def test_analysis_multi(args):
    multi_config_analysis.main(args)


def test_analysis_parallel(args):
    args.parallel = False
    multi_config_analysis.main(args)