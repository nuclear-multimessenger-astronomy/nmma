from argparse import Namespace
import os
import pytest
import shutil

from ..em import multi_config_analysis


WORKING_DIR = os.path.dirname(__file__)


@pytest.fixture(scope="module")
def args():
    WORKING_DIR = os.path.dirname(__file__)
    os.environ["WORKING_DIR"] = WORKING_DIR
    config = os.path.join(WORKING_DIR, "data/multi_config_analysis/config.yaml")

    args = Namespace(config=config, process=2, parallel=False)

    return args


def test_analysis_multi(args):
    multi_config_analysis.main(args)


# Randomly fails on GitHub runners
# def test_analysis_parallel(args):
#    args.parallel = True
#    multi_config_analysis.main(args)
