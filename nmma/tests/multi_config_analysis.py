from argparse import Namespace
import os
import pytest


from ..em import multi_config_analysis


@pytest.fixture(scope="module")
def args():
    workingDir = os.path.dirname(__file__)
    dataDir = os.path.join(workingDir, "data")
    configDir = os.path.join(workingDir, "data/multi_config_analysis")

    args = Namespace(config=configDir, process=2, parallel=True)

    return args


def test_analysis(args):
    multi_config_analysis.main(args)
