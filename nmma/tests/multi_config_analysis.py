from argparse import Namespace
import os
import pytest


from ..em import multi_config_analysis


@pytest.fixture(scope="module")
def args():
    workingDir = os.path.dirname(__file__)
    os.environ["workingDir"] = workingDir
    print(workingDir)
    config = os.path.join(workingDir, "data/multi_config_analysis/config.yaml")

    args = Namespace(config=config, process=1, parallel=False)

    return args


def test_analysis_multi(args):
    print(args)
    multi_config_analysis.main(args)

