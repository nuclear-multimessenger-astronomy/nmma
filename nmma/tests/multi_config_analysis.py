from argparse import Namespace
import os
import pytest


from nmma.em import multi_config_analysis


@pytest.fixture(scope="module")
def args():
    workingDir = os.path.dirname(__file__)
    config = os.path.join(workingDir, "data/multi_config_analysis/config.yaml")

    arguments = Namespace(config=config, process=1, parallel=False)

    return arguments


def test_analysis(args):
    multi_config_analysis.main(args)

