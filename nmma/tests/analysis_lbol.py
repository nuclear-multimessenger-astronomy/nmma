import os
import toml
import pytest


from ..em import analysis, em_parsing


@pytest.fixture(scope="module")
def args():
    workingDir = os.path.dirname(__file__)
    dataDir = os.path.join(workingDir, "data")


    non_default_args = dict(
        em_model="Arnett_modified",
        outdir="outdir",
        label="lbol_test",
        em_trigger_time=60168.79041667,
        data='example_files/lbol/ztf23bqun/23bqun_bbdata.csv',
        prior="example_files/lbol/ztf23bqun/Arnett_modified.priors",
        em_tmin=0.005,
        em_tmax=20.0,
        em_tstep=0.5,
        bestfit=True,
        Ebv_max=0.0,
        error_budget=0.0001,
        nlive=64,
        plot=True
    )
    non_default_file =os.path.join(dataDir, "config.toml")
    non_default_args = {k.replace("_", "-"): v for k, v in non_default_args.items()}
    with open(non_default_file, "w") as f:
        toml.dump(non_default_args, f)

    args = em_parsing.parsing_and_logging(em_parsing.bolometric_parser, [non_default_file])
    args.__dict__.update(non_default_args)
    return args


def test_analysis_lbol(args):

    analysis.lbol_main(args)
