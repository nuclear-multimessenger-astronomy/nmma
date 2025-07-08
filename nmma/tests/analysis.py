from argparse import Namespace
import os
import pytest
import shutil


from ..em import analysis, em_parsing
from tools import analysis_slurm


WORKING_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(WORKING_DIR, "data")


@pytest.fixture(autouse=True)
def cleanup_outdir(args):
    yield
    if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir)


@pytest.fixture(scope="module")
def args():
    args = em_parsing.parsing_and_logging(em_parsing.em_analysis_parser, [])
    non_default_args = dict(
        em_model="Bu2019nsbh",
        interpolation_type="tensorflow",
        svd_path=DATA_DIR,
        label="injection",
        prior="priors/Bu2019lm.prior",
        em_tmin=0.1,
        em_tmax=10.0,
        em_tstep=0.5,
        bestfit=True,
        filters="ztfr",
        Ebv_max=0.0,
        em_error_budget=0,
        nlive=64,
        injection=f"{DATA_DIR}/Bu2019lm_injection.json",
        injection_outfile="outdir/lc.csv",
        plot=True,
    )
    for key, value in non_default_args.items():
        setattr(args, key, value)

    return args


def test_analysis_systematics_with_time(args):
    args.systematics_file = f"{DATA_DIR}/systematics_with_time.yaml"
    analysis.main(args)


def test_analysis_systematics_without_time(args):

    args.systematics_file = f"{DATA_DIR}/systematics_without_time.yaml"
    analysis.main(args)


def test_analysis_tensorflow(args):

    analysis.main(args)


def test_analysis_sklearn_gp(args):
    args.systematics_file = None
    args.interpolation_type = "sklearn_gp"
    analysis.main(args)


def test_nn_analysis(args):

    args.em_model = "Ka2017"
    args.sampler = "neuralnet"
    args.prior = "priors/Ka2017.prior"
    args.em_tstep = 0.25
    args.filters = "ztfg,ztfr,ztfi"
    args.local_only = False
    args.injection = f"{DATA_DIR}/Ka2017_injection.json"
    analysis.main(args)


def test_analysis_slurm(args):

    args_slurm = dict(
        Ncore=8,
        job_name="lightcurve-analysis",
        logs_dir_name="slurm_logs",
        cluster_name="Expanse",
        partition_type="shared",
        nodes=1,
        gpus=0,
        memory_GB=64,
        time="24:00:00",
        mail_type="NONE",
        mail_user="",
        account_name="umn131",
        python_env_name="nmma_env",
        script_name="slurm.sub",
    )

    args.__dict__.update(args_slurm)

    analysis_slurm.main(args)
