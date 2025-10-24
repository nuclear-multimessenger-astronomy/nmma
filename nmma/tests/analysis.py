import os
import pytest
import shutil


from ..em import analysis, em_parsing, slurm_handling

WORKING_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(WORKING_DIR, "data")


@pytest.fixture(autouse=True)
def cleanup_outdir(args):
    yield
    if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir, ignore_errors=True)


@pytest.fixture(scope="module")
def args():
    args = em_parsing.parsing_and_logging(em_parsing.multi_wavelength_analysis_parser, [])
    non_default_args = dict(
        em_model="Bu2019nsbh",
        interpolation_type="tensorflow",
        svd_path=DATA_DIR,
        label="injection",
        prior_file="priors/Bu2019lm.prior",
        em_tmin=0.1,
        em_tmax=14.0,
        injection_em_tmax=12.0,
        em_tstep=0.5,
        bestfit=True,
        filters="ztfr",
        Ebv_max=0.0,
        em_error_budget=0,
        nlive=64,
        injection_file=f"{DATA_DIR}/Bu2019lm_injection.json",
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

def test_analysis_systematics_with_time_and_filters(args):

    args.filters = ["ztfr", "sdssu", "2massks"]
    args.systematics_file = f"{DATA_DIR}/systematics_with_time_combined_filters.yaml"
    analysis.main(args)

def test_analysis_tensorflow(args):
    args.systematics_file = None
    args.filters = "ztfr"
    analysis.main(args)


def test_analysis_sklearn_gp(args):
    args.interpolation_type = "sklearn_gp"
    analysis.main(args)


def test_nn_analysis(args):

    args.em_model = "Ka2017"
    args.sampler = "neuralnet"
    args.prior_file = "priors/Ka2017.prior"
    args.em_tstep = 0.25
    args.filters = ["ztfg", "ztfr", "ztfi"]
    args.local_only = False
    args.injection_file = f"{DATA_DIR}/Ka2017_injection.json"
    analysis.main(args)


def test_analysis_slurm(args):

    args_slurm = dict(
        Ncore=8,
        job_name="lightcurve-analysis",
        base_dir=os.getcwd(),
        logs_dir_name="slurm_logs",
        cluster_name="Expanse",
        partition_type="shared",
        nodes=1,
        gpus=0,
        memory_GB=64,
        time="24:00:00",
        mail_type="NONE",
        mail_user="",
        python_env_name="nmma_env",
        script_name="slurm.sub",
    )

    args.__dict__.update(args_slurm)

    slurm_handling.slurm_analysis(args) 
    shutil.rmtree(os.path.join(args.base_dir, args.logs_dir_name), ignore_errors=True)
