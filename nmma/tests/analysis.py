from argparse import Namespace
import os
import pytest
import shutil


from ..em import analysis
from tools import analysis_slurm


WORKING_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(WORKING_DIR, "data")


@pytest.fixture(autouse=True)
def cleanup_outdir(args):
    yield
    if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir, ignore_errors=True)


@pytest.fixture(scope="module")
def args():
    args = Namespace(
        model="Bu2019nsbh",
        interpolation_type="tensorflow",
        svd_path=DATA_DIR,
        outdir="outdir",
        label="injection",
        trigger_time=None,
        data=None,
        prior="priors/Bu2019lm.prior",
        tmin=0.1,
        tmax=10.0,
        dt=0.5,
        dt_inj=0.5,
        log_space_time=False,
        photometric_error_budget=0.1,
        soft_init=False,
        bestfit=True,
        svd_mag_ncoeff=10,
        svd_lbol_ncoeff=10,
        filters="ztfr",
        Ebv_max=0.0,
        grb_resolution=5,
        jet_type=0,
        error_budget="0",
        sampler="pymultinest",
        cpus=1,
        nlive=64,
        reactive_sampling=False,
        seed=42,
        injection=f"{DATA_DIR}/Bu2019lm_injection.json",
        injection_num=0,
        injection_detection_limit=None,
        injection_outfile="outdir/lc.csv",
        injection_model=None,
        ignore_timeshift=False,
        remove_nondetections=True,
        detection_limit=None,
        detection_limit_fits_file=None,
        with_grb_injection=False,
        prompt_collapse=False,
        ztf_sampling=False,
        ztf_uncertainties=False,
        ztf_ToO=None,
        train_stats=False,
        rubin_ToO=False,
        rubin_ToO_type=None,
        xlim="0,14",
        ylim="22,16",
        generation_seed=42,
        plot=True,
        bilby_zero_likelihood_mode=False,
        photometry_augmentation=False,
        photometry_augmentation_seed=0,
        photometry_augmentation_N_points=10,
        photometry_augmentation_filters=None,
        photometry_augmentation_times=None,
        conditional_gaussian_prior_thetaObs=False,
        conditional_gaussian_prior_N_sigma=1,
        sample_over_Hubble=False,
        sampler_kwargs="{}",
        verbose=False,
        local_only=True,
        skip_sampling=False,
        fits_file=None,
        cosiota_node_num=10,
        ra=None,
        dec=None,
        fetch_Ebv_from_dustmap=False,
        systematics_file=None,
    )

    return args


def test_analysis_systematics_with_time(args):

    args.filters = "ztfr"
    args.systematics_file = f"{DATA_DIR}/systematics_with_time.yaml"
    analysis.main(args)


def test_analysis_systematics_with_time_and_filters(args):

    args.filters = "ztfr,sdssu,2massks"
    args.systematics_file = f"{DATA_DIR}/systematics_with_time_combined_filters.yaml"
    analysis.main(args)


def test_analysis_systematics_without_time(args):

    args.filters = "ztfr"
    args.systematics_file = f"{DATA_DIR}/systematics_without_time.yaml"
    analysis.main(args)


def test_analysis_tensorflow(args):

    args.filters = "ztfr"
    analysis.main(args)


def test_analysis_sklearn_gp(args):

    args.filters = "ztfr"
    args.interpolation_type = "sklearn_gp"
    analysis.main(args)


def test_nn_analysis(args):

    args.model = "Ka2017"
    args.sampler = "neuralnet"
    args.prior = "priors/Ka2017.prior"
    args.dt = 0.25
    args.dt_inj = 0.25
    args.filters = "ztfg,ztfr,ztfi"
    args.local_only = False
    args.injection = f"{DATA_DIR}/Ka2017_injection.json"
    analysis.main(args)


def test_analysis_slurm(args):

    args_slurm = Namespace(
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

    args.__dict__.update(args_slurm.__dict__)

    args.filters = "ztfr"
    analysis_slurm.main(args)
