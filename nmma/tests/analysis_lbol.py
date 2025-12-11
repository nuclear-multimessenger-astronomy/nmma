from argparse import Namespace
import os
import pytest


from ..em import analysis_lbol


@pytest.fixture(scope="module")
def args():
    workingDir = os.path.dirname(__file__)
    dataDir = os.path.join(workingDir, "data")

    args = Namespace(
        model="Arnett_modified",
        outdir="outdir",
        label="lbol_test",
        trigger_time=60168.79041667,
        data="example_files/lbol/ztf23bqun/23bqun_bbdata.csv",
        prior="example_files/lbol/ztf23bqun/Arnett_modified.priors",
        tmin=0.005,
        tmax=20.0,
        dt=0.5,
        log_space_time=False,
        injection=None,
        soft_init=False,
        bestfit=True,
        svd_lbol_ncoeff=10,
        Ebv_max=0.0,
        error_budget=0.0001,
        sampler="pymultinest",
        cpus=1,
        nlive=64,
        reactive_sampling=False,
        seed=42,
        plot=True,
        bilby_zero_likelihood_mode=False,
        conditional_gaussian_prior_thetaObs=False,
        sample_over_Hubble=False,
        sampler_kwargs="{}",
        verbose=False,
        skip_sampling=False,
        fits_file=None,
        cosiota_node_num=10,
        ra=None,
        dec=None,
        fetch_Ebv_from_dustmap=False,
        systematics_file=None,
    )

    return args


def test_analysis_lbol(args):

    analysis_lbol.main(args)
