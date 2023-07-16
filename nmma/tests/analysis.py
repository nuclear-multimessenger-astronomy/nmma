from argparse import Namespace
import os

from ..em import analysis


def test_analysis():

    dataDir = f"{os.path.dirname(__file__)}/data"

    args = Namespace(
        model="Me2017",
        interpolation_type="sklearn_gp",
        svd_path="svdmodels",
        outdir="outdir",
        label="injection",
        trigger_time=None,
        data=None,
        prior="priors/Me2017.prior",
        tmin=0.1,
        tmax=20.0,
        dt=0.5,
        photometric_error_budget=0.1,
        soft_init=False,
        bestfit=True,
        svd_mag_ncoeff=10,
        svd_lbol_ncoeff=10,
        filters="sdssu",
        Ebv_max=0.0,
        grb_resolution=5,
        jet_type=0,
        error_budget="1",
        sampler="pymultinest",
        cpus=1,
        nlive=512,
        seed=42,
        injection=f"{dataDir}/injection.json",
        injection_num=0,
        injection_detection_limit=None,
        injection_outfile="outdir/lc.csv",
        injection_model=None,
        remove_nondetections=True,
        detection_limit=None,
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
        verbose=False,
    )

    analysis.main(args)
