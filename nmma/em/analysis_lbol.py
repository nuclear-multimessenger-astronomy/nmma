import argparse
import json
import os
from pathlib import Path
import yaml
from ast import literal_eval
import copy

import bilby
import bilby.core
import matplotlib
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from astropy import time
from bilby.core.likelihood import ZeroLikelihood

from .likelihood import BolometricLightCurve
from .model_lbol import SimpleBolometricLightCurveModel
from .prior import create_prior_from_args
from .utils import running_in_ci

matplotlib.use("agg")
matplotlib.rcParams["text.usetex"] = not running_in_ci()


def get_parser(**kwargs):
    add_help = kwargs.get("add_help", True)

    parser = argparse.ArgumentParser(
        description="Inference on astronomical transient parameters with bolometric luminosity data.",
        add_help=add_help,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Name of the configuration file containing parameter values.",
    )
    parser.add_argument(
        "--model", type=str, help="Name of the kilonova model to be used"
    )
    parser.add_argument(
        "--interpolation-type",
        type=str,
        help="SVD interpolation scheme.",
        default="sklearn_gp",
    )
    parser.add_argument(
        "--svd-path",
        type=str,
        help="Path to the SVD directory, with {model}_mag.pkl and {model}_lbol.pkl",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Path to the output directory",
        default="outdir",
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Label for the run",
    )
    parser.add_argument(
        "--trigger-time",
        type=float,
        help="Trigger time in modified julian day, not required if injection set is provided",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the data file in [time(isot) filter magnitude error] format",
    )
    parser.add_argument("--prior", type=str, help="Path to the prior file")
    parser.add_argument(
        "--tmin",
        type=float,
        default=0.05,
        help="Days to start analysing from the trigger time (default: 0.05)",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=14.0,
        help="Days to stop analysing from the trigger time (default: 14)",
    )
    parser.add_argument(
        "--dt", type=float, default=0.1, help="Time step in day (default: 0.1)"
    )
    parser.add_argument(
        "--log-space-time",
        action="store_true",
        default=False,
        help="Create the sample_times to be uniform in log-space",
    )
    parser.add_argument(
        "--n-tstep",
        type=int,
        default=50,
        help="Number of time steps (used with --log-space-time, default: 50)",
    )
    parser.add_argument(
        "--error-budget",
        type=float,
        default=0.1,
        help="Bolometric error (default: 10 percent of relative error)",
    )
    parser.add_argument(
        "--svd-lbol-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for lbol evaluation (default: 10)",
    )
    parser.add_argument(
        "--use-Ebv",
        action="store_true",
        default=False,
        help="If using the Ebv extinction during the inference",
    )
    parser.add_argument(
        "--Ebv-max",
        type=float,
        default=0.5724,
        help="Maximum allowed value for Ebv (default:0.5724)",
    )
    parser.add_argument(
        "--conditional-gaussian-prior-thetaObs",
        action="store_true",
        default=False,
        help="The prior on the inclination is against to a gaussian prior centered at zero with sigma = thetaCore / N_sigma",
    )

    parser.add_argument(
        "--conditional-gaussian-prior-N-sigma",
        default=1,
        type=float,
        help="The input for N_sigma; to be used with conditional-gaussian-prior-thetaObs set to True",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="pymultinest",
        help="Sampler to be used (default: pymultinest)",
    )
    parser.add_argument(
        "--soft-init",
        action="store_true",
        default=False,
        help="To start the sampler softly (without any checking, default: False)",
    )
    parser.add_argument(
        "--sampler-kwargs",
        default="{}",
        type=str,
        help="Additional kwargs (e.g. {'evidence_tolerance':0.5}) for bilby.run_sampler, put a double quotation marks around the dictionary",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=1,
        help="Number of cores to be used, only needed for dynesty (default: 1)",
    )
    parser.add_argument(
        "--nlive", type=int, default=2048, help="Number of live points (default: 2048)"
    )
    parser.add_argument(
        "--reactive-sampling",
        action="store_true",
        default=False,
        help="To use reactive sampling in ultranest (default: False)",
    )
    parser.add_argument(
        "--seed",
        metavar="seed",
        type=int,
        default=42,
        help="Sampling seed (default: 42)",
    )
    parser.add_argument(
        "--injection", metavar="PATH", type=str, help="Path to the injection json file"
    )
    parser.add_argument(
        "--plot", action="store_true", default=False, help="add best fit plot"
    )
    parser.add_argument(
        "--bestfit",
        help="Save the best fit parameters and magnitudes to JSON",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--bilby-zero-likelihood-mode",
        action="store_true",
        default=False,
        help="enable prior run",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="print out log likelihoods",
    )
    parser.add_argument(
        "--skip-sampling",
        help="If analysis has already run, skip bilby sampling and compute results from checkpoint files. Combine with --plot to make plots from these files.",
        action="store_true",
        default=False,
    )
    parser.add_argument(  # no use in this script
        "--systematics-file",
        metavar="PATH",
        help="Path to systematics configuration file",
        default=None,
    )

    return parser


def analysis(args):

    if args.sampler == "pymultinest":
        if len(args.outdir) > 64:
            print(
                "WARNING: output directory name is too long, it should not be longer than 64 characters"
            )
            exit()

    bilby.core.utils.setup_logger(outdir=args.outdir, label=args.label)
    bilby.core.utils.check_directory_exists_and_if_not_mkdir(args.outdir)

    # initialize light curve model
    if args.log_space_time:
        if args.n_tstep:
            n_step = args.n_tstep
        else:
            n_step = int((args.tmax - args.tmin) / args.dt)
        sample_times = np.logspace(
            np.log10(args.tmin), np.log10(args.tmax + args.dt), n_step
        )
    else:
        sample_times = np.arange(args.tmin, args.tmax + args.dt, args.dt)

    # create the data if an injection set is given
    if args.injection:
        # FIXME with actual injection functionality
        pass
    else:
        # load the bolometric data
        data = pd.read_csv(args.data)

    error_budget = args.error_budget

    light_curve_model = SimpleBolometricLightCurveModel(
        model=args.model, sample_times=sample_times
    )

    # setup the prior
    priors = create_prior_from_args(args.model.split(","), args)

    # setup the likelihood
    likelihood_kwargs = dict(
        light_curve_model=light_curve_model,
        light_curve_data=data,
        tmin=args.tmin,
        tmax=args.tmax,
        error_budget=error_budget,
        verbose=args.verbose,
    )

    likelihood = BolometricLightCurve(**likelihood_kwargs)
    if args.bilby_zero_likelihood_mode:
        likelihood = ZeroLikelihood(likelihood)

    # fetch the additional sampler kwargs
    sampler_kwargs = literal_eval(args.sampler_kwargs)
    print("Running with the following additional sampler_kwargs:")
    print(sampler_kwargs)

    # check if it is running with reactive sampler
    if args.reactive_sampling:
        if args.sampler != "ultranest":
            print("Reactive sampling is only available in ultranest")
        else:
            print("Running with reactive-sampling in ultranest")
            nlive = None
    else:
        nlive = args.nlive

    if args.skip_sampling:
        print("Sampling for 1 iteration and plotting checkpointed results.")
        if args.sampler == "pymultinest":
            sampler_kwargs["max_iter"] = 1
        elif args.sampler == "ultranest":
            sampler_kwargs["niter"] = 1
        elif args.sampler == "dynesty":
            sampler_kwargs["maxiter"] = 1

    result = bilby.run_sampler(
        likelihood,
        priors,
        sampler=args.sampler,
        outdir=args.outdir,
        label=args.label,
        nlive=nlive,
        seed=args.seed,
        soft_init=args.soft_init,
        queue_size=args.cpus,
        check_point_delta_t=3600,
        **sampler_kwargs,
    )

    result.save_posterior_samples()

    if args.injection:
        # FIXME
        pass
    else:
        result.plot_corner()

    if args.bestfit or args.plot:
        import matplotlib.pyplot as plt

        posterior_file = os.path.join(
            args.outdir, f"{args.label}_posterior_samples.dat"
        )

        ##########################
        # Fetch bestfit parameters
        ##########################
        posterior_samples = pd.read_csv(posterior_file, header=0, delimiter=" ")
        bestfit_idx = np.argmax(posterior_samples.log_likelihood.to_numpy())
        bestfit_params = posterior_samples.to_dict(orient="list")
        for key in bestfit_params.keys():
            bestfit_params[key] = bestfit_params[key][bestfit_idx]
        print(
            f"Best fit parameters: {str(bestfit_params)}\nBest fit index: {bestfit_idx}"
        )

        #########################
        # Generate the lightcurve
        #########################
        lbol = light_curve_model.generate_lightcurve(sample_times, bestfit_params)
        lbol_dict = {}
        lbol_dict["lbol"] = lbol
        lbol_dict["bestfit_sample_times"] = sample_times

        if "timeshift" in bestfit_params:
            lbol_dict["bestfit_sample_times"] = (
                lbol_dict["bestfit_sample_times"] + bestfit_params["timeshift"]
            )

        matplotlib.rcParams.update({"font.size": 12, "font.family": "Times New Roman"})

        plt.figure(1)
        plotName = os.path.join(args.outdir, f"{args.label}_lightcurves.png")
        color = "coral"

        t = data["phase"].to_numpy()
        y = data["Lbb"].to_numpy()
        sigma_y = data["Lbb_unc"].to_numpy()

        idx = np.where(~np.isnan(y))[0]
        t, y, sigma_y = t[idx], y[idx], sigma_y[idx]

        idx = np.where(np.isfinite(sigma_y))[0]
        plt.errorbar(
            t[idx],
            y[idx],
            sigma_y[idx],
            fmt="o",
            color="k",
            markersize=12,
        )

        idx = np.where(~np.isfinite(sigma_y))[0]
        plt.errorbar(t[idx], y[idx], sigma_y[idx], fmt="v", color="k", markersize=12)

        plt.plot(
            lbol_dict["bestfit_sample_times"],
            lbol_dict["lbol"],
            color=color,
            linewidth=3,
            linestyle="--",
        )

        plt.ylabel("L [erg / s]")
        plt.xlabel("Time [days]")
        plt.tight_layout()
        plt.savefig(plotName)
        plt.close()

    return


def main(args=None):
    if args is None:
        parser = get_parser()
        args = parser.parse_args()
        if args.config is not None:
            yaml_dict = yaml.safe_load(Path(args.config).read_text())
            for analysis_set in yaml_dict.keys():
                params = yaml_dict[analysis_set]
                for key, value in params.items():
                    key = key.replace("-", "_")
                    if key not in args:
                        print(f"{key} not a known argument... please remove")
                        exit()
                    setattr(args, key, value)
                analysis(args)
        else:
            analysis(args)
    else:
        analysis(args)
