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

from ..utils.models import refresh_models_list
from .injection import create_light_curve_data
from .likelihood import OpticalLightCurve
from .model import create_light_curve_model_from_args, model_parameters_dict
from .prior import create_prior_from_args
from .utils import getFilteredMag, dataProcess, set_mission_name

from .io import loadEvent

matplotlib.use("agg")


def get_parser(**kwargs):
    add_help = kwargs.get("add_help", True)

    parser = argparse.ArgumentParser(
        description="Inference on kilonova ejecta parameters.",
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
        help="Path to the SVD directory with {model}.joblib",
        default="svdmodels",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Path to the output directory",
        default="outdir",
    )
    parser.add_argument(
        "--label", type=str, help="Label for the run", default="injection"
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
        "--dt-inj",
        type=float,
        default=1,
        help="Time step in day for injection (default: 1.0)",
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
        "--photometric-error-budget",
        type=float,
        default=0.1,
        help="Photometric error (mag) (default: 0.1)",
    )
    parser.add_argument(
        "--svd-mag-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for mag evaluation (default: 10)",
    )
    parser.add_argument(
        "--svd-lbol-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for lbol evaluation (default: 10)",
    )
    parser.add_argument(
        "--filters",
        type=str,
        help="A comma seperated list of filters to use (e.g. g,r,i). If none is provided, will use all the filters available",
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
        "--grb-resolution",
        type=float,
        default=5,
        help="The upper bound on the ratio between thetaWing and thetaCore (default: 5)",
    )
    parser.add_argument(
        "--jet-type",
        type=int,
        default=0,
        help="Jet type to used used for GRB afterglow light curve (default: 0)",
    )
    parser.add_argument(
        "--energy-injection",
        action="store_true",
        default=False,
        help="To include energy injection for GRB model (default: False)",
    )
    parser.add_argument(
        "--error-budget",
        type=str,
        default="1.0",
        help="Additional systematic error (mag) to be introduced (default: 1)",
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
        "--injection-num",
        metavar="eventnum",
        type=int,
        help="The injection number to be taken from the injection set",
    )
    parser.add_argument(
        "--injection-detection-limit",
        metavar="mAB",
        type=str,
        help="The highest mAB to be presented in the injection data set, any mAB higher than this will become a non-detection limit. Should be comma delimited list same size as injection set.",
    )
    parser.add_argument(
        "--injection-outfile",
        metavar="PATH",
        type=str,
        help="Path to the output injection lightcurve",
    )
    parser.add_argument(
        "--injection-model",
        type=str,
        help="Name of the kilonova model to be used for injection (default: the same as model used for recovery)",
    )
    parser.add_argument(
        "--remove-nondetections",
        action="store_true",
        default=False,
        help="remove non-detections from fitting analysis",
    )
    parser.add_argument(
        "--detection-limit",
        metavar="DICT",
        type=str,
        default=None,
        help="Dictionary for detection limit per filter, e.g., {'r':22, 'g':23}, put a double quotation marks around the dictionary",
    )
    parser.add_argument(
        "--with-grb-injection",
        help="If the injection has grb included",
        action="store_true",
    )
    parser.add_argument(
        "--prompt-collapse",
        help="If the injection simulates prompt collapse and therefore only dynamical",
        action="store_true",
    )
    parser.add_argument(
        "--ztf-sampling", help="Use realistic ZTF sampling", action="store_true"
    )
    parser.add_argument(
        "--ztf-uncertainties",
        help="Use realistic ZTF uncertainties",
        action="store_true",
    )
    parser.add_argument(
        "--ztf-ToO",
        help="Adds realistic ToO observations during the first one or two days. Sampling depends on exposure time specified. Valid values are 180 (<1000sq deg) or 300 (>1000sq deg). Won't work w/o --ztf-sampling",
        type=str,
        choices=["180", "300"],
    )
    parser.add_argument(
        "--train-stats",
        help="Creates a file too.csv to derive statistics",
        action="store_true",
    )
    parser.add_argument(
        "--rubin-ToO",
        help="Adds ToO obeservations based on the strategy presented in arxiv.org/abs/2111.01945.",
        action="store_true",
    )
    parser.add_argument(
        "--rubin-ToO-type",
        help="Type of ToO observation. Won't work w/o --rubin-ToO",
        type=str,
        choices=["platinum", "gold", "gold_z", "silver", "silver_z"],
    )
    parser.add_argument(
        "--xlim",
        type=str,
        default="0,14",
        help="Start and end time for light curve plot (default: 0-14)",
    )
    parser.add_argument(
        "--ylim",
        type=str,
        default="22,16",
        help="Upper and lower magnitude limit for light curve plot (default: 22-16)",
    )
    parser.add_argument(
        "--generation-seed",
        metavar="seed",
        type=int,
        default=42,
        help="Injection generation seed (default: 42)",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False, help="add best fit plot"
    )
    parser.add_argument(
        "--bilby-zero-likelihood-mode",
        action="store_true",
        default=False,
        help="enable prior run",
    )
    parser.add_argument(
        "--photometry-augmentation",
        help="Augment photometry to improve parameter recovery",
        action="store_true",
    )
    parser.add_argument(
        "--photometry-augmentation-seed",
        metavar="seed",
        type=int,
        default=0,
        help="Optimal generation seed (default: 0)",
    )
    parser.add_argument(
        "--photometry-augmentation-N-points",
        help="Number of augmented points to include",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--photometry-augmentation-filters",
        type=str,
        help="A comma seperated list of filters to use for augmentation (e.g. g,r,i). If none is provided, will use all the filters available",
    )
    parser.add_argument(
        "--photometry-augmentation-times",
        type=str,
        help="A comma seperated list of times to use for augmentation in days post trigger time (e.g. 0.1,0.3,0.5). If none is provided, will use random times between tmin and tmax",
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
        "--sample-over-Hubble",
        action="store_true",
        default=False,
        help="To sample over Hubble constant and redshift",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="print out log likelihoods",
    )

    parser.add_argument(
        "--refresh-models-list",
        type=bool,
        default=False,
        help="Refresh the list of models available on Gitlab",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        default=False,
        help="only look for local svdmodels (ignore Gitlab)",
    )
    parser.add_argument(
        "--ignore-timeshift",
        help="If you want to ignore the timeshift parameter in an injection file.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--bestfit",
        help="Save the best fit parameters and magnitudes to JSON",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--fits-file",
        help="Fits file output from Bayestar, to be used for constructing dL-iota prior",
    )
    # FIXME: to be remove latter
    parser.add_argument(
        "--detection-limit-fits-file",
        help="Fits file output from m4opt which contain the detection limit of a given sky location",
    )
    parser.add_argument(
        "--mission-name",
        help="The Telescope name. When running with the detection limit from M4OPT, this will be caught in utils.py to read the appropriate bandpass (UVEX or ULTRASAT)",
    )
    parser.add_argument(
        "--cosiota-node-num",
        help="Number of cos-iota nodes used in the Bayestar fits (default: 10)",
        default=10,
    )

    parser.add_argument(
        "--skip-sampling",
        help="If analysis has already run, skip bilby sampling and compute results from checkpoint files. Combine with --plot to make plots from these files.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ra",
        type=float,
        help="Right ascension of the sky location; to be used together with fits file",
    )
    parser.add_argument(
        "--dec",
        type=float,
        help="Declination of the sky location; to be used together with fits file",
    )
    parser.add_argument(
        "--dL",
        type=float,
        help="Distance of the location; to be used together with fits file",
    )
    parser.add_argument(
        "--fetch-Ebv-from-dustmap",
        help="Fetching Ebv from dustmap, to be used as fixed-value prior",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--systematics-file",
        metavar="PATH",
        help="Path to systematics configuration file",
        default=None,
    )

    return parser


def analysis(args):

    # Global mission name storage for M4OPT mission of specific filters
    # When a mission name is provided (ULTRASAT or UVEX), this global variable
    # stores it so that get_default_filts_lambdas() can dynamically add the
    # appropriate bandpass filters (NUV for ULTRASAT, FUV+NUV for UVEX).
    # Set once at the start of analysis() via set_mission_name(),
    # then accessed throughout the code via get_mission_name().
    set_mission_name(args.mission_name)

    if args.sampler == "pymultinest":
        if len(args.outdir) > 64:
            print(
                "WARNING: output directory name is too long, it should not be longer than 64 characters"
            )
            exit()

    refresh = False
    try:
        refresh = args.refresh_models_list
    except AttributeError:
        pass
    if refresh:
        refresh_models_list(
            models_home=args.svd_path if args.svd_path not in [None, ""] else None
        )

    bilby.core.utils.setup_logger(outdir=args.outdir, label=args.label)
    bilby.core.utils.check_directory_exists_and_if_not_mkdir(args.outdir)

    if args.filters:
        filters = args.filters.replace(" ", "")  # remove all whitespace
        filters = filters.split(",")
        if len(filters) == 0:
            raise ValueError("Need at least one valid filter.")
    elif args.rubin_ToO_type == "platinum":
        filters = ["ps1__g", "ps1__r", "ps1__i", "ps1__z", "ps1__y"]
    elif args.rubin_ToO_type == "gold":
        filters = ["ps1__g", "ps1__r", "ps1__i"]
    elif args.rubin_ToO_type == "gold_z":
        filters = ["ps1__g", "ps1__r", "ps1__z"]
    elif args.rubin_ToO_type == "silver":
        filters = ["ps1__g" "ps1__i"]
    elif args.rubin_ToO_type == "silver_z":
        filters = ["ps1__g", "ps1__z"]
    else:
        filters = None

    # initialize light curve model
    timeshift = 0
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

    print("Creating light curve model for inference")

    if args.filters:
        filters = args.filters.replace(" ", "")  # remove all whitespace
        filters = filters.split(",")
        if len(filters) == 0:
            raise ValueError("Need at least one valid filter.")
    else:
        filters = None

    # create the kilonova data if an injection set is given
    if args.injection:
        with open(args.injection, "r") as f:
            injection_dict = json.load(
                f, object_hook=bilby.core.utils.decode_bilby_json
            )
        injection_df = injection_dict["injections"]
        row = injection_df.loc[injection_df["simulation_id"] == args.injection_num]
        injection_parameters = row.squeeze().to_dict()

        if "geocent_time" in injection_parameters:
            tc_gps = time.Time(injection_parameters["geocent_time"], format="gps")
        elif "geocent_time_x" in injection_parameters:
            tc_gps = time.Time(injection_parameters["geocent_time_x"], format="gps")
        else:
            print("Need either geocent_time or geocent_time_x")
            exit(1)

        timeshift = 0
        trigger_time = tc_gps.mjd + timeshift

        if args.ignore_timeshift:
            if "timeshift" in injection_parameters:
                timeshift = injection_parameters["timeshift"]
            else:
                timeshift = 0

            trigger_time = tc_gps.mjd + timeshift

        # print("the trigger time from the injection file is: ", trigger_time)

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
            sample_times = np.arange(
                args.tmin + timeshift, args.tmax + timeshift + args.dt, args.dt
            )

        print("Creating light curve model for inference")

        injection_parameters["kilonova_trigger_time"] = trigger_time
        if args.prompt_collapse:
            injection_parameters["log10_mej_wind"] = -3.0

        # sanity check for eject masses
        if "log10_mej_dyn" in injection_parameters and not np.isfinite(
            injection_parameters["log10_mej_dyn"]
        ):
            injection_parameters["log10_mej_dyn"] = -3.0
        if "log10_mej_wind" in injection_parameters and not np.isfinite(
            injection_parameters["log10_mej_wind"]
        ):
            injection_parameters["log10_mej_wind"] = -3.0

        args.kilonova_tmin = args.tmin
        args.kilonova_tmax = args.tmax
        args.kilonova_tstep = args.dt_inj
        args.kilonova_error = args.photometric_error_budget

        if not args.injection_model:
            args.kilonova_injection_model = args.model
        else:
            args.kilonova_injection_model = args.injection_model
        args.kilonova_injection_svd = args.svd_path
        args.injection_svd_mag_ncoeff = args.svd_mag_ncoeff
        args.injection_svd_lbol_ncoeff = args.svd_lbol_ncoeff

        print("Creating injection light curve model")
        _, _, injection_model = create_light_curve_model_from_args(
            args.kilonova_injection_model,
            args,
            sample_times,
            filters=filters,
            sample_over_Hubble=args.sample_over_Hubble,
        )
        data = create_light_curve_data(
            injection_parameters, args, light_curve_model=injection_model
        )
        print(f"Injection generated with parameters {injection_parameters}")

        # checking produced data for magnitudes dimmer than the detection limit
        if filters is not None:
            if args.detection_limit is None:
                if args.rubin_ToO_type:
                    detection_limit = {
                        "ps1__g": 25.8,
                        "ps1__r": 25.5,
                        "ps1__i": 24.8,
                        "ps1__z": 24.1,
                        "ps1__y": 22.9,
                    }
                # elif args.ztf_sampling:
                #    detection_limit = {}

                # FIXME : to be remove we could just provide the detection limit from M4OPT instead of recalculat them
                # elif args.detection_limit_fits_file is not None:
                #     limit_given_radec = detection_limit_from_m4opt_fits_file(
                #         args.detection_limit_fits_file, args.ra, args.dec
                #     )
                #     print(f"Detection limit from {args.detection_limit_fits_file} is used")
                #     print(f"Given ra:{args.ra} and dec:{args.dec}, the limiting mag is {limit_given_radec}")
                #     detection_limit = {
                #         x: float(limit_given_radec)
                #         for x in filters
                #     }
                else:
                    detection_limit = {x: np.inf for x in filters}
            else:
                detection_limit = literal_eval(args.detection_limit)

            # print("the detection limits for this run are: ", detection_limit)

            for filt in filters:
                i = 0
                for row in data[filt]:
                    # print('the old data is {data}'.format(data=data[filt]))
                    mjd, mag, mag_unc = row
                    # print("the data for {f} is: ".format(f=filt), row)
                    if mag > detection_limit[filt]:
                        data[filt][i, :] = [mjd, detection_limit[filt], -np.inf]

                    # print("the new data is: ", data[filt])
                    i += 1

        if args.injection_outfile is not None:
            if filters is not None:
                if args.detection_limit is None:
                    detection_limit = {x: np.inf for x in filters}
                else:
                    detection_limit = literal_eval(args.detection_limit)
            else:
                detection_limit = {}
            data_out = np.empty((0, 6))
            for filt in data.keys():
                if filters:
                    if args.photometry_augmentation_filters:
                        filts = list(
                            set(
                                filters
                                + args.photometry_augmentation_filters.split(",")
                            )
                        )
                    else:
                        filts = filters
                    if filt not in filts:
                        continue
                for row in data[filt]:
                    mjd, mag, mag_unc = row
                    if not np.isfinite(mag_unc):
                        data_out = np.append(
                            data_out,
                            np.array([[mjd, 99.0, 99.0, filt, mag, 0.0]]),
                            axis=0,
                        )
                    else:
                        if filt in detection_limit:
                            data_out = np.append(
                                data_out,
                                np.array(
                                    [
                                        [
                                            mjd,
                                            mag,
                                            mag_unc,
                                            filt,
                                            detection_limit[filt],
                                            0.0,
                                        ]
                                    ]
                                ),
                                axis=0,
                            )
                        else:
                            data_out = np.append(
                                data_out,
                                np.array([[mjd, mag, mag_unc, filt, np.inf, 0.0]]),
                                axis=0,
                            )

            columns = ["jd", "mag", "mag_unc", "filter", "limmag", "programid"]
            lc = pd.DataFrame(data=data_out, columns=columns)
            lc.sort_values("jd", inplace=True)
            lc = lc.reset_index(drop=True)
            lc.to_csv(args.injection_outfile)

    else:
        # load the kilonova afterglow data
        try:
            data = loadEvent(args.data)

        except ValueError:
            with open(args.data) as f:
                data = json.load(f)
                for key in data.keys():
                    data[key] = np.array(data[key])

        if args.trigger_time is None:
            # load the minimum time as trigger time
            min_time = np.inf
            for key, array in data.items():
                min_time = np.minimum(min_time, np.min(array[:, 0]))
            trigger_time = min_time
            print(
                f"trigger_time is not provided, analysis will continue using a trigger time of {trigger_time}"
            )
        else:
            trigger_time = args.trigger_time

    if args.remove_nondetections:
        filters_to_check = list(data.keys())
        for filt in filters_to_check:
            idx = np.where(np.isfinite(data[filt][:, 2]))[0]
            data[filt] = data[filt][idx, :]
            if len(idx) == 0:
                del data[filt]

    # check for detections
    detection = False
    notallnan = False

    # print('data before checking for detections: ', data[filt])

    for filt in data.keys():
        idx = np.where(np.isfinite(data[filt][:, 2]))[0]
        if len(idx) > 0:
            detection = True
        idx = np.where(np.isfinite(data[filt][:, 1]))[0]
        if len(idx) > 0:
            notallnan = True
        if detection and notallnan:
            break
    if (not detection) or (not notallnan):
        raise ValueError("Need at least one detection to do fitting.")

    if type(args.error_budget) in [float, int]:
        error_budget = [args.error_budget]
    else:
        error_budget = [float(x) for x in args.error_budget.split(",")]
    if args.filters:
        if args.photometry_augmentation_filters:
            filters = list(
                set(
                    args.filters.split(",")
                    + args.photometry_augmentation_filters.split(",")
                )
            )
        else:
            filters = args.filters.split(",")

        values_to_indices = {v: i for i, v in enumerate(filters)}
        filters_to_analyze = sorted(
            list(set(filters).intersection(set(list(data.keys())))),
            key=lambda v: values_to_indices[v],
        )

        if len(error_budget) == 1:
            error_budget = dict(
                zip(filters_to_analyze, error_budget * len(filters_to_analyze))
            )
        elif len(args.filters.split(",")) == len(error_budget):
            error_budget = dict(zip(args.filters.split(","), error_budget))
        else:
            raise ValueError("error_budget must be the same length as filters")

    else:
        filters_to_analyze = list(data.keys())
        error_budget = dict(
            zip(filters_to_analyze, error_budget * len(filters_to_analyze))
        )

    print("Running with filters {0}".format(filters_to_analyze))
    model_names, models, light_curve_model = create_light_curve_model_from_args(
        args.model,
        args,
        sample_times,
        filters=filters_to_analyze,
        sample_over_Hubble=args.sample_over_Hubble,
    )

    # setup the prior
    priors = create_prior_from_args(model_names, args)

    # print('the data passed to likelihood is: ', data)

    # setup the likelihood
    if args.detection_limit:
        args.detection_limit = literal_eval(args.detection_limit)
    likelihood_kwargs = dict(
        light_curve_model=light_curve_model,
        filters=filters_to_analyze,
        light_curve_data=data,
        trigger_time=trigger_time,
        tmin=args.tmin + timeshift,
        tmax=args.tmax + timeshift,
        error_budget=error_budget,
        verbose=args.verbose,
        detection_limit=args.detection_limit,
        systematics_file=args.systematics_file,
    )

    likelihood = OpticalLightCurve(**likelihood_kwargs)
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

    # print("passing arguments to bilby")

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

    # check if it is running under mpi
    try:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            pass
        else:
            return

    except ImportError:
        pass

    result.save_posterior_samples()

    if args.injection:
        injlist_all = []
        for model_name in model_names:
            if model_name in ["Bu2019nsbh"]:
                injlist = [
                    "luminosity_distance",
                    "inclination_EM",
                    "log10_mej_dyn",
                    "log10_mej_wind",
                ]
            elif model_name in ["Bu2019lm"]:
                injlist = [
                    "luminosity_distance",
                    "inclination_EM",
                    "KNphi",
                    "log10_mej_dyn",
                    "log10_mej_wind",
                ]
            else:
                injlist = ["luminosity_distance"] + model_parameters_dict[model_name]

            injlist_all = list(set(injlist_all + injlist))

        constant_columns = []
        for column in result.posterior:
            if len(result.posterior[column].unique()) == 1:
                constant_columns.append(column)

        injlist_all = list(set(injlist_all) - set(constant_columns))
        injection = {
            key: injection_parameters[key]
            for key in injlist_all
            if key in injection_parameters
        }
        result.plot_corner(parameters=injection)
    else:
        result.plot_corner()

    if args.bestfit or args.plot:
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
        _, mag = light_curve_model.generate_lightcurve(sample_times, bestfit_params)
        for filt in mag.keys():
            if bestfit_params["luminosity_distance"] > 0:
                mag[filt] += 5.0 * np.log10(
                    bestfit_params["luminosity_distance"] * 1e6 / 10.0
                )
        mag["bestfit_sample_times"] = sample_times

        if "timeshift" in bestfit_params:
            mag["bestfit_sample_times"] = (
                mag["bestfit_sample_times"] + bestfit_params["timeshift"]
            )

        ######################
        # calculate the chi2 #
        ######################
        processed_data = dataProcess(
            data,
            filters_to_analyze,
            trigger_time,
            args.tmin + timeshift,
            args.tmax + timeshift,
        )
        chi2 = 0.0
        dof = 0.0
        chi2_per_dof_dict = {}
        for filt in filters_to_analyze:
            # make best-fit lc interpolation
            sample_times = mag["bestfit_sample_times"]
            mag_used = mag[filt]
            interp = interp1d(sample_times, mag_used)
            # fetch data
            samples = copy.deepcopy(processed_data[filt])
            t, y, sigma_y = samples[:, 0], samples[:, 1], samples[:, 2]
            print("the time values before adding timeshift are: ", t)
            # shift t values by timeshift
            if "timeshift" in bestfit_params:
                print(
                    "timeshift found in bestfit_params is: ",
                    bestfit_params["timeshift"],
                )
                t += bestfit_params["timeshift"]
            # only the detection data are needed
            finite_idx = np.where(np.isfinite(sigma_y))[0]
            print("the {f} data being analyzed is: ".format(f=filt), samples)
            print(
                "for {f} the length of the detections array is: ".format(f=filt),
                len(finite_idx),
            )
            if len(finite_idx) > 0:
                # fetch the erorr_budget
                if "em_syserr" in bestfit_params:
                    err = bestfit_params["em_syserr"]
                else:
                    err = error_budget[filt]
                t_det, y_det, sigma_y_det = (
                    t[finite_idx],
                    y[finite_idx],
                    sigma_y[finite_idx],
                )
                print("the time passes into the interp is: ", t_det)
                num = (y_det - interp(t_det)) ** 2
                den = sigma_y_det**2 + err**2
                chi2_per_filt = np.sum(num / den)
                # store the data
                chi2 += chi2_per_filt
                dof += len(finite_idx)
                print("the number of dof are: ", dof)
                chi2_per_dof_dict[filt] = chi2_per_filt / len(finite_idx)

        if dof == 0:
            print("Uh oh! the dof is zero")

        chi2_per_dof = chi2 / dof

    if args.bestfit:
        bestfit_to_write = bestfit_params.copy()
        bestfit_to_write["log_bayes_factor"] = result.log_bayes_factor
        bestfit_to_write["log_bayes_factor_err"] = result.log_evidence_err
        bestfit_to_write["Best fit index"] = int(bestfit_idx)
        bestfit_to_write["Magnitudes"] = {i: mag[i].tolist() for i in mag.keys()}
        bestfit_to_write["chi2_per_dof"] = chi2_per_dof
        bestfit_to_write["chi2_per_dof_per_filt"] = {
            i: chi2_per_dof_dict[i].tolist() for i in chi2_per_dof_dict.keys()
        }
        bestfit_file = os.path.join(args.outdir, f"{args.label}_bestfit_params.json")

        with open(bestfit_file, "w") as file:
            json.dump(bestfit_to_write, file, indent=4)

        print(f"Saved bestfit parameters and magnitudes to {bestfit_file}")

    if args.plot:
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if len(models) > 1:
            _, mag_all = light_curve_model.generate_lightcurve(
                sample_times, bestfit_params, return_all=True
            )

            for ii in range(len(mag_all)):
                for filt in mag_all[ii].keys():
                    if bestfit_params["luminosity_distance"] > 0:
                        mag_all[ii][filt] += 5.0 * np.log10(
                            bestfit_params["luminosity_distance"] * 1e6 / 10.0
                        )
            model_colors = cm.Spectral(np.linspace(0, 1, len(models)))[::-1]

        filters_plot = []
        for filt in filters_to_analyze:
            if filt not in data:
                continue
            samples = data[filt]
            t, y, sigma_y = samples[:, 0], samples[:, 1], samples[:, 2]
            idx = np.where(~np.isnan(y))[0]
            t, y, sigma_y = t[idx], y[idx], sigma_y[idx]
            if len(t) == 0:
                continue
            filters_plot.append(filt)

        colors = cm.Spectral(np.linspace(0, 1, len(filters_plot)))[::-1]

        plotName = os.path.join(args.outdir, f"{args.label}_lightcurves.png")

        # set up the geometry for the all-in-one figure
        wspace = 0.6  # All in inches.
        hspace = 0.3
        lspace = 1.0
        bspace = 0.7
        trspace = 0.2
        hpanel = 2.25
        wpanel = 3.0

        ncol = 2
        nrow = int(np.ceil(len(filters_plot) / ncol))
        fig, axes = plt.subplots(nrow, ncol)

        figsize = (
            1.5 * (lspace + wpanel * ncol + wspace * (ncol - 1) + trspace),
            1.5 * (bspace + hpanel * nrow + hspace * (nrow - 1) + trspace),
        )
        # Create the figure and axes.
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
        fig.subplots_adjust(
            left=lspace / figsize[0],
            bottom=bspace / figsize[1],
            right=1.0 - trspace / figsize[0],
            top=1.0 - trspace / figsize[1],
            wspace=wspace / wpanel,
            hspace=hspace / hpanel,
        )

        if len(filters_plot) % 2:
            axes[-1, -1].axis("off")

        cnt = 0
        for filt, color in zip(filters_plot, colors):
            cnt = cnt + 1

            # summary plot
            row = (cnt - 1) // ncol
            col = (cnt - 1) % ncol
            ax_sum = axes[row, col]
            # adding the ax for the Delta
            divider = make_axes_locatable(ax_sum)
            ax_delta = divider.append_axes("bottom", size="30%", sharex=ax_sum)

            # configuring ax_sum
            ax_sum.set_ylabel("AB magnitude", rotation=90)
            ax_delta.set_ylabel(r"$\Delta (\sigma)$")
            if cnt == len(filters_plot) or cnt == len(filters_plot) - 1:
                ax_delta.set_xlabel("Time [days]")
            else:
                ax_delta.set_xticklabels([])

            # plotting the best-fit lc and the data in ax1
            samples = data[filt]
            t, y, sigma_y = samples[:, 0], samples[:, 1], samples[:, 2]
            t -= trigger_time + timeshift
            idx = np.where(~np.isnan(y))[0]
            t, y, sigma_y = t[idx], y[idx], sigma_y[idx]

            idx = np.where(np.isfinite(sigma_y))[0]
            det_idx = idx
            ax_sum.errorbar(
                t[idx],
                y[idx],
                sigma_y[idx],
                fmt="o",
                color=color,
            )

            idx = np.where(~np.isfinite(sigma_y))[0]
            ax_sum.scatter(
                t[idx],
                y[idx],
                marker="v",
                color=color,
            )

            mag_plot = getFilteredMag(mag, filt)

            # calculating the chi2
            mag_per_data = np.interp(t[det_idx], mag["bestfit_sample_times"], mag_plot)
            diff_per_data = mag_per_data - y[det_idx]
            sigma_per_data = np.sqrt((sigma_y[det_idx] ** 2 + error_budget[filt] ** 2))
            chi2_per_data = diff_per_data**2
            chi2_per_data /= sigma_per_data**2
            chi2_total = np.sum(chi2_per_data)
            N_data = len(det_idx)

            # plot the mismatch between the model and the data
            ax_delta.scatter(t[det_idx], diff_per_data / sigma_per_data, color=color)
            ax_delta.axhline(0, linestyle="--", color="k")

            ax_sum.plot(
                mag["bestfit_sample_times"],
                mag_plot,
                color="coral",
                linewidth=3,
                linestyle="--",
            )

            if len(models) > 1:
                ax_sum.fill_between(
                    mag["bestfit_sample_times"],
                    mag_plot + error_budget[filt],
                    mag_plot - error_budget[filt],
                    facecolor="coral",
                    alpha=0.2,
                    label="combined",
                )
            else:
                ax_sum.fill_between(
                    mag["bestfit_sample_times"],
                    mag_plot + error_budget[filt],
                    mag_plot - error_budget[filt],
                    facecolor="coral",
                    alpha=0.2,
                )

            if len(models) > 1:
                for ii in range(len(mag_all)):
                    mag_plot = getFilteredMag(mag_all[ii], filt)
                    ax_sum.plot(
                        mag["bestfit_sample_times"],
                        mag_plot,
                        color="coral",
                        linewidth=3,
                        linestyle="--",
                    )
                    ax_sum.fill_between(
                        mag["bestfit_sample_times"],
                        mag_plot + error_budget[filt],
                        mag_plot - error_budget[filt],
                        facecolor=model_colors[ii],
                        alpha=0.2,
                        label=models[ii].model,
                    )

            ax_sum.set_title(
                f"{filt}: " + rf"$\chi^2 / d.o.f. = {round(chi2_total / N_data, 2)}$"
            )

            ax_sum.set_xlim([float(x) for x in args.xlim.split(",")])
            ax_sum.set_ylim([float(x) for x in args.ylim.split(",")])
            ax_delta.set_xlim([float(x) for x in args.xlim.split(",")])

        plt.savefig(plotName, bbox_inches="tight")
        plt.close()


def nnanalysis(args):

    set_mission_name(args.mission_name)

    # import functions
    from ..mlmodel.dataprocessing import (
        # gen_prepend_filler,  # noqa:  F401
        # gen_append_filler,  # noqa: F401
        pad_the_data,
    )

    # from ..mlmodel.resnet import ResNet  # noqa:  F401
    from ..mlmodel.embedding import SimilarityEmbedding
    from ..mlmodel.normalizingflows import normflow_params
    from ..mlmodel.inference import cast_as_bilby_result

    # need to add these packages:
    import torch

    # import torch.nn as nn  # noqa:  F401
    # from torch.utils.data import (
    #     Dataset,  # noqa: F401
    #     DataLoader,  # noqa:  F401
    #     TensorDataset,  # noqa:  F401
    #     random_split,  # noqa:  F401
    # )
    # import torch.nn.functional as F  # noqa: F401
    # from nflows.nn.nets.resnet import ResidualNet  # noqa: F401
    # from nflows import transforms, distributions, flows  # noqa: F401
    # from nflows.distributions import StandardNormal  # noqa: F401
    from nflows.flows import Flow

    # from nflows.transforms.autoregressive import (
    #     MaskedAffineAutoregressiveTransform,  # noqa: F401
    # )
    # from nflows.transforms import CompositeTransform, RandomPermutation  # noqa: F401
    # import nflows.utils as torchutils  # noqa: F401

    # only continue if the Kasen model is selected
    if args.model != "Ka2017":
        print(
            "WARNING: model selected is not currently compatible with this inference method"
        )
        exit()
    else:
        pass

    print("Starting LFI")

    # only can use ztfr, ztfg, and ztfi filters in the light curve data
    if args.filters:
        filters = args.filters.replace(" ", "")  # remove all whitespace
        filters = filters.split(",")
        if ("ztfr" in filters) and ("ztfi" in filters) and ("ztfg" in filters):
            pass
        else:
            raise ValueError("Need the ztfr, ztfi, and ztfg filters.")
    else:
        print(
            "Currently filters are hardcoded to ztfr, ztfi, and ztfg. Continuing with these filters."
        )
        filters = "ztfg,ztfi,ztfr"
        filters = filters.replace(" ", "")  # remove all whitespace
        filters = filters.split(",")

    refresh = False
    try:
        refresh = args.refresh_models_list
    except AttributeError:
        pass
    if refresh:
        refresh_models_list(
            models_home=args.svd_path if args.svd_path not in [None, ""] else None
        )

    # set up outdir
    bilby.core.utils.setup_logger(outdir=args.outdir, label=args.label)
    bilby.core.utils.check_directory_exists_and_if_not_mkdir(args.outdir)

    print("Setting up logger and storage directory")

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

    # create the kilonova data if an injection set is given
    if args.injection:
        with open(args.injection, "r") as f:
            injection_dict = json.load(
                f, object_hook=bilby.core.utils.decode_bilby_json
            )
        injection_df = injection_dict["injections"]
        injection_parameters = injection_df.iloc[args.injection_num].to_dict()

        if "geocent_time" in injection_parameters:
            tc_gps = time.Time(injection_parameters["geocent_time"], format="gps")
        elif "geocent_time_x" in injection_parameters:
            tc_gps = time.Time(injection_parameters["geocent_time_x"], format="gps")
        else:
            print("Need either geocent_time or geocent_time_x")
            exit(1)
        trigger_time = tc_gps.mjd

        injection_parameters["kilonova_trigger_time"] = trigger_time
        if args.prompt_collapse:
            injection_parameters["log10_mej_wind"] = -3.0

        # sanity check for eject masses
        if "log10_mej_dyn" in injection_parameters and not np.isfinite(
            injection_parameters["log10_mej_dyn"]
        ):
            injection_parameters["log10_mej_dyn"] = -3.0
        if "log10_mej_wind" in injection_parameters and not np.isfinite(
            injection_parameters["log10_mej_wind"]
        ):
            injection_parameters["log10_mej_wind"] = -3.0

        # need to interpolate between data points if time step is not 0.25
        if args.dt:
            time_step = args.dt
            if args.dt != 0.25:
                raise ValueError(
                    "Need dt to be 0.25 until interpolation feature is incorporated."
                )
                # currently no linear interpolation function
                do_lin_interpolation = True  # noqa:  F841
            else:
                do_lin_interpolation = False  # noqa:  F841

        args.kilonova_tmin = args.tmin
        args.kilonova_tmax = args.tmax
        args.kilonova_tstep = args.dt_inj
        args.kilonova_error = args.photometric_error_budget

        # current_points = int(round(args.tmax - args.tmin)) / args.dt + 1

        if not args.injection_model:
            args.kilonova_injection_model = args.model
        else:
            args.kilonova_injection_model = args.injection_model
        args.kilonova_injection_svd = args.svd_path
        args.injection_svd_mag_ncoeff = args.svd_mag_ncoeff
        args.injection_svd_lbol_ncoeff = args.svd_lbol_ncoeff

        print("Creating injection light curve model")
        _, _, injection_model = create_light_curve_model_from_args(
            args.kilonova_injection_model,
            args,
            sample_times,
            filters=filters,
            sample_over_Hubble=args.sample_over_Hubble,
        )
        data = create_light_curve_data(
            injection_parameters, args, light_curve_model=injection_model
        )
        print("Injection generated")
        res = next(iter(data))

        if args.injection_outfile is not None:
            if filters is not None:
                if args.injection_detection_limit is not None:
                    detection_limit = {
                        x: float(y)
                        for x, y in zip(
                            filters,
                            args.injection_detection_limit.split(","),
                        )
                    }
                # FIXME : to be remove
                # elif args.detection_limit_fits_file is not None:
                #     limit_given_radec = detection_limit_from_m4opt_fits_file(
                #         args.detection_limit_fits_file, args.ra, args.dec
                #     )
                #     print(f"Detection limit from {args.detection_limit_fits_file} is used")
                #     print(f"Given ra:{args.ra} and dec:{args.dec}, the limiting mag is {limit_given_radec}")
                #     detection_limit = {
                #         x: float(limit_given_radec)
                #         for x in filters
                #     }
                else:
                    detection_limit = {x: np.inf for x in filters}
            else:
                detection_limit = {}
            data_out = np.empty((0, 6))
            for filt in data.keys():
                if filters:
                    if args.photometry_augmentation_filters:
                        filts = list(
                            set(
                                filters
                                + args.photometry_augmentation_filters.split(",")
                            )
                        )
                    else:
                        filts = filters
                    if filt not in filts:
                        continue
                for row in data[filt]:
                    mjd, mag, mag_unc = row
                    if not np.isfinite(mag_unc):
                        data_out = np.append(
                            data_out,
                            np.array([[mjd, 99.0, 99.0, filt, mag, 0.0]]),
                            axis=0,
                        )
                    else:
                        if filt in detection_limit:
                            data_out = np.append(
                                data_out,
                                np.array(
                                    [
                                        [
                                            mjd,
                                            mag,
                                            mag_unc,
                                            filt,
                                            detection_limit[filt],
                                            0.0,
                                        ]
                                    ]
                                ),
                                axis=0,
                            )
                        else:
                            data_out = np.append(
                                data_out,
                                np.array([[mjd, mag, mag_unc, filt, np.inf, 0.0]]),
                                axis=0,
                            )

            columns = ["jd", "mag", "mag_unc", "filter", "limmag", "programid"]
            lc = pd.DataFrame(data=data_out, columns=columns)
            lc.sort_values("jd", inplace=True)
            lc = lc.reset_index(drop=True)
            lc.to_csv(args.injection_outfile)

    else:
        # load the lightcurve data
        data = loadEvent(args.data)
        res = next(iter(data))
        # current_points = len(data[res])

        if args.trigger_time is None:
            # load the minimum time as trigger time
            min_time = np.inf
            for key, array in data.items():
                min_time = np.minimum(min_time, np.min(array[:, 0]))
            trigger_time = min_time
            print(
                f"trigger_time is not provided, analysis will continue using a trigger time of {trigger_time}"
            )
        else:
            trigger_time = args.trigger_time

    if args.remove_nondetections:
        filters_to_check = list(data.keys())
        for filt in filters_to_check:
            idx = np.where(np.isfinite(data[filt][:, 2]))[0]
            data[filt] = data[filt][idx, :]
            if len(idx) == 0:
                del data[filt]

    # check for detections
    detection = False
    notallnan = False
    for filt in data.keys():
        idx = np.where(np.isfinite(data[filt][:, 2]))[0]
        if len(idx) > 0:
            detection = True
        idx = np.where(np.isfinite(data[filt][:, 1]))[0]
        if len(idx) > 0:
            notallnan = True
        if detection and notallnan:
            break
    if (not detection) or (not notallnan):
        raise ValueError("Need at least one detection to do fitting.")

    if type(args.error_budget) in [float, int]:
        error_budget = [args.error_budget]
    else:
        error_budget = [float(x) for x in args.error_budget.split(",")]
    if args.filters:
        if args.photometry_augmentation_filters:
            filters = list(
                set(
                    args.filters.split(",")
                    + args.photometry_augmentation_filters.split(",")
                )
            )
        else:
            filters = args.filters.split(",")

        values_to_indices = {v: i for i, v in enumerate(filters)}
        filters_to_analyze = sorted(
            list(set(filters).intersection(set(list(data.keys())))),
            key=lambda v: values_to_indices[v],
        )

        if len(error_budget) == 1:
            error_budget = dict(
                zip(filters_to_analyze, error_budget * len(filters_to_analyze))
            )
        elif len(args.filters.split(",")) == len(error_budget):
            error_budget = dict(zip(args.filters.split(","), error_budget))
        else:
            raise ValueError("error_budget must be the same length as filters")

    else:
        filters_to_analyze = list(data.keys())
        error_budget = dict(
            zip(filters_to_analyze, error_budget * len(filters_to_analyze))
        )

    print("Running with filters {0}".format(filters_to_analyze))
    model_names, models, light_curve_model = create_light_curve_model_from_args(
        args.model,
        args,
        sample_times,
        filters=filters_to_analyze,
        sample_over_Hubble=args.sample_over_Hubble,
    )

    # setup the prior
    priors = create_prior_from_args(model_names, args)

    # now that we have the kilonova light curve, we need to pad it with non-detections
    # this part is currently hard coded in terms of the times !!!! likely will need the most work
    # (so that the 'fixed' and 'shifted' are properly represented)
    num_points = 121
    num_channels = 3
    # bands = [
    #     "ztfg",
    #     "ztfr",
    #     "ztfi",
    # ]  # noqa: F841  # will need to edit to not be hardcoded
    # t_zero = 44242.00021937881  # noqa: F841
    # t_min = 44240.00050450478
    # t_max = 44269.99958898723
    # days = int(round(t_max - t_min))  # noqa: F841
    time_step = 0.25

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    if args.detection_limit:
        detection_limit = args.detection_limit
    else:
        detection_limit = 22.0

    data_df = pd.DataFrame()
    t_list = []
    for i in range(len(data[res])):
        t_list.append(data[res][i][0])
    data_df["t"] = t_list
    for key in data:
        mag_list = []
        for i, val in enumerate(data[key]):
            mag = data[key][i][1]
            mag_list.append(mag)
        data_df[key] = mag_list
    column_list = data_df.columns.to_list()

    # pad the data
    padded_data_df = pad_the_data(
        data_df,
        column_list,
        desired_count=num_points,
        filler_time_step=time_step,
        filler_data=detection_limit,
    )

    # change the data into pytorch tensors
    data_tensor = torch.tensor(
        padded_data_df.iloc[:, 1:4].values.reshape(1, num_points, num_channels),
        dtype=torch.float32,
    ).transpose(1, 2)

    # set up the embedding
    similarity_embedding = SimilarityEmbedding(
        num_dim=7,
        num_hidden_layers_f=1,
        num_hidden_layers_h=1,
        num_blocks=4,
        kernel_size=5,
        num_dim_final=5,
    ).to(device)
    num_dim = 7
    SAVEPATH = os.getcwd() + "/nmma/mlmodel/similarity_embedding_weights.pth"
    similarity_embedding.load_state_dict(torch.load(SAVEPATH, map_location=device))
    for name, param in similarity_embedding.named_parameters():
        param.requires_grad = False

    # set up the normalizing flows
    transform, base_dist, embedding_net = normflow_params(
        similarity_embedding, 9, 5, 90, context_features=num_dim, num_dim=num_dim
    )
    flow = Flow(transform, base_dist, embedding_net).to(device=device)
    PATH_nflow = os.getcwd() + "/nmma/mlmodel/frozen-flow-weights.pth"
    flow.load_state_dict(torch.load(PATH_nflow, map_location=device))

    nsamples = 20000
    with torch.no_grad():
        samples = flow.sample(nsamples, context=data_tensor)
        samples = samples.cpu().reshape(nsamples, 3)

    if args.injection:
        avail_parameters = injection_parameters.keys()
        if (
            ("log10_mej" in avail_parameters)
            and ("log10_vej" in avail_parameters)
            and ("log10_Xlan" in avail_parameters)
        ):
            param_tensor = torch.tensor(
                [
                    injection_parameters["log10_mej"],
                    injection_parameters["log10_vej"],
                    injection_parameters["log10_Xlan"],
                ],
                dtype=torch.float32,
            )
            with torch.no_grad():
                truth = param_tensor
            flow_result = cast_as_bilby_result(samples, truth, priors=priors)
            flow_result.plot_corner(save=True, label=args.label, outdir=args.outdir)
            print("saved posterior plot")
        else:
            raise ValueError(
                "The injection parameters provided do not match the parameters the flow has been trained on"
            )
    else:
        flow_result = cast_as_bilby_result(samples, truth=None, priors=priors)
        flow_result.plot_corner(save=True, label=args.label, outdir=args.outdir)
        print("saved posterior plot")


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
    if args.sampler == "neuralnet":
        nnanalysis(args)
    else:
        analysis(args)
