import os
import numpy as np
import argparse
import json
import pandas as pd
import matplotlib

from ast import literal_eval

from astropy import time

import bilby
import bilby.core
from bilby.core.likelihood import ZeroLikelihood

from .model import SVDLightCurveModel, GRBLightCurveModel, SupernovaLightCurveModel
from .model import ShockCoolingLightCurveModel
from .model import SimpleKilonovaLightCurveModel
from .model import GenericCombineLightCurveModel
from .model import model_parameters_dict
from .utils import loadEvent, getFilteredMag
from .injection import create_light_curve_data
from .likelihood import OpticalLightCurve

matplotlib.use("agg")


def get_parser():

    parser = argparse.ArgumentParser(
        description="Inference on kilonova ejecta parameters."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the kilonova model to be used"
    )
    parser.add_argument(
        "--interpolation_type",
        type=str,
        help="SVD interpolation scheme.",
        default="sklearn_gp",
    )
    parser.add_argument(
        "--svd-path",
        type=str,
        help="Path to the SVD directory, with {model}_mag.pkl and {model}_lbol.pkl",
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
    parser.add_argument(
        "--prior", type=str, required=True, help="Path to the prior file"
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=0.0,
        help="Days to start analysing from the trigger time (default: 0)",
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
        "--cpus",
        type=int,
        default=1,
        help="Number of cores to be used, only needed for dynesty (default: 1)",
    )
    parser.add_argument(
        "--nlive", type=int, default=2048, help="Number of live points (default: 2048)"
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
        choices=["BNS", "NSBH"],
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
        "--bilby_zero_likelihood_mode",
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
        "--verbose",
        action="store_true",
        default=False,
        help="print out log likelihoods",
    )

    return parser


def main(args=None):

    if args is None:
        parser = get_parser()
        args = parser.parse_args()

    bilby.core.utils.setup_logger(outdir=args.outdir, label=args.label)
    bilby.core.utils.check_directory_exists_and_if_not_mkdir(args.outdir)

    # initialize light curve model
    sample_times = np.arange(args.tmin, args.tmax + args.dt, args.dt)

    models = []
    # check if there are more than one model
    if "," in args.model:
        print("Running with combination of multiple light curve models")
        model_names = args.model.split(",")
    else:
        model_names = [args.model]

    for model_name in model_names:
        if model_name == "TrPi2018":
            lc_model = GRBLightCurveModel(
                sample_times=sample_times,
                resolution=args.grb_resolution,
                jetType=args.jet_type,
            )

        elif model_name == "nugent-hyper":
            lc_model = SupernovaLightCurveModel(
                sample_times=sample_times, model="nugent-hyper"
            )

        elif model_name == "salt2":
            lc_model = SupernovaLightCurveModel(
                sample_times=sample_times, model="salt2"
            )

        elif model_name == "Piro2021":
            lc_model = ShockCoolingLightCurveModel(sample_times=sample_times)

        elif model_name == "Me2017" or model_name == "PL_BB_fixedT":
            lc_model = SimpleKilonovaLightCurveModel(
                sample_times=sample_times, model=model_name
            )

        else:
            lc_kwargs = dict(
                model=model_name,
                sample_times=sample_times,
                svd_path=args.svd_path,
                mag_ncoeff=args.svd_mag_ncoeff,
                lbol_ncoeff=args.svd_lbol_ncoeff,
                interpolation_type=args.interpolation_type,
            )
            lc_model = SVDLightCurveModel(**lc_kwargs)

        models.append(lc_model)

        if len(models) > 1:
            light_curve_model = GenericCombineLightCurveModel(models, sample_times)
        else:
            light_curve_model = models[0]

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

        args.kilonova_tmin = args.tmin
        args.kilonova_tmax = args.tmax
        args.kilonova_tstep = args.dt
        args.kilonova_error = args.photometric_error_budget

        args.kilonova_injection_model = args.model
        args.kilonova_injection_svd = args.svd_path
        args.injection_svd_mag_ncoeff = args.svd_mag_ncoeff
        args.injection_svd_lbol_ncoeff = args.svd_lbol_ncoeff

        data = create_light_curve_data(
            injection_parameters, args, light_curve_model=light_curve_model
        )
        print("Injection generated")

        if args.injection_outfile is not None:
            if args.injection_detection_limit is None:
                detection_limit = {x: np.inf for x in args.filters.split(",")}
            else:
                detection_limit = {
                    x: float(y)
                    for x, y in zip(
                        args.filters.split(","),
                        args.injection_detection_limit.split(","),
                    )
                }
            data_out = np.empty((0, 6))
            for filt in data.keys():
                if args.filters:
                    if args.photometry_augmentation_filters:
                        filts = list(
                            set(
                                args.filters.split(",")
                                + args.photometry_augmentation_filters.split(",")
                            )
                        )
                    else:
                        filts = args.filters.split(",")
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
        data = loadEvent(args.data)

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
    for filt in data.keys():
        idx = np.where(np.isfinite(data[filt][:, 2]))[0]
        if len(idx) > 0:
            detection = True
            break
    if not detection:
        raise ValueError("Need at least one detection to do fitting.")

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

        filters_to_analyze = list(set(filters).intersection(set(list(data.keys()))))

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
    # setup the prior
    priors = bilby.gw.prior.PriorDict(args.prior)

    # setup for Ebv
    if args.Ebv_max > 0.0:
        Ebv_c = 1.0 / (0.5 * args.Ebv_max)
        priors["Ebv"] = bilby.core.prior.Interped(
            name="Ebv",
            minimum=0.0,
            maximum=args.Ebv_max,
            latex_label="$E(B-V)$",
            xx=[0, args.Ebv_max],
            yy=[Ebv_c, 0],
        )
    else:
        priors["Ebv"] = bilby.core.prior.DeltaFunction(
            name="Ebv", peak=0.0, latex_label="$E(B-V)$"
        )

    # setup the likelihood
    if args.detection_limit:
        args.detection_limit = literal_eval(args.detection_limit)
    likelihood_kwargs = dict(
        light_curve_model=light_curve_model,
        filters=filters_to_analyze,
        light_curve_data=data,
        trigger_time=trigger_time,
        tmin=args.tmin,
        tmax=args.tmax,
        error_budget=error_budget,
        verbose=args.verbose,
        detection_limit=args.detection_limit,
    )

    likelihood = OpticalLightCurve(**likelihood_kwargs)
    if args.bilby_zero_likelihood_mode:
        likelihood = ZeroLikelihood(likelihood)

    result = bilby.run_sampler(
        likelihood,
        priors,
        sampler=args.sampler,
        outdir=args.outdir,
        label=args.label,
        nlive=args.nlive,
        seed=args.seed,
        soft_init=True,
        queue_size=args.cpus,
        check_point_delta_t=3600,
    )

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

    if args.plot:
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import cm

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

        plotName = os.path.join(args.outdir, "lightcurves.png")
        plt.figure(figsize=(20, 16))
        color2 = "coral"

        cnt = 0
        for filt, color in zip(filters_plot, colors):
            cnt = cnt + 1
            if cnt == 1:
                ax1 = plt.subplot(len(filters_plot), 1, cnt)
            else:
                ax2 = plt.subplot(len(filters_plot), 1, cnt, sharex=ax1, sharey=ax1)

            samples = data[filt]
            t, y, sigma_y = samples[:, 0], samples[:, 1], samples[:, 2]
            t -= trigger_time
            idx = np.where(~np.isnan(y))[0]
            t, y, sigma_y = t[idx], y[idx], sigma_y[idx]

            idx = np.where(np.isfinite(sigma_y))[0]
            plt.errorbar(
                t[idx],
                y[idx],
                sigma_y[idx],
                fmt="o",
                color="k",
                markersize=16,
            )  # or color=color

            idx = np.where(~np.isfinite(sigma_y))[0]
            plt.errorbar(
                t[idx], y[idx], sigma_y[idx], fmt="v", color="k", markersize=16
            )  # or color=color

            mag_plot = getFilteredMag(mag, filt)

            plt.plot(sample_times, mag_plot, color=color2, linewidth=3, linestyle="--")

            if len(models) > 1:
                plt.fill_between(
                    sample_times,
                    mag_plot + error_budget[filt],
                    mag_plot - error_budget[filt],
                    facecolor=color2,
                    alpha=0.2,
                    label="Combined",
                )
            else:
                plt.fill_between(
                    sample_times,
                    mag_plot + error_budget[filt],
                    mag_plot - error_budget[filt],
                    facecolor=color2,
                    alpha=0.2,
                )

            if len(models) > 1:
                for ii in range(len(mag_all)):
                    mag_plot = getFilteredMag(mag_all[ii], filt)
                    plt.plot(
                        sample_times,
                        mag_plot,
                        color=color2,
                        linewidth=3,
                        linestyle="--",
                    )
                    plt.fill_between(
                        sample_times,
                        mag_plot + error_budget[filt],
                        mag_plot - error_budget[filt],
                        facecolor=model_colors[ii],
                        alpha=0.2,
                        label=models[ii].model,
                    )

            plt.ylabel("%s" % filt, fontsize=48, rotation=0, labelpad=40)

            plt.xlim([float(x) for x in args.xlim.split(",")])
            plt.ylim([float(x) for x in args.ylim.split(",")])
            plt.grid()

            if cnt == 1:
                ax1.set_yticks([26, 22, 18, 14])
                plt.setp(ax1.get_xticklabels(), visible=False)
                if len(models) > 1:
                    plt.legend(
                        loc="upper right",
                        prop={"size": 18},
                        numpoints=1,
                        shadow=True,
                        fancybox=True,
                    )
            elif not cnt == len(filters_plot):
                plt.setp(ax2.get_xticklabels(), visible=False)
            plt.xticks(fontsize=36)
            plt.yticks(fontsize=36)

        ax1.set_zorder(1)
        plt.xlabel("Time [days]", fontsize=48)
        plt.tight_layout()
        plt.savefig(plotName)
        plt.close()
