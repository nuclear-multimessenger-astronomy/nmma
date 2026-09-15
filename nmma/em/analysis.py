import numpy as np
import pandas as pd

from ..core.base import multi_analysis_loop
from ..core.utils import injection_from_args, read_trigger_time, set_filename
from . import io, model, systematics, utils
from .em_likelihood import EMTransientLikelihood
from .em_parsing import (
    bolometric_parser,
    multi_wavelength_analysis_parser,
    parsing_and_logging,
)
from .lightcurve_handling import adjust_injection_parameters, create_light_curve_data
from .prior import create_prior_from_args


def data_from_injection(args, filters):
    inj_model = model.create_injection_model(args, filters)
    injection_params = injection_from_args(args)
    injection_params = adjust_injection_parameters(injection_params, args, inj_model)
    inj_outfile = set_filename(args.label, args, "_lc")
    if inj_outfile.exists():
        print(f"Loading existing injection lc from {inj_outfile}")
        full_data = io.load_em_observations(inj_outfile, format="model")
    else:
        # keep_infinite_data=True (full time grid, non-detections marked
        full_data = create_light_curve_data(
            injection_params, args, inj_model, keep_infinite_data=True
        )
        io.write_em_observations(inj_outfile, full_data, format="model")
    data = {
        filt: {
            key: val[
                np.isfinite(full_data[filt]["mag"])
                & np.isfinite(full_data[filt]["mag_error"])
            ]
            for key, val in filt_dict.items()
        }
        for filt, filt_dict in full_data.items()
    }
    return data, injection_params


def inspect_detection_limit(detection_limit, data):
    # checking data for magnitudes dimmer than the detection limit
    for filt, filt_dict in data.items():
        non_detections = filt_dict["mag"] > detection_limit[filt]

        filt_dict["mag"] = np.where(
            non_detections, detection_limit[filt], filt_dict["mag"]
        )
        filt_dict["mag_error"] = np.where(
            non_detections, np.inf, filt_dict["mag_error"]
        )
    return data


def check_detections(data, remove_nondetections=False):
    if remove_nondetections:
        for filt, filt_dict in data.items():
            detections = np.isfinite(filt_dict["mag_error"])
            if detections.any():
                data[filt] = {k: v[detections] for k, v in filt_dict.items()}
            else:
                data.pop(filt)

    if not any(np.isfinite(data[filt]["mag_error"]).any() for filt in data):
        print("No detection available, fits only on non-detections.")
    return data


def set_analysis_filters(filters, data):
    if filters is None:
        return list(data.keys())

    filters_to_analyze = [filt for filt in data.keys() if filt in filters]
    print(f"Running with filters {filters_to_analyze}")
    return filters_to_analyze


def bolometric_setup(args):

    # create the data
    # FIXME add  injection functionality
    # if args.injection_file:
    #     pass
    injection_parameters = None

    # load the bolometric data
    data = pd.read_csv(args.light_curve_data)
    trigger_time = read_trigger_time(None, args)
    light_curve_data = utils.setup_bolometric_lc_data(data, trigger_time)

    light_curve_model = model.SimpleBolometricLightCurveModel(
        args.em_model,
        sample_times=utils.setup_sample_times(
            args
        ),  # usually None, defaults to model_times
    )
    systematics_handler = systematics.SystematicsHandler(
        args.systematics_file, args.em_error_budget, light_curve_data[0]
    )

    # setup the prior
    priors = create_prior_from_args(args, systematics_handler)

    # setup the likelihood
    likelihood_kwargs = dict(
        light_curve_model=light_curve_model,
        light_curve_data=light_curve_data,
        priors=priors,
        systematics_handler=systematics_handler,
        verbose=args.verbose,
    )
    likelihood = EMTransientLikelihood(**likelihood_kwargs)

    return priors, likelihood, injection_parameters


def analysis_setup(args):

    filters = utils.set_filters(args)
    if getattr(args, "light_curve_data", None):
        # load observational data
        data = io.load_em_observations(args, format="observations")
        trigger_time = read_trigger_time(None, args)
        injection_parameters = getattr(args, "injection_parameters", None)
    else:
        # try to work with injection data instead
        data, injection_parameters = data_from_injection(args, filters)
        trigger_time = injection_parameters.get("trigger_time", 0)
    data = utils.cut_data_to_time_range(data, args, trigger_time)
    detection_limit = utils.create_detection_limit(args, data.keys())
    if injection_parameters is not None:
        data = inspect_detection_limit(detection_limit, data)
    data = check_detections(data, args.remove_nondetections)
    filters_to_analyze = set_analysis_filters(filters, data)
    detection_limit = {filt: detection_limit[filt] for filt in filters_to_analyze}

    # initialize light curve model
    print("Creating light curve model for inference")
    light_curve_model = model.create_light_curve_model_from_args(
        args,
        filters=filters_to_analyze,
    )

    light_curve_data = utils.setup_filtered_lc_data(data, trigger_time)
    systematics_handler = systematics.FilterSystematicsHandler(
        filters_to_analyze,
        args.systematics_file,
        args.em_error_budget,
        light_curve_data[0],
    )
    priors = create_prior_from_args(args, systematics_handler)
    if injection_parameters is not None:
        injection_parameters = {
            k: injection_parameters.get(k, None) for k in priors.keys()
        }
    light_curve_data = utils.check_model_time_consistency(
        light_curve_data, 
        light_curve_model, 
        priors, 
        injection_parameters,
        allow_data_cuts=args.allow_data_cuts,
    )
    # check_model_time_consistency may cut the data; rebuild the handler so
    # its per-filter error_budget arrays match the cut light_curve_times.
    systematics_handler = systematics.FilterSystematicsHandler(
        filters_to_analyze,
        args.systematics_file,
        args.em_error_budget,
        light_curve_data[0],
    )
    # setup the likelihood
    likelihood_kwargs = dict(
        light_curve_model=light_curve_model,
        filters=filters_to_analyze,
        light_curve_data=light_curve_data,
        priors=priors,
        systematics_handler=systematics_handler,
        verbose=args.verbose,
        detection_limit=detection_limit,
    )

    likelihood = EMTransientLikelihood(**likelihood_kwargs)
    return priors, likelihood, injection_parameters


def main(args=None):
    if isinstance(args, dict):
        non_default = args.copy()
        args = []
    else:
        non_default = {}
    args = parsing_and_logging(multi_wavelength_analysis_parser, args)
    args.__dict__.update(non_default)

    multi_analysis_loop(args, analysis_setup)


def lbol_main(args=None):
    args = parsing_and_logging(bolometric_parser, args)
    multi_analysis_loop(args, bolometric_setup)
