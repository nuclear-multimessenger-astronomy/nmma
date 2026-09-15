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
    """Build or reload the simulated light curve of an injected transient.

    The light curve is cached as ``<outdir>/<label>_lc.<extension>``: it is
    read back if that file exists, and generated then written otherwise.
    Points with a non-finite magnitude or uncertainty are dropped from the
    returned data.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed command-line arguments, including the injection file, the
        label and the model settings.
    filters: list of str or None
        Photometric filters, e.g. ``ztfg`` or ``ps1::r``. If None, the model
        is built for every filter it supports.

    Returns
    -------
    data: dict
        Photometry per filter, each holding finite ``time``, ``mag`` and
        ``mag_error`` arrays.
    injection_params: dict
        Injection parameters actually used, after adjustment to the model.
    """

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
    """Clip photometry below the per-filter detection limit.

    Points fainter than their filter limit are set to that limit and their
    uncertainty to ``inf``, which marks them as non-detections for the
    likelihood.

    Parameters
    ----------
    detection_limit: dict
        Limiting magnitude per filter.
    data: dict
        Photometry per filter, holding ``mag`` and ``mag_error`` arrays.

    Returns
    -------
    dict
        The same object, modified in place.
    """

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
    """Report, and optionally drop, non-detections in the photometry.

    Non-detections are the points whose uncertainty is not finite. When
    ``remove_nondetections`` is set they are removed, and a filter left
    without any point is dropped altogether.

    Parameters
    ----------
    data: dict
        Photometry per filter, holding ``mag`` and ``mag_error`` arrays.
    remove_nondetections: bool, optional
        If True, discard non-detections instead of keeping them.

    Returns
    -------
    dict
        The photometry, possibly with filters removed.
    """

    if remove_nondetections:
        for filt, filt_dict in list(data.items()):
            detections = np.isfinite(filt_dict["mag_error"])
            if detections.any():
                data[filt] = {k: v[detections] for k, v in filt_dict.items()}
            else:
                data.pop(filt)
        # FIXME weizmann: to be activated separately, after the NMMA
        # documentation work.
        # if not data:
        #     raise ValueError("No filter left after removing non-detections.")

    if not any(np.isfinite(data[filt]["mag_error"]).any() for filt in data):
        print("No detection available, fits only on non-detections.")
    return data


def set_analysis_filters(filters, data):
    """Restrict the requested filters to those actually present in the data.

    Parameters
    ----------
    filters: list of str or None
        Filters requested by the user. If None, every filter in ``data`` is
        analysed.
    data: dict
        Photometry per filter.

    Returns
    -------
    list of str
        Filters to run the analysis on.
    """

    if filters is None:
        return list(data.keys())

    filters_to_analyze = [filt for filt in data.keys() if filt in filters]
    print(f"Running with filters {filters_to_analyze}")
    return filters_to_analyze


def bolometric_setup(args):
    """Assemble the prior and likelihood for a bolometric light curve fit.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed command-line arguments, including the bolometric data file,
        the model name and the systematics settings.

    Returns
    -------
    priors: bilby.core.prior.PriorDict or ConditionalPriorDict
        Priors on the model parameters. Conditional when
        ``--conditional-gaussian-prior-thetaObs`` is set.
    likelihood: `nmma.em.em_likelihood.EMTransientLikelihood`
        Likelihood of the bolometric data given the model.
    injection_parameters: None
        Always None; injections are not supported for bolometric fits yet.
    """

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
    """Assemble the prior and likelihood for a multi-band photometric fit.

    Photometry is read from ``args.light_curve_data`` when given, and
    simulated from an injection otherwise. It is then cut to the requested
    time range and restricted to the available filters, before the light
    curve model, the systematics handler and the priors are built. Clipping
    to the detection limits is applied to injections only.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    priors: bilby.core.prior.PriorDict or ConditionalPriorDict
        Priors on the model parameters. Conditional when
        ``--conditional-gaussian-prior-thetaObs`` is set.
    likelihood: `nmma.em.em_likelihood.EMTransientLikelihood`
        Likelihood of the photometry given the model.
    injection_parameters: dict or None
        Injection parameters restricted to the sampled parameters, or None
        when fitting observations.
    """

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
        light_curve_data, light_curve_model, priors, injection_parameters
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
    """Entry point of the ``lightcurve-analysis`` command.

    Parameters
    ----------
    args: dict or str or list of str or argparse.Namespace or None, optional
        Command-line arguments. A dict is applied on top of the parser
        defaults, which is convenient when driving the analysis from Python;
        a str is split on whitespace; a Namespace is used as-is. Defaults to
        ``sys.argv[1:]``.

    See Also
    --------
    analysis_setup: builds the prior and likelihood used here.
    nnanalysis: flow-based alternative, selected by ``--sampler neuralnet``.
    """

    if isinstance(args, dict):
        non_default = args.copy()
        args = []
    else:
        non_default = {}
    args = parsing_and_logging(multi_wavelength_analysis_parser, args)
    args.__dict__.update(non_default)

    multi_analysis_loop(args, analysis_setup)


def lbol_main(args=None):
    """Entry point of the ``lightcurve-analysis-lbol`` command.

    Parameters
    ----------
    args: str or list of str or argparse.Namespace or None, optional
        Command-line arguments. A str is split on whitespace; a Namespace is
        used as-is. Defaults to ``sys.argv[1:]``.

    See Also
    --------
    bolometric_setup: builds the prior and likelihood used here.
    """

    args = parsing_and_logging(bolometric_parser, args)
    multi_analysis_loop(args, bolometric_setup)
