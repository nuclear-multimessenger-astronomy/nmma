"""Parameter inference on electromagnetic transients.

This module assembles the data -> model -> prior -> likelihood chain and
hands the sampling over to :func:`nmma.core.base.multi_analysis_loop`.

It backs two commands: ``lightcurve-analysis`` for multi-band photometric
fits (:func:`main`) and ``lightcurve-analysis-lbol`` for bolometric ones
(:func:`lbol_main`).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from .lightcurve_handling import create_light_curve_data, adjust_injection_parameters
from .em_likelihood import EMTransientLikelihood
from .prior import create_prior_from_args
from . import io, model, utils, systematics
from .em_parsing import (
    parsing_and_logging,
    multi_wavelength_analysis_parser,
    bolometric_parser,
)
from ..core.base import multi_analysis_loop
from ..core.utils import injection_from_args, set_filename, read_trigger_time


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


def nnanalysis(args):
    """Infer kilonova parameters with a pre-trained normalizing flow.

    A likelihood-free alternative to the sampler-based path: the photometry
    is padded onto a fixed time grid, embedded, and passed to a frozen flow
    that yields posterior samples directly. A corner plot is written to
    ``args.outdir``.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed command-line arguments. ``args.em_model`` must be ``Ka2017``.

    Raises
    ------
    ValueError
        If the injection parameters do not match those the flow was trained
        on (``log10_mej``, ``log10_vej`` and ``log10_Xlan``).

    Notes
    -----
    Filters are hard-coded to ``ztfg``, ``ztfr`` and ``ztfi``, and the time
    grid to 121 points spaced by 0.25 day. The process exits if a model
    other than ``Ka2017`` is requested.
    """

    # import functions
    from ..mlmodel.dataprocessing import pad_the_data
    from ..mlmodel.embedding import SimilarityEmbedding
    from ..mlmodel.normalizingflows import normflow_params
    from ..mlmodel.inference import cast_as_bilby_result
    import torch
    from nflows.flows import Flow

    # only continue if the Kasen model is selected
    if isinstance(args.em_model, str):
        args.em_model = args.em_model.split(",")
    if args.em_model[0] != "Ka2017":
        print(
            "WARNING: model selected is not currently compatible with this inference method"
        )
        exit()

    # only can use ztfr, ztfg, and ztfi filters in the light curve data
    print(
        "Currently filters are hardcoded to ztfr, ztfi, and ztfg. Continuing with these filters."
    )
    filters = ["ztfg", "ztfi", "ztfr"]

    # create the kilonova data if an injection set is given
    if args.injection_file:
        data, injection_parameters = data_from_injection(args, filters)
    else:
        # load the lightcurve data
        data = io.load_em_observations(args)

    detection_limit = utils.create_detection_limit(args, filters, 22.0)
    data = inspect_detection_limit(detection_limit, data)
    data = check_detections(data, args.remove_nondetections)
    filters_to_analyze = set_analysis_filters(filters, data)
    detection_limit = {filt: detection_limit[filt] for filt in filters_to_analyze}

    model.create_light_curve_model_from_args(
        args,
        filters=filters_to_analyze,
    )

    # setup the prior
    systematics_handler = systematics.FilterSystematicsHandler(
        filters_to_analyze, args.systematics_file, error_budget=args.em_error_budget
    )
    priors = create_prior_from_args(args, systematics_handler)

    # now that we have the kilonova light curve, we need to pad it with non-detections
    # this part is currently hard coded in terms of the times !!!! likely will need the most work
    # (so that the 'fixed' and 'shifted' are properly represented)
    num_points = 121
    num_channels = 3
    time_step = 0.25

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Convert data dict to DataFrame with time and filter columns
    res = next(iter(data))
    t_list = (data[res]["time"]).tolist()
    data_df = pd.DataFrame({"t": t_list})
    for key in data:
        data_df[key] = data[key]["mag"]
    column_list = data_df.columns.to_list()

    # pad the data

    padded_data_df = pad_the_data(
        data_df,
        column_list,
        desired_count=num_points,
        filler_time_step=time_step,
        filler_data=detection_limit[
            column_list[-1]
        ],  # some value from the detection limit dict
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
    SAVEPATH = Path(__file__).parent.parent / "mlmodel/similarity_embedding_weights.pth"
    similarity_embedding.load_state_dict(torch.load(SAVEPATH, map_location=device))
    for name, param in similarity_embedding.named_parameters():
        param.requires_grad = False

    # set up the normalizing flows
    transform, base_dist, embedding_net = normflow_params(
        similarity_embedding, 9, 5, 90, context_features=num_dim, num_dim=num_dim
    )
    flow = Flow(transform, base_dist, embedding_net).to(device=device)
    PATH_nflow = Path(__file__).parent.parent / "mlmodel/frozen-flow-weights.pth"
    flow.load_state_dict(torch.load(PATH_nflow, map_location=device))

    nsamples = 20000
    with torch.no_grad():
        samples = flow.sample(nsamples, context=data_tensor)
        samples = samples.cpu().reshape(nsamples, 3)

    try:
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
    except NameError:
        truth = None
    except KeyError:
        raise ValueError(
            "The injection parameters provided do not match the parameters the flow has been trained on"
        )

    flow_result = cast_as_bilby_result(samples, truth, priors=priors)
    flow_result.plot_corner(save=True, label=args.label, outdir=args.outdir)
    print("saved posterior plot")


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

    if args.sampler == "neuralnet":
        nnanalysis(args)
    else:
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
