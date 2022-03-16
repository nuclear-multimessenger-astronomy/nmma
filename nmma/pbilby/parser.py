import bilby
import parallel_bilby
import bilby_pipe
import bilby_pipe.bilbyargparser
import argparse
import logging
from parallel_bilby.parser import (
    _add_dynesty_settings_to_parser,
    _add_slurm_settings_to_parser,
    _add_misc_settings_to_parser,
)


def purge_empty_argument_group(parser):
    non_empty_action_groups = []
    non_empty_mutually_exclusive_groups = []

    try:
        # Purge _action_groups
        for action_group in parser._action_groups:
            if action_group._group_actions != []:
                non_empty_action_groups.append(action_group)
        # Purge _mutually_exclusive_groups
        for action_group in parser._mutually_exclusive_groups:
            if action_group._group_actions != []:
                non_empty_mutually_exclusive_groups.append(action_group)
    except Exception:
        pass

    parser._action_groups = non_empty_action_groups
    parser._mutually_exclusive_groups = non_empty_mutually_exclusive_groups


def remove_arguments_from_parser(parser, args, prog):
    logger = logging.getLogger(prog)

    for arg in args:
        for action in parser._actions:
            if action.dest == arg.replace("-", "_"):
                try:
                    parser._handle_conflict_resolve(None, [("--" + arg, action)])
                except ValueError as e:
                    logger.warning("Error removing {}: {}".format(arg, e))
        logger.debug(
            "Request to remove arg {} from bilby_pipe args, but arg not found".format(
                arg
            )
        )

    purge_empty_argument_group(parser)


def keep_arguments_from_parser(parser, args, prog):
    original_args = [action.dest.replace("_", "-") for action in parser._actions]
    args_to_remove = list(set(original_args) - set(args))

    remove_arguments_from_parser(parser, args_to_remove, prog)


def _create_base_parser(prog, prog_version):
    base_parser = argparse.ArgumentParser(prog, add_help=False)
    base_parser.add(
        "--version",
        action="version",
        version=f"%(prog)s={prog_version}\nbilby={bilby.__version__}\nbilby_pipe={bilby_pipe.__version__}\nparallel_bilby={parallel_bilby.__version__}",
    )

    base_parser = _add_dynesty_settings_to_parser(base_parser)
    base_parser = _add_misc_settings_to_parser(base_parser)

    return base_parser


def _remove_arguments_from_bilby_pipe_parser_for_nmma(bilby_pipe_parser, prog):
    bilby_pipe_arguments_to_keep = [
        "ini",
        "help",
        "verbose",
        "sampler",
        "sampling-seed",
        "n-parallel",
        "sampler-kwargs",
        "accounting",
        "label",
        "local",
        "local-generation",
        "outdir",
        "periodic-restart-time",
        "request-memory",
        "request-memory-generation",
        "request-cpus",
        "singularity-image",
        "scheduler",
        "scheduler-args",
        "scheduler-module",
        "scheduler-env",
        "scheduler-analysis-time",
        "submit",
        "condor-job-priority",
        "transfer-files",
        "log-directory",
        "online-pe",
        "osg",
        "single-postprocessing-executable",
        "single-postprocessing-arguments",
        "create-summary",
        "notification",
        "email",
    ]

    keep_arguments_from_parser(bilby_pipe_parser, bilby_pipe_arguments_to_keep, prog)

    return bilby_pipe_parser


def _remove_arguments_from_bilby_pipe_parser_for_pbilby(bilby_pipe_parser, prog):
    bilby_pipe_arguments_to_ignore = [
        "version",
        "accounting",
        "local",
        "local-generation",
        "local-plot",
        "request-memory",
        "request-memory-generation",
        "request-cpus",
        "singularity-image",
        "scheduler",
        "scheduler-args",
        "scheduler-module",
        "scheduler-env",
        "condor-job-priority",
        "periodic-restart-time",
        "transfer-files",
        "online-pe",
        "osg",
        "email",
        "postprocessing-executable",
        "postprocessing-arguments",
        "sampler",
        "sampling-seed",
        "sampler-kwargs",
        "plot-calibration",
        "plot-corner",
        "plot-format",
        "plot-marginal",
        "plot-skymap",
        "plot-waveform",
    ]

    remove_arguments_from_parser(
        bilby_pipe_parser, bilby_pipe_arguments_to_ignore, prog
    )

    bilby_pipe_parser.add_argument(
        "--sampler",
        choices=["dynesty"],
        default="dynesty",
        type=str,
        help="The parallelised sampler to use, defaults to dynesty",
    )

    return bilby_pipe_parser


def _add_nmma_settings_to_parser(parser):
    em_input_parser = parser.add_argument_group(
        title="EM analysis input arguments", description="Specify EM analysis inputs"
    )
    em_input_parser.add("--binary-type", type=str, help="The binary is BNS or NSBH")
    em_input_parser.add(
        "--light-curve-data", type=str, help="Path to the observed light curve data"
    )
    em_input_parser.add(
        "--with-grb",
        action="store_true",
        help="Flag for including GRB afterglow in analysis",
    )
    em_input_parser.add(
        "--grb-jettype",
        type=int,
        default=0,
        help="GRB jet type (default: 0 (Gaussian))",
    )
    em_input_parser.add(
        "--grb-resolution",
        type=float,
        default=5,
        help="The upper bound on the ratio between thetaWing and thetaCore (default: 5)",
    )
    em_input_parser.add(
        "--kilonova-model", type=str, help="Name of the kilonova model to be used"
    )
    em_input_parser.add(
        "--kilonova-model-svd", type=str, help="Path to the kilonova model's SVD data"
    )
    em_input_parser.add(
        "--kilonova-interpolation-type",
        type=str,
        help="Interpolation method to be used for KN model (sklearn_gp or tensorflow)"
    )
    em_input_parser.add(
        "--svd-mag-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for mag evaluation (default: 10)",
    )
    em_input_parser.add(
        "--svd-lbol-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for lbol evaluation (default: 10)",
    )
    em_input_parser.add(
        "--filters",
        type=str,
        help="A comma seperated list of filters to use (e.g. g,r,i)."
        "If none is provided, will use all the filters available",
    )
    em_input_parser.add(
        "--with-grb-injection",
        action="store_true",
        help="Flag for including GRB afterglow in injection",
    )
    em_input_parser.add(
        "--kilonova-injection-model",
        type=str,
        help="Name of the kilonova model to be used for injection",
    )
    em_input_parser.add(
        "--kilonova-injection-svd",
        type=str,
        help="Path to the kilonova model's SVD data for injection",
    )
    em_input_parser.add(
        "--injection-svd-mag-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for mag evaluation (default: 10)",
    )
    em_input_parser.add(
        "--injection-svd-lbol-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for lbol evaluation (default: 10)",
    )
    em_input_parser.add(
        "--injection-detection-limit",
        type=float,
        default=28,
        help="Above which the injection light curve is expected to be not detectable (default: 28)",
    )
    em_input_parser.add(
        "--kilonova-trigger-time",
        type=float,
        help="Time for the kilonova trigger (in MJD)",
    )
    em_input_parser.add(
        "--kilonova-tmin",
        type=float,
        default=0.0,
        help="Days to be started analysing from the trigger time (default: 0)",
    )
    em_input_parser.add(
        "--kilonova-tmax",
        type=float,
        default=14.0,
        help="Days to be stoped analysing from the trigger time (default: 14)",
    )
    em_input_parser.add(
        "--kilonova-tstep",
        type=float,
        default=0.1,
        help="Time step (in days) for light curve initial evalution (default: 0.1)",
    )
    em_input_parser.add(
        "--kilonova-error",
        type=float,
        default=1.0,
        help="Additionaly statistical error (mag) to be introdouced (default: 1)",
    )

    eos_input_parser = parser.add_argument_group(
        title="EOS input arguments", description="Specify EOS inputs"
    )
    eos_input_parser.add(
        "--with-eos",
        action="store_true",
        help="Flag for sampling over EOS (default:True)",
    )
    eos_input_parser.add("--eos-data", type=str, help="Path to the EOS directory")
    eos_input_parser.add("--Neos", type=int, help="Number of EOSs to be used")
    eos_input_parser.add(
        "--eos-weight", type=str, help="Path to the precalculated EOS weighting"
    )

    return parser


def create_nmma_generation_parser(prog, prog_version):
    base_parser = _create_base_parser(prog, prog_version)
    bilby_pipe_parser = _remove_arguments_from_bilby_pipe_parser_for_pbilby(
        bilby_pipe.parser.create_parser(), prog
    )
    #    bilby_pipe_parser = _remove_arguments_from_bilby_pipe_parser_for_nmma(
    #        bilby_pipe_parser, prog
    #    )

    nmma_generation_parser = bilby_pipe.bilbyargparser.BilbyArgParser(
        prog=prog,
        ignore_unknown_config_file_keys=False,
        allow_abbrev=False,
        add_help=False,
        parents=[base_parser, bilby_pipe_parser],
    )

    nmma_generation_parser = _add_slurm_settings_to_parser(nmma_generation_parser)
    nmma_generation_parser = _add_nmma_settings_to_parser(nmma_generation_parser)

    purge_empty_argument_group(nmma_generation_parser)

    return nmma_generation_parser


def create_nmma_analysis_parser(prog, prog_version):
    parser = _create_base_parser(prog, prog_version)
    analysis_parser = argparse.ArgumentParser(prog="nmma_analysis", parents=[parser])
    analysis_parser.add_argument(
        "data_dump",
        type=str,
        help="The pickled data dump generated by nmma_generation",
    )
    analysis_parser.add_argument(
        "--outdir", default=None, type=str, help="Outdir to overwrite input label"
    )
    analysis_parser.add_argument(
        "--label", default=None, type=str, help="Label to overwrite input label"
    )
    return analysis_parser
