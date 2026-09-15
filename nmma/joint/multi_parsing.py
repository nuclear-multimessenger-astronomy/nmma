import argparse
import bilby
from bilby_pipe import parser as bp_parser

from nmma.core.parsing import (
    base_analysis_parsing,
    dynesty_parsing,
    check_for_config,
)
from nmma.joint.joint_parsing import joint_likelihood_parsing
from nmma.em.em_parsing import em_analysis_parsing
from nmma.eos.eos_parsing import eos_parsing, tabulated_eos_parsing
from nmma.gw.gw_parsing import gw_parsing


from .. import __version__  # noqa: E402

logger = bilby.core.utils.logger


def _create_base_nmma_parser(sampler="dynesty", parents=[]):
    """
    Build the parser shared by :func:`create_nmma_generation_parser` and
    :func:`create_nmma_analysis_parser`.

    Parameters
    ----------
    sampler : str, default="dynesty"
        ``"all"`` or ``"dynesty"`` additionally applies
        :func:`nmma.core.parsing.dynesty_parsing` and
        :func:`multi_dynesty_parsing`.
    parents : list, default=[]
        Passed to ``argparse.ArgumentParser`` as ``parents``.

    Returns
    -------
    argparse.ArgumentParser
        The newly created parser.
    """
    base_parser = argparse.ArgumentParser(
        "base", parents=parents, conflict_handler="resolve", add_help=False
    )

    base_parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s={__version__}\nbilby={bilby.__version__}",
    )

    base_parser = em_settings_parsing(base_parser)
    base_parser = base_analysis_parsing(base_parser)
    base_parser = em_analysis_parsing(base_parser)
    base_parser = gw_parsing(base_parser)
    base_parser = joint_likelihood_parsing(base_parser)
    base_parser = tabulated_eos_parsing(base_parser)
    base_parser = eos_parsing(base_parser)
    base_parser = add_misc_settings(base_parser)
    if sampler in ["all", "dynesty"]:
        base_parser = dynesty_parsing(base_parser)
        base_parser = multi_dynesty_parsing(base_parser)

    return base_parser


def em_settings_parsing(parser):
    # general args
    """
    Add ``--light-curve-data`` in an "EM analysis input arguments" group.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the argument to.

    Returns
    -------
    argparse.ArgumentParser
        The same parser, with the arguments added.
    """
    em_input_parser = parser.add_argument_group(
        title="EM analysis input arguments", description="Specify EM analysis inputs"
    )
    em_input_parser.add(
        "--light-curve-data", help="Path to the observed light curve data"
    )
    return parser


def multi_dynesty_parsing(parser):
    """
    Add the dynesty sampler arguments in their own group.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the arguments to.

    Returns
    -------
    argparse.ArgumentParser
        The same parser, with the arguments added.
    """
    sampler_group = parser.add_argument_group(title="Setting for the Dynesty Sampler")

    sampler_group.add_argument(
        "--sampler",
        default="dynesty",
        help="The parallelised sampler to use, defaults to dynesty",
    )
    sampler_group.add_argument(
        "--bound",
        "--dynesty-bound",
        default="live",
        help="Dynesty bounding method (default=live)",
    )
    sampler_group.add_argument(
        "--save-bounds",
        action="store_true",
        help="Whether to store bounds in the resume file. Not doing this can make resume files large (~GB)",
    )
    sampler_group.add_argument(
        "--sample",
        "--dynesty-sample",
        default="acceptance-walk",
        help="sampling method (default=acceptance-walk).",
    )
    sampler_group.add_argument(
        "--n-check-point",
        default=2000,
        type=int,
        help="Steps to take before attempting checkpoint",
    )

    return parser


def add_misc_settings(parser):
    """
    Add ``--clean`` and ``--plot`` in a "Misc. Settings" group.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the arguments to.

    Returns
    -------
    argparse.ArgumentParser
        The same parser, with the arguments added.
    """
    misc_group = parser.add_argument_group(title="Misc. Settings")
    misc_group.add_argument(
        "-c", "--clean", action="store_true", help="Run clean: ignore any resume files"
    )
    misc_group.add_argument(
        "--plot",
        action="store_true",
        help="Whether to generate the various data plots at the end of the run",
    )
    return parser


def run_parsing(parser):
    """
    Add the main run arguments in their own group.

    ``data_dump`` is added twice, as an optional positional and as
    ``--data-dump``, both writing to the ``data_dump`` destination.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the arguments to.

    Returns
    -------
    argparse.ArgumentParser
        The same parser, with the arguments added.
    """
    run_group = parser.add_argument_group(title="Setting for the Main run")
    run_group.add_argument("data_dump", nargs="?")  # nargs makes it optional
    run_group.add_argument(
        "--data-dump",
        dest="data_dump",
        help="The pickled data dump generated by nmma_generation",
    )
    run_group.add_argument("--outdir", help="Outdir to overwrite input label")
    run_group.add_argument("--label", help="Label to overwrite input label")
    run_group.add_argument(
        "--result-format", default="hdf5", help="Format to save the result"
    )

    return parser


def remove_argument_from_parser(parser, arg):
    """
    Remove an argument from ``parser`` by its destination.

    ``arg`` is matched with ``-`` replaced by ``_`` against each
    ``action.dest``. A ``ValueError`` raised during removal is caught and
    logged as a warning.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to remove the argument from.
    arg : str
        Argument name without leading dashes.
    """
    for action in parser._actions:
        if action.dest == arg.replace("-", "_"):
            try:
                parser._handle_conflict_resolve(None, [("--" + arg, action)])
            except ValueError as e:
                logger.warning(f"Error removing {arg}: {e}")
    # FIXME: Debug "arg not found" logged outside loop, fires on successful removals
    logger.debug(f"Request to remove arg {arg} from bilby_pipe args, but arg not found")


def _create_reduced_bilby_pipe_parser():
    """
    Build the ``bilby_pipe`` parser with a fixed list of arguments removed.

    The removed names cover versioning, scheduler and submission settings,
    post-processing hooks, sampler selection and the plotting options.

    Returns
    -------
    The parser from ``bilby_pipe.parser.create_parser`` with
    ``top_level=False``, after the removals.
    """
    bilby_pipe_parser = bp_parser.create_parser(top_level=False)
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
        # "transfer-files",
        "online-pe",
        "osg",
        "email",
        "postprocessing-executable",
        "postprocessing-arguments",
        "sampler",
        "sampling-seed",
        "sampler-kwargs",
        "sampler_kwargs",
        "plot-calibration",
        "plot-corner",
        "plot-format",
        "plot-marginal",
        "plot-skymap",
        "plot-waveform",
    ]
    for arg in bilby_pipe_arguments_to_ignore:
        remove_argument_from_parser(bilby_pipe_parser, arg)

    return bilby_pipe_parser


def create_nmma_generation_parser():
    """
    Build the parser for ``nmma_generation``.

    Returns
    -------
    argparse.ArgumentParser
        :func:`_create_base_nmma_parser` with ``sampler="all"`` and
        :func:`_create_reduced_bilby_pipe_parser` as a parent.
    """
    bilby_pipe_parser = _create_reduced_bilby_pipe_parser()
    generation_parser = _create_base_nmma_parser(
        sampler="all", parents=[bilby_pipe_parser]
    )
    # generation_parser = argparse.ArgumentParser(
    #     prog="nmma_generation",
    #     usage=__doc__,
    #     ignore_unknown_config_file_keys=False,
    #     allow_abbrev=False,
    #     parents=[parser, bilby_pipe_parser],
    #     add_help=False,
    # )
    return generation_parser


def parse_generation_args(cli_args=[""]):
    """
    Parse arguments for ``nmma_generation``.

    Parameters
    ----------
    cli_args : list of str, default=[""]
        Arguments to parse, passed through
        :func:`nmma.core.parsing.check_for_config` first.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    argparse.ArgumentParser
        The parser they were parsed with, as returned by
        :func:`nmma.core.parsing.check_for_config`.
    """
    generation_parser = create_nmma_generation_parser()
    generation_parser, cli_args = check_for_config(cli_args, [generation_parser], False)
    args = generation_parser.parse_args(args=cli_args)
    return args, generation_parser


def create_nmma_analysis_parser(sampler="dynesty"):
    """
    Build the parser for ``nmma_analysis``.

    Parameters
    ----------
    sampler : str, default="dynesty"
        Passed to :func:`_create_base_nmma_parser`.

    Returns
    -------
    argparse.ArgumentParser
        The base parser with :func:`run_parsing` applied.
    """
    parser = _create_base_nmma_parser(sampler=sampler)
    parser = run_parsing(parser)
    return parser


def parse_analysis_args(parser, args=None):
    """
    Parse arguments for ``nmma_analysis``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to parse with.
    args : list of str, default=None
        Passed to ``parser.parse_args`` as ``args``.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.

    Raises
    ------
    ValueError
        If ``walks`` exceeds ``maxmcmc``, or if ``nact`` is below 1.
    """
    args = parser.parse_args(args=args)

    if args.walks > args.maxmcmc:
        raise ValueError(
            f"You have maxmcmc ({args.maxmcmc}) > walks ({args.walks}, minimum mcmc)"
        )
    if args.nact < 1:
        raise ValueError(f"Your nact ({args.nact}) < 1 (must be >= 1)")

    return args
