import argparse
import bilby
import bilby_pipe.data_generation

from nmma.joint.base_parsing import StoreBoolean, base_analysis_parsing, nonestr
from nmma.em.em_parsing import em_analysis_parsing
from nmma.eos.eos_parsing import eos_parsing, tabulated_eos_parsing
from nmma.gw.gw_parsing import gw_parsing


from ... import __version__  # noqa: E402
from numpy import inf
logger = bilby.core.utils.logger


def em_settings_parsing(parser):
    # general args
    em_input_parser = parser.add_argument_group(
        title="EM analysis input arguments", description="Specify EM analysis inputs"
    )
    em_input_parser.add("--light-curve-data", type=str,
        help="Path to the observed light curve data")
    return parser

 
def dynesty_parsing(parser):
    dynesty_group = parser.add_argument_group(title="Dynesty Settings")
    dynesty_group.add_argument( "-n", "--nlive", default=1000, type=int, help="Number of live points" )
    dynesty_group.add_argument("--dlogz", default=0.1, type=float,
        help="Stopping criteria: remaining evidence, (default=0.1)" )
    dynesty_group.add_argument("--n-effective", default=inf, type=float,
        help="Stopping criteria: effective number of samples, (default=inf)" )
    dynesty_group.add_argument( "--dynesty-sample", default="acceptance-walk", type=str,
        help="Dynesty sampling method (default=acceptance-walk). "
            "Note, the dynesty rwalk method is overwritten by parallel bilby for an optimised version ")
    dynesty_group.add_argument("--dynesty-bound", default="live", type=str,
        help="Dynesty bounding method (default=live)" )
    dynesty_group.add_argument( "--walks", default=100, type=int,
        help="Minimum number of walks, defaults to 100" )
    dynesty_group.add_argument( "--proposals", type=nonestr, action="append", 
        help="The jump proposals to use, the options are 'diff' and 'volumetric'" )
    dynesty_group.add_argument("--maxmcmc", default=5000, type=int,
        help="Maximum number of walks, defaults to 5000" )
    dynesty_group.add_argument( "--nact", default=2, type=int,
        help="Number of autocorrelation times to take, defaults to 2")
    dynesty_group.add_argument("--naccept", default=60, type=int,
        help="The average number of accepted steps per MCMC chain, defaults to 60")
    dynesty_group.add_argument("--min-eff", default=10, type=float,
        help="The minimum efficiency at which to switch from uniform sampling.")
    dynesty_group.add_argument("--facc", default=0.5, type=float,
        help="See dynesty.NestedSampler")
    dynesty_group.add_argument("--enlarge", default=1.5, type=float,
        help="See dynesty.NestedSampler")
    dynesty_group.add_argument("--n-check-point", default=1000, type=int,
        help="Steps to take before attempting checkpoint")
    dynesty_group.add_argument("--max-its", default=10**10, type=int,
        help="Maximum number of iterations to sample for (default=1.e10)")
    dynesty_group.add_argument("--max-run-time", default=1.0e10, type=float,
        help="Maximum time to run for (default=1.e10 s)")
    dynesty_group.add_argument("--fast-mpi", default=False, action=StoreBoolean,
        help="Fast MPI communication pattern (default=False)")
    dynesty_group.add_argument("--mpi-timing", default=False, action=StoreBoolean,
        help="Print MPI timing when finished (default=False)")
    dynesty_group.add_argument("--mpi-timing-interval", default=0, type=int,
        help="Interval to write timing snapshot to disk (default=0 -- disabled)")
    dynesty_group.add_argument("--nestcheck", default=False, action=StoreBoolean,
        help=(
            "Save a 'nestcheck' pickle in the outdir (default=False). "
            "This pickle stores a `nestcheck.data_processing.process_dynesty_run` "
            "object, which can be used during post processing to compute the "
            "implementation and bootstrap errors explained by Higson et al (2018) "
            "in “Sampling Errors In Nested Sampling Parameter Estimation”."
        ),
    )
    dynesty_group.add_argument("--rejection-sample-posterior", default=True,
        action=StoreBoolean, help=(
            "Whether to generate the posterior samples by rejection sampling the "
            "nested samples or resampling with replacement" ) )
    return parser


def add_misc_settings(parser):
    misc_group = parser.add_argument_group(title="Misc. Settings")
    misc_group.add_argument("-c", "--clean", action=StoreBoolean, help="Run clean: ignore any resume files")
    misc_group.add_argument("--checkpoint-plot", default=True, action=StoreBoolean,
        help="Whether to generate analytical check-point plots")
    misc_group.add_argument("--save-bounds", default=False, action=StoreBoolean,
        help="Whether to store bounds in the resume file. Not doing this can make resume files large (~GB)")
    misc_group.add_argument("--check-point-deltaT", default=3600, type=float,
        help="Write a checkpoint resume file and diagnostic plots every deltaT [s]. Default: 1 hour.")
    misc_group.add_argument("--plot", default=False, action=StoreBoolean,
        help="Whether to generate the various data plots at the end of the run")
    return parser


def _create_base_nmma_parser(sampler="dynesty"):
    base_parser = argparse.ArgumentParser("base", add_help=False)

    base_parser.add_argument("--version", action="version",
        version=f"%(prog)s={__version__}\nbilby={bilby.__version__}",
    )

    if sampler in ["all", "dynesty"]:
        base_parser = dynesty_parsing(base_parser)

    base_parser = em_settings_parsing(base_parser)
    base_parser = base_analysis_parsing(base_parser)
    base_parser = em_analysis_parsing(base_parser)
    base_parser = gw_parsing(base_parser)
    base_parser = tabulated_eos_parsing(base_parser)
    base_parser = eos_parsing(base_parser)
    base_parser = add_misc_settings(base_parser)

    return base_parser


def slurm_parsing(parser):
    slurm_group = parser.add_argument_group(title="Slurm Settings")
    slurm_group.add_argument("--nodes", type=int, default=1,
        help="Number of nodes to use (default 1)")
    slurm_group.add_argument("--ntasks-per-node", type=int, default=2,
        help="Number of tasks per node (default 2)")
    slurm_group.add_argument("--time", type=str, default="24:00:00",
        help="Maximum wall time (defaults to 24:00:00)")
    slurm_group.add_argument("--mem-per-cpu", type=str, default="2G",
        help="Memory per CPU (defaults to 2GB)")
    slurm_group.add_argument("--extra-lines", type=str, default=None,
        help="Additional lines, separated by ';', use for setting up conda env or module imports",
    )
    slurm_group.add_argument("--slurm-extra-lines", type=str, default=None,
        help="additional slurm args (args that need #SBATCH in front) of the form arg=val separated by sapce",
    )
    return parser


def remove_argument_from_parser(parser, arg):
    for action in parser._actions:
        if action.dest == arg.replace("-", "_"):
            try:
                parser._handle_conflict_resolve(None, [("--" + arg, action)])
            except ValueError as e:
                logger.warning(f"Error removing {arg}: {e}")
    logger.debug(f"Request to remove arg {arg} from bilby_pipe args, but arg not found")

def _create_reduced_bilby_pipe_parser():
    bilby_pipe_parser = bilby_pipe.parser.create_parser()
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
        "transfer-files",
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

    bilby_pipe_parser.add_argument("--sampler", choices=["dynesty"], default="dynesty",
        help="The parallelised sampler to use, defaults to dynesty")
    return bilby_pipe_parser

def create_nmma_generation_parser():
    """Parser for nmma_generation"""
    parser = _create_base_nmma_parser(sampler="all")
    bilby_pipe_parser = _create_reduced_bilby_pipe_parser()
    generation_parser = bilby_pipe.parser.BilbyArgParser(
        prog="nmma_generation",
        usage=__doc__,
        ignore_unknown_config_file_keys=False,
        allow_abbrev=False,
        parents=[parser, bilby_pipe_parser],
        add_help=False,
    )
    generation_parser = slurm_parsing(generation_parser)
    return generation_parser


def parse_generation_args(parser, cli_args=[""], as_namespace=False):
    """
    Returns dictionary of arguments, as specified in the
    parser.

    If no cli_args arguments are specified, returns the default arguments
    (by running the parser with no ini file and no CLI arguments)

    Parameters
    ----------
    parser: generation-parser
    cli_args: list of strings (default: [""])
        List of arguments to be parsed. If empty, returns default arguments
    as_namespace: bool (default False)
        If True, returns the arguments as a Namespace object. If False, returns a dict

    Returns
    -------
    args: dict or Namespace

    """

    args = parser.parse_args(args=cli_args)

    pipe_args, _ = bilby_pipe.data_generation.parse_args(
        cli_args, bilby_pipe.data_generation.create_generation_parser()
    )
    for key, val in vars(pipe_args).items():
        if key not in args:
            setattr(args, key, val)
    if as_namespace:
        return args
    return vars(args)


def create_nmma_analysis_parser(sampler="dynesty"):
    """Parser for nmma_analysis"""
    parser = _create_base_nmma_parser(sampler=sampler)

    analysis_parser = argparse.ArgumentParser(prog="nmma_analysis", parents=[parser])
    analysis_parser.add_argument("--data-dump", type=str,
        help="The pickled data dump generated by nmma_generation" )
    analysis_parser.add_argument("--outdir", default=None, type=str, 
        help="Outdir to overwrite input label" )
    analysis_parser.add_argument("--label", default=None, type=str, 
        help="Label to overwrite input label" )
    analysis_parser.add_argument("--result-format", default="hdf5", type=str,
        help="Format to save the result" )
    return analysis_parser



def parse_analysis_args(parser, cli_args=[""]):
    """Parse the command line arguments for nmma_analysis and nmma_gw_analysis"""
    args = parser.parse_args(args=cli_args)

    if args.walks > args.maxmcmc:
        raise ValueError(
            f"You have maxmcmc ({args.maxmcmc}) > walks ({args.walks}, minimum mcmc)"
        )
    if args.nact < 1:
        raise ValueError(f"Your nact ({args.nact}) < 1 (must be >= 1)")

    return args
