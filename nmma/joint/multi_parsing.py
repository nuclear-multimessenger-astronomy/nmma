import argparse
from ast import literal_eval
import bilby
from bilby_pipe import parser as bp_parser

from nmma.core.parsing import base_analysis_parsing, check_for_config, nonestr
from nmma.joint.joint_parsing import joint_likelihood_parsing
from nmma.em.em_parsing import em_analysis_parsing
from nmma.eos.eos_parsing import eos_parsing, tabulated_eos_parsing
from nmma.gw.gw_parsing import gw_parsing


from .. import __version__  # noqa: E402
from numpy import inf
logger = bilby.core.utils.logger

def _create_base_nmma_parser(sampler="dynesty", parents=[]):
    base_parser = argparse.ArgumentParser("base", parents=parents, 
                        conflict_handler="resolve", add_help=False)

    base_parser.add_argument("--version", action="version",
        version=f"%(prog)s={__version__}\nbilby={bilby.__version__}",
    )

    if sampler in ["all", "dynesty"]:
        base_parser = sampler_parsing(base_parser)
        base_parser = dynesty_parsing(base_parser)

    base_parser = em_settings_parsing(base_parser)
    base_parser = base_analysis_parsing(base_parser)
    base_parser = em_analysis_parsing(base_parser)
    base_parser = gw_parsing(base_parser)
    base_parser = joint_likelihood_parsing(base_parser)
    base_parser = tabulated_eos_parsing(base_parser)
    base_parser = eos_parsing(base_parser)
    base_parser = add_misc_settings(base_parser)

    return base_parser

def em_settings_parsing(parser):
    # general args
    em_input_parser = parser.add_argument_group(
        title="EM analysis input arguments", description="Specify EM analysis inputs"
    )
    em_input_parser.add("--light-curve-data", help="Path to the observed light curve data")
    return parser


def sampler_parsing(parser):
    sampler_group = parser.add_argument_group(title = "Setting for the Sampler")

    sampler_group.add_argument("--init-sampler-kwargs", default="{}", 
        help="Additional keyword arguments to pass to the sampler as a dictionary" )
    sampler_group.add_argument("--sampler", choices=["dynesty"], default="dynesty",
        help="The parallelised sampler to use, defaults to dynesty")
    sampler_group.add_argument( "-n", "--nlive", default=1000, type=int, help="Number of live points" )
    sampler_group.add_argument("--dlogz", default=0.1, type=float,
        help="Stopping criteria: remaining evidence, (default=0.1)" )
    sampler_group.add_argument("--n-effective", default=inf, type=float,
        help="Stopping criteria: effective number of samples, (default=inf)" )
    sampler_group.add_argument("--bound","--dynesty-bound", default="live", 
        help="Dynesty bounding method (default=live)" )
    sampler_group.add_argument( "--sample", "--dynesty-sample", default="acceptance-walk",
        help="sampling method (default=acceptance-walk).")
    return parser

def dynesty_parsing(parser):
    dynesty_group = parser.add_argument_group(title="Dynesty Settings")
    dynesty_group.add_argument("--n-check-point", default=1000, type=int,
        help="Steps to take before attempting checkpoint")
    dynesty_group.add_argument("--max-its", default=10**10, type=int,
        help="Maximum number of iterations to sample for (default=1.e10)")
    dynesty_group.add_argument("--max-run-time", default=1.0e10, type=float,
        help="Maximum time to run for (default=1.e10 s)")
    dynesty_group.add_argument("--rejection-sample-posterior", action='store_false', help=(
            "Whether to generate the posterior samples by rejection sampling the "
            "nested samples or resampling with replacement" ) )
    dynesty_group.add_argument( "--walks", default=100, type=int,
        help="Minimum number of walks, defaults to 100" )
    dynesty_group.add_argument( "--proposals",  action="append", 
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
    return parser


def add_misc_settings(parser):
    misc_group = parser.add_argument_group(title="Misc. Settings")
    misc_group.add_argument("-c", "--clean", action='store_true', help="Run clean: ignore any resume files")
    misc_group.add_argument("--checkpoint-plot", action='store_true',
        help="Whether to generate analytical check-point plots")
    misc_group.add_argument("--save-bounds", action='store_true',
        help="Whether to store bounds in the resume file. Not doing this can make resume files large (~GB)")
    misc_group.add_argument("--check-point-delta-t","--check-point-deltaT", default=3600, type=float,
        help="Write a checkpoint resume file and diagnostic plots every deltaT [s]. Default: 1 hour.")
    misc_group.add_argument("--plot", action='store_true',
        help="Whether to generate the various data plots at the end of the run")
    return parser


def run_parsing(parser):
    run_group = parser.add_argument_group(title = "Setting for the Main run")
    run_group.add_argument("data_dump", nargs="?") # nargs makes it optional
    run_group.add_argument("--data-dump", dest ="data_dump",
        help="The pickled data dump generated by nmma_generation" )
    run_group.add_argument("--outdir", help="Outdir to overwrite input label" )
    run_group.add_argument("--label", help="Label to overwrite input label" )
    run_group.add_argument("--result-format", default="hdf5",
        help="Format to save the result" )

    return parser

def slurm_parsing(parser):
    slurm_group = parser.add_argument_group(title="Slurm Settings")
    slurm_group.add_argument("--nodes", type=int, default=1,
        help="Number of nodes to use (default 1)")
    slurm_group.add_argument("--ntasks-per-node", type=int, default=2,
        help="Number of tasks per node (default 2)")
    slurm_group.add_argument("--time", default="24:00:00",
        help="Maximum wall time (defaults to 24:00:00)")
    slurm_group.add_argument("--mem-per-cpu", default="2G",
        help="Memory per CPU (defaults to 2GB)")
    slurm_group.add_argument("--extra-lines", type=nonestr,
        help="Additional lines, separated by ';', use for setting up conda env or module imports",
    )
    slurm_group.add_argument("--slurm-extra-lines", type=nonestr,
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
    """Parser for nmma_generation"""
    bilby_pipe_parser = _create_reduced_bilby_pipe_parser()
    generation_parser = _create_base_nmma_parser(sampler="all", parents=[bilby_pipe_parser])
    # generation_parser = argparse.ArgumentParser(
    #     prog="nmma_generation",
    #     usage=__doc__,
    #     ignore_unknown_config_file_keys=False,
    #     allow_abbrev=False,
    #     parents=[parser, bilby_pipe_parser],
    #     add_help=False,
    # )
    generation_parser = slurm_parsing(generation_parser)
    return generation_parser


def parse_generation_args(cli_args=[""]):
    """
    Returns dictionary of arguments, as specified in the
    parser.

    If no cli_args arguments are specified, returns the default arguments
    (by running the parser with no ini file and no CLI arguments)

    Parameters
    ----------
    cli_args: list of strings (default: [""])
        List of arguments to be parsed. If empty, returns default arguments

    Returns
    -------
    args: dict or Namespace

    """
    generation_parser = create_nmma_generation_parser()
    generation_parser, cli_args = check_for_config(cli_args, [generation_parser], False)
    args = generation_parser.parse_args(args=cli_args)
    return args, generation_parser

def process_sampler_kwargs(init_sampler_kwargs, run_sampler_kwargs, kwargs):
    # Set defaults here to avoid inconsistent values between main.py and MainRun
    default_kwargs = dict(dlogz=0.1, save_bounds=False,
        min_eff=10, sample="acceptance-walk", nlive=1000, bound="live",
        walks=100, facc=0.5, enlarge=1.5)
    
    def_run_kwargs = {key: kwargs.get(key, default_kwargs[key]) 
        for key in ['dlogz', 'save_bounds']}
    if isinstance(run_sampler_kwargs, str):
        run_sampler_kwargs = literal_eval(run_sampler_kwargs)
    run_sampler_kwargs = def_run_kwargs | run_sampler_kwargs
    

    def_init_kwargs = {key: kwargs.get(key, default_kwargs[key]) 
        for key in ['min_eff','sample', 'nlive', 'bound', 'walks', 'facc', 'enlarge']}
    if isinstance(init_sampler_kwargs, str):
        init_sampler_kwargs = literal_eval(init_sampler_kwargs)
    init_sampler_kwargs = def_init_kwargs | init_sampler_kwargs
    init_sampler_kwargs['first_update'] = dict(min_eff=init_sampler_kwargs.pop('min_eff'), 
                        min_ncall= 2 * init_sampler_kwargs['nlive'])
        
    return init_sampler_kwargs, run_sampler_kwargs


def create_nmma_analysis_parser(sampler="dynesty"):
    """Parser for nmma_analysis"""
    parser = _create_base_nmma_parser(sampler=sampler)
    parser = run_parsing(parser)
    return parser   


def parse_analysis_args(parser, args=None):
    """Parse the command line arguments for nmma_analysis and nmma_gw_analysis"""
    args = parser.parse_args(args=args)

    if args.walks > args.maxmcmc:
        raise ValueError(
            f"You have maxmcmc ({args.maxmcmc}) > walks ({args.walks}, minimum mcmc)"
        )
    if args.nact < 1:
        raise ValueError(f"Your nact ({args.nact}) < 1 (must be >= 1)")

    return args
