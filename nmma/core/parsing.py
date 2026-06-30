import argparse
import configargparse
import yaml
import sys
import os
import operator

from bilby.core.utils import setup_logger
from .gitlab import refresh_models_list

# If an argument is not specified while its type is given, 
# it will still be parsed as None. If, however, we then reparse
# this argument, e.g. from a stored config-file, it would raise an error.
# These classes are a workaround to avoid this error.
from bilby_pipe.utils import nonestr, nonefloat, noneint  # we forward import these elsewhere, do not remove

def yaml_parse(s):
    return yaml.safe_load(s)

def parsing_and_logging(parser_func, args= None):

    if not isinstance(args, argparse.Namespace):
        args = nmma_base_parsing(parser_func, args)

    if getattr(args, 'sampler', None) == "pymultinest":
        if len(args.outdir) > 64:
            raise ValueError("output directory name is longer than 64 characters" )

    if getattr(args, 'refresh_model_list', False):
        refresh_models_list(args.svd_path)

    try:
        setup_logger(outdir=args.outdir, label=args.label)
        os.makedirs(args.outdir, exist_ok=True)
        print('Setting up logger and storage directory')
    except Exception as e:
        pass
    return args

def nmma_base_parsing(parsing_func, cli_args=None, return_parser=False):
    """Base parsing function for nmma.
    Takes a parsing function as input and returns the corresponding namespace, 
    potentially inferred from a config file."""

    # Determine if a config file is given and set up the parser accordingly
    if cli_args is None:
        cli_args = sys.argv[1:]
    elif isinstance(cli_args, str):
        cli_args = cli_args.split()
    else:
        cli_args = list(cli_args)
    
    parser, cli_args = check_for_config(cli_args)
       

    parser.add_argument("--multi", type=yaml_parse,
        help="YAML formatted dict specifying multiple runs with different parameter changes. \n"
             "E.g.: '{run1: {param1: value1, param2: value2}, run2: {param1: value3}}' ")
    parser.add_argument("--matrix", type=yaml_parse,
        help="YAML formatted dict specifying multiple runs with cross-wise parameter changes. \n"
             "E.g.: '{run1: {param1: value1, param2: value2}, run2: {param1: value3}}' ")
    
    if isinstance(parsing_func, (list, tuple)):
        for pars_func in tuple(parsing_func):
            parser = pars_func(parser)
    else:
        parser = parsing_func(parser)
    if return_parser:
        return parser
    return parser.parse_args(cli_args)

def check_for_config(cli_args, parents=[], drop_config =True): 
    if cli_args:
        first_arg = cli_args[0]
        if first_arg.startswith("-c") or first_arg.startswith("--config") or first_arg.startswith("--ini"):
            cli_args.pop(0)  
            first_arg = cli_args[0]
            config_given = True
        else:
            config_given = False

        if os.path.isfile(first_arg):
            if first_arg.endswith((".yaml", ".yml")):
                pc = configargparse.YAMLConfigFileParser
            elif first_arg.endswith((".ini", ".toml", ".cfg", '.tml')):
                pc = configargparse.DefaultConfigFileParser
            try:
                parser = configargparse.ArgumentParser(
                    add_help=False,
                    default_config_files=[first_arg],
                    parents=parents,
                    config_file_parser_class=pc )
                if drop_config:
                    cli_args.pop(0)  # remove config file from args
                return parser, cli_args
            except Exception:
                if config_given:
                    print(f"Tried to parse {first_arg} as a config file, but failed.")
                    print("Please check the file format and try again.")
                    sys.exit(1)
    if parents:
        return configargparse.ArgumentParser(add_help=False, parents=parents), cli_args
    return argparse.ArgumentParser( ), cli_args
    
def base_analysis_parsing(parser):
    parser.add_argument("--Hubble", "--with-Hubble", "--sample-over-Hubble", action='store_true', 
        help="To sample over Hubble constant and redshift (default:False)")

    parser.add_argument("--Hubble-weight", help="Path to the precalculated Hubble weighting")

    parser.add_argument("--cosmology", help="Name of the cosmology to be used, see astropy.cosmology for available cosmologies (implicit default: Planck18)")
    parser.add_argument("--sampling-seed","--seed", type=int, default=42,
        help="Sampling seed (default: 42)")
    parser.add_argument("--sampler-kwargs", default="{}", type = yaml_parse,
        help="Additional keyword arguments to pass to the sampler as a dictionary" )
<<<<<<< HEAD
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


def single_messenger_analysis_parsing(parser):
    parser = base_analysis_parsing(parser)
    parser = dynesty_parsing(parser)

    parser.add_argument("--config", help="Name of the configuration file containing parameter values.")
    parser.add_argument("-o", "--outdir", default="outdir",  help="Path to the output directory")
    parser.add_argument("--label", default ="nmma_transient", help="Label for the run")
    parser.add_argument("--plot", action='store_true', help="create characteristic plot")
    parser.add_argument("--verbose", action='store_true', help="print out log likelihoods" )
    parser.add_argument("--prior-file","--prior", help="Path to the prior file")
    parser.add_argument("--sampler", default="pymultinest",
        help="Sampler to be used (default: pymultinest)")
=======
    parser.add_argument("--skip-sampling", action='store_true', 
        help="If analysis has already run, skip bilby sampling and compute results from checkpoint files. Combine with --plot to make plots from these files.")
>>>>>>> dev
    parser.add_argument("--dlogz", default=0.1, type=float,
        help="Stopping criteria: remaining evidence, (default=0.1)" )
    parser.add_argument("--soft-init", action='store_true',
        help="To start the sampler softly (without any checking, default: False)")
    parser.add_argument("--cpus", type=int, default=1,
        help="Number of cores to be used, only needed for dynesty (default: 1)")
    parser.add_argument("-n","--nlive", "--n-live", type=int, default=2048, help="Number of live points (default: 2048)")
<<<<<<< HEAD
    parser.add_argument("--reactive-sampling", action='store_true',
        help="To use reactive sampling in ultranest (default: False)")
    parser.add_argument("--skip-sampling", action='store_true', 
        help="If analysis has already run, skip bilby sampling and compute results from checkpoint files. Combine with --plot to make plots from these files.")
    parser.add_argument("--bestfit", "--best-fit", action='store_true',
        help="Save the best fit parameters to JSON")
    
    return parser

=======
    parser.add_argument("--check-point-delta-t","--check-point-deltaT", default=1800, type=float,
        help="Write a checkpoint resume file and diagnostic plots every deltaT [s]. Default: 0.5 hour.")
    parser.add_argument("--checkpoint-plot", action='store_true',
        help="Whether to generate analytical check-point plots")
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


def single_messenger_analysis_parsing(parser):
    parser = base_analysis_parsing(parser)
    parser = dynesty_parsing(parser)

    parser.add_argument("--config", help="Name of the configuration file containing parameter values.")
    parser.add_argument("-o", "--outdir", default="outdir",  help="Path to the output directory")
    parser.add_argument("--label", default ="nmma_transient", help="Label for the run")
    parser.add_argument("--plot", action='store_true', help="create characteristic plot")
    parser.add_argument("--plot-kwargs", type=yaml_parse, default={}, help="Additional keyword arguments to pass to the plotting routine as a dictionary" )
    parser.add_argument("--verbose", action='store_true', help="print out log likelihoods" )
    parser.add_argument("--prior-file","--prior", help="Path to the prior file")
    parser.add_argument("--sampler", default="pymultinest",
        help="Sampler to be used (default: pymultinest)")
    parser.add_argument("--reactive-sampling", action='store_true',
        help="To use reactive sampling in ultranest (default: False)")
    parser.add_argument("--bestfit", "--best-fit", action='store_true',
        help="Save the best fit parameters to JSON")
    parser.add_argument("--result-format", default="json",
        help="Format to save the result" )
    
    return parser

>>>>>>> dev
    
def base_injection_parsing(parser):
    parser.add_argument("-f", "--injection-file","--filename", default="injection", 
        help="Path to the file with injection parameters, default: 'injection'.")
    parser.add_argument("-e", "--extension","--outfile-type", default="json",
        choices=["json", "dat", "csv"], help="Injection file format", )
    parser.add_argument("--generation-seed", type=int, default=42, 
        help="Injection generation seed (default: 42)")
    return parser

def pipe_inj_parsing(parser):
    ### bilby-pipe injectionCreator parameters
    parser.add_argument( "--prior-file", type=nonestr, 
        help="The prior file from which to generate injections. Alternatively, a prior-dict must be given.")
    parser.add_argument("--prior-dict", type=nonestr, 
        help=("A prior dict to use for generating injections. If not given, the prior file is used. "))
    parser.add_argument("-n", "--n-injection", type=int, default=20,
        help=("The number of injections to generate: not required" 
              "if --gps-file or injection file is also given"), )
        
    # bilby-pipe Optional Time parameters
    parser.add_argument( "-t", "--trigger-time", type=float, default=0.,
        help=("The trigger time to use for setting a geocent_time prior "
            "(default=0). Ignored if a geocent_time prior exists in the "
            "prior_file or --gps-file is given." ), )
    parser.add_argument( "--deltaT", type=float, default=0.2,
        help=( "The symmetric width (in s) around the trigger time to"
            " search over the coalesence time. Ignored if a geocent_time prior"
            " exists in the prior_file" ), )
    

    parser.add_argument( "-g", "--gps-file", type=nonestr, 
        help=( "A list of gps start times to use for setting a geocent_time"
            "prior. Note, the trigger time is obtained from "
            " start_time + duration - post_trigger_duration." ))
    parser.add_argument("--duration", type=float, default=4., help=("The segment "
            "duration (default=4s), used only in conjunction with --gps-file" ), )
    parser.add_argument( "--post-trigger-duration", type=float, default=2.,
        help=("The post trigger duration (default=2s), used only in conjunction "
            "with --gps-file" ))
    return parser

def slurm_setup_parser(parser):
    parser.description="Create files from nmma injection file"
    parser = pipe_inj_parsing(parser)
    
    parser.add_argument("--injection-file","--injection",
        required=True, help="The bilby injection json file to be used")
    parser.add_argument("-o", "--outdir", default="outdir", 
        help="Path to the output directory")
    parser.add_argument("--analysis-file", required=True,
        help="The analysis bash script to be replicated")
    parser.add_argument("--n-per-job","--lightcurves-per-job", type=int, default=100,
        help="Number of services per job")
    ### Dummy for intermediate setup until we decide on whether to abandon or extend these routines
    parser.add_argument("--simple-setup", default=True, choices=[True])
    return parser



def slurm_analysis_parser(parser):
    slurm_args = parser.add_argument_group(
        title="Slurm arguments",
        description="Arguments for running the lightcurve analysis on a Slurm HPC cluster",
    )
    # Slurm-specific arguments
    slurm_args.add_argument("--Ncore", default=8, type=int,
        help="number of cores for mpiexec")
    slurm_args.add_argument("--base-dir", default=os.getcwd(),
        help="base directory for the job")
    slurm_args.add_argument("--job-name")
    slurm_args.add_argument("--logs-dir-name", default="slurm_logs",
        help="directory name for slurm logs")
    slurm_args.add_argument("--cluster-name", default="Expanse",
        help="Name of HPC cluster")
    slurm_args.add_argument("--partition-type", default="shared",
        help="Partition name to request for computing")
    slurm_args.add_argument("--nodes", type=int, default=1,
        help="Number of nodes to request for computing")
    slurm_args.add_argument("--gpus", type=int, default=0,
        help="Number of GPUs to request")
    slurm_args.add_argument("--memory-GB", type=int, default=64,
        help="Memory allocation to request for computing")
    slurm_args.add_argument("--time", default="24:00:00",
        help="Walltime for instance")
    slurm_args.add_argument("--mail-type", default="NONE",
        help="slurm mail type (e.g. NONE, FAIL, ALL)")
    slurm_args.add_argument("--mail-user", help="contact email address")
    slurm_args.add_argument("--python-env-name", default="nmma_env",
        help="Name of python environment to activate")
    slurm_args.add_argument("--script-name", default="slurm.sub")

    return parser


<<<<<<< HEAD
def process_sampler_kwargs(sampler_kwargs, kwargs):
=======
def process_sampler_kwargs(args):
>>>>>>> dev
    # Set defaults here to avoid inconsistent values
    default_kwargs = dict(dlogz=0.1, save_bounds=False,
        min_eff=10, sample="acceptance-walk", nlive=1000, bound="live",
        walks=100, facc=0.5, enlarge=1.5)
    
<<<<<<< HEAD
    run_sampler_kwargs = {key: kwargs.get(key, default_kwargs[key]) 
        for key in ['dlogz', 'save_bounds']}
    

    def_init_kwargs = {key: kwargs.get(key, default_kwargs[key]) 
        for key in ['min_eff','sample', 'nlive', 'bound', 'walks', 'facc', 'enlarge']}
    
    sampler_init_kwargs = def_init_kwargs | sampler_kwargs
    sampler_init_kwargs['first_update'] = dict(min_eff=sampler_init_kwargs.pop('min_eff'), 
                        min_ncall= 2 * sampler_init_kwargs['nlive'])
=======
    run_sampler_kwargs = {key: getattr(args, key, default_kwargs[key]) 
                          for key in ['dlogz', 'save_bounds']}
    

    def_init_kwargs = {key: getattr(args, key, default_kwargs[key]) 
        for key in ['min_eff','sample', 'nlive', 'bound', 'walks', 'facc', 'enlarge']}
    
    sampler_init_kwargs = def_init_kwargs | args.sampler_kwargs 
    sampler_init_kwargs['first_update'] = dict(min_eff=sampler_init_kwargs.pop('min_eff'), 
                        min_ncall= 2 * args.nlive)
>>>>>>> dev
        
    return sampler_init_kwargs, run_sampler_kwargs



############# UTILS #############
def process_multi_condition_string(multi_condition_string):
    # Supported operators 
    operators = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
}   

    out = {}
    if isinstance(multi_condition_string, str):
        multi_condition_string = multi_condition_string.split(",")

    for item in multi_condition_string:
        matched = False
        for op in operators.keys(): # then check for num relation
            if op in item:
                parts = item.split(op)
                out[parts[0].strip()] = (operators[op], float(parts[1].strip()) )
                matched = True
                break
        
        if not matched:
            if "=" in item:
                parts = item.split("=")
                out[parts[0].strip()] = parts[1].strip() 
            else:
                # Treat as boolean flag
                out[item] = True 

    return out