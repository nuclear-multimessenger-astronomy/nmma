import argparse
import configargparse
import sys
import os

from bilby.core.utils import setup_logger
from ..utils.models import refresh_models_list

# If an argument is not specified while its type is given, 
# it will still be parsed as None. If, however, we then reparse
# this argument, e.g. from a stored config-file, it would raise an error.
# These classes are a workaround to avoid this error.
from bilby_pipe.utils import nonestr, nonefloat, noneint  # we forward import these elsewhere, do not remove

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
    parser = argparse.ArgumentParser() # Default

    # Determine if a config file is given and set up the parser accordingly
    if cli_args is None:
        cli_args = sys.argv[1:]
    elif isinstance(cli_args, str):
        cli_args = [cli_args]
    config_given = False
    
    if cli_args:
        first_arg = cli_args[0]
        if first_arg.startswith("-c") or first_arg.startswith("--config"):
            cli_args.pop(0)  
            config_given = True
        if os.path.isfile(first_arg):
            if first_arg.endswith((".yaml", ".yml")):
                pc = configargparse.YAMLConfigFileParser
            elif first_arg.endswith((".ini", ".toml", ".cfg", '.tml')):
                pc = configargparse.DefaultConfigFileParser
            try:
                parser = configargparse.ArgumentParser(
                    default_config_files=[first_arg],
                    config_file_parser_class=pc )
                cli_args.pop(0)  # Remove the config file argument
            except Exception:
                print(f"Tried to parse {first_arg} as a config file, but failed.")
                if config_given:
                    print("Please check the file format and try again.")
                    sys.exit(1)

    if isinstance(parsing_func, (list, tuple)):
        for pars_func in tuple(parsing_func):
            parser = pars_func(parser)
    else:
        parser = parsing_func(parser)
        
    if return_parser:
        return parser
    return parser.parse_args(cli_args)


def base_analysis_parsing(parser):
    parser.add_argument("--bilby-zero-likelihood-mode",
        action='store_true', help="enable prior run")

    parser.add_argument("--Hubble", "--with-Hubble", "--sample-over-Hubble", action='store_true', 
        help="To sample over Hubble constant and redshift (default:False)")

    parser.add_argument("--Hubble-weight", help="Path to the precalculated Hubble weighting")

    parser.add_argument("--sampling-seed","--seed", type=int, default=42,
        help="Sampling seed (default: 42)")
    return parser

def single_messenger_analysis_parsing(parser):
    parser = base_analysis_parsing(parser)

    parser.add_argument("--config", help="Name of the configuration file containing parameter values.")
    parser.add_argument("-o", "--outdir", default="outdir",  help="Path to the output directory")
    parser.add_argument("--label", default ="em_transient", help="Label for the run")
    parser.add_argument("--plot", action='store_true', help="create characteristic plot")
    parser.add_argument("--verbose", action='store_true', help="print out log likelihoods" )
    parser.add_argument("--prior-file","--prior", help="Path to the prior file")
    parser.add_argument("--sampler", default="pymultinest",
        help="Sampler to be used (default: pymultinest)")
    parser.add_argument("--sampler-kwargs", default="{}", 
        help="Additional kwargs (e.g. {'evidence_tolerance':0.5}) for bilby.run_sampler, put a double quotation marks around the dictionary")
    parser.add_argument("--soft-init", action='store_true',
        help="To start the sampler softly (without any checking, default: False)")
    parser.add_argument("--cpus", type=int, default=1,
        help="Number of cores to be used, only needed for dynesty (default: 1)")
    parser.add_argument("-n","--nlive", "--n-live", type=int, default=2048, help="Number of live points (default: 2048)")
    parser.add_argument("--reactive-sampling", action='store_true',
        help="To use reactive sampling in ultranest (default: False)")
    parser.add_argument("--skip-sampling", action='store_true', 
        help="If analysis has already run, skip bilby sampling and compute results from checkpoint files. Combine with --plot to make plots from these files.")

    return parser

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
    parser.add_argument( "-t", "--trigger-time", type=int, default=0,
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
    parser.add_argument("--duration", type=float, default=4, help=("The segment "
            "duration (default=4s), used only in conjunction with --gps-file" ), )
    parser.add_argument( "--post-trigger-duration", type=float, default=2,
        help=("The post trigger duration (default=2s), used only in conjunction "
            "with --gps-file" ))
    return parser



############# UTILS #############
def process_multi_condition_string(multi_condition_string):
    # Supported operators 
    operators = [">=", "<=", "==", ">", "<", "="]

    out = {}

    for item in multi_condition_string:
        matched = False
        for op in operators: # then check for num relation
            if op in item:
                parts = item.split(op, 1)
                out[parts[0].strip()] = (op, parts[1].strip())
                matched = True
                break
        if not matched:
            # Treat as boolean flag
            out[item] = True 

    return out
