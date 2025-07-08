import argparse
import configargparse
import sys
import os


# If an argument is not specified while its type is given, 
# it will still be parsed as None. If, however, we then reparse
# this argument, e.g. from a stored config-file, it would raise an error.
# These classes are a workaround to avoid this error.
from bilby_pipe.utils import nonestr, nonefloat, noneint  # we forward import these elsewhere, do not remove

def nmma_base_parsing(parsing_func, cli_args=None):
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

    parser = parsing_func(parser)
    args = parser.parse_args(cli_args)
    return args



class StoreBoolean(argparse.Action):
    """argparse class for robust handling of booleans with configargparse

    When using configargparse, if the argument is setup with
    action="store_true", but the default is set to True, then there is no way,
    in the config file to switch the parameter off. To resolve this, this class
    handles the boolean properly.

    """

    def __call__(self, parser, namespace, value, option_string=None):
        value = str(value).lower()
        if value in ["true"]:
            setattr(namespace, self.dest, True)
        else:
            setattr(namespace, self.dest, False)


def Hubble_parsing(parser):
    H0_input_parser = parser.add_argument_group(
        title="Hubble input arguments", description="Specify Hubble inputs"
    )
    H0_input_parser.add(
        "--with-Hubble",
        action=StoreBoolean,
        default=False,
        help="Flag for sampling over Hubble constants (default:False)",
    )
    H0_input_parser.add(
        "--Hubble-weight", type=str, help="Path to the precalculated Hubble weighting"
    )

    return parser

def base_analysis_parsing(parser):
    parser.add_argument("--bilby-zero-likelihood-mode",
        action=StoreBoolean, default=False, help="enable prior run")
    
    parser.add_argument("--sampling-seed","--seed", type=int, default=42, 
        help="Sampling seed (default: 42)")
    return parser


