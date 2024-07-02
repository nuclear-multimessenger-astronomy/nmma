"""
Module to generate/prepare data, likelihood, and priors for parallel runs.

This will create a directory structure for your parallel runs to store the
output files, logs and plots. It will also generate a `data_dump` that stores
information on the run settings and data to be analysed.
"""
import os
import pickle
import subprocess
from argparse import Namespace
import json

import bilby
import bilby_pipe
import parallel_bilby
import bilby_pipe.data_generation
import dynesty
import lalsimulation
import numpy as np

from parallel_bilby import slurm
from parallel_bilby.utils import get_cli_args

from .parser import (
    create_nmma_generation_parser,
    parse_generation_args,
)
from .parser.generation_parser import remove_argument_from_parser
from ..em.io import loadEvent
from ..em.injection import create_light_curve_data

from .. import __version__


def find_sh_scripts(file_path):
    # Open the file
    with open(file_path, "r") as f:
        # Read the content
        content = f.read()

    # Split the content into words using any whitespace as a separator
    words = content.split()

    # Filter out the words that end with '.sh'
    sh_scripts = [word for word in words if word.endswith(".sh)")]

    return sh_scripts


def replace_pbilby_in_file(file_path, name):
    # Read the contents of the file
    with open(file_path, "r") as file:
        file_contents = file.read()

    # Replace the text
    new_contents = file_contents.replace("parallel_bilby", name)

    # Write the updated contents back to the file
    with open(file_path, "w+") as file:
        file.write(new_contents)

    return


def get_version_info():
    return dict(
        bilby_version=bilby.__version__,
        bilby_pipe_version=bilby_pipe.__version__,
        parallel_bilby_version=parallel_bilby.__version__,
        dynesty_version=dynesty.__version__,
        lalsimulation_version=lalsimulation.__version__,
        nmma_version=__version__,
    )

def remove_expandable_args(parser, category, inp):
    # for cat in msg_list:
    #     if messenger not in inputs.messengers:
    #         #  identify argument_group corresponding to non-sampled msg.
    #         for ag in parser._action_groups:
    #             if f'with_{messenger}' in [act.dest for act in ag._group_actions]:
    #                 parser._action_groups.remove(ag)
    for ag in parser._action_groups:
        if 

def write_complete_config_file(parser, args, inputs):
    """Wrapper function that uses bilby_pipe's complete config writer.

    Note: currently this function does not verify that the written complete config is
    identical to the source config

    :param parser: The argparse.ArgumentParser to parse user input
    :param args: The parsed user input in a Namespace object
    :param inputs: The bilby_pipe.input.Input object storing user args
    :return: None
    """
    inputs.request_cpus = 1
    inputs.sampler_kwargs = "{}"
    inputs.mpi_timing_interval = 0
    inputs.log_directory = None

    ##eliminate args from config we do not use
    #   iterate through possible messengers
    msg_list =['em','eos', 'grb','gw']
    ana_mod_list = ['Hubble', 'tabulated_eos']



    for analysis_modifier in ana_mod_list:
        if analysis_modifier in inputs.analysis_modifiers: 
            for ag in parser._action_groups:
                if f'with_{messenger}' in [act.dest for act in ag._group_actions]:
                    parser._action_groups.remove(ag)

    try:
        bilby_pipe.main.write_complete_config_file(parser, args, inputs)
    except AttributeError:
        # bilby_pipe expects the ini to have "online_pe" and some other non pBilby args
        pass


def create_generation_logger(outdir, label):
    logger = bilby.core.utils.logger
    bilby.core.utils.setup_logger(
        outdir=os.path.join(outdir, "log_data_generation"), label=label
    )
    bilby_pipe.data_generation.logger = logger
    return logger

def read_constraint_from_args(args, constraint_kind):
    ##preferred: Have the dict with the subconstraints already set up
    prep_dict= getattr(args, constraint_kind, None) 
    if prep_dict is not None:
        return bilby_pipe.utils.convert_string_to_dict(prep_dict)
    
    ###otherwise try to construct it:
        ### read in provided attributes 
    prel_dict= {key.removeprefix(constraint_kind+'_'): ##cut identifier
                bilby_pipe.utils.convert_string_to_list(getattr(args, key)) ## read in list or float
                for key in dir(args)            ## search args for attrs 
                if key.startswith(constraint_kind+'_')} ## related with kind
    new_constraints = prel_dict.pop('name', None)
    if new_constraints is not None: ## there needs to be a unique label
        ext_dict={} 
        ### iterate through constrs.
        for i, name in enumerate(new_constraints): 
            ext_dict[name] ={k:v[i] for k,v in prel_dict.items()} 
        return ext_dict

def compose_eos_constraints(args, constraint_kinds=['lower_mtov', 'upper_mtov', 'mass_radius']):
    if args.eos_constraint_dict:
        try:
            with open(args.eos_constraint_dict, 'r') as f:
                constraint_dict = json.load(f) 
        except:
            constraint_dict = {}

        for constraint_kind in constraint_kinds:
            new_dict= read_constraint_from_args(args,constraint_kind)
            try:
                constraint_dict[constraint_kind].update(new_dict)
            except KeyError:
                constraint_dict[constraint_kind] = new_dict
            except AttributeError:
                constraint_dict[constraint_kind] = new_dict

    with open(args.eos_constraint_dict, "w") as f:
        json.dump(constraint_dict, f, indent=4)
    return constraint_dict

class NMMADataGenerationInput(bilby_pipe.data_generation.DataGenerationInput):
    ###FIXME get rid of compulsory GW structure
    def __init__(self, args, unknown_args):
        super().__init__(args, unknown_args)
        self.args = args
        self.sampler = "dynesty"
        self.sampling_seed = args.sampling_seed
        self.data_dump_file = f"{self.data_directory}/{self.label}_data_dump.pickle"

        ### identify messengers and modifiers, to be extended
        messengers=[]
        if args.with_eos_parameters:
            messengers.append("eos")
        if args.with_em:
            messengers.append('em')
        if args.with_grb:
            messengers.append("grb")
        if args.with_gw:
            #### resetting likelihood type is an unpleasant bilby_pipe remnant
            self.gw_likelihood_type = self.likelihood_type
            messengers.append("gw")
        self.messengers = messengers

        analysis_modifiers= []
        if args.with_Hubble:
            analysis_modifiers.append("Hubble")
        if args.with_tabulated_eos:
            analysis_modifiers.append('tabulated_eos')
        self.analysis_modifiers= analysis_modifiers

        self.setup_inputs()

    @property
    def sampling_seed(self):
        return self._sampling_seed

    @sampling_seed.setter
    def sampling_seed(self, sampling_seed):
        if sampling_seed is None:
            sampling_seed = np.random.randint(1, 1e6)
        self._sampling_seed = sampling_seed
        np.random.seed(sampling_seed)

    def save_data_dump(self):
        data_dump= dict(
                prior_file=self.prior_file,
                args=self.args,
                messengers = self.messengers,
                analysis_modifiers= self.analysis_modifiers,
                data_dump_file=self.data_dump_file,
                meta_data=self.meta_data,
                injection_parameters=self.injection_parameters,
        )
        if "em" in self.messengers:
            if self.injection_parameters:
                light_curve_data = create_light_curve_data(
                    self.injection_parameters, self.args
                )
            else:
                light_curve_data = loadEvent(self.args.light_curve_data)

            data_dump |=  dict(
                light_curve_data=light_curve_data
            )

        if "gw" in self.messengers:
            data_dump |= dict(
                waveform_generator=self.waveform_generator,
                ifo_list=self.interferometers,
            )
        if "eos" in self.messengers:
            data_dump |= dict(
                eos_constraint_dict=self.eos_constraint_dict
            )
        with open(self.data_dump_file, "wb+") as file:
            pickle.dump(data_dump, file)

    def setup_inputs(self):
        if "gw" in self.messengers:
            if self.gw_likelihood_type == "ROQGravitationalWaveTransient":
                self.save_roq_weights()
            self.interferometers.plot_data(outdir=self.data_directory, label=self.label)

        # This is done before instantiating the likelihood so that it is the full prior
        self.priors.to_json(outdir=self.data_directory, label=self.label)
        self.prior_file = f"{self.data_directory}/{self.label}_prior.json"

        if "eos" in self.messengers:
            self.eos_constraint_dict = compose_eos_constraints(self.args)

        # We build the likelihood here to ensure the distance marginalization exist
        # before sampling
        self.likelihood

        self.meta_data.update(
            dict(
                config_file=self.ini,
                data_dump_file=self.data_dump_file,
                **get_version_info(),
            )
        )

        self.save_data_dump()


def generate_runner(parser=None, **kwargs):
    """
    API for running the generation from Python instead of the command line.
    It takes all the same options as the CLI, specified as keyword arguments,
    and combines them with the defaults in the parser.

    Parameters
    ----------
    parser: generation-parser
    **kwargs:
        Any keyword arguments that can be specified via the CLI

    Returns
    -------
    inputs: NMMADataGenerationInput
    logger: bilby.core.utils.logger

    """

    # Get default arguments from the parser
    default_args = parse_generation_args(parser)

    # Take the union of default_args and any input arguments,
    # and turn it into a Namespace
    args = Namespace(**{**default_args, **kwargs})

    logger = create_generation_logger(outdir=args.outdir, label=args.label)
    for package, version in get_version_info().items():
        logger.info(f"{package} version: {version}")

    inputs = NMMADataGenerationInput(args, [])

    logger.info(
        "Setting up likelihood with marginalizations: "
        f"distance={inputs.distance_marginalization}, "
        f"time={inputs.time_marginalization}, "
        f"phase={inputs.phase_marginalization}."
    )
    logger.info(f"Setting sampling-seed={inputs.sampling_seed}")
    logger.info(f"prior-file save at {inputs.prior_file}")
    logger.info(
        f"Initial meta_data ="
        f"{bilby_pipe.utils.pretty_print_dictionary(inputs.meta_data)}"
    )

    write_complete_config_file(parser=parser, args=args, inputs=inputs)
    logger.info(f"Complete ini written: {inputs.complete_ini_file}")

    return inputs, logger


def nmma_generation():
    """
    nmma_generation entrypoint.

    This function is a wrapper around generate_runner(),
    giving it a command line interface.
    """

    # Parse command line arguments
    cli_args = get_cli_args()
    generation_parser = create_nmma_generation_parser()
    args = parse_generation_args(generation_parser, cli_args, as_namespace=True)

    # Initialise run
    inputs, logger = generate_runner(parser=generation_parser, **vars(args)
    )

    # Write slurm script
    bash_file = slurm.setup_submit(inputs.data_dump_file, inputs, args, cli_args)
    # change the parallel_bilby_analysis to nmma_analysis
    sh_scripts = find_sh_scripts(bash_file)
    for sh_script in sh_scripts:
        replace_pbilby_in_file(sh_script.replace(")", ""), "nmma")
    if args.submit:
        subprocess.run([f"bash {bash_file}"], shell=True)
    else:
        logger.info(f"Setup complete, now run:\n $ bash {bash_file}")

