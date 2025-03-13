"""
Module to generate/prepare data, likelihood, and priors for parallel runs.

This will create a directory structure for your parallel runs to store the
output files, logs and plots. It will also generate a `data_dump` that stores
information on the run settings and data to be analysed.
"""
import os
import sys
import pickle
from argparse import Namespace

import bilby
import bilby_pipe
import bilby_pipe.data_generation
import dynesty
import lalsimulation
import numpy as np


from .parser import (
    create_nmma_generation_parser,
    parse_generation_args,
)

from ..em.io import loadEvent
from ..em.model import create_injection_model
from ..em.lightcurve_generation import create_light_curve_data
from ..eos.eos_likelihood import compose_eos_constraints

from .. import __version__


def get_version_info():
    return dict(
        bilby_version=bilby.__version__,
        bilby_pipe_version=bilby_pipe.__version__,
        dynesty_version=dynesty.__version__,
        lalsimulation_version=lalsimulation.__version__,
        nmma_version=__version__,
    )

def remove_expandable_args(parser, required_arg_groups):
    # for cat in msg_list:
    #     if messenger not in inputs.messengers:
    #         #  identify argument_group corresponding to non-sampled msg.
    #         for ag in parser._action_groups:
    #             if f'with_{messenger}' in [act.dest for act in ag._group_actions]:
    #                 parser._action_groups.remove(ag)
    for ag in parser._action_groups:
        if ag.title not in required_arg_groups:
            parser._action_groups.remove(ag)
        
    return parser

def determine_required_args(analysis_categories):
    required_args=[
        'positional arguments', 'options', 'Dynesty Settings','GW input arguments', 'Misc. Settings',  'Data generation arguments', 'Injection arguments', 'Job submission arguments', 'Likelihood arguments', 'Output arguments', 'Prior arguments',  'Sampler arguments','Slurm Settings']
    if "gw" in analysis_categories:
        required_args.extend(
            ['Calibration arguments','Waveform arguments', 
             'Detector arguments', 'Post processing arguments'])
    if 'em' in analysis_categories:
        required_args.append('EM analysis input arguments')
    if 'eos' in analysis_categories:
        required_args.append('EOS analysis input arguments')
    if 'Hubble' in analysis_categories:
        required_args.append('Hubble input arguments')
    if 'tabulated_eos' in analysis_categories:
        required_args.append('Tabulated EOS input arguments')
    return required_args
    

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
    check_messengers= inputs.messengers + inputs.analysis_modifiers

    # clean parser to only use setups that we apply
    required_arg_groups= determine_required_args(check_messengers)
    parser= remove_expandable_args(parser, required_arg_groups)

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

class NMMADataGenerationInput(bilby_pipe.input.Input):
    """
    NMMADataGenerationInput class. 
    Inherits from bilby_pipe.input.Input. 
    Note that many of the args are not specified in the NMMA parsing,
    but are required by bilby_pipe and are set as the default there.
    """
    ###FIXME get rid of compulsory GW structure
    def __init__(self, args, unknown_args):
        
        super().__init__(args, unknown_args)
        # Generic setup, ripped from bilby pipe
        # Admin arguments
        self.injection_parameters=None
        self.ini = args.ini
        self.transfer_files = args.transfer_files

        # Run index arguments
        self.idx = args.idx
        self.generation_seed = args.generation_seed
        self.trigger_time = args.trigger_time

        # Naming arguments
        self.outdir = args.outdir
        self.label = args.label

        # Prior arguments
        self.reference_frame = args.reference_frame
        self.time_reference = args.time_reference
        self.phase_marginalization = args.phase_marginalization
        self.prior_file = args.prior_file
        self.prior_dict = args.prior_dict
        self.deltaT = args.deltaT
        self.default_prior = args.default_prior

        # Marginalization
        self.distance_marginalization = args.distance_marginalization
        self.distance_marginalization_lookup_table = (
            args.distance_marginalization_lookup_table
        )
        self.phase_marginalization = args.phase_marginalization
        self.time_marginalization = args.time_marginalization
        self.calibration_marginalization = args.calibration_marginalization
        self.calibration_lookup_table = args.calibration_lookup_table
        self.number_of_response_curves = args.number_of_response_curves
        self.jitter_time = args.jitter_time

        # Plotting
        self.plot_data = args.plot_data
        self.plot_spectrogram = args.plot_spectrogram
        self.plot_injection = args.plot_injection


        
        self.args = args
        self.sampler = "dynesty"
        self.sampling_seed = args.sampling_seed
        self.data_dump_file = f"{self.data_directory}/{self.label}_data_dump.pickle"


        # This is done before instantiating the likelihood so that it is the full prior
        self._priors=self._get_priors()
        self.priors.to_json(outdir=self.data_directory, label=self.label)
        self.prior_file = f"{self.data_directory}/{self.label}_prior.json"


        self.create_data(args, unknown_args)

        self.meta_data = dict(
                config_file=self.ini,
                data_dump_file=self.data_dump_file,
                **get_version_info(),
                command_line_args=args.__dict__,
                unknown_command_line_args=unknown_args,
                injection_parameters= self.injection_parameters,
        )

        ### identify messengers, to be extended
        messengers=[]
        if args.with_eos:
            messengers.append("eos")
        if args.with_em:
            messengers.append("em")
        if args.with_gw:
            messengers.append("gw")

        self.messengers = messengers

        analysis_modifiers= []
        if args.with_Hubble:
            analysis_modifiers.append("Hubble")
        if args.with_tabulated_eos:
            analysis_modifiers.append('tabulated_eos')
        self.analysis_modifiers= analysis_modifiers
        


        self.save_data_dump()

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
            data_dump |=  dict(
                light_curve_data=self.light_curve_data
            )

        if "gw" in self.messengers:
            data_dump |= dict(
                waveform_generator=self.gw_inputs.waveform_generator,
                ifo_list=self.gw_inputs.interferometers,
            )
        if "eos" in self.messengers:
            data_dump |= dict(
                eos_constraint_dict=self.eos_constraint_dict
            )
        with open(self.data_dump_file, "wb+") as file:
            pickle.dump(data_dump, file)

    def create_data(self, args, unknown_args):
        self.data_set = False
        self.injection = args.injection
        self.injection_numbers = args.injection_numbers
        self.injection_file = args.injection_file
        self.injection_dict = args.injection_dict
        if self.injection:
            self.injection_parameters = self.injection_df.iloc[self.idx].to_dict()
        else: 
            self.injection_parameters=None

        if args.with_eos:
            #FIXME add routine for eos model training!
            self.eos_constraint_dict = compose_eos_constraints(self.args)

        if args.with_em:
            if self.injection_parameters:
                injection_model = create_injection_model(args)
                self.light_curve_data = create_light_curve_data(
                    self.injection_parameters, self.args, injection_model
                )
            else:
                self.light_curve_data = loadEvent(self.args.light_curve_data)
            

        if args.with_gw:
            self.gw_inputs= bilby_pipe.data_generation.DataGenerationInput(args, unknown_args)
            #### FIXME resetting likelihood type is an unpleasant bilby_pipe remnant
            args.gw_likelihood_type = self.gw_inputs.likelihood_type
            self.gw_inputs.interferometers.plot_data(outdir=self.data_directory, label=self.label)

            # # We build the likelihood here to ensure the 
            # # distance marginalization exist before sampling
            # self.gw_inputs.likelihood

            if args.gw_likelihood_type == "ROQGravitationalWaveTransient":
                self.gw_inputs.save_roq_weights()


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
    cli_args = sys.argv[1:]
    generation_parser = create_nmma_generation_parser()
    args = parse_generation_args(generation_parser, cli_args, as_namespace=True)

    # Initialise run
    inputs, logger = generate_runner(parser=generation_parser, **vars(args)
    )
    logger.info(f"Setup complete")

