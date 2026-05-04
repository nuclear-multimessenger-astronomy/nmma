"""
Module to generate/prepare data, likelihood, and priors for parallel runs.

This will create a directory structure for your parallel runs to store the
output files, logs and plots. It will also generate a `data_dump` that stores
information on the run settings and data to be analysed.
"""
import os
import sys
import pickle

import bilby
import bilby_pipe
import bilby_pipe.data_generation
import dynesty
import lalsimulation
import numpy as np


from  .multi_parsing import  parse_generation_args
from ..core.constants import set_cosmology
from ..core.conversion import KilonovaEjectaFitting
from ..core.base import adjust_priors_for_nmma, adjust_hubble_prior
from ..core.utils import read_trigger_time
from ..gw.gw_inputs import NMMAGravitationalWaveInput
from ..em.prior import extinction_prior
from ..em.io import load_em_observations
from ..em.model import create_injection_model
from ..em.lightcurve_generation import create_light_curve_data
from ..em.systematics import FilterSystematicsHandler
from ..em import utils as em_utils
from ..eos.eos_likelihood import (compose_eos_constraints, 
        EoSConverter, JointEoSConstraint, setup_tabulated_eos_priors)
from .joint_likelihood import MultiMessengerLikelihood

import matplotlib    
matplotlib.rcParams['text.usetex'] = False

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

    args_dict = vars(args).copy()
    for key, val in args_dict.items():
        if key == "label":
            continue
        if isinstance(val, str):
            if os.path.isfile(val) or os.path.isdir(val):
                setattr(args, key, os.path.abspath(val))
        if isinstance(val, list):
            if len(val) == 0:
                setattr(args, key, "[]")
            elif isinstance(val[0], str):
                setattr(args, key, f"[{', '.join(val)}]")
    args.sampler_kwargs = str(inputs.sampler_kwargs)
    args.submit = False
    parser.write_config_file(args, [inputs.complete_ini_file])

def create_generation_logger(outdir, label):
    logger = bilby.core.utils.logger
    bilby.core.utils.setup_logger(
        outdir=os.path.join(outdir, "data"), label=label
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
    def __init__(self, args, unknown_args, logger=None):
        
        # nmma-defaults that might conflict with bilby/bilby_pipe defaults
        gen_cosmo = set_cosmology(getattr(args, "cosmology", None))
        args.cosmology = gen_cosmo.name
        self.cosmology = args.cosmology

        super().__init__(args, unknown_args)
        # Generic setup, ripped from bilby pipe
        # Admin arguments
        self.ini = args.ini

        # Run index arguments
        self.idx = args.idx
        self.generation_seed = args.generation_seed

        # Naming arguments
        self.outdir = args.outdir
        self.label = args.label
        self.unknown_args = unknown_args

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


        self.sampler = "dynesty"
        self.sampling_seed = args.sampling_seed
        self.data_dump_file = f"{self.data_directory}/{self.label}_data_dump.pickle"

        self.data_set = False
        self.injection_numbers = args.injection_numbers
        self.injection_file = args.injection_file
        self.injection_dict = args.injection_dict
        if self.injection_file or self.injection_dict:
            self.injection = True
            args.injection = True
            self.injection_parameters = self.injection_df.iloc[self.idx].to_dict()
        else: 
            args.injection = False
            self.injection_parameters = None

        self.trigger_time = read_trigger_time(self.injection_parameters, args, 'gps')
        args.trigger_time = self.trigger_time

        self.meta_data = dict(
                config_file=self.ini,
                data_dump_file=self.data_dump_file,
                **get_version_info(),
                command_line_args=args.__dict__,
                unknown_command_line_args=self.unknown_args,
                injection_parameters=self.injection_parameters,
        )
        self.adjust_priors_and_data(args, logger)
        
        #test-build likelihood 
        self.lhood = MultiMessengerLikelihood.setup_from_args(
            self.data_dump, self._priors, self.args, logger)
        self.lhood.log_likelihood(self._priors.sample())
        
        self.save_data_dump()

    def adjust_priors_and_data(self, args, logger):
        messengers, analysis_modifiers = [], []
        data_dump = dict(injection_parameters = self.injection_parameters)
        if self.injection_parameters:
            converted_injection = self.injection_parameters.copy()
            
        # GW SETUP
        if args.detectors:
            messengers.append("gw")
            # get a BBHPriorDict only if GW parameters are present
            priors = super()._get_priors()
            if self.injection_parameters:
                converted_injection = bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters(
                    converted_injection)[0]
            else:
                self.gw_inputs= NMMAGravitationalWaveInput(args, self.unknown_args)
                data_dump |= dict(waveform_generator=self.gw_inputs.waveform_generator,
                    ifo_list=self.gw_inputs.interferometers)
        else:
            priors = self._get_priors()

        priors = adjust_priors_for_nmma(priors, logger)
        priors = adjust_hubble_prior(priors, args, logger)
        if args.Hubble or any(['hubble' in key.lower() for key in priors.keys()]):
            analysis_modifiers.append("Hubble")

        # EOS Setup
        if args.emulator_metadata:
            messengers.append("eos")
            logger.info("Setting up EOS constraints")
            data_dump |= dict(eos_constraint_dict= compose_eos_constraints(args))
            eos_converter = EoSConverter(args, 'emulated')

        elif args.eos_data:
            analysis_modifiers.append('tabulated_eos')
            eos_constraint_dict = compose_eos_constraints(args)
            if eos_constraint_dict:
                eos_converter = EoSConverter(args, 'tabulated')
                constraint = JointEoSConstraint(eos_constraint_dict, eos_converter=eos_converter)
                args.eos_weight, args.eos_data, args.Neos = constraint.tabulate_weighted_eos(
                    args.Neos, args.outdir, args.eos_weight)
            priors = setup_tabulated_eos_priors(args, priors, logger)
        
        if self.injection_parameters:
            try: 
                converted_injection = eos_converter(converted_injection)
                converted_injection = KilonovaEjectaFitting()(converted_injection)
            except Exception as e:
                logger.warning("eos and ejecta fitting failed for injection parameters. Continuing without conversion.")
                logger.warning(f"Error was {e}")
                pass
            finally:
                # correct injection only once lambdas are properly set
                # some routines return np.float32 which raises errors downstream in results
                # processing, so we convert to float here. Should be handled more elegantly
                args.injection_dict = {k: float(v) for k, v in converted_injection.items()}
                self.gw_inputs= NMMAGravitationalWaveInput(args, self.unknown_args)
                data_dump |= dict(waveform_generator=self.gw_inputs.waveform_generator,
                    ifo_list=self.gw_inputs.interferometers)

        # EM SETUP
        if args.em_model or args.em_transient_class:
            messengers.append('em')
            if self.injection_parameters:
                injection_model = create_injection_model(args)
                converted_injection = injection_model.parameter_conversion(
                    converted_injection)
                for param in injection_model.model_parameters:
                    if param not in self.injection_parameters:
                        try:
                            self.injection_parameters[param] = converted_injection[param]
                        except KeyError:
                            raise KeyError(f"Required parameter {param} could not be derived from conversion.")
                light_curve_data = create_light_curve_data(
                        self.injection_parameters, args, injection_model
                    )
            else:
                light_curve_data = load_em_observations(args)

            filters = em_utils.set_filters(args)
            if not filters:
                filters = list(light_curve_data.keys())
            
            sys_handler = FilterSystematicsHandler(filters, 
                args.systematics_file, error_budget=args.em_error_budget)
            
            priors = sys_handler.setup_systematics_priors(priors)
            priors = extinction_prior(priors, args)
            data_dump |= dict(light_curve_data=light_curve_data, filters = filters,
                    systematics_dict = sys_handler.systematics_dict)

        self.args = args
        self.messengers = messengers
        self.analysis_modifiers= analysis_modifiers
        self._priors = priors
        self.priors.to_json(outdir=self.data_directory, label=self.label)
        self.prior_file = f"{self.data_directory}/{self.label}_prior.json"

        self.data_dump = data_dump | dict(
                prior_file=self.prior_file,
                args=self.args,
                messengers = self.messengers,
                analysis_modifiers= self.analysis_modifiers,
                data_dump_file=self.data_dump_file,
                meta_data=self.meta_data,
                injection_parameters=self.injection_parameters,
        )
        logger.info(f"Set up data dump with messengers: {self.messengers} "
            f"and analysis modifiers: {self.analysis_modifiers}"
            f"data dump is {data_dump}")


    def save_data_dump(self):        

        with open(self.data_dump_file, "wb+") as file:
            pickle.dump(self.data_dump, file)

    @property
    def sampling_seed(self):
        return self._sampling_seed

    @sampling_seed.setter
    def sampling_seed(self, sampling_seed):
        if sampling_seed is None:
            sampling_seed = np.random.randint(1, 1e6)
        self._sampling_seed = sampling_seed
        np.random.seed(sampling_seed)

    def _get_priors(self):
        # agnostic prior setup
        return bilby.core.prior.PriorDict(self.prior_file)
    

def generate_runner(cli_args=[""], **kwargs):
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
    args, generation_parser = parse_generation_args(cli_args)
    for key, val in kwargs.items():
        setattr(args, key, val)

    logger = create_generation_logger(outdir=args.outdir, label=args.label)
    for package, version in get_version_info().items():
        logger.info(f"{package} version: {version}")

    inputs = NMMADataGenerationInput(args, [], logger)

    write_complete_config_file(parser=generation_parser, args=args, inputs=inputs)
    logger.info(f"Complete ini written: {inputs.complete_ini_file}")
    logger.info(f"Setup complete")

    return inputs, logger


def nmma_generation():
    """
    nmma_generation entrypoint.

    This function is a wrapper around generate_runner(),
    giving it a command line interface.
    """
    # Parse command line arguments
    cli_args = sys.argv[1:]

    # Initialise run
    generate_runner(cli_args=cli_args)

