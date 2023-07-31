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
    create_nmma_gw_generation_parser,
    parse_generation_args
)
from ..em.io import loadEvent
from ..em.injection import create_light_curve_data

from .._version import __version__

def find_sh_scripts(file_path):
    # Open the file
    with open(file_path, 'r') as f:
        # Read the content
        content = f.read()
        
    # Split the content into words using any whitespace as a separator
    words = content.split()
    
    # Filter out the words that end with '.sh'
    sh_scripts = [word for word in words if word.endswith('.sh)')]
    
    return sh_scripts


def replace_pbilby_in_file(file_path, name):
    # Read the contents of the file
    with open(file_path, 'r') as file:
        file_contents = file.read()

    # Replace the text
    new_contents = file_contents.replace("parallel_bilby", name)

    # Write the updated contents back to the file
    with open(file_path, 'w+') as file:
        file.write(new_contents)

    return


def get_version_info():
    return dict(
        bilby_version=bilby.__version__,
        bilby_pipe_version=bilby_pipe.__version__,
        parallel_bilby_version=parallel_bilby.__version__,
        dynesty_version=dynesty.__version__,
        lalsimulation_version=lalsimulation.__version__,
        nmma_version=__version__
    )


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


class NMMADataGenerationInput(bilby_pipe.data_generation.DataGenerationInput):
    def __init__(self, args, unknown_args, inference_favour):
        super().__init__(args, unknown_args, inference_favour)
        self.args = args
        self.sampler = "dynesty"
        self.sampling_seed = args.sampling_seed
        self.data_dump_file = f"{self.data_directory}/{self.label}_data_dump.pickle"
        self.inference_favour = inference_favour
        assert inference_favour in ['nmma', 'nmma_gw'], "invalid inference_favour"
        self.setup_inputs()

    @property
    def sampling_seed(self):
        return self._samplng_seed

    @sampling_seed.setter
    def sampling_seed(self, sampling_seed):
        if sampling_seed is None:
            sampling_seed = np.random.randint(1, 1e6)
        self._samplng_seed = sampling_seed
        np.random.seed(sampling_seed)

    def save_data_dump(self):
        if self.inference_favour == 'nmma':
            if self.injection_parameters:
                light_curve_data = create_light_curve_data(
                    self.injection_parameters, self.args
                )
            else:
                light_curve_data = loadEvent(self.args.light_curve_data)

            with open(self.data_dump_file, "wb+") as file:
                data_dump = dict(
                    waveform_generator=self.waveform_generator,
                    ifo_list=self.interferometers,
                    prior_file=self.prior_file,
                    light_curve_data=light_curve_data,
                    args=self.args,
                    data_dump_file=self.data_dump_file,
                    meta_data=self.meta_data,
                    injection_parameters=self.injection_parameters,
                )
                pickle.dump(data_dump, file)

        elif self.inference_favour == 'nmma_gw':
            with open(self.data_dump_file, "wb+") as file:
                data_dump = dict(
                    waveform_generator=self.waveform_generator,
                    ifo_list=self.interferometers,
                    prior_file=self.prior_file,
                    args=self.args,
                    data_dump_file=self.data_dump_file,
                    meta_data=self.meta_data,
                    injection_parameters=self.injection_parameters,
                )
                pickle.dump(data_dump, file)

    def setup_inputs(self):
        if self.likelihood_type == "ROQGravitationalWaveTransient":
            self.save_roq_weights()
        self.interferometers.plot_data(outdir=self.data_directory, label=self.label)

        # This is done before instantiating the likelihood so that it is the full prior
        self.priors.to_json(outdir=self.data_directory, label=self.label)
        self.prior_file = f"{self.data_directory}/{self.label}_prior.json"

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


def generate_runner(inference_favour=None, parser=None, **kwargs):
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

    inputs = NMMADataGenerationInput(args, [], inference_favour)
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


def main_nmma():
    """
    nmma_generation entrypoint.

    This function is a wrapper around generate_runner(),
    giving it a command line interface.
    """

    # Parse command line arguments
    cli_args = get_cli_args()
    generation_parser = create_nmma_generation_parser()
    args = parse_generation_args(generation_parser,
                                 cli_args, as_namespace=True)

    # Initialise run
    inputs, logger = generate_runner(inference_favour='nmma',
                                     parser=generation_parser,
                                     **vars(args))

    # Write slurm script
    bash_file = slurm.setup_submit(inputs.data_dump_file, inputs, args, cli_args)
    # change the parallel_bilby_analysis to nmma_analysis
    sh_scripts = find_sh_scripts(bash_file)
    for sh_script in sh_scripts:
        replace_pbilby_in_file(sh_script.replace(')', ''), 'nmma')
    if args.submit:
        subprocess.run([f"bash {bash_file}"], shell=True)
    else:
        logger.info(f"Setup complete, now run:\n $ bash {bash_file}")


def main_nmma_gw():
    """
    nmma_gw_generation entrypoint.

    This function is a wrapper around generate_runner(),
    giving it a command line interface.
    """

    # Parse command line arguments
    cli_args = get_cli_args()
    generation_parser = create_nmma_gw_generation_parser()
    args = parse_generation_args(generation_parser,
                                 cli_args, as_namespace=True)

    # Initialise run
    inputs, logger = generate_runner(inference_favour='nmma_gw',
                                     parser=generation_parser,
                                     **vars(args))

    # Write slurm script
    bash_file = slurm.setup_submit(inputs.data_dump_file, inputs, args, cli_args)
    # change the parallel_bilby_analysis to nmma_gw_analysis
    sh_scripts = find_sh_scripts(bash_file)
    for sh_script in sh_scripts:
        replace_pbilby_in_file(sh_script.replace(')', ''), 'nmma_gw')

    if args.submit:
        subprocess.run([f"bash {bash_file}"], shell=True)
    else:
        logger.info(f"Setup complete, now run:\n $ bash {bash_file}")
