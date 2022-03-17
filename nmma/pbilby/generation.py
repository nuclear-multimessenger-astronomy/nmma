#!/usr/bin/env python
import matplotlib
matplotlib.use("agg")
import subprocess
import copy
import logging
import bilby_pipe
import shutil
import pickle
import dynesty
from bilby_pipe import data_generation as bilby_pipe_datagen
from bilby_pipe.data_generation import parse_args
import bilby
from parallel_bilby.utils import get_cli_args
from .slurm import setup_submit
from .parser import create_nmma_generation_parser
from .utils import write_complete_config_file as _write_complete_config_file

from .._version import __version__
from ..em.utils import loadEvent
from ..em.injection import create_light_curve_data
__prog__ = "nmma_generation"

logger = logging.getLogger(__prog__)


def add_extra_args_from_bilby_pipe_namespace(args):
    """
    :param args: args from parallel_bilby
    :return: Namespace argument object
    """
    pipe_args, _ = parse_args(
        get_cli_args(), bilby_pipe_datagen.create_generation_parser()
    )
    for key, val in vars(pipe_args).items():
        if key not in args:
            setattr(args, key, val)
    return args


def write_complete_config_file(parser, args, inputs, prog):
    # Hack
    inputs_for_writing_config = copy.deepcopy(inputs)
    inputs_for_writing_config.request_cpus = 1
    inputs_for_writing_config.sampler_kwargs = "{}"
    inputs_for_writing_config.mpi_timing_interval = 0

    _write_complete_config_file(parser, args, inputs_for_writing_config, prog)


def main():
    cli_args = get_cli_args()
    parser = create_nmma_generation_parser(__prog__, __version__)
    args = parser.parse_args(args=cli_args)
    args = add_extra_args_from_bilby_pipe_namespace(args)

    inputs = bilby_pipe_datagen.DataGenerationInput(args, [])
    if inputs.likelihood_type == "ROQGravitationalWaveTransient":
        inputs.save_roq_weights()
    inputs.log_directory = None
    shutil.rmtree(inputs.data_generation_log_directory)  # Hack to remove unused dir

    ifo_list = inputs.interferometers
    data_dir = inputs.data_directory
    label = inputs.label
    ifo_list.plot_data(outdir=data_dir, label=label)

    logger.info(
        "Setting up likelihood with marginalizations: "
        f"distance={args.distance_marginalization} "
        f"time={args.time_marginalization} "
        f"phase={args.phase_marginalization} "
    )

    # This is done before instantiating the likelihood so that it is the full prior
    prior_file = f"{data_dir}/{label}_prior.json"
    inputs.priors.to_json(outdir=data_dir, label=label)

    # We build the likelihood here to ensure the distance marginalization exist
    # before sampling
    inputs.likelihood

    # load/create the light curve data
    if inputs.injection_parameters:
        light_curve_data = create_light_curve_data(
            inputs.injection_parameters, args
        )
    else:
        light_curve_data = loadEvent(args.light_curve_data)

    data_dump_file = f"{data_dir}/{label}_data_dump.pickle"

    meta_data = inputs.meta_data
    meta_data.update(
        dict(
            config_file=args.ini,
            data_dump_file=data_dump_file,
            nmma_version=__version__,
            bilby_version=bilby.__version__,
            bilby_pipe_version=bilby_pipe.__version__,
            parallel_bilby_version=__version__,
            dynesty_version=dynesty.__version__,
        )
    )
    logger.info("Initial meta_data = {}".format(meta_data))

    data_dump = dict(
        waveform_generator=inputs.waveform_generator,
        ifo_list=ifo_list,
        prior_file=prior_file,
        light_curve_data=light_curve_data,
        args=args,
        data_dump_file=data_dump_file,
        meta_data=meta_data,
        injection_parameters=inputs.injection_parameters,
    )

    with open(data_dump_file, "wb+") as file:
        pickle.dump(data_dump, file)

    write_complete_config_file(parser, args, inputs, __prog__)

    # Generate bash file for slurm submission
    bash_file = setup_submit(data_dump_file, inputs, args)
    if args.submit:
        subprocess.run(["bash {}".format(bash_file)], shell=True)
    else:
        logger.info("Setup complete, now run:\n $ bash {}".format(bash_file))
