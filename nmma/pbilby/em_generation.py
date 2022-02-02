#!/usr/bin/env python
import copy
import json
import logging
import bilby_pipe
import pickle
import dynesty
from astropy import time
from bilby_pipe import data_generation as bilby_pipe_datagen
from bilby_pipe.data_generation import parse_args
import bilby
from parallel_bilby.utils import get_cli_args
from .parser import create_nmma_generation_parser
from .utils import write_complete_config_file as _write_complete_config_file

from .._version import __version__
from ..em.utils import loadEvent
from ..em.injection import create_light_curve_data
__prog__ = "parallel_em_generation"

logger = logging.getLogger(__prog__)


def add_extra_args_for_label_and_outdir(args):
    """
    :param args: args from parallel_bilby
    :return: Namespace argument object
    """
    pipe_args, _ = parse_args(
        get_cli_args(), bilby_pipe_datagen.create_generation_parser()
    )
    accepted_args = [
        'label',
        'outdir',
        'injection',
        'injection-file',
        'injection-numbers',
    ]

    for key, val in vars(pipe_args).items():
        if key not in args and key in accepted_args:
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
    args = add_extra_args_for_label_and_outdir(args)

    bilby.core.utils.setup_logger(outdir=args.outdir, label=args.label)
    bilby.core.utils.check_directory_exists_and_if_not_mkdir(args.outdir)

    label = args.label

    if args.injection:
        # create light curve data
        with open(args.injection_file, 'r') as f:
            injection_dict = json.load(
                f, object_hook=bilby.core.utils.decode_bilby_json
            )
        injection_df = injection_dict["injections"]
        injection_parameters = injection_df.iloc[args.injection_numbers[0]].to_dict()

        try:
            tc_gps = time.Time(
                injection_parameters['geocent_time_x'],
                format='gps'
            )
        except KeyError:
            tc_gps = time.Time(
                injection_parameters['geocent_time'],
                format='gps'
            )
        trigger_time = tc_gps.mjd
        injection_parameters['kilonova_trigger_time'] = trigger_time

        light_curve_data = create_light_curve_data(
            injection_parameters, args
        )

    else:
        # load the kilonova afterglow data
        light_curve_data = loadEvent(args.light_curve_data)

    data_dump_file = f"{args.outdir}/{label}_data_dump.pickle"

    meta_data = \
        dict(
            config_file=args.ini,
            data_dump_file=data_dump_file,
            nmma_version=__version__,
            bilby_version=bilby.__version__,
            bilby_pipe_version=bilby_pipe.__version__,
            parallel_bilby_version=__version__,
            dynesty_version=dynesty.__version__,
        )
    logger.info("Initial meta_data = {}".format(meta_data))

    data_dump = dict(
        prior_file=args.prior_file,
        light_curve_data=light_curve_data,
        args=args,
        data_dump_file=data_dump_file,
        meta_data=meta_data,
        injection_parameters=None,
    )

    with open(data_dump_file, "wb+") as file:
        pickle.dump(data_dump, file)

    args.complete_ini_file = f"{args.outdir}/config.ini"
    # write_complete_config_file(parser, args, __prog__)
