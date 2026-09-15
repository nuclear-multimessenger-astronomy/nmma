import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import pickle

try:
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    rank = 0

if rank != 0:
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)

from bilby.core.prior import PriorDict
from ..core.mpi_setup import pbilby_sampling
from ..core.base import bilby_sampling
from .multi_parsing import create_nmma_analysis_parser, parse_analysis_args
from .joint_likelihood import MultiMessengerLikelihood
from ..core.utils import logger


def analysis_runner(
    data_dump,
    outdir=None,
    label=None,
    plot=False,
    **kwargs,
):
    """
    Run the analysis described by a data dump.

    Loads the data dump, rebuilds the priors and the likelihood from it, and
    samples with ``pbilby_sampling`` when ``args.sampler`` is ``"dynesty"``
    and ``bilby_sampling`` otherwise.

    Parameters
    ----------
    data_dump : str
        Path to the pickled data dump. If it does not end in
        ``_dump.pickle``, the first ``*_dump.pickle`` found under
        ``{data_dump}/data`` is used instead.
    outdir : str, default=None
        If truthy, overwrites ``outdir`` on the arguments taken from the
        data dump.
    label : str, default=None
        If truthy, overwrites ``label`` on the arguments taken from the
        data dump.
    plot : bool, default=False
        Assigned to ``args.plot``, and passed on to ``pbilby_sampling``.
    **kwargs
        Passed to ``pbilby_sampling``; not forwarded on the
        ``bilby_sampling`` branch.

    Returns
    -------
    The return value of ``pbilby_sampling`` or ``bilby_sampling``.
    """

    ## Load the data dump
    if not data_dump.endswith("_dump.pickle"):
        test_out = Path(data_dump, "data")
        data_dump = next(test_out.glob("*_dump.pickle"))
    with open(data_dump, "rb") as file:
        data_dump = pickle.load(file)

    ## Set properties from the data dump
    args = data_dump["args"]
    args.plot = plot

    # If the run dir has not been specified, get it from the args
    if outdir:
        args.outdir = outdir

    # If the label has not been specified, get it from the args
    if label:
        args.label = label

    priors = PriorDict.from_json(data_dump["prior_file"])

    ## Set up the likelihood
    likelihood = MultiMessengerLikelihood.setup_from_args(
        data_dump, priors, args, logger
    )

    ## adjust meta data to storable format
    meta_data = data_dump.copy()
    waveform_generator = meta_data.pop("waveform_generator", None)
    if waveform_generator is not None:
        meta_data["waveform_generator"] = waveform_generator.__repr__()
    ifo_list = meta_data.pop("ifo_list", None)
    if ifo_list is not None:
        meta_data["ifo_list"] = [ifo.__repr__() for ifo in ifo_list]

    if args.sampler == "dynesty":
        logger.info("Using dynesty sampler")
        return pbilby_sampling(
            likelihood,
            priors,
            args,
            data_dump.get("injection_parameters", None),
            rank,
            plot=plot,
            meta_data=meta_data,
            **kwargs,
        )
    else:
        return bilby_sampling(
            likelihood, priors, args, data_dump.get("injection_parameters", None), rank
        )


def nmma_analysis():
    """
    Entry point for ``nmma-analysis``.

    Builds the parser with :func:`nmma.joint.multi_parsing.create_nmma_analysis_parser`,
    parses the command line with
    :func:`nmma.joint.multi_parsing.parse_analysis_args`, and calls
    :func:`analysis_runner` with the parsed arguments as keyword arguments.
    """
    # Parse command line arguments
    analysis_parser = create_nmma_analysis_parser(sampler="dynesty")
    input_args = parse_analysis_args(analysis_parser)

    # Run the analysis
    analysis_runner(**vars(input_args))
