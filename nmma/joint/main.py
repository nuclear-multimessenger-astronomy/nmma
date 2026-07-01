import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from glob import glob
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
    API for running the analysis from Python instead of the command line.
    It takes all the same options as the CLI, specified as keyword arguments.
    """

    ## Load the data dump
    if not data_dump.endswith("_dump.pickle"):
        test_out = os.path.join(os.getcwd(), data_dump)
        test_dump = glob(f"{test_out}/data/*_dump.pickle")
        data_dump = test_dump[0]
    with open(data_dump, "rb") as file:
        data_dump = pickle.load(file)

    ## Set properties from the data dump
    args = data_dump["args"]
    args.plot = plot

    # If the run dir has not been specified, get it from the args
    if outdir:
        args.outdir  = outdir
        
    # If the label has not been specified, get it from the args
    if label:
        args.label = label

    priors = PriorDict.from_json(data_dump["prior_file"])
    
    ## Set up the likelihood
    likelihood = MultiMessengerLikelihood.setup_from_args(
        data_dump, priors, args, logger)
    
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
            likelihood, priors, args, 
            data_dump.get("injection_parameters", None), rank,
            plot=plot, meta_data=meta_data, **kwargs)
    else:
        return bilby_sampling(
            likelihood, priors, args, 
            data_dump.get("injection_parameters", None), rank)

def nmma_analysis():
    """
    nmma_analysis entrypoint.

    This function is a wrapper around analysis_runner(),
    giving it a command line interface.
    """
    # Parse command line arguments
    analysis_parser = create_nmma_analysis_parser(sampler="dynesty")
    input_args = parse_analysis_args(analysis_parser)

    # Run the analysis
    analysis_runner(**vars(input_args))

