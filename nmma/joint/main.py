"""
Module to run parallel bilby using MPI
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import signal
import io
import contextlib
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    rank = 0

if rank != 0:
    # Create buffers
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    # Redirect python output into buffers
    redirect_out = contextlib.redirect_stdout(stdout_buffer)
    redirect_err = contextlib.redirect_stderr(stderr_buffer)
else:
    redirect_out = contextlib.nullcontext()
    redirect_err = contextlib.nullcontext()


with redirect_out, redirect_err:
    from schwimmbad import MPIPool, MultiPool
    from .multi_parsing import create_nmma_analysis_parser, parse_analysis_args, process_sampler_kwargs
    from .analysis_run import Dynesty, Worker

def analysis_runner(
    data_dump,
    outdir=None,
    label=None,
    maxmcmc=5000,
    naccept=60,
    nact=2,
    init_sampler_kwargs={},
    run_sampler_kwargs={},
    sampling_seed=42,
    plot=True,
    #
    check_point_delta_t=1800,
    n_check_point=2000,
    max_its=1e10,
    max_run_time=1e10,
    checkpoint_plot=False,
    #
    rejection_sample_posterior=True,
    result_format="hdf5",
    pool_type ='mpi',
    **kwargs,
):
    """
    API for running the analysis from Python instead of the command line.
    It takes all the same options as the CLI, specified as keyword arguments.
    """
    # Initialise a worker. this needs a global scope to allow 
    # persistence of states beyond the pool's scope.
    # Otherwise emulators retrace on each evaluation.
    global worker
    with redirect_out, redirect_err:
        if rank == 0:
            init_sampler_kwargs, run_sampler_kwargs = process_sampler_kwargs(
                init_sampler_kwargs, run_sampler_kwargs, kwargs)

            worker = Dynesty(
                data_dump, outdir, label,
                maxmcmc=maxmcmc,
                nact=nact,
                naccept=naccept,
                sampling_seed=sampling_seed,
                sampler_kwargs = run_sampler_kwargs,
                sampler_init_kwargs=init_sampler_kwargs,
                plot=plot,
            )

        else:
            worker = Worker(data_dump, outdir, label)

    ## graceful handling of preemptive shutdowns
    def handle_sigterm(signum, frame):
        try:
            worker.checkpointing(False,
                'Received termination signal. Checkpointing and exiting gracefully.')
        except Exception:
            pass

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT , handle_sigterm)
    signal.signal(signal.SIGUSR1, handle_sigterm) 

    POOL = MPIPool if pool_type == 'mpi' else MultiPool
    with POOL() as pool:
        result = None
        if pool.is_master():           
            worker.start_sampler(
                pool,
                pooled_log_likelihood, 
                pooled_prior_transform,
                pooled_initial_point_from_prior)

            results = worker.run_sampler(
                check_point_delta_t, n_check_point, max_its,
                max_run_time, checkpoint_plot)
            result = worker.format_result(results, result_format,
                rejection_sample_posterior)
    return result


# Worker functions. These are read in the global scope by each worker
def pooled_initial_point_from_prior(args):
    return worker.get_initial_point_from_prior(args)

def pooled_log_likelihood(v_array):
    return worker.log_likelihood(v_array)

def pooled_prior_transform(u_array):
    return worker.prior_transform(u_array)

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
    
