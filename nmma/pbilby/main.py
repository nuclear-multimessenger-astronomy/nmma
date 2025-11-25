"""
Module to run parallel bilby using MPI
"""

import mpi4py
mpi4py.rc.threads = False
mpi4py.rc.recv_mprobe = False
del mpi4py

import sys
import os
from time import time
from bilby.core.utils import logger

import io
import contextlib
# Create buffers
stdout_buffer = io.StringIO()
stderr_buffer = io.StringIO()

# Redirect python output into buffers
redirect_out = contextlib.redirect_stdout(stdout_buffer)
redirect_err = contextlib.redirect_stderr(stderr_buffer)

redirect_out.__enter__()
redirect_err.__enter__()
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from schwimmbad import MPIPool, MultiPool
import signal
from .multi_parsing import create_nmma_analysis_parser, parse_analysis_args
from .analysis_run import MainRun, WorkerRun

def analysis_runner(
    data_dump,
    outdir=None,
    label=None,
    sample="acceptance-walk",
    nlive=1000,
    bound="live",
    walks=100,
    maxmcmc=5000,
    naccept=60,
    nact=2,
    facc=0.5,
    min_eff=10,
    enlarge=1.5,
    sampling_seed=0,
    bilby_zero_likelihood_mode=False,
    rejection_sample_posterior=True,
    #
    check_point_delta_t=3600,
    dlogz=0.1,
    save_bounds=False,
    n_check_point=1000,
    max_its=1e10,
    max_run_time=1e10,
    checkpoint_plot=False,
    result_format="hdf5",
    pool_type ='mpi',
    **kwargs,
):
    """
    API for running the analysis from Python instead of the command line.
    It takes all the same options as the CLI, specified as keyword arguments.

    Returns
    -------
    exit_reason: integer u
        Used during testing, to determine the reason the code halted:
            0 = run completed normally, based on convergence criteria
            1 = reached max iterations
            2 = reached max runtime
        MPI worker tasks always return -1

    """
    # Initialise a WorkerRun. this needs a global scope to allow 
    # persistence of states beyond the pool's scope.
    # Otherwise emulators retrace on each evaluation.
    global worker_run
    worker_run = WorkerRun(data_dump, bilby_zero_likelihood_mode)
    t0 = time()
    sampling_time = 0
    POOL = MPIPool if pool_type == 'mpi' else MultiPool
    # Restore normal stdout/stderr
    redirect_out.__exit__(None, None, None)
    redirect_err.__exit__(None, None, None)

    with POOL(use_dill=True) as pool:
        if pool.is_master():
            sys.stdout.write(stdout_buffer.getvalue())
            sys.stderr.write(stderr_buffer.getvalue())

            POOL_SIZE = pool.size
            run = MainRun(
                worker_run.sampling_keys,
                pooled_log_likelihood, 
                pooled_prior_transform,
                pooled_initial_point_from_prior,
                args=worker_run.args,
                outdir=outdir,
                label=label,
                periodic=worker_run.periodic,
                reflective=worker_run.reflective,
                sample=sample,
                nlive=nlive,
                bound=bound,
                walks=walks,
                maxmcmc=maxmcmc,
                nact=nact,
                naccept=naccept,
                facc=facc,
                min_eff=min_eff,
                enlarge=enlarge,
                sampling_seed=sampling_seed,
                sampler_kwargs = dict(dlogz=dlogz, save_bounds=save_bounds)
            )
            logger.info("Using priors:")
            for k, p in worker_run.priors.items():
                logger.info(f"{k}: {p}")
            
            sampler, sampling_time = run.start_sampler(pool)

            ## graceful handling of preemptive shutdowns
            def handle_sigterm(signum, frame):
                logger.info("Received SIGTERM, writing checkpoint and exiting.")
                ## no time for plotting when file_size becomes larger
                run.checkpointing(sampler, sampling_time, False)
                pool.close()
                pool.wait()
                logger.info("Exited gracefully.")
                sys.exit(0)

            signal.signal(signal.SIGTERM, handle_sigterm)
            signal.signal(signal.SIGINT , handle_sigterm)
            signal.signal(signal.SIGUSR1, handle_sigterm)

            run_time = 0
            last_checkpoint_time= t0
            early_stop = False
            logger.info(f"Run criteria: {run.sampler_kwargs}")

            logger.info(f"Starting sampling for job {run.label}, with pool size={POOL_SIZE} "
                f"and time between checkpoints ={check_point_delta_t}s" )
            for it, res in enumerate(sampler.sample(**run.sampler_kwargs)):
                run.stdout_sampling_log(
                    results=res, niter=it, ncall=sampler.ncall, dlogz=dlogz
                )

                iteration_time = time() - t0
                t0 = time()

                sampling_time += iteration_time
                run_time += iteration_time
                last_checkpoint_s = t0 - last_checkpoint_time

                """
                Criteria for writing checkpoints:
                a) time since last checkpoint > check_point_delta_t
                b) reached an integer multiple of n_check_point
                c) reached max iterations
                d) reached max runtime
                """

                if (
                    last_checkpoint_s > check_point_delta_t
                    or (it % n_check_point == 0 and it != 0)
                    or it == max_its
                    or run_time > max_run_time
                ):
                    run.checkpointing(sampler, sampling_time, checkpoint_plot)
                    if it == max_its:
                        exit_reason = 1
                        logger.info(
                            f"Max iterations ({it}) reached; stopping sampling (exit_reason={exit_reason})."
                        )
                        early_stop = True
                        break

                    if run_time > max_run_time:
                        exit_reason = 2
                        logger.info(
                            f"Max run time ({max_run_time}) reached; stopping sampling (exit_reason={exit_reason})."
                        )
                        early_stop = True
                        break
                    last_checkpoint_time = time() 

            if not early_stop:
                exit_reason = 0
                # Adding the final set of live points.
                for it_final, res in enumerate(sampler.add_live_points()):
                    pass

                # Create a final checkpoint and set of plots
                run.temp_ext = 'dat'
                run.checkpointing(sampler, sampling_time, checkpoint_plot)

                sampling_time += time() - t0

                run.format_result(
                    worker_run,
                    data_dump,
                    sampler.results,
                    result_format,
                    sampling_time,
                    rejection_sample_posterior=rejection_sample_posterior
                )
        else:
            exit_reason = -1
        return exit_reason


# Worker functions. These are read in the global scope by each worker
def pooled_initial_point_from_prior(args):
    return worker_run.get_initial_point_from_prior(args)

def pooled_log_likelihood(v_array):
    return worker_run.log_likelihood_function(v_array)

def pooled_prior_transform(u_array):
    return worker_run.prior_transform_function(u_array)

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
