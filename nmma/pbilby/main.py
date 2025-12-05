"""
Module to run parallel bilby using MPI
"""

# import mpi4py
# mpi4py.rc.threads = False
# mpi4py.rc.recv_mprobe = False
# del mpi4py

import sys
import os
from time import time
import traceback

import io
import contextlib
# Create buffers
stdout_buffer = io.StringIO()
stderr_buffer = io.StringIO()

# Redirect python output into buffers
redirect_out = contextlib.redirect_stdout(stdout_buffer)
redirect_err = contextlib.redirect_stderr(stderr_buffer)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

with redirect_out, redirect_err:
    from schwimmbad import MPIPool, MultiPool
    import signal
    from .multi_parsing import create_nmma_analysis_parser, parse_analysis_args, process_sampler_kwargs
    from .analysis_run import MainRun, WorkerRun

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
    bilby_zero_likelihood_mode=False,
    rejection_sample_posterior=True,
    #
    check_point_delta_t=1800,
    n_check_point=2000,
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
    """
    POOL = MPIPool if pool_type == 'mpi' else MultiPool
    global worker_run
    # Initialise a WorkerRun. this needs a global scope to allow 
    # persistence of states beyond the pool's scope.
    # Otherwise emulators retrace on each evaluation.
    try:
        with redirect_out, redirect_err:
            worker_run = WorkerRun(data_dump, bilby_zero_likelihood_mode)
    except:
        with POOL() as pool:
            if pool.is_master():
                sys.stdout.write(stdout_buffer.getvalue())
                traceback.print_exc()
                sys.stderr.write(stderr_buffer.getvalue())
                pool.close()
            else:
                pool.wait()
            sys.exit() 
        
    
    with POOL() as pool:
        if pool.is_master():
            sys.stdout.write(stdout_buffer.getvalue())
            sys.stderr.write(stderr_buffer.getvalue())
            prelim_sampler_kwargs, run_sampler_kwargs = process_sampler_kwargs(
                init_sampler_kwargs, run_sampler_kwargs, kwargs)
            init_sampler_kwargs = dict(
                periodic    =worker_run.periodic,
                reflective  =worker_run.reflective,
                ndim=len(worker_run.sampling_keys)
            ) | prelim_sampler_kwargs

            run = MainRun(
                worker_run.sampling_keys,
                pooled_log_likelihood, 
                pooled_prior_transform,
                pooled_initial_point_from_prior,
                args=worker_run.args,
                outdir=outdir,
                label=label,
                maxmcmc=maxmcmc,
                nact=nact,
                naccept=naccept,
                sampling_seed=sampling_seed,
                sampler_kwargs = run_sampler_kwargs,
                sampler_init_kwargs=init_sampler_kwargs,
            )

            sampler, sampling_time = run.start_sampler(pool)
            
            ## graceful handling of preemptive shutdowns
            def handle_sigterm(signum, frame):
                ## no time for plotting when file_size becomes larger
                run.checkpointing(sampler, sampling_time, False, 
                    "succesfully created checkpoint after termination signal.")
                pool.close()
                pool.wait()
                sys.exit(0)

            signal.signal(signal.SIGTERM, handle_sigterm)
            signal.signal(signal.SIGINT , handle_sigterm)
            signal.signal(signal.SIGUSR1, handle_sigterm)

            t0 = time()
            run_time = 0
            last_checkpoint_time= t0
            last_checkpoint_it = 0
            
            for it, res in enumerate(sampler.sample(**run.sampler_kwargs)):

                run.stdout_sampling_log(results=res, niter=it, ncall=sampler.ncall)
                iteration_time = time() - t0
                sampling_time += iteration_time
                run_time += iteration_time
                t0 = time()

                if it >= max_its or run_time > max_run_time:
                    run.checkpointing(sampler, sampling_time, checkpoint_plot,
                        f"{it} of max {max_its} iterations completed after {sampling_time:.2f}s" 
                        f" sampling time of max {max_run_time}s. Stopping." )
                    return 
                
                elif (
                    # checkpoint criteria
                    t0 - last_checkpoint_time > check_point_delta_t
                    or (it - last_checkpoint_it > n_check_point) 
                ):
                    run.checkpointing(sampler, sampling_time, checkpoint_plot, 
                        run.get_step_info_str(results=res, niter=it, ncall=sampler.ncall))
                    last_checkpoint_time = time() 
                    last_checkpoint_it = it

            # Adding the final set of live points.
            for it_final, res in enumerate(sampler.add_live_points()):
                pass

            # Create a final checkpoint in case anything happens during the formatting
            run.sampling_time = sampling_time + time() - t0
            run.write_current_state(sampler, run.sampling_time )
            run.plot_current_state(sampler)

            run.format_result(worker_run, sampler.results, result_format,
                rejection_sample_posterior=rejection_sample_posterior
            )
            pool.close()
        else:
            pool.wait()
        exit_code = 0
        return exit_code


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
    
