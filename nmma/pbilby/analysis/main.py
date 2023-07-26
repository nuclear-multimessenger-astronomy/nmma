"""
Module to run parallel bilby using MPI
"""
import datetime
import json
import os
import pickle
import time

import bilby
import numpy as np
import pandas as pd
from bilby.core.utils import logger
from bilby.gw import conversion
from nestcheck import data_processing
from pandas import DataFrame

from parallel_bilby.schwimmbad_fast import MPIPoolFast as MPIPool
from parallel_bilby.utils import get_cli_args, stdout_sampling_log
from parallel_bilby.analysis.plotting import plot_current_state
from parallel_bilby.analysis.read_write import (
    format_result,
    read_saved_state,
    write_current_state,
    write_sample_dump,
)
from parallel_bilby.analysis.sample_space import fill_sample

from ..parser import (
    create_nmma_analysis_parser,
    create_nmma_gw_analysis_parser,
    parse_analysis_args
    )
from .analysis_run import AnalysisRun


def analysis_runner(
    data_dump,
    inference_favour,
    outdir=None,
    label=None,
    dynesty_sample="acceptance-walk",
    nlive=5,
    dynesty_bound="live",
    walks=100,
    proposals=None,
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
    fast_mpi=False,
    mpi_timing=False,
    mpi_timing_interval=0,
    check_point_deltaT=3600,
    n_effective=np.inf,
    dlogz=10,
    do_not_save_bounds_in_resume=True,
    n_check_point=1000,
    max_its=1e10,
    max_run_time=1e10,
    rotate_checkpoints=False,
    no_plot=False,
    nestcheck=False,
    result_format="hdf5",
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

    # Initialise a run
    run = AnalysisRun(
        data_dump=data_dump,
        inference_favour=inference_favour,
        outdir=outdir,
        label=label,
        dynesty_sample=dynesty_sample,
        nlive=nlive,
        dynesty_bound=dynesty_bound,
        walks=walks,
        maxmcmc=maxmcmc,
        nact=nact,
        naccept=naccept,
        facc=facc,
        min_eff=min_eff,
        enlarge=enlarge,
        sampling_seed=sampling_seed,
        proposals=proposals,
        bilby_zero_likelihood_mode=bilby_zero_likelihood_mode,
    )

    t0 = datetime.datetime.now()
    sampling_time = 0
    with MPIPool(
        parallel_comms=fast_mpi,
        time_mpi=mpi_timing,
        timing_interval=mpi_timing_interval,
        use_dill=True,
    ) as pool:
        if pool.is_master():
            POOL_SIZE = pool.size

            logger.info(f"sampling_keys={run.sampling_keys}")
            if run.periodic:
                logger.info(
                    f"Periodic keys: {[run.sampling_keys[ii] for ii in run.periodic]}"
                )
            if run.reflective:
                logger.info(
                    f"Reflective keys: {[run.sampling_keys[ii] for ii in run.reflective]}"
                )
            logger.info("Using priors:")
            for key in run.priors:
                logger.info(f"{key}: {run.priors[key]}")

            resume_file = f"{run.outdir}/{run.label}_checkpoint_resume.pickle"
            samples_file = f"{run.outdir}/{run.label}_samples.dat"

            sampler, sampling_time = read_saved_state(resume_file)

            if sampler is False:
                logger.info(f"Initializing sampling points with pool size={POOL_SIZE}")
                live_points = run.get_initial_points_from_prior(pool)
                logger.info(
                    f"Initialize NestedSampler with "
                    f"{json.dumps(run.init_sampler_kwargs, indent=1, sort_keys=True)}"
                )
                sampler = run.get_nested_sampler(live_points, pool, POOL_SIZE)
            else:
                # Reinstate the pool and map (not saved in the pickle)
                logger.info(f"Read in resume file with sampling_time = {sampling_time}")
                sampler.pool = pool
                sampler.queue_size = pool.size
                sampler.M = pool.map
                sampler.loglikelihood.pool = pool

            logger.info(
                f"Starting sampling for job {run.label}, with pool size={POOL_SIZE} "
                f"and check_point_deltaT={check_point_deltaT}"
            )

            sampler_kwargs = dict(
                n_effective=n_effective,
                dlogz=dlogz,
                save_bounds=not do_not_save_bounds_in_resume,
            )
            logger.info(f"Run criteria: {json.dumps(sampler_kwargs)}")

            run_time = 0
            early_stop = False

            for it, res in enumerate(sampler.sample(**sampler_kwargs)):
                stdout_sampling_log(
                    results=res, niter=it, ncall=sampler.ncall, dlogz=dlogz
                )

                iteration_time = (datetime.datetime.now() - t0).total_seconds()
                t0 = datetime.datetime.now()

                sampling_time += iteration_time
                run_time += iteration_time

                if os.path.isfile(resume_file):
                    last_checkpoint_s = time.time() - os.path.getmtime(resume_file)
                else:
                    last_checkpoint_s = np.inf

                """
                Criteria for writing checkpoints:
                a) time since last checkpoint > check_point_deltaT
                b) reached an integer multiple of n_check_point
                c) reached max iterations
                d) reached max runtime
                """

                if (
                    last_checkpoint_s > check_point_deltaT
                    or (it % n_check_point == 0 and it != 0)
                    or it == max_its
                    or run_time > max_run_time
                ):

                    write_current_state(
                        sampler,
                        resume_file,
                        sampling_time,
                        rotate_checkpoints,
                    )
                    write_sample_dump(sampler, samples_file, run.sampling_keys)
                    if no_plot is False:
                        plot_current_state(
                            sampler, run.sampling_keys, run.outdir, run.label
                        )

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

            if not early_stop:
                exit_reason = 0
                # Adding the final set of live points.
                for it_final, res in enumerate(sampler.add_live_points()):
                    pass

                # Create a final checkpoint and set of plots
                write_current_state(
                    sampler, resume_file, sampling_time, rotate_checkpoints
                )
                write_sample_dump(sampler, samples_file, run.sampling_keys)
                if no_plot is False:
                    plot_current_state(
                        sampler, run.sampling_keys, run.outdir, run.label
                    )

                sampling_time += (datetime.datetime.now() - t0).total_seconds()

                out = sampler.results

                if nestcheck is True:
                    logger.info("Creating nestcheck files")
                    ns_run = data_processing.process_dynesty_run(out)
                    nestcheck_path = os.path.join(run.outdir, "Nestcheck")
                    bilby.core.utils.check_directory_exists_and_if_not_mkdir(
                        nestcheck_path
                    )
                    nestcheck_result = f"{nestcheck_path}/{run.label}_nestcheck.pickle"

                    with open(nestcheck_result, "wb") as file_nest:
                        pickle.dump(ns_run, file_nest)

                weights = np.exp(out["logwt"] - out["logz"][-1])
                nested_samples = DataFrame(out.samples, columns=run.sampling_keys)
                nested_samples["weights"] = weights
                nested_samples["log_likelihood"] = out.logl

                result = format_result(
                    run,
                    data_dump,
                    out,
                    weights,
                    nested_samples,
                    sampler_kwargs,
                    sampling_time,
                    rejection_sample_posterior=True
                )

                posterior = conversion.fill_from_fixed_priors(
                    result.posterior, run.priors
                )

                logger.info(
                    "Generating posterior from marginalized parameters for"
                    f" nsamples={len(posterior)}, POOL={pool.size}"
                )
                #fill_args = [
                #    (ii, row, run.likelihood) for ii, row in posterior.iterrows()
                #]
                #samples = pool.map(fill_sample, fill_args)
                posterior, _ = run.likelihood.parameter_conversion(
                    posterior,
                )

                result.posterior = conversion._generate_all_cbc_parameters(
                    posterior,
                    run.likelihood.GWLikelihood.waveform_generator.waveform_arguments,
                    conversion.convert_to_lal_binary_neutron_star_parameters,
                )

                logger.debug(
                    "Updating prior to the actual prior (undoing marginalization)"
                )
                for par, name in zip(
                    ["distance", "phase", "time"],
                    ["luminosity_distance", "phase", "geocent_time"],
                ):
                    if getattr(run.likelihood, f"{par}_marginalization", False):
                        run.priors[name] = run.likelihood.priors[name]
                result.priors = run.priors

                result.posterior = result.posterior.applymap(
                    lambda x: x[0] if isinstance(x, list) else x
                )
                result.posterior = result.posterior.select_dtypes([np.number])
                logger.info(
                    f"Saving result to {run.outdir}/{run.label}_result.{result_format}"
                )
                if result_format != "json":  # json is saved by default
                    result.save_to_file(extension="json")
                result.save_to_file(extension=result_format)
                print(
                    f"Sampling time = {datetime.timedelta(seconds=result.sampling_time)}s"
                )
                print(f"Number of lnl calls = {result.num_likelihood_evaluations}")
                print(result)
                if no_plot is False:
                    result.plot_corner()

        else:
            exit_reason = -1
        return exit_reason


def main_nmma():
    """
    nmma_analysis entrypoint.

    This function is a wrapper around analysis_runner(),
    giving it a command line interface.
    """
    cli_args = get_cli_args()

    # Parse command line arguments
    analysis_parser = create_nmma_analysis_parser(sampler="dynesty")
    input_args = parse_analysis_args(analysis_parser, cli_args=cli_args)

    # Run the analysis
    analysis_runner(**vars(input_args), inference_favour='nmma')


def main_nmma_gw():
    """
    nmma_analysis entrypoint.

    This function is a wrapper around analysis_runner(),
    giving it a command line interface.
    """
    cli_args = get_cli_args()

    # Parse command line arguments
    analysis_parser = create_nmma_gw_analysis_parser(sampler="dynesty")
    input_args = parse_analysis_args(analysis_parser, cli_args=cli_args)

    # Run the analysis
    analysis_runner(**vars(input_args), inference_favour='nmma_gw')
