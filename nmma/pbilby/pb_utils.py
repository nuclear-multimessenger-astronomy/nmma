### parallel_bilby_utils that we use without buying in legacy issues
import os
from io import BufferedWriter
import functools
import sys
import dill
from time import time
import numpy as np
from pandas import DataFrame
from bilby.core.utils import logger
from bilby.core.result import  Result
from bilby.core.sampler.dynesty import dynesty_stats_plot



import dynesty.plotting as dyplot 
from dynesty.utils import get_print_fn_args, resample_equal

import matplotlib.pyplot as plt

def time_storage(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} took {time()-start}s")
        return result

    return wrapper



def stdout_sampling_log(**kwargs):
    """Logs will look like:
    #:282|eff(%):26.406|logl*:-inf<-160.2<inf|logz:-165.5+/-0.1|dlogz:1038.1>0.1

    Adapted from dynesty
    https://github.com/joshspeagle/dynesty/blob/bb1c5d5f9504c9c3bbeffeeba28ce28806b42273/py/dynesty/utils.py#L349
    """
    niter, short_str, mid_str, long_str = get_print_fn_args(**kwargs)
    custom_str = [f"#: {niter:d}"] + mid_str
    custom_str = "|".join(custom_str).replace(" ", "")
    sys.stdout.write("\033[K" + custom_str + "\r")
    sys.stdout.flush()

@time_storage
def write_sample_dump(sampler, samples_file, search_parameter_keys):
    """Writes a checkpoint file """
    weights = np.exp(sampler.saved_run.D["logwt"] - sampler.saved_run.D["logz"][-1])
    samples, keep = rejection_sample(sampler.saved_run.D["v"], weights)

    logger.info(f"Writing {np.sum(keep)} current samples to {samples_file}")
    df = DataFrame(samples, columns=search_parameter_keys)
    if samples_file.endswith(".dat"):
        df.to_csv(samples_file, index=False, header=True, sep=" ")
    else:
        df.to_parquet(samples_file, index=False)

@time_storage
def plot_current_state(sampler, search_parameter_keys, outdir, label):
    labels = [label.replace("_", " ") for label in search_parameter_keys]
    try:
        filename = f"{outdir}/{label}_checkpoint_trace.png"
        fig = dyplot.traceplot(sampler.results, labels=labels)[0]
        fig.tight_layout()
        fig.savefig(filename)
    except (
        AssertionError,
        RuntimeError,
        np.linalg.linalg.LinAlgError,
        ValueError,
    ) as e:
        logger.warning(e)
        logger.warning("Failed to create dynesty state plot at checkpoint")
    finally:
        plt.close("all")
    try:
        filename = f"{outdir}/{label}_checkpoint_run.png"
        fig, axs = dyplot.runplot(sampler.results, mark_final_live=False)
        fig.tight_layout()
        plt.savefig(filename)
    except (RuntimeError, np.linalg.linalg.LinAlgError, ValueError) as e:
        logger.warning(e)
        logger.warning("Failed to create dynesty run plot at checkpoint")
    finally:
        plt.close("all")
    try:
        filename = f"{outdir}/{label}_checkpoint_stats.png"
        fig, _ = dynesty_stats_plot(sampler)
        fig.tight_layout()
        plt.savefig(filename)
    except (RuntimeError, ValueError) as e:
        logger.warning(e)
        logger.warning("Failed to create dynesty stats plot at checkpoint")
    finally:
        plt.close("all")


@time_storage
def write_current_state(sampler, resume_file, sampling_time):
    """Writes a checkpoint file

    Parameters
    ----------
    sampler: dynesty.NestedSampler
        The sampler object itself
    resume_file: str
        The name of the resume/checkpoint file to use
    sampling_time: float
        The total sampling time in seconds
    """
    print("")
    try:
        seconds = time() - os.path.getmtime(resume_file)
        m, s = divmod(seconds, 60)
        strtime = f"{m:02.0f}m {s:02.0f}s"

        logger.info(
            "Start checkpoint writing" + f" (last checkpoint {strtime} ago)"
        )
    except FileNotFoundError:
        logger.info("Start checkpoint writing" + " (no previous checkpoint)")

    sampler.kwargs["sampling_time"] = sampling_time

    # Get random state and package it into the resume object
    sampler.kwargs["random_state"] = sampler.rstate.bit_generator.state

    if dill.pickles(sampler):
        temp_filename = resume_file + ".temp"
        with open(temp_filename, "wb") as file:
            with BufferedWriter(file) as buffer:
                dill.dump(sampler, buffer, protocol=dill.HIGHEST_PROTOCOL)
        os.rename(temp_filename, resume_file)

        logger.info(f"Written checkpoint file {resume_file}")
    else:
        logger.warning("Cannot write pickle resume file!")

    # Delete the random state so that the object is unchanged
    del sampler.kwargs["random_state"]


def read_saved_state(resume_file):
    """
    Read a saved state of the sampler to disk.

    The required information to reconstruct the state of the run is read from a
    pickle file.

    Parameters
    ----------
    resume_file: str
        The path to the resume file to read

    Returns
    -------
    sampler: dynesty.NestedSampler
        If a resume file exists and was successfully read, the nested sampler
        instance updated with the values stored to disk. If unavailable,
        returns False
    sampling_time: int
        The sampling time from previous runs
    """

    if os.path.isfile(resume_file):
        logger.info(f"Reading resume file {resume_file}")
        with open(resume_file, "rb") as file:
            sampler = dill.load(file)
            if sampler.added_live:
                sampler._remove_live_points()

            # Create random number generator and restore state
            # from file, then remove it from kwargs because it
            # is not useful after the generator has been cycled
            sampler.rstate = np.random.Generator(np.random.PCG64())
            sampler.rstate.bit_generator.state = sampler.kwargs.pop("random_state")

            sampling_time = sampler.kwargs.pop("sampling_time")
        return sampler, sampling_time
    else:
        logger.info(f"Resume file {resume_file} does not exist.")
        return False, 0


def format_result(
    run,
    worker_run,
    data_dump,
    out,
    nested_samples,
    sampler_kwargs,
    sampling_time,
    rejection_sample_posterior=True,
):
    """
    Packs the variables from the run into a bilby result object

    Parameters
    ----------
    run: MainRun
    worker_run: WorkerRun
    data_dump: str
        Path to the *_data_dump.pickle file
    out: dynesty.results.Results
        Results from the dynesty sampler
    nested_samples: pandas.core.frame.DataFrame
        DataFrame of the weights and likelihoods
    sampler_kwargs: dict
        Dictionary of keyword arguments for the sampler
    sampling_time: float
        Time in seconds spent sampling
    rejection_sample_posterior: bool
        Whether to generate the posterior samples by rejection sampling the
        nested samples or resampling with replacement

    Returns
    -------
    result: bilby.core.result.Result
        result object with values written into its attributes
    """

    result = Result(
        label=run.label, outdir=run.outdir, search_parameter_keys=run.sampling_keys
    )
    result.priors = worker_run.priors
    result.nested_samples = nested_samples
    result.meta_data = worker_run.data_dump["meta_data"]
    result.meta_data["command_line_args"]["sampler"] = "parallel_bilby"
    result.meta_data["data_dump"] = data_dump
    result.meta_data["likelihood"] = worker_run.likelihood.meta_data
    result.meta_data["sampler_kwargs"] = run.init_sampler_kwargs
    result.meta_data["run_sampler_kwargs"] = sampler_kwargs
    result.meta_data["injection_parameters"] = worker_run.injection_parameters
    result.injection_parameters = worker_run.injection_parameters

    weights = np.exp(out["logwt"] - out["logz"][-1])
    if rejection_sample_posterior:
        result.samples, keep = rejection_sample(out.samples, weights)
        result.log_likelihood_evaluations = out.logl[keep]
        logger.info(
            f"Rejection sampling nested samples to obtain {sum(keep)} posterior samples"
        )
    else:
        result.samples = resample_equal(out.samples, weights)
        result.log_likelihood_evaluations = reorder_loglikelihoods(
            unsorted_loglikelihoods=out.logl,
            unsorted_samples=out.samples,
            sorted_samples=result.samples,
        )
        logger.info("Resampling nested samples to posterior samples in place.")

    result.log_evidence = out.logz[-1] + worker_run.likelihood.noise_log_likelihood()
    result.log_evidence_err = out.logzerr[-1]
    result.log_bayes_factor = result.log_evidence - result.log_noise_evidence
    result.sampling_time = sampling_time
    result.num_likelihood_evaluations = np.sum(out.ncall)

    result.samples_to_posterior(likelihood=worker_run.likelihood, priors=result.priors)
    return result

def rejection_sample(posterior, weights, seed =42):
    rng = np.random.default_rng(seed)
    keep = (weights > rng.uniform(0, max(weights), weights.shape))
    return np.array(posterior)[keep], keep


def reorder_loglikelihoods(unsorted_loglikelihoods, unsorted_samples, sorted_samples):
    """Reorders the stored log-likelihood after they have been reweighted

    This creates a sorting index by matching the reweights `result.samples`
    against the raw samples, then uses this index to sort the
    loglikelihoods

    Parameters
    ----------
    sorted_samples, unsorted_samples: array-like
        Sorted and unsorted values of the samples. These should be of the
        same shape and contain the same sample values, but in different
        orders
    unsorted_loglikelihoods: array-like
        The loglikelihoods corresponding to the unsorted_samples

    Returns
    -------
    sorted_loglikelihoods: array-like
        The loglikelihoods reordered to match that of the sorted_samples


    """

    idxs = []
    for ii in range(len(unsorted_loglikelihoods)):
        idx = np.where(np.all(sorted_samples[ii] == unsorted_samples, axis=1))[0]
        if len(idx) > 1:
            print(
                "Multiple likelihood matches found between sorted and "
                "unsorted samples. Taking the first match."
            )
        idxs.append(idx[0])
    return unsorted_loglikelihoods[idxs]