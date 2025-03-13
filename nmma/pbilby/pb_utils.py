### parallel_bilby_utils that we use without buying in legacy issues
import os
import sys
import dill
from time import time
import numpy as np
from pandas import DataFrame
from bilby.core.utils import logger
from bilby.core.result import rejection_sample
from bilby.core.sampler.dynesty import dynesty_stats_plot



import dynesty.plotting as dyplot 
from dynesty.utils import get_print_fn_args

import matplotlib.pyplot as plt


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

def write_sample_dump(sampler, samples_file, search_parameter_keys):
    """Writes a checkpoint file

    Parameters
    ----------
    sampler: dynesty.NestedSampler
        The sampler object itself
    """

    ln_weights = sampler.saved_run.D["logwt"] - sampler.saved_run.D["logz"][-1]
    weights = np.exp(ln_weights)
    samples = rejection_sample(
        np.array(sampler.saved_run.D["v"]), weights
    )
    nsamples = len(samples)

    # If we don't have enough samples, don't dump them
    if nsamples < 100:
        return

    logger.info(f"Writing {nsamples} current samples to {samples_file}")
    df = DataFrame(samples, columns=search_parameter_keys)
    df.to_csv(samples_file, index=False, header=True, sep=" ")


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
            dill.dump(sampler, file)
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

