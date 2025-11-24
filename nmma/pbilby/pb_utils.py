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
from bilby.core.sampler.dynesty import dynesty_stats_plot

from dynesty.plotting import traceplot, runplot 
from dynesty.utils import get_print_fn_args

from ..joint.utils import rejection_sample

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
def write_sample_dump(sampler, search_parameter_keys):
    """Writes a checkpoint file """
    samples_file= sampler.samples_file
    weights = np.exp(sampler.saved_run.D["logwt"] - sampler.saved_run.D["logz"][-1])
    samples, keep = rejection_sample(sampler.saved_run.D["v"], weights, sampler.rstate)

    logger.info(f"Writing {np.sum(keep)} current samples to {samples_file}")
    df = DataFrame(samples, columns=search_parameter_keys)
    if samples_file.endswith(".dat"):
        df.to_csv(samples_file, index=False, header=True, sep=" ")
    else:
        df.to_parquet(samples_file, index=False)

@time_storage
def plot_current_state(sampler, outdir, label):
    # labels = [label.replace("_", " ") for label in search_parameter_keys]
    for name, func in zip (
        ["trace", "run", "stats"],
        [traceplot, runplot, dynesty_stats_plot]):

        try: 
            fig, _ = func(sampler.results)
            fig.tight_layout()
            fig.savefig(f"{outdir}/{label}_checkpoint_{name}.png")

        except Exception as e:
            logger.warning(e)
            logger.warning(f"Failed to create dynesty {name} plot at checkpoint")
        finally:
            plt.close("all")


@time_storage
def write_current_state(sampler, sampling_time):
    """Writes a checkpoint file

    Parameters
    ----------
    sampler: dynesty.NestedSampler
        The sampler object itself
    sampling_time: float
        The total sampling time in seconds
    """
    print("")
    resume_file = sampler.resume_file
    pool = sampler.pool
    try:
        seconds = time() - os.path.getmtime(resume_file)
        m, s = divmod(seconds, 60)
        strtime = f"{m:02.0f}m {s:02.0f}s"

        logger.info(
            "Start checkpoint writing" + f" (last checkpoint {strtime} ago)"
        )
    except FileNotFoundError:
        logger.info("Start checkpoint writing" + " (no previous checkpoint)")


    if dill.pickles(sampler):
        # Temporarily remove the pool to avoid pickling issues
        sampler.pool = None
        sampler.mapper = map
        sampler.sampling_time = sampling_time
        temp_filename = resume_file + ".temp"
        with open(temp_filename, "wb") as file:
            with BufferedWriter(file) as buffer:
                dill.dump(sampler, buffer, protocol=dill.HIGHEST_PROTOCOL)
        os.rename(temp_filename, resume_file)
        logger.info(f"Written checkpoint file {resume_file}")
        sampler.pool = pool
        sampler.mapper = pool.map
    else:
        logger.warning("Cannot write pickle resume file!")



def read_saved_state(resume_file, pool):
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

            #reset pool
            sampler.nqueue = -1
            sampler.pool = pool
            sampler.queue_size = pool.size
            sampler.mapper = pool.map
            try:
                sampling_time = sampler.sampling_time
            except AttributeError:
                sampling_time = sampler.resume_kwargs["sampling_time"]
        return sampler, sampling_time
    else:
        logger.info(f"Resume file {resume_file} does not exist.")
        return False, 0


def checkpointing( run, sampler, sampling_time, checkpoint_plot=False):
    write_current_state(sampler, sampling_time )
    write_sample_dump(sampler, run.sampling_keys)
    if checkpoint_plot:
        plot_current_state(sampler, run.outdir, run.label)