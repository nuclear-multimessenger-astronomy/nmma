import os
import json
import h5py
import numpy as np
import pandas as pd
from argparse import Namespace
from bilby.core.utils import decode_bilby_json

def read_injection_file(file):
    #work for both file-str and Namespace
    if isinstance(file, Namespace):
        if isinstance(file.injection, str):
            file.injection_file = file.injection
        if not file.injection_file.endswith('.json'):
            file.injection_file = os.path.join(file.outdir, f"{file.injection_file}.json")
        inj_file = file.injection_file
    else:
        inj_file = file
    with open(inj_file, "r") as f:
        injection_dict = json.load(f, object_hook=decode_bilby_json)
    return injection_dict["injections"]

def get_posteriors(posterior_samples, outdir = None):
    """
    Load posterior samples from a file or DataFrame.
    """
    lc_model = None
    if isinstance(posterior_samples, str):
        if not os.path.isfile(posterior_samples):
            posterior_samples = os.path.join(outdir, posterior_samples)
        base, ext = os.path.splitext(posterior_samples)
        format_str = ext[1:].lower()
        if format_str in ['csv', 'txt', 'dat']:
            posterior_samples = pd.read_csv(posterior_samples, sep='\s+', header=0)
        elif format_str == 'json':
            with open(posterior_samples, 'r') as f:
                samples_dict = json.load(f, object_hook=decode_bilby_json)
            posterior_samples = samples_dict["posterior"]
        elif format_str == 'hdf5':
            with h5py.File(posterior_samples, 'r') as f:
                posterior_group = f['posterior']
                posterior_samples = pd.DataFrame({key: np.array(posterior_group[key]) for key in posterior_group.keys()})
        else:
            raise ValueError("Unsupported file format, must be csv, txt, dat, json or hdf5")
    return posterior_samples

def set_filename(basename, args, identifier=''):
    base, ext = os.path.splitext(basename)
    if not ext:
        ext = getattr(args, "extension", "json")
        return os.path.join(args.outdir, f"{basename}{identifier}.{ext}"
    )
    elif ext[1:] not in ["json", "csv", "dat"]:
        raise ValueError(f"Unsupported output file type: {ext}")
    elif os.path.dirname(basename)=='':
        return os.path.join(args.outdir, f"{base}{identifier}{ext}"
    )
    else:
        return  f"{base}{identifier}{ext}"


def read_bestfit_from_posterior(args):
    posterior_file = os.path.join(
        args.outdir, f"{args.label}_posterior_samples.dat"
    )
    posterior_samples = pd.read_csv(posterior_file, header=0, delimiter=" ")
    bestfit_idx = np.argmax(posterior_samples.log_likelihood.to_numpy())
    bestfit_params = posterior_samples.to_dict(orient="list")
    for key in bestfit_params.keys():
        bestfit_params[key] = bestfit_params[key][bestfit_idx]
    print(f"Best fit parameters: {str(bestfit_params)}\nBest fit index: {bestfit_idx}")
    bestfit_params["best_fit_index"] = int(bestfit_idx)
    return bestfit_params

def read_bestfit_from_json(bestfit_file_json, cols, verbose=False):
    df = pd.read_json(bestfit_file_json, typ="series")
    truths = np.array([df[col] for col in cols if col in df])
    if verbose:
        print("\nLoaded Bestfit:")
        print(f"Truths from bestfit: {truths}")
    return truths.flatten()

def rejection_sample(posterior, weights, rng):
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

def sig_lims(values, quantiles=None, sig_unc=2):
    "get limits to significant figures"
    if quantiles is None:
        quantiles = [0.16, 0.5, 0.84]
    q_low, q_mean, q_high = np.quantile(values, quantiles)
    low_err     = q_mean - q_low
    high_err    = q_high - q_mean
    ord_error   = sig_unc -1 - int(np.log10(min(low_err, high_err)))
    if ord_error>=0:
        fmt = f".{ord_error}f"
        return f"${{{q_mean:{fmt}}}}_{{-{low_err:{fmt}}}}^{{+{high_err:{fmt}}}}$"
    else:
        return f"${{{int(q_mean)}}}_{{-{int(low_err)}}}^{{+{int(high_err)}}}$"
