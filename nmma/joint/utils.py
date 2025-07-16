import os
import json
import numpy as np
import pandas as pd
from bilby.core.utils import decode_bilby_json

def read_injection_file(args):
    if isinstance(args.injection, str):
        args.injection_file = args.injection
    if not args.injection_file.endswith('.json'):
        args.injection_file = os.path.join(args.outdir, f"{args.injection_file}.json")
    with open(args.injection_file, "r") as f:
        injection_dict = json.load(f, object_hook=decode_bilby_json)
    return injection_dict["injections"]

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


def fetch_bestfit(args):
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