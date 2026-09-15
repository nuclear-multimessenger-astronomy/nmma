import numpy as np
import pandas as pd
import scipy.stats
from tqdm import tqdm

from ..core.constants import c_kms
from ..core.conversion import reweight_to_flat_mass_prior
from ..core.parsing import nmma_base_parsing
from ..core.utils import read_injection_file
from .parser import Hubble_parser
from .resampling import find_spread_from_resampling


def H0_resampling(prior_dist, weight, post_samplesize):
    """Resample from a weighted KDE built on prior_dist.

    Builds a Gaussian KDE over `prior_dist` weighted by `weight`
    (interpreted as an unnormalized posterior weight per prior sample,
    i.e. importance-sampling reweighting), then draws fresh samples
    from it.

    Parameters
    ----------
    prior_dist : array_like
        Prior H0 samples to reweight (the proposal distribution).
    weight : array_like
        Same length as `prior_dist`; typically exp(cumulative log
        posterior) from `get_cumprod_rowwise`.
    post_samplesize : int
        Number of samples to draw from the reweighted KDE.

    Returns
    -------
    numpy.ndarray
        `post_samplesize` samples approximating the reweighted
        posterior.
    """

    kde = scipy.stats.gaussian_kde(prior_dist, weights=weight)
    return kde.resample(size=post_samplesize)[0]


def generate_logprob(probs, H0sample, index):
    """Incrementally combine per-event log-posteriors over H0, in a given order.

    For each event in `index` (in order), adds that event's log-pdf at
    `H0sample` to a running total, renormalizes (log-sum-exp) after
    each step, and records the running combined log-posterior. Every
    event after the first also gets a `+3*log(H0sample)` correction,
    undoing the H0^-3 selection effect introduced when each individual
    event's H0 samples were constructed from a uniform-in-volume
    distance prior (see `load_in_posteriors`).

    Parameters
    ----------
    probs : dict
        Maps event index -> an object with a `.logpdf(H0sample)`
        method (e.g. a `scipy.stats.gaussian_kde`).
    H0sample : array_like
        H0 grid/samples to evaluate log-pdfs on.
    index : array_like of int
        Event keys into `probs`, in the order they should be combined.

    Returns
    -------
    numpy.ndarray
        Shape (len(index), len(H0sample)): row k is the renormalized,
        combined log-posterior after including the first k+1 events
        (in `index` order).
    """

    log_prob_list = []
    logprob_combined = np.zeros_like(H0sample)
    for idx, i in enumerate(index):
        logprob_combined += probs[i].logpdf(H0sample)
        if idx != 0:
            logprob_combined += 3 * np.log(H0sample)
        logprob_combined -= scipy.special.logsumexp(logprob_combined)
        log_prob_list.append(logprob_combined)

    return np.array(log_prob_list)


def get_cumprod_rowwise(logprob):
    """Exponentiate each row of a log-probability array back to probability space.

    Parameters
    ----------
    logprob : array_like
        2D array of log-probabilities (e.g. from `generate_logprob`).

    Returns
    -------
    numpy.ndarray
        Same shape as `logprob`, exponentiated row-wise.
    """

    return np.array([np.exp(row) for row in logprob])


def generate_cumprods(gw_prob, em_prob, H0sample, index):
    """Combine GW-only, EM-only, and joint GW+EM running H0 posteriors.

    Parameters
    ----------
    gw_prob : dict
        Per-event GW H0 log-pdf objects (see `generate_logprob`).
    em_prob : dict
        Per-event EM H0 log-pdf objects.
    H0sample : array_like
        H0 grid/samples to evaluate on.
    index : array_like of int
        Event ordering to combine in.

    Returns
    -------
    list of numpy.ndarray
        [gw_cumprod, em_cumprod, total_cumprod], each shape
        (len(index), len(H0sample)) in probability (not log) space,
        one row per "after including k+1 events" checkpoint.
    """

    gw_logprobs = generate_logprob(gw_prob, H0sample, index)
    em_logprobs = generate_logprob(em_prob, H0sample, index)
    total_logprob = gw_logprobs + em_logprobs
    total_logprob[0, :] = total_logprob[0, :] + 3 * np.log(H0sample)
    total_logprobs = np.array(
        [logprob - scipy.special.logsumexp(logprob) for logprob in total_logprob]
    )

    return [
        get_cumprod_rowwise(logprob)
        for logprob in (gw_logprobs, em_logprobs, total_logprobs)
    ]


def H0_means_from_probs(gw_prob, em_prob, H0_sample, args, idx):
    """Estimate the running median/credible-interval H0 trend, averaged over random event orderings.

    Repeats `args.N_reordering` times: shuffles `idx`, builds the
    running GW/EM/combined posteriors via `generate_cumprods`, and for
    each "after k events" checkpoint resamples via `H0_resampling` to
    get a median and `args.cred_interval` credible interval. Returns
    the median (over reorderings) of each of these per-checkpoint
    estimates.

    Note: relies on `cumprodlist.append(cumprod)` where `cumprod` is a
    2D array; `find_spread_from_resampling`/`H0_resampling` expect one
    1D row at a time, so this currently raises `ValueError` from
    `scipy.stats.gaussian_kde` before completing even one reordering
    (see review notes).

    Parameters
    ----------
    gw_prob, em_prob : dict
        Per-event H0 log-pdf objects, keyed by event index.
    H0_sample : array_like
        Prior/proposal H0 samples used for resampling.
    args : argparse.Namespace
        Needs `N_reordering`, `N_posterior_samples`, `cred_interval`,
        `rng` (a `numpy.random.Generator`, used to shuffle `idx`).
    idx : array_like of int
        Event indices to combine (shuffled in place each reordering).

    Returns
    -------
    list of list of numpy.ndarray
        [[GW_med, GW_uplim, GW_lowlim], [EM_med, EM_uplim, EM_lowlim],
        [total_med, total_uplim, total_lowlim]], each array of length
        len(idx) (one value per "after k events" checkpoint).
    """

    H0_med_tot = []
    H0_uplim_tot = []
    H0_lowlim_tot = []

    H0_med_GW = []
    H0_uplim_GW = []
    H0_lowlim_GW = []

    H0_med_EM = []
    H0_uplim_EM = []
    H0_lowlim_EM = []
    gw_H0_estimates = [H0_med_GW, H0_uplim_GW, H0_lowlim_GW]
    em_H0_estimates = [H0_med_EM, H0_uplim_EM, H0_lowlim_EM]
    tot_H0_estimates = [H0_med_tot, H0_uplim_tot, H0_lowlim_tot]
    all_estimates = [gw_H0_estimates, em_H0_estimates, tot_H0_estimates]

    for _ in tqdm(range(args.N_reordering)):
        # randomly shuffle the ordering of the index to mimic different ordering realisation
        args.rng.shuffle(idx)
        #  generate cumulative probability product
        gw_cumprods, em_cumprods, total_cumprods = ([], [], [])
        cumprods = (gw_cumprods, em_cumprods, total_cumprods)
        for cumprod, cumprodlist in zip(
            generate_cumprods(gw_prob, em_prob, H0_sample, idx), cumprods
        ):
            cumprodlist.append(cumprod)
        ### FIXME: H0_means_from_probs uses .append() where it needs .extend(); generate_cumprods returns three 2D arrays (shape (N_events, len(H0sample)) — one row per "after including N events" checkpoint). .append(cumprod) wraps each entire 2D array as a single list element, instead of .extend(cumprod), which would add each row as its own element.

        for cumprod, estimates_list in zip(cumprods, all_estimates):
            for sub_list, spread_estimate in zip(
                estimates_list,
                find_spread_from_resampling(
                    H0_resampling,
                    cumprod,
                    H0_sample,
                    args.N_posterior_samples,
                    args.cred_interval,
                ),
            ):
                sub_list.append(spread_estimate)

    return [
        [np.median(sub_list, axis=0) for sub_list in estimate]
        for estimate in all_estimates
    ]


def load_in_posteriors(injection_df, event_to_loop, args):
    """Load per-event GW/EM distance posteriors and convert them to H0 posteriors.

    For each event in `event_to_loop`, loads
    '{args.EMsamples}/posterior_samples_{i}.dat' and
    '{args.GWsamples}/posterior_samples_{i}.dat' (skipping missing
    files), reweights the GW samples to flat-in-component-mass,
    optionally rejects events whose GW distance posterior is
    inconsistent with the injected distance (`args.p_value_threshold`,
    a two-sided p-value on `distance_injected`'s percentile in the GW
    posterior), converts luminosity distance to H0 samples assuming
    the true redshift is centered on the injected value with 1e-3
    scatter, and builds a Gaussian KDE H0 posterior per event (EM
    reweighted by distance^2 to undo an assumed non-uniform-in-volume
    EM distance prior; GW left as-is).

    Parameters
    ----------
    injection_df : pandas.DataFrame
        Must have a `luminosity_distance` column indexed by event id.
    event_to_loop : iterable of int
        Event indices to attempt to load.
    args : argparse.Namespace
        Needs `inject_Hubble`, `EMsamples`, `GWsamples`, `rng`, and
        optionally `p_value_threshold`.

    Returns
    -------
    tuple of dict
        (probs_EM, probs_GW): event index -> `scipy.stats.gaussian_kde`
        H0 posterior, for whichever events had both files and (if
        requested) passed the p-value cut.
    """

    # load the injected Hubble constant
    H0_true = args.inject_Hubble

    # get the combined posterior
    probs_EM = dict()
    probs_GW = dict()
    p_values = []
    for i in event_to_loop:
        try:
            samplesEM = pd.read_csv(
                "{0}/posterior_samples_{1}.dat".format(args.EMsamples, i),
                header=0,
                delimiter=" ",
            )
            samplesGW = pd.read_csv(
                "{0}/posterior_samples_{1}.dat".format(args.GWsamples, i),
                header=0,
                delimiter=" ",
            )
        except IOError:  # this is designed for running on incomplete dataset
            continue

        distanceEM = samplesEM.luminosity_distance.to_numpy()
        # resample GW to flat in component mass
        distanceGW = reweight_to_flat_mass_prior(
            samplesGW
        ).luminosity_distance.to_numpy()

        # Have the true redshift centered be the center of the posterior
        distance_injected = injection_df.luminosity_distance[i]
        z_true = H0_true * distance_injected / c_kms

        if args.p_value_threshold:
            # check if the GW posterior is good
            p_value = (
                scipy.stats.percentileofscore(distanceGW, distance_injected) / 100
            )  # as it function is returning percentage, we divide it by 100
            p_value = 2 * min(p_value, 1 - p_value)
            p_values.append(p_value)
            # remove bad injections
            # event 10 and 34 are two unconverged GW runs
            if p_value < args.p_value_threshold:
                continue

        # generate redshift samples and Hubble samples
        redshift_samples = args.rng.normal(float(z_true), 1e-3, size=len(distanceEM))
        H0_samples_EM = (
            redshift_samples * c_kms / distanceEM
        )  # Hubble samples in km s^-1 Mpc^-1

        # generate the posterior on H0 with a gaussian_kde with weighting back to the uniform in volume
        # the reason for reweighting back to uniform in volume (p(d) ~ d^2) is because the selection effect
        # for such prior is known (N(H0) ~ H0^-3), which makes the further analysis easier
        # if the EM analysis is already using a uniform-in-volume prior for distance, this line is to be commented out
        probs_EM[i] = scipy.stats.gaussian_kde(
            H0_samples_EM, weights=distanceEM * distanceEM
        )
        # now do it for GW
        redshift_samples = args.rng.normal(float(z_true), 1e-3, size=len(distanceGW))
        H0_samples_GW = redshift_samples * c_kms / distanceGW
        probs_GW[i] = scipy.stats.gaussian_kde(H0_samples_GW)

    return probs_EM, probs_GW


def main():
    """CLI entrypoint: estimate the GW+EM H0 trend across injected events and write it to disk.

    Parses via `Hubble_parser`, loads the event list (from
    `--detections-file` or `range(args.Nevent)`) and injection table,
    builds per-event H0 posteriors via `load_in_posteriors`, runs
    `H0_means_from_probs` over a flat H0 prior in [5, 120], and writes
    the per-event-count GW/EM/total median and asymmetric error trend
    to 'GW_EM_H0_trend_{args.output_label}.dat'.
    """

    args = nmma_base_parsing(Hubble_parser)
    # load the detectable events
    try:
        if args.detections_file:
            event_to_loop = np.loadtxt(args.detections_file).astype(int)
        else:
            event_to_loop = np.arange(args.Nevent)
    except:
        raise ValueError("Input for detectable or number of injection is required")

    # read the injection table
    injection_df = read_injection_file(args)
    # setup the seed
    rng = np.random.default_rng(args.seed)
    args.rng = rng
    ##get posterios
    probs_EM, probs_GW = load_in_posteriors(injection_df, event_to_loop, args)

    # preset a H0 prior samples to be used for further resampling
    H0_prior_samples = rng.uniform(5, 120, size=args.N_prior_samples)

    # get the list of index with result
    index = list(probs_GW.keys())
    index = np.array(index)

    ###main resampling routine
    GW_H0_estimates, EM_H0_estimates, total_H0_estimates = H0_means_from_probs(
        probs_GW, probs_EM, H0_prior_samples, args, index
    )

    # output the result using pandas
    df_dict = dict(
        GW_med=GW_H0_estimates[0],
        GW_uperr=GW_H0_estimates[1] - GW_H0_estimates[0],
        GW_lowerr=GW_H0_estimates[0] - GW_H0_estimates[2],
        EM_med=EM_H0_estimates[0],
        EM_uperr=EM_H0_estimates[1] - EM_H0_estimates[0],
        EM_lowerr=EM_H0_estimates[0] - EM_H0_estimates[2],
        total_med=total_H0_estimates[0],
        total_uperr=total_H0_estimates[1] - total_H0_estimates[0],
        total_lowerr=total_H0_estimates[0] - total_H0_estimates[2],
    )
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv("GW_EM_H0_trend_{0}.dat".format(args.output_label), index=False, sep=" ")
