import numpy as np
import pandas as pd
import scipy.stats
import scipy.constants
import arviz
import bilby
from tqdm import tqdm
import json
import argparse


def reweight_to_flat_mass_prior(df):
    total_mass = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(df.chirp_mass, df.mass_ratio)
    m1 = total_mass / (1. + df.mass_ratio)
    jacobian = m1 * m1 / df.chirp_mass
    df_new = df.sample(frac=0.3, weights=jacobian)
    return df_new


def main():

    parser = argparse.ArgumentParser(description="Calculate the combination and seperate trend of estimated Hubble constant with GW and EM input")
    parser.add_argument("--output-label", metavar="NAME", type=str, required=True)
    parser.add_argument("--GWsamples", metavar='PATH', type=str, required=True, help="Path to the GW posterior samples directory, the samples files are expected to be in the format of posterior_samples_{i}.dat")
    parser.add_argument("--EMsamples", metavar='PATH', type=str, required=True, help="Same as the GW samples but for EM samples")
    parser.add_argument("--injection", metavar='PATH', type=str, required=True)
    parser.add_argument("--inject-Hubble", metavar='H0', type=float, required=True)
    parser.add_argument("--detectable", type=str, required=False)
    parser.add_argument("--Nevent", type=int, required=False)
    parser.add_argument("--N-prior-samples", type=int, required=False, default=10000, help="Number of prior samples to be used for resampling (default: 10000)")
    parser.add_argument("--N-posterior-samples", type=int, required=False, default=6000, help="Number of posterior samples to be drawn during the resampling (default: 6000)")
    parser.add_argument("--N-reordering", type=int, default=100, help="Number of reodering realisation to be comsidered (default: 100)")
    parser.add_argument("--cred-interval", type=float, default=0.95, help="Credible interval to be calculated (default: 0.95)")
    parser.add_argument("--seed", type=int, required=False, default=42, help="Random seed (default: 42)")
    parser.add_argument("--p-value-threshold", type=float, required=False, help="p-value threshold used to remove badly recoved injections")
    args = parser.parse_args()

    assert args.detectable or args.Nevent, \
           "Input for detectable or number of injection is required"

    # load the injected Hubble constant
    H0_true = args.inject_Hubble

    # load the detectable events
    if args.detectable:
        detectable = np.loadtxt(args.detectable)

    # read the injection table
    with open(args.injection, 'r') as f:
        injection_dict = json.load(
            f, object_hook=bilby.core.utils.decode_bilby_json
        )
    injection_df = injection_dict["injections"]

    if args.detectable:
        event_to_loop = detectable
    else:
        event_to_loop = np.arange(args.Nevent)

    # setup the seed
    np.random.seed(args.seed)

    # get the combined posterior
    probs_EM = dict()
    probs_GW = dict()
    p_values = []
    for i in event_to_loop:
        try:
            samplesEM = pd.read_csv('{0}/posterior_samples_{1}.dat'.format(args.EMsamples, i), header=0, delimiter=' ')
            samplesGW = pd.read_csv('{0}/posterior_samples_{1}.dat'.format(args.GWsamples, i), header=0, delimiter=' ')
        except IOError:  # this is designed for running on incomplete dataset
            continue

        distanceEM = samplesEM.luminosity_distance.to_numpy()
        # resample GW to flat in component mass
        distanceGW = reweight_to_flat_mass_prior(samplesGW).luminosity_distance.to_numpy()

        # Have the true redshift centered be the center of the posterior
        distance_injected = injection_df.luminosity_distance[i]
        z_true = H0_true * distance_injected * (1e3 / scipy.constants.c)  # the factor 100 is to convert speed of light to km s^-1

        if args.p_value_threshold:
            # check if the GW posterior is good
            p_value = scipy.stats.percentileofscore(distanceGW, distance_injected) / 100  # as it function is returning percentage, we divide it by 100
            p_value = 2 * min(p_value, 1 - p_value)
            p_values.append(p_value)
            # remove bad injections
            # event 10 and 34 are two unconverged GW runs
            if p_value < args.p_value_threshold:
                continue

        # generate redshift samples and Hubble samples
        redshift_samples = np.random.normal(float(z_true), 1e-3, size=len(distanceEM))
        velocity_samples = redshift_samples * scipy.constants.c / 1e3  # velocity in km s^-1
        H0_samples_EM = velocity_samples / distanceEM  # Hubble samples in km s^-1 Mpc^-1

        # generate the posterior on H0 with a gaussian_kde with weighting back to the uniform in volume
        # the reason for reweighting back to uniform in volume (p(d) ~ d^2) is because the selection effect
        # for such prior is known (N(H0) ~ H0^-3), which makes the further analysis easier
        # if the EM analysis is already using a uniform-in-volume prior for distance, this line is to be commented out
        probs_EM[i] = scipy.stats.gaussian_kde(H0_samples_EM, weights=distanceEM * distanceEM)
        # now do it for GW
        redshift_samples = np.random.normal(float(z_true), 1e-3, size=len(distanceGW))
        velocity_samples = redshift_samples * scipy.constants.c / 1e3
        H0_samples_GW = velocity_samples / distanceGW
        probs_GW[i] = scipy.stats.gaussian_kde(H0_samples_GW)

    # preset a H0 prior samples to be used for further resampling
    H0_prior_samples = np.random.uniform(5, 120, size=args.N_prior_samples)

    # get the list of index with result
    index = list(probs_GW.keys())
    index = np.array(index)

    # empty array for the final result
    H0_med_total = []
    H0_uplim_total = []
    H0_lowlim_total = []

    H0_med_GW = []
    H0_uplim_GW = []
    H0_lowlim_GW = []

    H0_med_EM = []
    H0_uplim_EM = []
    H0_lowlim_EM = []

    for _ in tqdm(range(args.N_reordering)):
        # randomly shuffle the ordering of the index to mimic different ordering realisation
        np.random.shuffle(index)
        prob_cumprod_GW = []
        prob_cumprod_EM = []
        prob_cumprod_total = []
        for idx, i in enumerate(index):
            if idx == 0:
                # the calculation is done in the log-space to prevent both overflowing and underflowing
                logprob_combined_GW = 0.
                logprob_combined_EM = 0.
                logprob_combined_total = 0.

                logprob_combined_GW = probs_GW[i].logpdf(H0_prior_samples)
                logprob_combined_GW -= scipy.special.logsumexp(logprob_combined_GW)
                prob_cumprod_GW.append(np.exp(logprob_combined_GW))

                logprob_combined_EM = probs_EM[i].logpdf(H0_prior_samples)
                logprob_combined_EM -= scipy.special.logsumexp(logprob_combined_EM)
                prob_cumprod_EM.append(np.exp(logprob_combined_EM))

                # FIXME now the combination only consider distance
                # a better implantation should do the combination
                # on the distance-inclination plane

                # Although the selection correction is not needed for one event
                # with GW+EM it is effectively two events
                logprob_combined_total = logprob_combined_GW + logprob_combined_EM + 3 * np.log(H0_prior_samples)
                logprob_combined_total -= scipy.special.logsumexp(logprob_combined_total)
                prob_cumprod_total.append(np.exp(logprob_combined_total))
            else:
                # when combing events, we are multiplying the posterior distribution and divided by the selection correction
                # in general we also need to divide the prior on H0, but since we are using a flat prior on it (line 147), it can be ignored
                logprob_combined_GW = logprob_combined_GW + probs_GW[i].logpdf(H0_prior_samples)
                logprob_combined_GW = logprob_combined_GW + 3 * np.log(H0_prior_samples)
                logprob_combined_GW -= scipy.special.logsumexp(logprob_combined_GW)
                prob_cumprod_GW.append(np.exp(logprob_combined_GW))

                logprob_combined_EM = logprob_combined_EM + probs_EM[i].logpdf(H0_prior_samples)
                logprob_combined_EM = logprob_combined_EM + 3 * np.log(H0_prior_samples)
                logprob_combined_EM -= scipy.special.logsumexp(logprob_combined_EM)
                prob_cumprod_EM.append(np.exp(logprob_combined_EM))

                logprob_combined_total = logprob_combined_total + logprob_combined_GW + logprob_combined_EM
                logprob_combined_total -= scipy.special.logsumexp(logprob_combined_total)
                prob_cumprod_total.append(np.exp(logprob_combined_total))

        # cast the list back to array for easier element-wise calculation
        prob_cumprod_GW = np.array(prob_cumprod_GW)
        prob_cumprod_EM = np.array(prob_cumprod_EM)
        prob_cumprod_total = np.array(prob_cumprod_total)

        # the lists for result with GW, EM and GW+EM with increasing number of events
        H0_med_per_event_GW = []
        H0_uplim_per_event_GW = []
        H0_lowlim_per_event_GW = []

        H0_med_per_event_EM = []
        H0_uplim_per_event_EM = []
        H0_lowlim_per_event_EM = []

        H0_med_per_event_total = []
        H0_uplim_per_event_total = []
        H0_lowlim_per_event_total = []

        for i in range(len(prob_cumprod_GW)):
            # calculate the posterior distribution using the prior samples
            # and the weighting that we previously calculated
            GW_kde = scipy.stats.gaussian_kde(H0_prior_samples, weights=prob_cumprod_GW[i])
            # generate sample based on this distribution
            samples_GW = GW_kde.resample(size=args.N_posterior_samples)[0]
            # calculate the median and append it to the list
            H0_med_per_event_GW.append(np.median(samples_GW))
            # calculate the 95% credible interval
            cred_range = arviz.hdi(samples_GW, hdi_prob=args.cred_interval)
            # append the bound to the list
            H0_uplim_per_event_GW.append(cred_range[1])
            H0_lowlim_per_event_GW.append(cred_range[0])

            EM_kde = scipy.stats.gaussian_kde(H0_prior_samples, weights=prob_cumprod_EM[i])
            samples_EM = EM_kde.resample(size=args.N_posterior_samples)[0]
            H0_med_per_event_EM.append(np.median(samples_EM))
            cred_range = arviz.hdi(samples_EM, hdi_prob=args.cred_interval)
            H0_uplim_per_event_EM.append(cred_range[1])
            H0_lowlim_per_event_EM.append(cred_range[0])

            total_kde = scipy.stats.gaussian_kde(H0_prior_samples, weights=prob_cumprod_total[i])
            samples_total = total_kde.resample(size=args.N_posterior_samples)[0]
            H0_med_per_event_total.append(np.median(samples_total))
            cred_range = arviz.hdi(samples_total, hdi_prob=args.cred_interval)
            H0_uplim_per_event_total.append(cred_range[1])
            H0_lowlim_per_event_total.append(cred_range[0])

        # again, cast all the lists to array
        H0_med_per_event_GW = np.array(H0_med_per_event_GW)
        H0_uplim_per_event_GW = np.array(H0_uplim_per_event_GW)
        H0_lowlim_per_event_GW = np.array(H0_lowlim_per_event_GW)

        H0_med_per_event_EM = np.array(H0_med_per_event_EM)
        H0_uplim_per_event_EM = np.array(H0_uplim_per_event_EM)
        H0_lowlim_per_event_EM = np.array(H0_lowlim_per_event_EM)

        H0_med_per_event_total = np.array(H0_med_per_event_total)
        H0_uplim_per_event_total = np.array(H0_uplim_per_event_total)
        H0_lowlim_per_event_total = np.array(H0_lowlim_per_event_total)

        # append the trend for each realisation to the final result list
        H0_med_GW.append(H0_med_per_event_GW)
        H0_uplim_GW.append(H0_uplim_per_event_GW)
        H0_lowlim_GW.append(H0_lowlim_per_event_GW)

        H0_med_EM.append(H0_med_per_event_EM)
        H0_uplim_EM.append(H0_uplim_per_event_EM)
        H0_lowlim_EM.append(H0_lowlim_per_event_EM)

        H0_med_total.append(H0_med_per_event_total)
        H0_uplim_total.append(H0_uplim_per_event_total)
        H0_lowlim_total.append(H0_lowlim_per_event_total)

    # take the median behaviour across ordering realisation
    H0_med_GW_final = np.median(H0_med_GW, axis=0)
    H0_uplim_GW_final = np.median(H0_uplim_GW, axis=0)
    H0_lowlim_GW_final = np.median(H0_lowlim_GW, axis=0)

    H0_med_EM_final = np.median(H0_med_EM, axis=0)
    H0_uplim_EM_final = np.median(H0_uplim_EM, axis=0)
    H0_lowlim_EM_final = np.median(H0_lowlim_EM, axis=0)

    H0_med_total_final = np.median(H0_med_total, axis=0)
    H0_uplim_total_final = np.median(H0_uplim_total, axis=0)
    H0_lowlim_total_final = np.median(H0_lowlim_total, axis=0)

    # output the result using pandas
    df_dict = dict(GW_med=H0_med_GW_final,
                   GW_uperr=H0_uplim_GW_final - H0_med_GW_final,
                   GW_lowerr=H0_med_GW_final - H0_lowlim_GW_final,
                   EM_med=H0_med_EM_final,
                   EM_uperr=H0_uplim_EM_final - H0_med_EM_final,
                   EM_lowerr=H0_med_EM_final - H0_lowlim_EM_final,
                   total_med=H0_med_total_final,
                   total_uperr=H0_uplim_total_final - H0_med_total_final,
                   total_lowerr=H0_med_total_final - H0_lowlim_total_final)
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv('GW_EM_H0_trend_{0}.dat'.format(args.output_label), index=False, sep=' ')
