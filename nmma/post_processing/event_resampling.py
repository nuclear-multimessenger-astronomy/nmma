from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_total_mass
import numpy as np
from arviz import hdi
from tqdm import tqdm


def reweight_to_flat_mass_prior(df):
    total_mass = chirp_mass_and_mass_ratio_to_total_mass(df.chirp_mass, df.mass_ratio)
    m1 = total_mass / (1. + df.mass_ratio)
    jacobian = m1 * m1 / df.chirp_mass
    df_new = df.sample(frac=0.3, weights=jacobian)
    return df_new


def find_spread_from_resampling(resampling_method, cumprod, prior_dist, post_samplesize, cred_interval):
    med, uplim, lowlim = [], [], []
    for weight in cumprod:
        samples = resampling_method(prior_dist, weight, post_samplesize)
        # calculate the posterior distribution using the prior samples
        # and the weighting that we previously calculated
        samples = resampling_method(prior_dist, weight, post_samplesize)

        # calculate the median and append it to the list
        med.append(np.median(samples))
        # calculate the 95% credible interval
        cred_range = hdi(samples, hdi_prob=cred_interval)
        # append the bound to the list
        uplim.append(cred_range[1])
        lowlim.append(cred_range[0])
    return np.array(med), np.array(uplim), np.array(lowlim)


def estimate_observable_trend(resampling_method, prior_dist, posterior_probs, index, args, cumprod_method, *extra_cumprod_args):
    obs_med, obs_uplim, obs_lowlim =  [], [], []
    np.random.seed(args.seed) 
    for _ in tqdm(range(args.N_reordering)):
        # randomly shuffle the ordering of the index to mimic different ordering realisation
        np.random.shuffle(index)
        prob_cumprod = cumprod_method(posterior_probs, index, *extra_cumprod_args)
        observable_spread = find_spread_from_resampling(resampling_method, prob_cumprod, prior_dist, args.N_posterior_samples, args.cred_interval)
        for obs_sub_list, spread_estimate in zip((obs_med, obs_uplim, obs_lowlim), observable_spread):
                obs_sub_list.append(spread_estimate)
        
    return [np.median(sub_array, axis=0) for  sub_array in [obs_med, obs_uplim, obs_lowlim] ]