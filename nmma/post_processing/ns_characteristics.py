import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import scipy.interpolate
from .plotting_routines import plot_R14_trend
from .parser import R14_parser
from .resampling import find_spread_from_resampling
from ..joint.base_parsing import nmma_base_parsing
from ..eos.eos_processing import load_macro_characteristics_from_tabulated_eos_set
from ..joint.conversion import reweight_to_flat_mass_prior
# matplotlib.use("agg")




def estimate_observable_trend(prior_dist, posterior_probs, prior_prob, args):
    obs_med, obs_uplim, obs_lowlim =  [], [], []
    rng = np.random.default_rng(args.seed)
    def R14_resampling(R14_prior, weights, post_samplesize):
        return rng.choice(R14_prior, p=weights, size=post_samplesize, replace=True)
    
    for _ in tqdm(range(args.N_reordering)):
        # randomly shuffle the ordering of the index to mimic different ordering realisation
        random.shuffle(posterior_probs)
        prob_cumprod = generate_EOS_cumprods(posterior_probs, prior_prob)
        observable_spread = find_spread_from_resampling(R14_resampling, prob_cumprod, prior_dist, args.N_posterior_samples, args.cred_interval)
        for obs_sub_list, spread_estimate in zip((obs_med, obs_uplim, obs_lowlim), observable_spread):
                obs_sub_list.append(spread_estimate)
        
    return [np.median(sub_array, axis=0) for  sub_array in [obs_med, obs_uplim, obs_lowlim] ]

def load_in_posteriors(detectable, args):
    probs = []
    for i in detectable:
        try:
            sample = pd.read_csv(
                f"{args.GWEMsamples}/{i}/posterior_samples.dat", header=0, sep="\s+",
            )
        except IOError:  # this is designed for running on incomplete dataset
            continue
        sample = reweight_to_flat_mass_prior(sample)
        # during the sampling, EOS is treated as a continous variable
        # we convert them back to integer
        EOSTrue = sample.EOS.to_numpy().astype(int) + 1
        # calulcate the posterior probability by counting the samples
        counts = []
        for j in range(1, args.Neos + 1):
            counts.append(len(np.where(EOSTrue == j)[0]))
        counts = np.array(counts)
        # normalization
        prob = counts / len(EOSTrue)
        probs.append(prob)
    return probs

def generate_EOS_cumprods(probs, prior_prob):
    prob_cumprod = []
    prob_combined = prior_prob
    for prob in probs:
        prob_combined = prob_combined * prob/ prior_prob
        prob_combined /= np.sum(prob_combined)
        prob_cumprod.append(prob_combined)
    return prob_cumprod



def main():
    args = nmma_base_parsing(R14_parser)

    # load the detectable events
    detectable = np.loadtxt(args.detections_file, usecols=[0]).astype(int)

    # load the EOS prior
    EOS_prior = np.loadtxt(args.EOS_prior)
    # get the R14, Mmax prior samples
    Mmax_prior, R14_prior = load_macro_characteristics_from_tabulated_eos_set(args.EOSPath, args.Neos, 1.4)

    # load the pdet
    Mmax, pdet = np.loadtxt(args.pdet, usecols=[0, 1], unpack=True)
    fit = scipy.interpolate.UnivariateSpline(Mmax, pdet, s=0.3)  # smooth interpolation
    pdet_of_EOS = fit(Mmax_prior) #calculate the selection effect correction for each EOS
    
    prior_prob = EOS_prior * pdet_of_EOS
    
    # get the combined posteriors
    probs = load_in_posteriors(detectable, args)

    R14_med , R14_uplim, R14_lowlim = estimate_observable_trend(R14_prior, probs, prior_prob, args)

    df_dict = dict(
        R14_med=R14_med, R14_uperr=R14_uplim - R14_med, R14_lowerr=R14_med - R14_lowlim
    )
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(f"{args.outdir}/GW_EM_R14trend_{args.label}.dat", index=False, sep="\s+" )

    plot_R14_trend(args)


if __name__ == "__main__":
    main()
