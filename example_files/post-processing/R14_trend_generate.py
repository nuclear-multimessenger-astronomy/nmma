import numpy as np
import pandas as pd
import scipy.stats
import scipy.interpolate
import arviz
import bilby
import os
from tqdm import tqdm


def reweight_to_flat_mass_prior(df):
    total_mass = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(df.chirp_mass, df.mass_ratio)
    m1 = total_mass / (1. + df.mass_ratio)
    jacobian = m1 * m1 / df.chirp_mass
    df_new = df.sample(frac=0.3, weights=jacobian, random_state=42)
    return df_new


# user specify input
pdet_Mmax_path = './example_files/eos/pdet_of_Mmax.dat'
EOS_prior_path = './example_files/eos/EOS_sorted_weight.dat'
EOSset_path = './example_files/eos/eos_sorted'
GWEMdata_path =  './output/GW_EMdata'
label = 'ZTF'
Neos = 5000
# load the detectable events
detectable = np.loadtxt('./example_files/csv_lightcurve/detectable.txt').astype(int)
# load the EOS prior
prior = np.loadtxt(EOS_prior_path)
prior = prior[:-1]  # to remove the duplicate at the last line added for it to work with pbilby

# get the R14, Mmax prior samples
R14_prior = []
Mmax_prior = []
for i in range(1, Neos + 1):
    m, r = np.loadtxt('{0}/{1}.dat'.format(EOSset_path, i), usecols=[1, 0], unpack=True)
    interp = scipy.interpolate.interp1d(m, r)
    R14_prior.append(interp(1.4))
    Mmax_prior.append(m[-1])
R14_prior = np.array(R14_prior)
Mmax_prior = np.array(Mmax_prior)

# load the pdet
Mmax, pdet = np.loadtxt(pdet_Mmax_path, usecols=[0, 1], unpack=True)
fit = scipy.interpolate.UnivariateSpline(Mmax, pdet, s=0.3)  # smooth interpolation
pdet_of_EOS = fit(Mmax_prior)  # calculate the selection effect correction for each EOS

# get the combined posterior
probs = dict()
for i in detectable:
    try:
        sample = pd.read_csv('{0}/{1}/posterior_samples.dat'.format(GWEMdata_path, i), header=0, delimiter=' ')
    except IOError:  # this is designed for running on incomplete dataset
        continue
    sample = reweight_to_flat_mass_prior(sample)
    # during the sampling, EOS is treated as a continous variable
    # we convert them back to integer
    EOSTrue = sample.EOS.to_numpy().astype(int) + 1
    # calulcate the posterio probability by counting the samples
    counts = []
    for j in range(1, Neos + 1):
        counts.append(len(np.where(EOSTrue == j)[0]))
    counts = np.array(counts)
    # normalization
    prob = counts / len(EOSTrue)
    probs[i] = prob

index = list(probs.keys())
index = np.array(index)
R14_med_total = []
R14_uplim_total = []
R14_lowlim_total = []
np.random.seed(42)
for _ in tqdm(range(1000)):
    # randomly shuffle the ordering of the index to mimic different ordering realisation
    np.random.shuffle(index)
    prob_cumprod = []
    for idx, i in enumerate(index):
        if idx == 0:
            prob_combined = probs[i]
            prob_cumprod.append(prob_combined)
        else:
            prob_combined = prob_combined * probs[i] / prior / pdet_of_EOS
            prob_combined /= np.sum(prob_combined)
            prob_cumprod.append(prob_combined)

    prob_cumprod = np.array(prob_cumprod)

    R14_med = []
    R14_uplim = []
    R14_lowlim = []
    for i in range(len(prob_cumprod)):
        # obtain the posterior samples using good old resampling
        samples = np.random.choice(R14_prior, p=prob_cumprod[i], size=10000, replace=True)
        # calculate the median and append it to the list
        R14_med.append(np.median(samples))
        # calculate the 95% credible interval
        range95 = arviz.hdi(samples, hdi_prob=0.95)
        # append the bound to the list
        R14_uplim.append(range95[1])
        R14_lowlim.append(range95[0])

    R14_med = np.array(R14_med)
    R14_uplim = np.array(R14_uplim)
    R14_lowlim = np.array(R14_lowlim)

    R14_med_total.append(R14_med)
    R14_uplim_total.append(R14_uplim)
    R14_lowlim_total.append(R14_lowlim)

R14_med = np.median(R14_med_total, axis=0)
R14_uplim = np.median(R14_uplim_total, axis=0)
R14_lowlim = np.median(R14_lowlim_total, axis=0)

df_dict = dict(R14_med=R14_med,
               R14_uperr=R14_uplim - R14_med,
               R14_lowerr=R14_med - R14_lowlim)
df = pd.DataFrame.from_dict(df_dict)
df.to_csv('./output/Figures/GW_EM_R14trend_{0}.dat'.format(label), index=False, sep=' ')
