import numpy as np
import pandas as pd
import scipy.stats
import scipy.interpolate
import arviz
import bilby
from tqdm import tqdm
import seaborn

import matplotlib
import matplotlib.pyplot as plt
from .event_resampling import (reweight_to_flat_mass_prior, find_spread_from_resampling,
                               estimate_observable_trend)
from .parser import R14_parser
from ..eos.eos_processing import load_macro_characteristics_from_tabulated_eos_set
matplotlib.use("agg")

c = seaborn.color_palette("colorblind")

fig_width_pt = 750.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = 0.9 * fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]
params = {
    "backend": "pdf",
    "axes.labelsize": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "text.usetex": True,
    "font.family": "Times New Roman",
    "figure.figsize": fig_size,
}
matplotlib.rcParams.update(params)


def R14_resampling(R14_prior, weights, post_samplesize):
    return np.random.choice(R14_prior, p=weights, size=post_samplesize, replace=True)

def load_in_posteriors(detectable, args):
    probs = dict()
    for i in detectable:
        try:
            sample = pd.read_csv(
                f"{args.GWEMsamples}/{i}/posterior_samples.dat",
                header=0,
                delimiter=" ",
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
        probs[i] = prob

    index = list(probs.keys())
    return probs, np.array(index)

def generate_EOS_cumprods(probs, index, EOS_prior, pdet_of_EOS):
    prob_cumprod = []
    prob_combined = EOS_prior*pdet_of_EOS
    for i in index:
        prob_combined = prob_combined * probs[i] / EOS_prior / pdet_of_EOS
        prob_combined /= np.sum(prob_combined)
        prob_cumprod.append(prob_combined)
    return prob_cumprod


def plot_R14_trend(args):
    data_GWEM = pd.read_csv(
        "{0}/GW_EM_R14trend_{1}.dat".format(args.outdir, args.label),
        header=0,
        delimiter=" ",
    )
    data_GW = pd.read_csv(
        "{0}/GW_R14trend.dat".format(args.gwR14trend), header=0, delimiter=" "
    )

    fig = plt.figure()
    fig.suptitle("Constrain EoS using EM + GW ", fontname="Times New Roman Bold")
    ax1 = plt.subplot2grid((4, 5), (0, 0), rowspan=3, colspan=4)
    ax2 = plt.subplot2grid((4, 5), (3, 0), colspan=4, sharex=ax1)
    ax1.set_xlim([0.5, len(data_GWEM) + 0.5])
    ax1.set_ylabel(r"$R_{1.4} \ [{\rm km}]$")
    ax2.set_ylabel(r"$\delta R_{1.4} / R_{1.4} \ [\%]$")
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.set_xticks(np.arange(1, len(data_GWEM) + 1, 2))
    ax2.set_xlabel("Events")

    axis_GW = np.arange(1, len(data_GW) + 1)
    axis_GWEM = np.arange(1, len(data_GWEM) + 1)

    ax1.errorbar(
        axis_GW,
        data_GW.R14_med,
        yerr=[data_GW.R14_lowerr, data_GW.R14_uperr],
        label="GW",
        color=c[3],
        fmt="o",
        capsize=5,
    )
    ax1.errorbar(
        axis_GWEM,
        data_GWEM.R14_med,
        yerr=[data_GWEM.R14_lowerr, data_GWEM.R14_uperr],
        label="GW+EM",
        color=c[0],
        fmt="o",
        capsize=5,
    )
    ax1.axhline(args.R14_true, linestyle="--", color=c[1], label="Injected value")
    ax1.legend()

    GW_mean_error = np.mean([data_GW.R14_lowerr, data_GW.R14_uperr], axis=0)
    GWEM_mean_error = np.mean([data_GWEM.R14_lowerr, data_GWEM.R14_uperr], axis=0)
    ax2.plot(axis_GW, GW_mean_error / data_GW.R14_med * 100, color=c[3], marker="o")
    ax2.plot(
        axis_GWEM, GWEM_mean_error / data_GWEM.R14_med * 100, color=c[0], marker="o"
    )
    ax2.set_yscale("log")
    ax2.axhline(10, color="grey", linestyle="--", alpha=0.5)
    ax2.axhline(5, color="grey", linestyle="--", alpha=0.5)
    ax2.axhline(1, color="grey", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    plt.savefig(
        "{0}/R14_trend_GW_EM_{1}.pdf".format(args.outdir, args.label),
        bbox_inches="tight",
    )

def main():
    parser = R14_parser()
    args = parser.parse_args()

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

    
    # get the combined posterior
    probs, index= load_in_posteriors(detectable, args)

    R14_med , R14_uplim, R14_lowlim = estimate_observable_trend(R14_resampling, R14_prior, probs, index, args, generate_EOS_cumprods, EOS_prior, pdet_of_EOS)

    df_dict = dict(
        R14_med=R14_med, R14_uperr=R14_uplim - R14_med, R14_lowerr=R14_med - R14_lowlim
    )
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(
        "{0}/GW_EM_R14trend_{1}.dat".format(args.outdir, args.label),
        index=False,
        sep=" ",
    )

    plot_R14_trend(args)


if __name__ == "__main__":
    main()
