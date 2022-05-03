import numpy as np
import pandas as pd
import scipy.stats
import scipy.interpolate
import arviz
import bilby
from tqdm import tqdm
import argparse
import seaborn

import matplotlib
import matplotlib.pyplot as plt

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


def reweight_to_flat_mass_prior(df):
    total_mass = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(
        df.chirp_mass, df.mass_ratio
    )
    m1 = total_mass / (1.0 + df.mass_ratio)
    jacobian = m1 * m1 / df.chirp_mass
    df_new = df.sample(frac=0.3, weights=jacobian, random_state=42)
    return df_new


def main():

    parser = argparse.ArgumentParser(
        description="Calculate the trend of estimated R14 with GW+EM input"
    )
    parser.add_argument("--outdir", metavar="PATH", type=str, required=True)
    parser.add_argument("--label", metavar="NAME", type=str, required=True)
    parser.add_argument(
        "--R14_true",
        type=float,
        default=11.55,
        help="The true value of Neutron stars's raduis (default:11.55)",
    )
    parser.add_argument(
        "--gwR14trend",
        metavar="PATH",
        type=str,
        required=True,
        help="Path to the R14trend GW  posterior samples directory, the  file are expected to be in the format of GW_R14trend.dat ",
    )
    parser.add_argument(
        "--GWEMsamples",
        metavar="PATH",
        type=str,
        required=True,
        help="Path to the GWEM posterior samples directory, the samples files are expected to be in the format of posterior_samples_{i}.dat",
    )
    parser.add_argument("-d", "--detections-file", type=str, required=False)
    parser.add_argument("--Neos", type=int, required=True, help="Number of EOS")
    parser.add_argument(
        "--EOS-prior",
        type=str,
        required=False,
        help="Path to the EOS prior file, if None, assuming fla prior across EOSs",
    )
    parser.add_argument("--EOSpath", type=str, required=True, help="The EOS data")
    parser.add_argument(
        "--pdet",
        type=str,
        required=False,
        help="Path to the probability of detection as a function of maximum mass (for correcting selection bias)",
    )
    parser.add_argument(
        "--cred-interval",
        type=float,
        default=0.95,
        help="Credible interval to be calculated (default: 0.95)",
    )
    parser.add_argument(
        "--seed", type=int, required=False, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    # load the detectable events
    detectable = np.loadtxt(args.detections_file, usecols=[0]).astype(int)

    # load the EOS prior
    prior = np.loadtxt(args.EOS_prior)
    prior = prior[
        :-1
    ]  # to remove the duplicate at the last line added for it to work with pbilby

    # get the R14, Mmax prior samples
    R14_prior = []
    Mmax_prior = []
    for i in range(1, args.Neos + 1):
        m, r = np.loadtxt(
            "{0}/{1}.dat".format(args.EOSpath, i), usecols=[1, 0], unpack=True
        )
        interp = scipy.interpolate.interp1d(m, r)
        R14_prior.append(interp(1.4))
        Mmax_prior.append(m[-1])
    R14_prior = np.array(R14_prior)
    Mmax_prior = np.array(Mmax_prior)

    # load the pdet
    Mmax, pdet = np.loadtxt(args.pdet, usecols=[0, 1], unpack=True)
    fit = scipy.interpolate.UnivariateSpline(Mmax, pdet, s=0.3)  # smooth interpolation
    pdet_of_EOS = fit(
        Mmax_prior
    )  # calculate the selection effect correction for each EOS

    # get the combined posterior
    probs = dict()
    for i in detectable:
        try:
            sample = pd.read_csv(
                "{0}/{1}/posterior_samples.dat".format(args.GWEMsamples, i),
                header=0,
                delimiter=" ",
            )
        except IOError:  # this is designed for running on incomplete dataset
            continue
        sample = reweight_to_flat_mass_prior(sample)
        # during the sampling, EOS is treated as a continous variable
        # we convert them back to integer
        EOSTrue = sample.EOS.to_numpy().astype(int) + 1
        # calulcate the posterio probability by counting the samples
        counts = []
        for j in range(1, args.Neos + 1):
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
    np.random.seed(args.seed)
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
            samples = np.random.choice(
                R14_prior, p=prob_cumprod[i], size=10000, replace=True
            )
            # calculate the median and append it to the list
            R14_med.append(np.median(samples))
            # calculate the 95% credible interval
            range95 = arviz.hdi(samples, hdi_prob=args.cred_interval)
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

    df_dict = dict(
        R14_med=R14_med, R14_uperr=R14_uplim - R14_med, R14_lowerr=R14_med - R14_lowlim
    )
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(
        "{0}/GW_EM_R14trend_{1}.dat".format(args.outdir, args.label),
        index=False,
        sep=" ",
    )

    # ==============================================================
    # plot of EOSs
    # ==============================================================

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


if __name__ == "__main__":
    main()
