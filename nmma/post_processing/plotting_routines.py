import corner
import numpy as np
import pandas as pd
import json
import h5py
import os

import matplotlib
import seaborn
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

from ..joint.conversion import chirp_mass_and_eta_to_component_masses, tidal_deformabilities_and_mass_ratio_to_eff_tidal_deformabilities


def corner_plot(plot_samples, labels, limits, outdir, fig = None, save=False, **kwargs):

    matplotlib.rcParams.update({'font.size': 16, 'text.usetex': True, 'font.family': 'Times New Roman'})
    default_kwargs = dict(bins=50, smooth=1.3, label_kwargs=dict(fontsize=16), show_titles=True,
                  title_kwargs=dict(fontsize=16), color='#0072C1',
                  truth_color='tab:orange', quantiles=[0.05, 0.5, 0.95],
                  levels=(0.10, 0.32, 0.68, 0.95), median_line=True, title_fmt=".2f",
                  plot_density=False, plot_datapoints=False, fill_contours=True,
                  max_n_ticks=4, hist_kwargs={'density': True})
    default_kwargs.update(kwargs) 
    # plt.figure(1)
    fig = corner.corner(plot_samples, labels=labels, range=limits, fig = fig, **default_kwargs)
    if save:
        plt.savefig(f'{outdir}/{save}.pdf', bbox_inches='tight')
    return fig

def get_posteriors(posterior_samples, outdir = None):
    """
    Load posterior samples from a file or DataFrame.
    """
    if isinstance(posterior_samples, str):
        if not os.path.isfile(posterior_samples):
            posterior_samples = os.path.join(outdir, posterior_samples)
        format_str = posterior_samples.split('.')[-1]
        if format_str in ['csv', 'txt', 'dat']:
            posterior_samples = pd.read_csv(posterior_samples, sep='\s+', header=0)
        elif format_str == 'json':
            with open(posterior_samples, 'r') as f:
                samples_dict = json.load(f)
            posterior_samples = pd.DataFrame(samples_dict['posterior']['content'])
        elif format_str == 'hdf5':
            with h5py.File(posterior_samples, 'r') as f:
                posterior_group = f['posterior']
                posterior_samples = pd.DataFrame({key: np.array(posterior_group[key]) for key in posterior_group.keys()})
        else:
            raise ValueError("Unsupported file format, must be csv, txt, dat, json or hdf5")
    return posterior_samples

def setup_corner_plot(posterior_samples, outdir, messengers,plot_keys = None, fig = None, injection=None, post_dir = None, **plot_kwargs):
    if post_dir is None:
        post_dir = outdir
    posterior_samples = get_posteriors(posterior_samples, post_dir)
    plottable_keys, labels = [], []
    for std_messenger, sample_func in zip(
        ['gw', 'eos', 'kn', 'grb'], 
        [get_gw_posterior_samples, get_eos_posterior_samples, get_kn_posterior_samples, get_grb_posterior_samples]
    ):
        if std_messenger in messengers:
            posterior_samples, new_keys, new_labels = sample_func(posterior_samples)
            plottable_keys += new_keys
            labels += new_labels

    if plot_keys is None:
        plot_keys = plottable_keys
    plot_samples, plot_labels, limits = [], [], []
    for key in plot_keys:
        if key in plottable_keys:
            plot_samples.append(posterior_samples[key].to_numpy())
            plot_labels.append(labels[plottable_keys.index(key)])
            limits.append((np.amin(posterior_samples[key]), np.amax(posterior_samples[key])))
        else:
            try:
                plot_samples.append(posterior_samples[key].to_numpy())
                plot_labels.append(key)
                limits.append((np.amin(posterior_samples[key]), np.amax(posterior_samples[key])))
            except KeyError:
                plot_samples.append(np.full(posterior_samples.shape[0], np.nan))
                print(f"key {key} was not found in the posterior samples; Inserting empty plot.")
                plot_labels.append('')
                limits.append((0, 1))
    plot_samples = np.column_stack(plot_samples)
    # plot_samples = np.column_stack([posterior_samples[key].to_numpy() for key in plot_keys])
    if injection is not None:
        truths = [injection.get(key, None) for key in plot_keys]
    else:
        truths = None
    # limits = ((np.amin(posterior_samples[k]), np.amax(posterior_samples[k])) for k in plot_keys)

    fig = corner_plot(plot_samples, plot_labels, limits, outdir, fig=fig, truths= truths, **plot_kwargs)
    if 'label' in plot_kwargs:
        fig.legends.clear()
        fig.axes[0].plot([], [], label = plot_kwargs['label'], color= plot_kwargs['color'])
        # fig.legend(loc ="upper right", frameon=False, fontsize=40)
    #      ,
    #     handles=[
    #         mlines.Line2D([], [], color=colors[i], label=sample_labels[i])
    #         for i in range(n)
    #     ],
    #     fontsize=20, frameon=False,
    #     bbox_to_anchor=(1, ndim), loc="upper right"
    # )
    #     if 'color' in plot_kwargs:
    #         label = mlines.Line2D([], [], color=plot_kwargs['color'], label=plot_kwargs['label'])
    #     else:
    #         label = mlines.Line2D([], [], label=plot_kwargs['label'])
    #     plt.legend(bbox_to_anchor=(0., 1.0, 1., .0), loc=4)
    return fig

def get_gw_posterior_samples(posterior_samples):
    """
    Extract and return the keys and labels for the gravitational wave posterior samples.
    """
    if "chi_eff" not in posterior_samples:
        q = posterior_samples['mass_ratio'].to_numpy()
        chi_1 = posterior_samples['chi_1'].to_numpy()
        chi_2 = posterior_samples['chi_2'].to_numpy()
        # Calculate the effective tidal deformability and chirp mass
        posterior_samples['chi_eff'] = (chi_1 + q*chi_2)/(1+q)
    if "lambdaT" not in posterior_samples:
        lambda1 = posterior_samples['lambda_1'].to_numpy()
        lambda2 = posterior_samples['lambda_2'].to_numpy()
        q = posterior_samples['mass_ratio'].to_numpy()
        # Calculate the effective tidal deformability
    posterior_samples['lambdaT'], _ = tidal_deformabilities_and_mass_ratio_to_eff_tidal_deformabilities(lambda1, lambda2, q)
    plot_keys = ["chirp_mass", "mass_ratio", "luminosity_distance", "chi_eff", "lambdaT", "mass_1_source", "mass_2_source", 'theta_jn']

    labels = [r'$\mathcal{M}_c{\rm [M_{\odot}]}$', r'$q$', r'$d_L{\rm [Mpc]}$', r'$\chi_{\rm{eff}}$', r'$\tilde{\Lambda}$', r'$m_{1,s}{\rm [M_{\odot}]}$', r'$m_{2,s}{\rm [M_{\odot}]}$', r'$\theta_{jn}$']
    return posterior_samples, plot_keys, labels

def get_eos_posterior_samples(posterior_samples):
    """
    Extract and return the keys and labels for the EOS posterior samples.
    """
    plot_keys = ["L_sym", "K_sym", "K_sat", "3n_sat", "5n_sat", "TOV_mass", "R_14"]
    labels = [r'$L_{\rm{sym}}{\rm [MeV]}$', r'$K_{\rm{sym}}{\rm [MeV]}$', r'$K_{\rm{sat}}{\rm [MeV]}$', r'$c_{3n_{\rm{sat}}}{\rm [c]}$', r'$c_{5n_{\rm{sat}}}{\rm [c]}$', r'$M_{\rm{TOV}}{\rm [M_{\odot}]}$', r'$R_{1.4}{\rm[km]}$']
    return posterior_samples, plot_keys, labels

def get_kn_posterior_samples(posterior_samples):
    """
    Extract and return the keys and labels for the KN posterior samples.
    """
    plot_keys = ['log10_mej','log10_mej_dyn', 'log10_mej_wind', 'ratio_zeta', 'alpha', 'KNtheta', 'KNphi']
    posterior_samples['log10_mej'] = np.log10(
        10**(posterior_samples['log10_mej_wind'].to_numpy())
        + 10**(posterior_samples['log10_mej_dyn'].to_numpy())
    )
    labels = [r'$\log_{10}(M_{\rm{ej}}{\rm [M_{\odot}]})$',r'$\log_{10}(M_{\rm{ej,dyn}}{\rm [M_{\odot}]})$', r'$\log_{10}(M_{\rm{ej,wind}}{\rm [M_{\odot}]})$', r'$\zeta$', r'$\alpha$', r'$\theta_{KN}$', r'$\phi_{KN}$']
    return posterior_samples, plot_keys, labels

def get_grb_posterior_samples(posterior_samples):
    """
    Extract and return the keys and labels for the GRB posterior samples.
    """

    if 'thetaWing' in posterior_samples:
        posterior_samples['alphaWing'] = posterior_samples['thetaWing'].to_numpy() / posterior_samples['thetaCore'].to_numpy()
    elif 'alphaWing' in posterior_samples:
        posterior_samples['thetaWing'] = posterior_samples['alphaWing'] * posterior_samples['thetaCore']
    plot_keys = ['ratio_epsilon', 'log10_E0', 'thetaCore', 'thetaWing', 'alphaWing', 'log10_n0', 'p', 'log10_epsilon_e', 'log10_epsilon_B']
    labels = [r'$\epsilon$', r'$\log_{10}(E_{0})$', r'$\theta_{c}$', r'$\theta_{w}$',r'$\alpha_{w}$', r'$\log_{10}(n_{0})$', r'$p$', r'$\log_{10}(\epsilon_{e})$', r'$\log_{10}(\epsilon_{B})$']
    return posterior_samples, plot_keys, labels

def eos_only_corner_plot(posterior_samples, outdir):
    return setup_corner_plot(posterior_samples, outdir, messengers='eos')
    
def resampling_corner_plot(posterior_samples, solution, outdir, withNSBH): 
        
    mc = posterior_samples['chirp_mass'].to_numpy()
    invq = posterior_samples['mass_ratio'].to_numpy()
    alpha = posterior_samples['alpha'].to_numpy()
    zeta = posterior_samples['zeta'].to_numpy()

    eta = invq / np.power(1. + invq, 2)
    m1, m2 = chirp_mass_and_eta_to_component_masses(mc, eta)
    q = m1 / m2  # inverted



    if withNSBH:#NSBH resampling result corner plot
        labels = [r'$\mathcal{M}_c[M_{\odot}]$', r'$q$', r'$\alpha[M_{\odot}]$', r'$\zeta$']
        plot_samples = np.vstack((mc, q, alpha, zeta))
        limits = ((np.amin(mc), np.amax(mc)), (np.amin(q), 3), (np.amin(alpha), np.amax(alpha)), (np.amin(zeta), np.amax(zeta)))

    else: #BNS resampling result corner plot
        labels = [r'$\mathcal{M}_c[M_{\odot}]$', r'$q$', r'$\tilde{\Lambda}$', r'$\alpha[M_{\odot}]$', r'$\zeta$', r'$M_{\rm{max}}[M_{\odot}]$']

        EOS = posterior_samples['EOS'].astype(int) + 1
        lambda1 = []
        lambda2 = []
        for i in range(len(EOS)):
            EOSsample = EOS[i]
            lam1, lam2 = np.interp([m1[i], m2[i]], solution.EOS_masses_dict[EOSsample], solution.EOS_lambda_dict[EOSsample])
            lambda1.append(lam1)
            lambda2.append(lam2)
        lambda1 = np.array(lambda1)
        lambda2 = np.array(lambda2)
        lambdaT, _ = tidal_deformabilities_and_mass_ratio_to_eff_tidal_deformabilities(lambda1, lambda2, q)

        MTOV = []
        for EOSsample in EOS:
            MTOV.append(solution.EOS_masses_dict[EOSsample][-1])
        MTOV = np.array(MTOV)

        print("The 90% upper bound for lambdaT is {0}".format(np.quantile(lambdaT, 0.9)))
        plot_samples = np.vstack((mc, q, lambdaT, alpha, zeta, MTOV))
        limits = ((np.amin(mc), np.amax(mc)), (np.amin(q), 3), (np.amin(lambdaT), np.amax(lambdaT)), (np.amin(alpha), np.amax(alpha)), (np.amin(zeta), np.amax(zeta)), (np.amin(MTOV), 2.7))
    corner_plot(plot_samples.T, labels, limits, outdir)

def plot_R14_trend(args):
    # load the data
    data_GWEM = pd.read_csv(f"{args.outdir}/GW_EM_R14trend_{args.label}.dat",
        header=0, delimiter=" ")
    data_GW = pd.read_csv(f"{args.gwR14trend}/GW_R14trend.dat", header=0, delimiter=" ")

    # initialise the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.suptitle("Constrain EoS using EM + GW", fontname="Times New Roman Bold")

    ax1.set_ylabel(r"$R_{1.4} \ [{\rm km}]$")
    ax2.set_ylabel(r"$\delta R_{1.4} / R_{1.4} \ [\%]$")
    ax2.set_xlabel("Events")

    # set reference lines
    c = seaborn.color_palette("colorblind")
    ax1.axhline(args.R14_true, linestyle="--", color=c[1], label="Injected value")
    for y in [10, 5, 1]:
        ax2.axhline(y, color="grey", linestyle="--", alpha=0.5)
    ax2.set_yscale("log")


    # plot the data
    for label, color, data_set in zip(["GW", "GW+EM"],[c[3], c[0]], [data_GW, data_GWEM]):
        x_values = np.arange(1, len(data_set) + 1)
        ax1.errorbar(
            x_values, data_set.R14_med, yerr=[data_set.R14_lowerr, data_set.R14_uperr],
            label=label, color=color, fmt="o", capsize=5
        )
        mean_error = np.mean([data_set.R14_lowerr, data_set.R14_uperr], axis=0)
        ax2.plot(x_values, mean_error/data_set.R14_med *100, color=color, marker="o" )

    # legend and limits
    ax1.legend()
    ax1.set_xlim([0.5, len(data_GWEM) + 0.5])
    ax2.set_xticks(np.arange(1, len(data_GWEM) + 1, 2))

    fig.tight_layout()
    # fig.subplots_adjust(hspace=0.1)
    plt.savefig(f"{args.outdir}/R14_trend_GW_EM_{args.label}.pdf", bbox_inches="tight")
