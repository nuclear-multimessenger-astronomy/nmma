import corner
import os
import numpy as np
import pandas as pd
import matplotlib
import seaborn
from matplotlib import gridspec, pyplot as plt
from ast import literal_eval
import itertools

from ..core.conversion import chirp_mass_and_eta_to_component_masses, tidal_deformabilities_and_mass_ratio_to_eff_tidal_deformabilities, label_mapping
from ..core import utils, parsing
from ..core import plotting_utils as corepu
from.parser import corner_plot_parser
color_array = corepu.fig_setup()
nmma_colors = itertools.cycle(color_array)


def plot_multi_corner(args, key_selection=None):

    plot_kwargs = literal_eval(args.kwargs)
    quantiles = [0.16, 0.5, 0.84]
    fig = None
    labels = [lab for lab in args.label_name] if args.label_name is not None else [f for f in args.posterior_files]
    for i, f in enumerate(args.posterior_files):
        plot_keys, plot_labels = corepu.plotting_parameters_from_priors(args.prior, keys=key_selection).items()
        if args.injection_json is not None:
            truths = utils.read_injection_file(args.injection_json)
            truths = truths.iloc[args.injection_num].to_dict()
            truths = np.array([truths[k] for k in plot_keys])
            if args.verbose:
                print("\nLoaded Injection:")
                print(f"Truths from injection: {truths}")
        elif args.bestfit_params is not None:
            truths = utils.read_bestfit_from_json(args.bestfit_json, plot_keys, args.verbose)
        else:
            truths = None
            
        fig = setup_corner_plot(f, [], label =labels[i], truths = truths, fig=fig, 
                quantiles=quantiles, plot_keys=plot_keys, default_labels = plot_labels, **plot_kwargs)

    filename, ext = os.path.splitext(args.output)
    if not ext:
        filename = os.path.join(os.getcwd(), f"{filename}.png")
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    print("\nSaved corner plot:", filename)


def setup_corner_plot(posterior_samples,limits = None, plot_keys = None, fig = None, 
                      injection=None, post_dir = None, default_labels=None, **plot_kwargs):
    #load samples
    posterior_samples = utils.get_posteriors(posterior_samples, post_dir)

    if plot_keys is None:
        plot_keys = posterior_samples.columns.tolist() # show all we can
    if limits is None:
        limits = [(np.inf, -np.inf) for key in plot_keys] # will adjust more permissively later
    # find what to actually plot
    plot_samples, plot_labels, titles = [], [], []
    for i, k in enumerate(plot_keys):
        try: # we have data to show
            show_data = posterior_samples[k].to_numpy()
            plot_samples.append(show_data)
            titles.append(utils.sig_lims(show_data, plot_kwargs.get('quantiles', None)))

            cur_min, cur_max = limits[i]
            limits[i] = (min(cur_min, np.amin(show_data)), max(cur_max, np.amax(show_data)))

            label = label_mapping.get(k, default_labels.get(k,k))
            plot_labels.append(label )
        except KeyError:
            print(f"key {k} was not found in the posterior samples; Inserting dummy plot.")
            cur_min, cur_max = limits[i]
            if cur_min > cur_max: # meaning no data seen so far, so we have to take chances
                cur_min, cur_max = 1e42, -1e42# just set crazy limits that we can still work on.
                limits[i] = (cur_min, cur_max) 
            # some dummy values that should remain out of frame
            plot_samples.append(np.linspace(100*cur_max, 1001*cur_max, posterior_samples.shape[0]))
            plot_labels.append('')
            titles.append('')
    plot_samples = np.column_stack(plot_samples)
    
    if injection is not None:
        truths = [injection.get(key, None) for key in plot_keys]
    else:
        truths = None
    # limits = ((np.amin(posterior_samples[k]), np.amax(posterior_samples[k])) for k in plot_keys)
    color = plot_kwargs.pop('color', next(nmma_colors))
    fig = corner_plot(plot_samples, plot_labels, limits, fig=fig, truths= truths, color = color, titles=titles, show_titles = False, **plot_kwargs)

    # allow joint legend
    if 'label' in plot_kwargs:
        fig.legends.clear()
        fig.axes[0].plot([], [], label = plot_kwargs['label'], color= color)
        fig.legend()
    return fig, limits


def corner_plot(plot_samples, labels, limits, fig = None, save=False, **kwargs):

    matplotlib.rcParams.update({'font.size': 16, 'font.family': 'Times New Roman'})
    matplotlib.rcParams['text.usetex'] = (os.environ.get("CI") != 'true')
    default_kwargs = dict(bins=50, smooth=1.3, label_kwargs=dict(fontsize=16), show_titles=True,
                  title_kwargs=dict(fontsize=16), color = color_array[0], #color='#0072C1',
                  truth_color='tab:orange', quantiles=[0.05, 0.5, 0.95],
                  levels=(0.10, 0.32, 0.68, 0.95), median_line=True, title_fmt=".2f",
                  plot_density=False, plot_datapoints=False, fill_contours=True,
                  max_n_ticks=4, hist_kwargs={'density': True})
    default_kwargs.update(kwargs) 
    # plt.figure(1)
    fig = corner.corner(plot_samples, labels=labels, range=limits, fig = fig, **default_kwargs)
    if save:
        plt.savefig(save, bbox_inches='tight')
    return fig




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

if __name__ == "__main__":
    args = parsing.nmma_base_parsing(corner_plot_parser)
    plot_multi_corner(args)