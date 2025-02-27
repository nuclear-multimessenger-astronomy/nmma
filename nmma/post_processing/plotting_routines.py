import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner
import numpy as np
import pandas as pd
import json
import h5py
import os


from ..joint.conversion import chirp_mass_and_eta_to_component_masses, tidal_deformabilities_and_mass_ratio_to_eff_tidal_deformabilities

def corner_plot(plot_samples, labels, limits, outdir, **kwargs):

    matplotlib.rcParams.update({'font.size': 16, 'text.usetex': True, 'font.family': 'Times New Roman'})
    default_kwargs = dict(bins=50, smooth=1.3, label_kwargs=dict(fontsize=16), show_titles=True,
                  title_kwargs=dict(fontsize=16), color='#0072C1',
                  truth_color='tab:orange', quantiles=[0.05, 0.5, 0.95],
                  levels=(0.10, 0.32, 0.68, 0.95), median_line=True,
                  plot_density=False, plot_datapoints=False, fill_contours=True,
                  max_n_ticks=4, hist_kwargs={'density': True})
    default_kwargs.update(kwargs) 

    plt.figure(1)
    corner.corner(plot_samples, labels=labels, range=limits, **default_kwargs)
    plt.savefig(f'{outdir}/corner.pdf', bbox_inches='tight')

def eos_only_corner_plot(posterior_samples, outdir):
    if isinstance(posterior_samples, str):
        if not os.path.isfile(posterior_samples):
            posterior_samples = os.path.join(outdir, posterior_samples)
        format_str = posterior_samples.split('.')[-1]
        if format_str in ['csv', 'txt', 'dat']:
            posterior_samples = pd.read_csv(posterior_samples)
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
        
    plot_keys = ["L_sym", "K_sym", "K_sat", "3n_sat", "5n_sat", "TOV_mass", "R_14"]
    plot_samples = np.vstack([posterior_samples[key].to_numpy() for key in plot_keys]).T
    labels = [r'$L_{\rm{sym}}$', r'$K_{\rm{sym}}$', r'$K_{\rm{sat}}$', r'$3n_{\rm{sat}}$', r'$5n_{\rm{sat}}$', r'$M_{\rm{TOV}}[M_{\odot}]$', r'$R_{1.4}[km]$']
    limits = ((np.amin(posterior_samples[k]), np.amax(posterior_samples[k])) for k in plot_keys)
    corner_plot(plot_samples, labels, limits, outdir)
    

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