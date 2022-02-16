import numpy as np
import scipy.stats
import scipy.special
import scipy.interpolate
import lal
from bilby.gw.prior import PriorDict
from bilby.gw.conversion import luminosity_distance_to_redshift
from bilby.core.prior import Uniform

from ..joint.conversion import BNSEjectaFitting

from pymultinest.solve import Solver


def construct_EM_KDE_seperate(EMsamples):

    kde = scipy.stats.gaussian_kde((EMsamples.log10_mej_dyn.to_numpy(), EMsamples.log10_mej_wind.to_numpy()))

    return kde


def construct_EM_KDE(EMsamples):
    total_eject_mass = 10**EMsamples.log10_mej_dyn.to_numpy() + 10**EMsamples.log10_mej_wind.to_numpy()

    kde = scipy.stats.gaussian_kde(total_eject_mass)

    return kde


def mceta2m1m2(mc, eta):
    M = mc / np.power(eta, 3. / 5.)
    q = (1 - np.sqrt(1. - 4. * eta) - 2 * eta) / (2. * eta)

    m1 = M / (1. + q)
    m2 = M * q / (1. + q)

    return m1, m2


def lambdas2lambdaTs(lambda1, lambda2, q):
    eta = q / np.power(1. + q, 2.)
    eta2 = eta * eta
    eta3 = eta2 * eta
    root14eta = np.sqrt(1. - 4 * eta)

    lambdaT = (8. / 13.) * ((1. + 7 * eta - 31 * eta2) * (lambda1 + lambda2) + root14eta * (1. + 9 * eta - 11. * eta2) * (lambda1 - lambda2))
    dlambdaT = 0.5 * (root14eta * (1. - 13272. * eta / 1319. + 8944. * eta2 / 1319.) * (lambda1 + lambda2) + (1. - 15910. * eta / 1319. + 32850. * eta2 / 1319. + 3380. * eta3 / 1319.) * (lambda1 - lambda2))

    return lambdaT, dlambdaT


def corner_plot(posterior_samples, solution, outdir):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import corner
    matplotlib.rcParams.update({'font.size': 16, 'text.usetex': True, 'font.family': 'Times New Roman'})
    kwargs = dict(bins=50, smooth=1.3, label_kwargs=dict(fontsize=16), show_titles=True,
                  title_kwargs=dict(fontsize=16), color='#0072C1',
                  truth_color='tab:orange', quantiles=[0.05, 0.95],
                  levels=(0.10, 0.32, 0.68, 0.95), median_line=True,
                  plot_density=False, plot_datapoints=False, fill_contours=True,
                  max_n_ticks=4, hist_kwargs={'density': True})

    labels = [r'$\mathcal{M}_c[M_{\odot}]$', r'$q$', r'$\tilde{\Lambda}$', r'$\alpha[M_{\odot}]$', r'$\zeta$', r'$M_{\rm{max}}[M_{\odot}]$']
    mc = posterior_samples['chirp_mass'].to_numpy()
    invq = posterior_samples['mass_ratio'].to_numpy()
    alpha = posterior_samples['alpha'].to_numpy()
    zeta = posterior_samples['zeta'].to_numpy()

    eta = invq / np.power(1. + invq, 2)
    m1, m2 = mceta2m1m2(mc, eta)
    q = m1 / m2  # inverted

    EOS = posterior_samples['EOS'].astype(int) + 1
    lambda1 = []
    lambda2 = []
    for i in range(len(EOS)):
        EOSsample = EOS[i]
        lambda1.append(solution.EOS_lambda_interp_dict[EOSsample](m1[i]))
        lambda2.append(solution.EOS_lambda_interp_dict[EOSsample](m2[i]))
    lambda1 = np.array(lambda1)
    lambda2 = np.array(lambda2)
    lambdaT, _ = lambdas2lambdaTs(lambda1, lambda2, q)

    MTOV = []
    for EOSsample in EOS:
        MTOV.append(solution.EOS_lambda_interp_dict[EOSsample].x[-1])
    MTOV = np.array(MTOV)

    plotSamples = np.vstack((mc, q, lambdaT, alpha, zeta, MTOV))
    limits = ((np.amin(mc), np.amax(mc)), (np.amin(q), 3), (np.amin(lambdaT), np.amax(lambdaT)), (np.amin(alpha), np.amax(alpha)), (np.amin(zeta), np.amax(zeta)), (np.amin(MTOV), 2.7))
    plt.figure(1)
    corner.corner(plotSamples.T, labels=labels, range=limits, **kwargs)
    plt.savefig('{0}/corner.pdf'.format(outdir), bbox_inches='tight')

    print("The 90% upper bound for lambdaT is {0}".format(np.quantile(lambdaT, 0.9)))


class TotalEjectaMassInference(Solver):

    def __init__(self, GWsamples, EMsamples, GWprior, EMprior, Neos, EOSpath, **kwargs):
        self.GWsamples = GWsamples
        self.EMsamples = EMsamples
        self.priors = {'chirp_mass': GWprior['chirp_mass'],
                       'mass_ratio': GWprior['mass_ratio'],
                       'EOS': Uniform(minimum=0, maximum=Neos + 1),
                       'alpha': EMprior['alpha'],
                       'zeta': EMprior['zeta']}
        self.priors = PriorDict(self.priors)
        self._search_parameter_keys = self.priors.keys()
        self.Neos = Neos

        EOS_radius_interp_dict = {}
        EOS_lambda_interp_dict = {}
        for i in range(1, Neos + 1):
            r, m, Lambda = np.loadtxt('{0}/{1}.dat'.format(EOSpath, i), usecols=[0, 1, 2], unpack=True)

            interp_r_of_m = scipy.interpolate.interp1d(m, r, kind='linear')
            interp_lambda_of_m = scipy.interpolate.interp1d(m, Lambda, kind='linear')

            EOS_radius_interp_dict[i] = interp_r_of_m
            EOS_lambda_interp_dict[i] = interp_lambda_of_m

        self.EOS_radius_interp_dict = EOS_radius_interp_dict
        self.EOS_lambda_interp_dict = EOS_lambda_interp_dict

        z = luminosity_distance_to_redshift(self.GWsamples.luminosity_distance.to_numpy())
        mc = self.GWsamples.chirp_mass.to_numpy() / (1 + z)
        q = self.GWsamples.mass_ratio.to_numpy()
        EOS = self.GWsamples.EOS.to_numpy()

        self.mcKDE = scipy.stats.gaussian_kde(mc)
        self.invqKDE = scipy.stats.gaussian_kde(1. / q)
        self.EOSsamples = EOS.astype(int) + 1
        self.EMKDE = construct_EM_KDE(self.EMsamples)
        self.BNSEjectaFitting = BNSEjectaFitting()

        Solver.__init__(self, **kwargs)

    def Prior(self, x):
        return self.priors.rescale(self._search_parameter_keys, x)

    def LogLikelihood(self, x):
        mc, q, EOS, alpha, zeta = x
        EOS = int(EOS) + 1
        eta = q / np.power(1. + q, 2.)
        m1, m2 = mceta2m1m2(mc, eta)
        total_mass = m1 + m2
        mass_ratio = m2 / m1

        if len(np.where(self.EOSsamples == EOS)[0]) == 0:
            return np.nan_to_num(-np.inf)

        logprior = self.mcKDE.logpdf(mc) + self.invqKDE.logpdf(m1 / m2) + np.log(len(np.where(self.EOSsamples == EOS)[0]))

        try:
            r1 = self.EOS_radius_interp_dict[EOS](m1)
            r2 = self.EOS_radius_interp_dict[EOS](m2)
        except ValueError:
            return np.nan_to_num(-np.inf)

        C1 = m1 / (r1 * 1e3 / lal.MRSUN_SI)
        C2 = m2 / (r2 * 1e3 / lal.MRSUN_SI)

        R16 = self.EOS_radius_interp_dict[EOS](1.6) * 1e3 / lal.MRSUN_SI
        MTOV = self.EOS_radius_interp_dict[EOS].x[-1]

        # estimated the ejecta masses from the posterior samples
        mdyn = self.BNSEjectaFitting.dynamic_mass_fitting_KrFo(m1, m2, C1, C2)
        log10_mdisk = self.BNSEjectaFitting.log10_disk_mass_fitting(total_mass, mass_ratio, MTOV, R16)

        log10_mwind = np.log10(zeta) + log10_mdisk

        total_ejecta_mass = mdyn + 10**log10_mwind + alpha

        loglikelihood = self.EMKDE.logpdf(total_ejecta_mass)

        return logprior + loglikelihood


class EjectaMassInference(Solver):

    def __init__(self, GWsamples, EMsamples, GWprior, EMprior, Neos, EOSpath, **kwargs):
        self.GWsamples = GWsamples
        self.EMsamples = EMsamples
        self.priors = {'chirp_mass': GWprior['chirp_mass'],
                       'mass_ratio': GWprior['mass_ratio'],
                       'EOS': Uniform(minimum=0, maximum=Neos + 1),
                       'alpha': EMprior['alpha'],
                       'zeta': EMprior['zeta']}
        self.priors = PriorDict(self.priors)
        self._search_parameter_keys = self.priors.keys()
        self.Neos = Neos

        EOS_radius_interp_dict = {}
        EOS_lambda_interp_dict = {}
        for i in range(1, Neos + 1):
            r, m, Lambda = np.loadtxt('{0}/{1}.dat'.format(EOSpath, i), usecols=[0, 1, 2], unpack=True)

            interp_r_of_m = scipy.interpolate.interp1d(m, r, kind='linear')
            interp_lambda_of_m = scipy.interpolate.interp1d(m, Lambda, kind='linear')

            EOS_radius_interp_dict[i] = interp_r_of_m
            EOS_lambda_interp_dict[i] = interp_lambda_of_m

        self.EOS_radius_interp_dict = EOS_radius_interp_dict
        self.EOS_lambda_interp_dict = EOS_lambda_interp_dict

        z = luminosity_distance_to_redshift(self.GWsamples.luminosity_distance.to_numpy())
        mc = self.GWsamples.chirp_mass.to_numpy() / (1 + z)
        q = self.GWsamples.mass_ratio.to_numpy()
        EOS = self.GWsamples.EOS.to_numpy()

        self.mcKDE = scipy.stats.gaussian_kde(mc)
        self.invqKDE = scipy.stats.gaussian_kde(1. / q)
        self.EOSsamples = EOS.astype(int) + 1
        self.EMKDE = construct_EM_KDE_seperate(self.EMsamples)
        self.BNSEjectaFitting = BNSEjectaFitting()

        Solver.__init__(self, **kwargs)

    def Prior(self, x):
        return self.priors.rescale(self._search_parameter_keys, x)

    def LogLikelihood(self, x):
        mc, q, EOS, alpha, zeta = x
        EOS = int(EOS) + 1
        eta = q / np.power(1. + q, 2.)
        m1, m2 = mceta2m1m2(mc, eta)
        total_mass = m1 + m2
        mass_ratio = m2 / m1

        if len(np.where(self.EOSsamples == EOS)[0]) == 0:
            return np.nan_to_num(-np.inf)

        logprior = self.mcKDE.logpdf(mc) + self.invqKDE.logpdf(m1 / m2) + np.log(len(np.where(self.EOSsamples == EOS)[0]))

        try:
            r1 = self.EOS_radius_interp_dict[EOS](m1)
            r2 = self.EOS_radius_interp_dict[EOS](m2)
        except ValueError:
            return np.nan_to_num(-np.inf)

        C1 = m1 / (r1 * 1e3 / lal.MRSUN_SI)
        C2 = m2 / (r2 * 1e3 / lal.MRSUN_SI)

        R16 = self.EOS_radius_interp_dict[EOS](1.6) * 1e3 / lal.MRSUN_SI
        MTOV = self.EOS_radius_interp_dict[EOS].x[-1]

        # estimated the ejecta masses from the posterior samples
        mdyn = self.BNSEjectaFitting.dynamic_mass_fitting_KrFo(m1, m2, C1, C2) + alpha
        if mdyn < 0.:
            return -np.inf
        log10_mdisk = self.BNSEjectaFitting.log10_disk_mass_fitting(total_mass, mass_ratio, MTOV, R16)

        log10_mwind = np.log10(zeta) + log10_mdisk

        loglikelihood = self.EMKDE.logpdf((np.log10(mdyn), log10_mwind))

        return np.nan_to_num(logprior + loglikelihood)
