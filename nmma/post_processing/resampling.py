import numpy as np
import pandas as pd
import os
from arviz import hdi
import scipy.stats
from bilby.gw.prior import PriorDict
from bilby.core.prior import Uniform

from .parser import resampling_parser
from ..joint.base_parsing import nmma_base_parsing
from ..joint.conversion import (
    BNSEjectaFitting, NSBHEjectaFitting,
    luminosity_distance_to_redshift, chirp_mass_and_eta_to_component_masses,
        )
from ..joint.constants import geom_msun_km
from .plotting_routines import resampling_corner_plot

from pymultinest.solve import Solver

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



def construct_EM_KDE(EMsamples, combine_ejecta_mass):
    if "log10_mej" in EMsamples.columns:
        return scipy.stats.gaussian_kde(10**EMsamples.log10_mej.to_numpy())

    elif "log10_mej_dyn" in EMsamples.columns and "log10_mej_wind" in EMsamples.columns:
        if combine_ejecta_mass:
            total_eject_mass = 10**EMsamples.log10_mej_dyn.to_numpy() + 10**EMsamples.log10_mej_wind.to_numpy()
            return scipy.stats.gaussian_kde(total_eject_mass)
        
        else:
            return scipy.stats.gaussian_kde((10**EMsamples.log10_mej_dyn.to_numpy(), 10**EMsamples.log10_mej_wind.to_numpy()))

    else:
        raise ValueError("EM samples must either contain ejecta mass as 'log10_mej' or seperate ejecta masses as 'log10_mej_dyn' and 'log10_mej_wind'.")


class EjectaResampler(Solver):

    def __init__(self, GWsamples, EMsamples, GWprior, EMprior, Neos, EOSpath, withNSBH, combine_ejecta_mass=False, **kwargs):
        self.GWsamples = GWsamples
        self.EMsamples = EMsamples
        self.withNSBH = withNSBH
        self.combine_ejecta_mass = combine_ejecta_mass
        #### FIXME: allow free EOS-params?
        priors = {'chirp_mass': GWprior['chirp_mass'],
                       'mass_ratio': GWprior['mass_ratio'],
                       'EOS': Uniform(minimum=0, maximum=Neos + 1),
                       'alpha': EMprior['alpha'],
                       'zeta': EMprior['zeta']}
        
        if withNSBH:
            priors.update({'chi_1': GWprior['chi_1'], 'chi_2': GWprior['chi_2']})
 
        
            
        self.priors = PriorDict(priors)
        self._search_parameter_keys = self.priors.keys()

        ###tabulated EOS_treatment, to be fixed for on-the-fly EOSs -> this read-in seems to occur somewhat frequently, could be an outside function
        self.Neos = Neos

        EOS_radius_dict = {}
        EOS_lambda_dict = {}
        EOS_masses_dict = {}
        for i in range(1, Neos + 1):
            r, m, Lambda = np.loadtxt('{0}/{1}.dat'.format(EOSpath, i), usecols=[0, 1, 2], unpack=True)
            EOS_radius_dict[i] = r
            EOS_lambda_dict[i] = Lambda
            EOS_masses_dict[i] = m


        self.EOS_radius_dict = EOS_radius_dict
        self.EOS_lambda_dict = EOS_lambda_dict
        self.EOS_masses_dict = EOS_masses_dict

        EOS = self.GWsamples.EOS.to_numpy()
        self.EOSsamples = EOS.astype(int) + 1

        z = luminosity_distance_to_redshift(self.GWsamples.luminosity_distance.to_numpy())
        mc = self.GWsamples.chirp_mass.to_numpy() / (1 + z)
        q = self.GWsamples.mass_ratio.to_numpy()

        if (withNSBH):
            chi_1 = self.GWsamples.chi_1.to_numpy()
            chi_2 = self.GWsamples.chi_2.to_numpy()
            chi_eff = (chi_1 + q*chi_2)/(1+q)
            self.chi_1KDE = scipy.stats.gaussian_kde(chi_1)
            self.chi_2KDE = scipy.stats.gaussian_kde(chi_2)

        self.mcKDE = scipy.stats.gaussian_kde(mc)
        self.invqKDE = scipy.stats.gaussian_kde(1. / q)
        self.EMKDE = construct_EM_KDE(self.EMsamples, self.combine_ejecta_mass)
        
        self.NSBHEjectaFitting = NSBHEjectaFitting()
        self.BNSEjectaFitting = BNSEjectaFitting()

        Solver.__init__(self, **kwargs)
    
    def Prior(self, x):
        return self.priors.rescale(self._search_parameter_keys, x)
    
    def LogLikelihood(self, x):
        if self.withNSBH:
            mc, q, EOS, alpha, zeta, chi_1, chi_2 = x
            chi_eff = (chi_1 + q*chi_2)/(1+q)
        else:
            mc, q, EOS, alpha, zeta = x

        EOS = int(EOS) + 1
        eta = q / np.power(1. + q, 2.)
        m1, m2 = chirp_mass_and_eta_to_component_masses(mc, eta)
        total_mass = m1 + m2
        mass_ratio = m2 / m1

        r1, r2, R16 = np.interp((m1, m2, 1.6), self.EOS_masses_dict[EOS], self.EOS_radius_dict[EOS], right = 0)
        R16 /= geom_msun_km ###needed in geo units for BNSEjectaFitting
        try:
            C2 = m2 / r2 * geom_msun_km   ### disfavour EOS if secondary cannot be supported as NS
        except ZeroDivisionError:
            return np.nan_to_num(-np.inf)
        if not self.withNSBH:
            try:
                C1 = m1 / r1 * geom_msun_km   ### disfavour EOS if primary cannot be supported as NS
            except ZeroDivisionError:
                return np.nan_to_num(-np.inf)
        MTOV = self.EOS_masses_dict[EOS][-1]

        if len(np.where(self.EOSsamples == EOS)[0]) == 0:
            return np.nan_to_num(-np.inf)
        

        
        if self.withNSBH:
            logprior = self.chi_1KDE.logpdf(chi_1) + self.chi_2KDE.logpdf(chi_2) + self.mcKDE.logpdf(mc) + self.invqKDE.logpdf(m1 / m2) + np.log(len(np.where(self.EOSsamples == EOS)[0]))
            mdyn = self.NSBHEjectaFitting.dynamic_mass_fitting(m1, m2, C2, chi_eff) + alpha
            if mdyn < 0.:
                return np.nan_to_num(-np.inf)
            mdisk = self.NSBHEjectaFitting.remnant_disk_mass_fitting(m1, m2, C2, chi_eff)
            log10_mwind = np.log10(zeta) + np.log10(mdisk)

        else:
            logprior = self.mcKDE.logpdf(mc) + self.invqKDE.logpdf(m1 / m2) + np.log(len(np.where(self.EOSsamples == EOS)[0]))
            mdyn = self.BNSEjectaFitting.dynamic_mass_fitting_KrFo(m1, m2, C1, C2) + alpha
            if mdyn < 0.:
                return np.nan_to_num(-np.inf)
            log10_mdisk = self.BNSEjectaFitting.log10_disk_mass_fitting(total_mass, mass_ratio, MTOV, R16)
            log10_mwind = np.log10(zeta) + log10_mdisk

        if self.combine_ejecta_mass:
            ejecta_mass =  mdyn + 10**log10_mwind
        else:
            ejecta_mass = (mdyn, 10**log10_mwind)
        loglikelihood = self.EMKDE.logpdf(ejecta_mass)

        return np.nan_to_num(logprior + loglikelihood)


class TotalEjectaMassInference(EjectaResampler):
    def __init__(self, GWsamples, EMsamples, GWprior, EMprior, Neos, EOSpath, withNSBH, **kwargs):
        super().__init__(GWsamples, EMsamples, GWprior, EMprior, Neos, EOSpath, withNSBH, combine_ejecta_mass=True, **kwargs)


class EjectaMassInference(EjectaResampler):
    def __init__(self, GWsamples, EMsamples, GWprior, EMprior, Neos, EOSpath, withNSBH, **kwargs):
        super().__init__(GWsamples, EMsamples, GWprior, EMprior, Neos, EOSpath, withNSBH, combine_ejecta_mass=False, **kwargs)

def main_resampling():
    args = nmma_base_parsing(resampling_parser)

    # read the GW samples
    GWsamples = pd.read_csv(args.GWsamples, header=0, delimiter=" ")
    # down sample
    weights = np.ones(len(GWsamples))
    weights /= np.sum(weights)
    GWsamples = GWsamples.sample(
        frac=30000 / len(GWsamples), weights=weights, random_state=42
    )

    # read the EM samples
    EMsamples = pd.read_csv(args.EMsamples, header=0, delimiter=" ")

    # read the prior files
    GWprior = PriorDict(args.GWprior)
    EMprior = PriorDict(args.EMprior)
    
    try:
        os.makedirs(args.outdir + "/pm/")
    except Exception:
        pass

    pymulti_kwargs = dict(
        outputfiles_basename=args.outdir + "/pm/",
        n_dims=5,
        n_live_points=args.nlive,
        verbose=True,
        resume=True,
        seed=42,
        importance_nested_sampling=False,
        )
    if args.withNSBH:
        pymulti_kwargs['n_dims'] = 7

    solution = EjectaResampler(
            GWsamples,
            EMsamples,
            GWprior,
            EMprior,
            args.Neos,
            args.EOSpath,
            args.withNSBH,
            args.total_ejecta_mass,
            **pymulti_kwargs
        )
    

    samples = solution.samples.T
    posterior_samples = dict()
    posterior_samples["chirp_mass"] = samples[0]
    posterior_samples["mass_ratio"] = samples[1]
    posterior_samples["EOS"] = samples[2]
    posterior_samples["alpha"] = samples[3]
    posterior_samples["zeta"] = samples[4]
    if args.withNSBH:
        posterior_samples["chi_1"] = samples[5]
        posterior_samples["chi_2"] = samples[6]

    posterior_samples = pd.DataFrame.from_dict(posterior_samples)
    posterior_samples.to_csv(f"{args.outdir}/posterior_samples.dat", sep=" ", index=False)

    resampling_corner_plot(posterior_samples, solution, args.outdir, args.withNSBH)


if __name__ == "__main__":
    main_resampling()
