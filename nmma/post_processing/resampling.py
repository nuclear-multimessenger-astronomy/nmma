from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from arviz import hdi
from bilby.core.likelihood import Likelihood
from bilby.gw.prior import PriorDict

from ..core.base import (
    NMMALikelihood,
    bilby_sampling,
    check_priors_and_likelihood_for_nmma,
)
from ..core.constants import geom_msun_km
from ..core.conversion import (
    BNSEjectaFitting,
    CosmologyConverter,
    NSBHEjectaFitting,
    chirp_mass_and_eta_to_component_masses,
)
from ..core.parsing import nmma_base_parsing
from ..eos.eos_likelihood import setup_tabulated_eos_priors
from ..eos.eos_processing import EoSConverter, TabulatedEoSGenerator
from .parser import resampling_parser
from .plotting_routines import resampling_corner_plot


def find_spread_from_resampling(
    resampling_method, cumprod, prior_dist, post_samplesize, cred_interval
):
    """
    Summarise a resampled distribution for each set of weights.

    Parameters
    ----------
    resampling_method : callable
        Called as ``resampling_method(prior_dist, weight, post_samplesize)``;
        its return value is passed to ``numpy.median`` and ``arviz.hdi``.
    cumprod : iterable
        Iterated over, each entry passed to ``resampling_method`` as
        ``weight``.
    prior_dist : array_like
        Passed to ``resampling_method`` unchanged.
    post_samplesize
        Passed to ``resampling_method`` unchanged.
    cred_interval : float

    Returns
    -------
    numpy.ndarray
        Median of each resampled distribution.
    numpy.ndarray
        Element 1 of each ``arviz.hdi`` result, named ``uplim`` here.
    numpy.ndarray
        Element 0 of each ``arviz.hdi`` result, named ``lowlim`` here.
    """
    med, uplim, lowlim = [], [], []
    for weight in cumprod:
        # calculate the posterior distribution using the prior samples
        # and the weighting that we previously calculated
        samples = resampling_method(prior_dist, weight, post_samplesize)

        # calculate the median and append it to the list
        med.append(np.median(samples))
        # calculate the 95% credible interval
        cred_range = hdi(samples, prob=cred_interval)
        # append the bound to the list
        uplim.append(cred_range[1])
        lowlim.append(cred_range[0])
    return np.array(med), np.array(uplim), np.array(lowlim)


def construct_EM_KDE(EMsamples, combine_ejecta_mass):
    """
    Build a Gaussian KDE of the ejecta mass from EM posterior samples.

    Parameters
    ----------
    EMsamples : pandas.DataFrame
        EM posterior samples. Must contain either ``log10_mej``, or both
        ``log10_mej_dyn`` and ``log10_mej_wind``.
    combine_ejecta_mass : bool
        True sums the dynamical and wind masses into a one-dimensional KDE;
        False builds a two-dimensional KDE over the pair. Only consulted on
        the ``log10_mej_dyn``/``log10_mej_wind`` branch.

    Returns
    -------
    scipy.stats.gaussian_kde
        KDE built on ``10 ** column``, not on the logarithms themselves.

    Raises
    ------
    ValueError
        If neither ``log10_mej`` nor the pair ``log10_mej_dyn`` and
        ``log10_mej_wind`` is present.
    """
    if "log10_mej" in EMsamples.columns:
        return scipy.stats.gaussian_kde(10 ** EMsamples.log10_mej.to_numpy())

    elif "log10_mej_dyn" in EMsamples.columns and "log10_mej_wind" in EMsamples.columns:
        if combine_ejecta_mass:
            total_eject_mass = (
                10 ** EMsamples.log10_mej_dyn.to_numpy()
                + 10 ** EMsamples.log10_mej_wind.to_numpy()
            )
            return scipy.stats.gaussian_kde(total_eject_mass)

        else:
            return scipy.stats.gaussian_kde(
                (
                    10 ** EMsamples.log10_mej_dyn.to_numpy(),
                    10 ** EMsamples.log10_mej_wind.to_numpy(),
                )
            )

    else:
        raise ValueError(
            "EM samples must either contain ejecta mass as 'log10_mej' or seperate ejecta masses as 'log10_mej_dyn' and 'log10_mej_wind'."
        )


class EjectaResamplingLikelihood(Likelihood):
    """
    Likelihood of the GW-EM resampling: the binary and EOS parameters are
    mapped to ejecta masses, which are compared to the EM ejecta posterior.

    Parameters
    ----------
    GWsamples : pandas.DataFrame
        GW posterior samples. Reads ``luminosity_distance``, ``chirp_mass``
        and ``mass_ratio``, plus ``chi_1`` and ``chi_2`` when ``withNSBH``.
    EMsamples : pandas.DataFrame
        EM posterior samples, passed to :func:`construct_EM_KDE`.
    eos_converter : EoSConverter
        Converter on the tabulated EOSs, selected by the 0-based "EOS".
    withNSBH : bool
        True treats the binary as NSBH, False as BNS.
    combine_ejecta_mass : bool, default=False
        Passed to :func:`construct_EM_KDE`.
    """

    def __init__(
        self,
        GWsamples,
        EMsamples,
        eos_converter,
        withNSBH,
        combine_ejecta_mass=False,
    ):
        super().__init__()
        self.eos_converter = eos_converter
        self.withNSBH = withNSBH
        self.combine_ejecta_mass = combine_ejecta_mass

        cosmo_converter = CosmologyConverter()
        z = cosmo_converter.redshift(GWsamples.luminosity_distance.to_numpy())
        mc = GWsamples.chirp_mass.to_numpy() / (1 + z)
        q = GWsamples.mass_ratio.to_numpy()

        if withNSBH:
            self.chi_1KDE = scipy.stats.gaussian_kde(GWsamples.chi_1.to_numpy())
            self.chi_2KDE = scipy.stats.gaussian_kde(GWsamples.chi_2.to_numpy())
            self.components = [2]
        else:
            self.components = [1, 2]

        self.mcKDE = scipy.stats.gaussian_kde(mc)
        self.qKDE = scipy.stats.gaussian_kde(q)
        self.EMKDE = construct_EM_KDE(EMsamples, combine_ejecta_mass)

        self.NSBHEjecta = NSBHEjectaFitting()
        self.BNSEjecta = BNSEjectaFitting()

    def log_likelihood(self, parameters):
        """
        Evaluate the log likelihood of one point in parameter space.

        Parameters
        ----------
        parameters : dict
            ``chirp_mass``, ``mass_ratio``, ``EOS``, ``alpha`` and ``zeta``,
            plus ``chi_1`` and ``chi_2`` when ``withNSBH``.

        Returns
        -------
        float
            ``numpy.nan_to_num`` of the log likelihood plus the KDE log prior.
        """
        parameters = self.binary_and_eos_parameters(parameters)
        if not all(parameters[f"radius_{i}"] > 0 for i in self.components):
            return np.nan_to_num(-np.inf)

        logl = self.mcKDE.logpdf(parameters["chirp_mass"])
        logl += self.qKDE.logpdf(parameters["mass_ratio"])
        if self.withNSBH:
            logl += self.chi_1KDE.logpdf(parameters["chi_1"])
            logl += self.chi_2KDE.logpdf(parameters["chi_2"])

        ejecta_masses = self.ejecta_masses(parameters)
        if ejecta_masses is None:
            return np.nan_to_num(-np.inf)
        mdyn, mwind = ejecta_masses

        if self.combine_ejecta_mass:
            ejecta_mass = mdyn + mwind
        else:
            ejecta_mass = (mdyn, mwind)
        logl += self.EMKDE.logpdf(ejecta_mass)
        return np.nan_to_num(logl.item())

    def binary_and_eos_parameters(self, parameters):
        """Add the component masses and the EOS properties (TOV_mass, R_16,
        radius_i, is_ns_i, ...) to a copy of `parameters`."""
        q = parameters["mass_ratio"]
        eta = q / np.power(1.0 + q, 2.0)
        m1, m2 = chirp_mass_and_eta_to_component_masses(parameters["chirp_mass"], eta)
        # spins only enter the ejecta fits; the NS structure is taken non-rotating
        parameters.update(mass_1_source=m1, mass_2_source=m2, a_1=0.0, a_2=0.0)
        return self.eos_converter(parameters)

    def ejecta_masses(self, parameters):
        """Dynamical and wind ejecta mass, or None for a negative dynamical
        ejecta mass."""
        m1 = parameters["mass_1_source"]
        m2 = parameters["mass_2_source"]
        C2 = m2 / parameters["radius_2"] * geom_msun_km
        alpha, zeta = parameters["alpha"], parameters["zeta"]

        if self.withNSBH:
            q = parameters["mass_ratio"]
            chi_eff = (parameters["chi_1"] + q * parameters["chi_2"]) / (1 + q)
            mdyn = self.NSBHEjecta.dynamic_mass_fitting(m1, m2, C2, chi_eff) + alpha
            mdisk = self.NSBHEjecta.remnant_disk_mass_fitting(m1, m2, C2, chi_eff)
            log10_mdisk = np.log10(mdisk)
        else:
            C1 = m1 / parameters["radius_1"] * geom_msun_km
            R16 = parameters["R_16"] / geom_msun_km  # BNSEjectaFitting: geo units
            mdyn = self.BNSEjecta.dynamic_mass_fitting_KrFo(m1, m2, C1, C2) + alpha
            log10_mdisk = self.BNSEjecta.log10_disk_mass_fitting(
                m1 + m2, m2 / m1, parameters["TOV_mass"], R16
            )
        if mdyn < 0.0:
            return None
        return mdyn, 10 ** (np.log10(zeta) + log10_mdisk)

    def final_diagnostics(self, bestfit_params, args, result=None):
        """Draw the resampling corner plot of the posterior in `result`.

        Parameters
        ----------
        bestfit_params : dict
            Unused; part of the ``final_diagnostics`` interface.
        args : argparse.Namespace
            Reads ``outdir`` and ``label`` for the output file.
        result : bilby.core.result.Result, optional
            Nothing is done without it.

        Returns
        -------
        matplotlib.figure.Figure or None
        """
        if result is None:
            return None
        save = Path(args.outdir) / f"{args.label}_resampling_corner.png"
        return resampling_corner_plot(
            result.posterior, self.eos_converter, self.withNSBH, save
        )


def eos_sample_counts(eos_samples, Neos):
    """Number of samples per 0-based EOS index.

    Parameters
    ----------
    eos_samples : array_like
        EOS indices of the samples; cast to int.
    Neos : int
        Number of tabulated EOSs.

    Returns
    -------
    numpy.ndarray
        Counts, shape (Neos,).

    Raises
    ------
    ValueError
        If an index lies outside ``[0, Neos)``.
    """
    indices = np.asarray(eos_samples).astype(int)
    if indices.min() < 0 or indices.max() >= Neos:
        raise ValueError(
            f"EOS indices of the samples lie in [{indices.min()}, {indices.max()}], "
            f"outside the {Neos} tabulated EOSs."
        )
    return np.bincount(indices, minlength=Neos)


def setup_resampling_priors(GWprior, EMprior, eos_counts, withNSBH):
    """Priors of the resampling: binary parameters from `GWprior`, ejecta
    parameters from `EMprior`, and a categorical EOS prior weighted by the
    EOS counts of the GW posterior.

    Parameters
    ----------
    GWprior : bilby.core.prior.PriorDict
        Must contain chirp_mass and mass_ratio, plus chi_1 and chi_2 when
        `withNSBH`.
    EMprior : bilby.core.prior.PriorDict
        Must contain alpha and zeta.
    eos_counts : array_like
        Weights of the EOSs, see :func:`eos_sample_counts`.
    withNSBH : bool

    Returns
    -------
    bilby.core.prior.PriorDict
    """
    gw_keys = ["chirp_mass", "mass_ratio"]
    if withNSBH:
        gw_keys += ["chi_1", "chi_2"]
    priors = {key: GWprior[key] for key in gw_keys}
    priors.update({key: EMprior[key] for key in ["alpha", "zeta"]})
    return setup_tabulated_eos_priors(
        Namespace(Neos=len(eos_counts)), PriorDict(priors), weights=eos_counts
    )


def main_resampling(args=None):
    """
    Run the GW-EM resampling from the parsed arguments via ``bilby_sampling``.

    Parameters
    ----------
    args : argparse.Namespace, optional
        Pre-parsed args; if None, parsed via
        ``nmma_base_parsing(resampling_parser)``.

    Returns
    -------
    bilby.core.result.Result or None
    """
    if args is None:
        args = nmma_base_parsing(resampling_parser)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    GWsamples = pd.read_csv(args.GWsamples, header=0, delimiter=" ")
    if len(GWsamples) > args.max_samples:
        GWsamples = GWsamples.sample(args.max_samples, random_state=args.sampling_seed)
    EMsamples = pd.read_csv(args.EMsamples, header=0, delimiter=" ")

    ejecta_grid = {"n_points": 100, "m_min": 0.5, "power": 1.5}
    eos_converter = EoSConverter(
        generator=TabulatedEoSGenerator(args, mass_grid=ejecta_grid)
    )
    eos_counts = eos_sample_counts(GWsamples.EOS, eos_converter.generator.Neos)
    priors = setup_resampling_priors(
        PriorDict(args.GWprior), PriorDict(args.EMprior), eos_counts, args.withNSBH
    )

    sub_model = EjectaResamplingLikelihood(
        GWsamples, EMsamples, eos_converter, args.withNSBH, args.total_ejecta_mass
    )
    likelihood = NMMALikelihood(sub_model, priors)
    priors, likelihood = check_priors_and_likelihood_for_nmma(priors, likelihood)

    return bilby_sampling(likelihood, priors, args)


if __name__ == "__main__":
    main_resampling()
