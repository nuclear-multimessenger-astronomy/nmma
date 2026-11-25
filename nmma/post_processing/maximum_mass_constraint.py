from argparse import Namespace
from pathlib import Path

import bilby.gw.conversion as conversion
import numpy as np
import pandas as pd
import scipy.integrate
import scipy.stats
from bilby.core.likelihood import Likelihood
from bilby.core.prior import PriorDict
from matplotlib import pyplot as plt

from ..core.base import (
    NMMALikelihood,
    bilby_sampling,
    check_priors_and_likelihood_for_nmma,
)
from ..core.constants import MeV_per_fm3_to_Msun_per_km3, geom_msun_km, particle_mass
from ..core.parsing import nmma_base_parsing
from ..eos.eos_likelihood import setup_tabulated_eos_priors
from ..eos.eos_processing import EoSConverter, TabulatedEoSGenerator
from .parser import maximum_mass_parser


def baryonic_Kepler_mass(mTOV, R_14, ratio_R, delta):
    """
    see https://arxiv.org/abs/2307.03225 and https://arxiv.org/abs/1905.03784
    """
    m_max = ratio_R * mTOV
    m_max_b = m_max + 0.78 / R_14 * m_max**2
    m_max_b *= 1 + delta

    return m_max_b


def _tov_equations(y, x, pressure, energy_density):
    """Right-hand side of the TOV equations for pressure and enclosed mass,
    with the energy density interpolated from the micro EOS table."""
    p, m = y
    eps = np.interp(p, pressure, energy_density)

    Dp = -geom_msun_km * m * eps / x**2 * (1 + p / eps)
    Dp *= 1 + 4 * np.pi * (x**3 * p) * MeV_per_fm3_to_Msun_per_km3 / m
    Dp *= (1 - 2 * geom_msun_km * m / x) ** (-1)
    Dm = (4 * np.pi * x**2 * eps) * MeV_per_fm3_to_Msun_per_km3

    return [Dp, Dm]


def baryonic_mass(gravitational_mass, macro_table, micro_table, dr=0.001):
    """Compute the baryonic mass (in Msun) of a star with the given gravitational mass.

    The central pressure and radius belonging to `gravitational_mass` are
    interpolated from the macro table. The TOV equations are then
    re-integrated outward from the centre to recover the pressure profile,
    which is mapped to the baryon number density via the micro table. The
    total baryon number is the integral of the density over the proper
    volume (including the GR volume-element correction
    1/sqrt(1 - 2GM/rc^2)), and the baryonic mass is that number times the
    particle mass.

    If the ODE integration goes unstable (produces NaNs) before reaching the
    stellar surface, the profile is truncated at the first NaN.

    Parameters
    ----------
    gravitational_mass : float
        Gravitational (ADM) mass, in solar masses.
    macro_table : array_like
        Rows (radius [km], mass [Msun], central pressure [MeV/fm^3]).
    micro_table : array_like
        Rows (baryon number density [fm^-3], energy density [MeV/fm^3],
        pressure [MeV/fm^3]).
    dr : float, default 0.001
        Radial step of the integration, in km.

    Returns
    -------
    float
        Baryonic mass, in solar masses; ``np.inf`` if the volume integral
        does not converge.
    """
    R, M, P0 = macro_table
    N, EPS, P = micro_table

    r = np.interp(gravitational_mass, M, R)
    p0 = np.interp(gravitational_mass, M, P0)
    eps0 = np.interp(p0, P, EPS)
    m0 = (eps0 * 4 * np.pi / 3 * dr**3) * MeV_per_fm3_to_Msun_per_km3
    x = np.arange(dr, r + dr, dr)

    p_solv, m_solv = scipy.integrate.odeint(
        _tov_equations, y0=[p0, m0], t=x, args=(P, EPS)
    ).T
    n_solv = np.interp(p_solv, P, N)

    if np.any(np.isnan(p_solv)):
        cut = np.where(np.isnan(p_solv))[0][0]
        if np.any(np.isnan(m_solv)):
            cut = min(cut, np.where(np.isnan(m_solv))[0][0])
        n_solv = n_solv[:cut]
        m_solv = m_solv[:cut]
        x = x[:cut]

    n_solv = n_solv * 1e54  # convert from fm**(-3) to km**(-3)
    integral_factor = scipy.integrate.simpson(
        y=n_solv * x**2 / np.sqrt(1 - 2 * geom_msun_km * m_solv / x), x=x
    )
    if np.isnan(integral_factor):
        return np.inf
    return 4 * np.pi * integral_factor * particle_mass


class TabulatedMicroEoS(TabulatedEoSGenerator):
    """Microphysical tables of tabulated EOSs, for computing baryonic masses.

    Holds, per EOS, the micro table (baryon number density, energy density,
    pressure) and the central pressures of the macro table (radius, mass,
    central pressure), both selected by the 0-based "EOS" parameter, as in
    ``TabulatedEoSGenerator``. All tables are read once at initialisation.

    Parameters
    ----------
    eos_path_micro : str or Path
        Directory (or glob pattern) of micro tables with columns
        [n, energy density, pressure, speed of sound squared].
    eos_path_macro : str or Path
        Directory (or glob pattern) of macro tables with columns
        [R, M, Lambda, central pressure].
    Neos : int, optional
        Number of EOSs; default all files found.
    """

    def __init__(self, eos_path_micro, eos_path_macro, Neos=None):
        self.eos_parameters = ["EOS"]

        self.read_eos_files(eos_path_macro, Neos)
        self.macro_tables = [
            self.read_eos_table(f, usecols=[0, 1, 3], test_scale=False)
            for f in self.eos_files
        ]
        self.read_eos_files(eos_path_micro, self.Neos)
        self.tables = [
            self.read_eos_table(f, usecols=[0, 1, 2], test_scale=False)
            for f in self.eos_files
        ]
        self.Neos = len(self.tables)
        self._tov_baryonic_masses = {}

    def __call__(self, parameters):
        """The micro tables of the requested EOSs."""
        return [self.tables[i] for i in self.selected_indices(parameters)]

    def baryonic_mass(self, gravitational_mass, eos_index):
        """Baryonic mass at `gravitational_mass` for EOS `eos_index`,
        see :func:`baryonic_mass`."""
        eos_index = int(eos_index)
        return baryonic_mass(
            gravitational_mass, self.macro_tables[eos_index], self.tables[eos_index]
        )

    def tov_baryonic_mass(self, eos_index):
        """Baryonic mass of the maximum-mass configuration of EOS `eos_index`,
        cached per EOS."""
        eos_index = int(eos_index)
        if eos_index not in self._tov_baryonic_masses:
            mTOV = self.macro_tables[eos_index][1].max()
            self._tov_baryonic_masses[eos_index] = self.baryonic_mass(mTOV, eos_index)
        return self._tov_baryonic_masses[eos_index]


class PostmergerLikelihood(Likelihood):
    """
    Likelihood for whether the remnant of a BNS merger collapsed to a black hole.

    It is assumed that the remnant collapsed to a black hole, so its baryonic
    mass must exceed the collapse threshold of the EOS: the baryonic TOV mass,
    or the baryonic Kepler mass if `use_M_max`.
    See https://arxiv.org/abs/2402.04172 for details.

    A joint GW+EM posterior enters as an informative "prior" density through a
    Gaussian KDE over (chirp_mass, eta_star, EOS, log10_mdisk, log10_mej_dyn),
    which is added to the log likelihood. The log evidence of a run with this
    likelihood is therefore not a physical evidence.

    Parameters
    ----------
    posterior_samples : Dataframe
        Joint posterior samples.
    eos_converter : EoSConverter
        Converter on the macro tables, providing TOV_mass and R_14.
    micro_eos : TabulatedMicroEoS
        Micro tables for the baryonic masses.
    use_M_max : bool, optional
        If True, use the quasi-universal Kepler-mass relation
        (`baryonic_Kepler_mass`) for the collapse threshold instead of the
        baryonic TOV mass (default False).
    """

    # parameters taken from the joint GW+EM posterior, in the order of the KDE
    kde_params = ["chirp_mass", "eta_star", "EOS", "log10_mdisk", "log10_mej_dyn"]

    def __init__(self, posterior_samples, eos_converter, micro_eos, use_M_max=False):
        super().__init__()

        kde_samples = posterior_samples[self.kde_params].to_numpy().T
        self.KDE = scipy.stats.gaussian_kde(kde_samples)
        self.eos_converter = eos_converter
        self.micro_eos = micro_eos
        self.use_M_max = use_M_max

    def log_likelihood(self, parameters):
        f"""KDE log density of the joint posterior plus 0 if the remnant
        collapses (baryonic remnant mass above the threshold), -inf otherwise.

        Parameters
        ----------
        parameters : dict
            Must contain {self.kde_params}, plus ratio_R and delta if
            `use_M_max`.

        Returns
        -------
        float
        """
        parameters = dict(parameters)
        logl = self.KDE.logpdf(
            [parameters[key] for key in self.kde_params]
        ).item()  # use the joint GW+EM posterior as "prior" here

        eos_index = int(parameters["EOS"])
        parameters = self.eos_converter.compute_macro_parameters(parameters)

        q = conversion.symmetric_mass_ratio_to_mass_ratio(
            0.25 - np.exp(parameters["eta_star"])
        )
        mass_1, mass_2 = conversion.chirp_mass_and_mass_ratio_to_component_masses(
            parameters["chirp_mass"], q
        )
        mdisk = 10 ** parameters["log10_mdisk"]
        mej_dyn = 10 ** parameters["log10_mej_dyn"]

        b1 = self.micro_eos.baryonic_mass(mass_1, eos_index)
        b2 = self.micro_eos.baryonic_mass(mass_2, eos_index)
        m_rem_b = b1 + b2 - mdisk - mej_dyn  # baryonic remnant mass

        if self.use_M_max:
            # the Kepler limit is the threshold: use the quasi-universal relation
            m_threshold = baryonic_Kepler_mass(
                parameters["TOV_mass"],
                parameters["R_14"],
                parameters["ratio_R"],
                parameters["delta"],
            )
        else:
            # the TOV mass is the limit: use its baryonic mass
            m_threshold = self.micro_eos.tov_baryonic_mass(eos_index)

        # a failed baryonic-mass integration (inf) on either side is rejected
        if (
            not np.isfinite(m_rem_b)
            or not np.isfinite(m_threshold)
            or m_threshold > m_rem_b
        ):
            return np.nan_to_num(-np.inf)
        return logl

    def final_diagnostics(self, bestfit_params, args, result=None):
        """Plot the posterior of the TOV mass and print its median and 90%
        credible interval.

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
        _, _, mTOV = self.eos_converter.generator({"EOS": result.posterior["EOS"]})
        low, med, high = np.quantile(mTOV, [0.05, 0.5, 0.95])
        print(
            f"TOV mass: {med:.3f} (+{high - med:.3f} / -{med - low:.3f}) Msun (90% CI)"
        )

        fig, ax = plt.subplots()
        ax.hist(mTOV, bins=30, density=True, histtype="step")
        for val in (low, med, high):
            ax.axvline(val, color="k", linestyle="--" if val != med else "-")
        ax.set_xlabel(r"$M_{\rm TOV}\,[M_\odot]$")
        ax.set_ylabel("probability density")
        save = Path(args.outdir) / f"{args.label}_TOV_mass.png"
        fig.savefig(save, bbox_inches="tight")
        return fig


def setup_postmerger_priors(prior, Neos, use_M_max=False):
    """Priors of the postmerger inference: the KDE parameters from `prior`
    (plus ratio_R and delta if `use_M_max`), and a categorical EOS prior.

    Parameters
    ----------
    prior : bilby.core.prior.PriorDict
        Must contain chirp_mass, eta_star, log10_mdisk, log10_mej_dyn, plus
        ratio_R and delta if `use_M_max`. Other entries are ignored.
    Neos : int
        Number of tabulated EOSs.
    use_M_max : bool, optional

    Returns
    -------
    bilby.core.prior.PriorDict

    Raises
    ------
    ValueError
        If a required prior is missing.
    """
    keys = [key for key in PostmergerLikelihood.kde_params if key != "EOS"]
    if use_M_max:
        keys += ["ratio_R", "delta"]
    missing = [key for key in keys if key not in prior]
    if missing:
        hint = (
            " With use_M_Kepler, priors for ratio_R and delta are required for the "
            "quasi-universal relation."
            if use_M_max
            else ""
        )
        raise ValueError(f"Prior is missing {missing}.{hint}")

    priors = PriorDict({key: prior[key] for key in keys})
    return setup_tabulated_eos_priors(Namespace(Neos=Neos), priors)


def maximum_mass_resampling(args):
    """CLI routine: run the postmerger inference via ``bilby_sampling``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed by ``maximum_mass_parser``: needs ``joint_posterior``,
        ``prior_file``, ``eos_path_macro``, ``eos_path_micro``,
        ``use_M_Kepler`` and the sampler settings of
        ``single_messenger_analysis_parsing``.

    Returns
    -------
    bilby.core.result.Result or None
    """
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    eos_converter = EoSConverter(
        generator=TabulatedEoSGenerator(Namespace(eos_data=args.eos_path_macro))
    )
    Neos = eos_converter.generator.Neos
    micro_eos = TabulatedMicroEoS(args.eos_path_micro, args.eos_path_macro, Neos)

    priors = setup_postmerger_priors(
        PriorDict(str(args.prior_file)), Neos, args.use_M_Kepler
    )

    posterior_samples = pd.read_csv(args.joint_posterior, header=0, delimiter=" ")
    sub_model = PostmergerLikelihood(
        posterior_samples, eos_converter, micro_eos, use_M_max=args.use_M_Kepler
    )
    likelihood = NMMALikelihood(sub_model, priors)
    priors, likelihood = check_priors_and_likelihood_for_nmma(priors, likelihood)

    return bilby_sampling(likelihood, priors, args)


def main(args=None):
    """CLI entrypoint: parse args (if not given) and run `maximum_mass_resampling`.

    Parameters
    ----------
    args : argparse.Namespace, optional
        Pre-parsed args; if None, parsed via
        `nmma_base_parsing(maximum_mass_parser)`.
    """
    if args is None:
        args = nmma_base_parsing(maximum_mass_parser)
    return maximum_mass_resampling(args)


if __name__ == "__main__":
    main()
