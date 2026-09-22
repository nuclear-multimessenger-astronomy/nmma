import warnings
from pathlib import Path

import bilby.gw.conversion as conversion
import numpy as np
import pandas as pd
import scipy.integrate
import scipy.stats
from bilby.core.prior import Uniform
from bilby.gw.prior import PriorDict

from ..core.constants import MeV_per_fm3_to_Msun_per_km3, geom_msun_km, particle_mass
from ..core.parsing import nmma_base_parsing
from .parser import maximum_mass_parser


def baryonic_Kepler_mass(mTOV, R_14, ratio_R, delta):
    """
    see https://arxiv.org/abs/2307.03225 and https://arxiv.org/abs/1905.03784
    """
    m_max = ratio_R * mTOV
    m_max_b = m_max + 0.78 / R_14 * m_max**2
    m_max_b *= 1 + delta

    return m_max_b


class PostmergerInferenceMixIn:
    """
    Class to sample over a joint GW+EM posterior and to determine the remnant mass of the system.
    It is assumed that the remnant collapsed to a black hole and thus the TOV mass of the EOS must be smaller than this number.
    See https://arxiv.org/abs/2402.04172 for details.

    Set up the postmerger sampler: priors, KDE prior-density estimate, and EOS paths.

    Builds a joint prior over (chirp_mass, eta_star, EOS, log10_mdisk,
    log10_mej_dyn[, ratio_R, delta] if use_M_max), and a Gaussian KDE
    over `posterior_samples`' corresponding columns used as an
    informative "prior" density inside `LogLikelihood`.

    Parameters
    ----------
    prior : dict or bilby.core.prior.PriorDict
        Must supply chirp_mass, eta_star, log10_mdisk, log10_mej_dyn
        priors, plus ratio_R/delta if `use_M_max`.
    posterior_samples : pandas.DataFrame
        Joint GW+EM posterior samples with chirp_mass, eta_star, EOS,
        log10_mej_dyn, log10_mdisk columns, used to build the KDE.
    Neos : int
        Number of tabulated EOS files (sets the EOS prior's upper bound).
    eos_path_macro, eos_path_micro : str or pathlib.Path
        Directories of per-EOS macro (R, M, Lambda, P0) and micro
        (N, EPS, P, CS2) tables, named "<EOS_index>.dat" (1-indexed).
    use_M_max : bool, optional
        If True, use the quasi-universal Kepler-mass relation
        (`baryonic_Kepler_mass`) for the remnant-collapse threshold
        instead of the EOS's own TOV mass (default False).
    **kwargs
        Forwarded to the sampler mixin (e.g. pymultinest.solve.Solver).
    """

    def __init__(
        self,
        prior,
        posterior_samples,
        Neos,
        eos_path_macro,
        eos_path_micro,
        use_M_max=False,
        **kwargs,
    ):
        self.posterior_samples = posterior_samples
        self.use_M_max = use_M_max

        self.priors = {
            "chirp_mass": prior["chirp_mass"],
            "eta_star": prior["eta_star"],
            "EOS": Uniform(name="EOS", minimum=0, maximum=Neos),
            "log10_mdisk": prior["log10_mdisk"],
            "log10_mej_dyn": prior["log10_mej_dyn"],
        }
        if self.use_M_max:
            self.priors["ratio_R"] = prior["ratio_R"]
            self.priors["delta"] = prior["delta"]

        self.priors = PriorDict(self.priors)
        self._search_parameter_keys = self.priors.keys()
        self.Neos = Neos
        self.eos_path_macro = Path(eos_path_macro)
        self.eos_path_micro = Path(eos_path_micro)

        chirp_mass = self.posterior_samples.chirp_mass.to_numpy()
        eta_star = self.posterior_samples.eta_star.to_numpy()
        EOS = self.posterior_samples.EOS.to_numpy()
        log10_mej_dyn = self.posterior_samples.log10_mej_dyn.to_numpy()
        log10_mdisk = self.posterior_samples.log10_mdisk.to_numpy()

        self.KDE = scipy.stats.gaussian_kde(
            (chirp_mass, eta_star, EOS, log10_mdisk, log10_mej_dyn)
        )
        super().__init__(**kwargs)

    def Prior(self, x):
        """Rescale unit-cube samples to the physical prior, for the nested sampler.

        Parameters
        ----------
        x : array_like
            Unit-cube samples, one per search-parameter key.

        Returns
        -------
        array_like
            Physical parameter values.
        """
        return self.priors.rescale(self._search_parameter_keys, x)

    def LogLikelihood(self, x):
        """Log-likelihood for whether the remnant collapses to a black hole.

        Uses the joint GW+EM posterior KDE as an informative prior term,
        then imposes a hard cut: the baryonic remnant mass (from the
        binary's baryonic masses minus the ejected/disk mass) must exceed
        the collapse threshold (the EOS's baryonic TOV mass, or the
        baryonic Kepler mass if `use_M_max`) for the point to be allowed
        (loglikelihood 0 if satisfied, -inf otherwise).

        Parameters
        ----------
        x : tuple
            (chirp_mass, eta_star, EOS, log10_mdisk, log10_mej_dyn[,
            ratio_R, delta] if use_M_max) -- physical parameter values
            from `Prior`.

        Returns
        -------
        float
            logprior (from the posterior-sample KDE) + 0 or -inf.
        """

        if self.use_M_max:
            chirp_mass, eta_star, EOS, log10_mdisk, log10_mej_dyn, ratio_R, delta = x

        else:
            chirp_mass, eta_star, EOS, log10_mdisk, log10_mej_dyn = x

        logprior = self.KDE.logpdf(
            (chirp_mass, eta_star, EOS, log10_mdisk, log10_mej_dyn)
        ).item()  # use the joint GW+EM posterior as "prior" here

        EOS = int(EOS) + 1
        R, M = np.loadtxt(
            self.eos_path_macro / f"{EOS}.dat", unpack=True, usecols=[0, 1], skiprows=0
        )
        mTOV = M.max()
        R_14 = np.interp(1.4, M, R)

        q = conversion.symmetric_mass_ratio_to_mass_ratio(0.25 - np.exp(eta_star))
        mass_1, mass_2 = conversion.chirp_mass_and_mass_ratio_to_component_masses(
            chirp_mass, q
        )
        mdisk = 10**log10_mdisk
        mej_dyn = 10**log10_mej_dyn

        b1 = self.baryonic_mass(mass_1, EOS)
        b2 = self.baryonic_mass(mass_2, EOS)
        m_rem_b = b1 + b2 - mdisk - mej_dyn  # calculate the baryonic remnant mass

        if self.use_M_max:
            m_threshold = baryonic_Kepler_mass(
                mTOV, R_14, ratio_R, delta
            )  # if the Kepler limit is the threshold, use the quasiuniversal relation

        else:
            m_threshold = self.baryonic_mass(
                mTOV, EOS
            )  # if the TOV mass is the limit just determine its baryonic mass

        if m_threshold > m_rem_b:
            loglikelihood = np.nan_to_num(-np.inf)
        else:
            loglikelihood = 0

        return logprior + loglikelihood

    def baryonic_mass(self, gravitational_mass, EOS):
        """Compute the baryonic mass (in Msun) of a given EOS at a given gravitational mass.

        Loads the EOS's macro (R, M, Lambda, central pressure) and micro
        (baryon density, energy density, pressure, sound speed squared)
        tables, locates the central pressure/radius corresponding to
        `gravitational_mass` via interpolation on the macro M-R/M-P0
        relations, then re-integrates the TOV equations for pressure and
        enclosed mass outward from the stellar center (`TOVeq`, via
        `scipy.integrate.odeint`) to recover the full pressure profile.
        The pressure profile is mapped back to baryon number density via
        the micro EOS table, and the total baryon number is obtained by
        integrating baryon density over the proper volume (including the
        GR volume-element correction 1/sqrt(1 - 2GM/rc^2)) via Simpson's
        rule; the baryonic mass is that baryon number times the particle
        mass.

        If the ODE integration goes unstable (produces NaNs) before
        reaching the stellar surface, the profile is truncated at the
        first NaN and the integral is computed over the truncated range.

        Note: this reloads both EOS tables from disk and re-solves the
        full TOV ODE on every call, even for a repeated (gravitational_mass,
        EOS) pair -- e.g. `LogLikelihood` calls this 2-3 times per
        evaluation for the *same* EOS (mass_1, mass_2, and mTOV), with no
        caching (see review notes on performance).

        Parameters
        ----------
        gravitational_mass : float
            Gravitational (ADM) mass, in solar masses, at which to
            evaluate the baryonic mass.
        EOS : int
            1-indexed EOS identifier; loads
            "{eos_path_macro}/{EOS}.dat" and "{eos_path_micro}/{EOS}.dat".

        Returns
        -------
        float
            Baryonic mass, in solar masses. May be `nan` if the volume
            integral doesn't converge (a `UserWarning` is emitted in that
            case via `warnings.warn`).
        """

        R, M, L, P0 = np.loadtxt(
            self.eos_path_macro / f"{EOS}.dat", unpack=True, skiprows=0
        )
        N, EPS, P, CS2 = np.loadtxt(
            self.eos_path_micro / f"{EOS}.dat", unpack=True, skiprows=0
        )

        def TOVeq(y, x):
            p, m = y
            eps = np.interp(p, P, EPS)

            Dp = -geom_msun_km * m * eps / x**2 * (1 + p / eps)
            Dp *= 1 + 4 * np.pi * (x**3 * p) * MeV_per_fm3_to_Msun_per_km3 / m
            Dp *= (1 - 2 * geom_msun_km * m / x) ** (-1)
            Dm = (4 * np.pi * x**2 * eps) * MeV_per_fm3_to_Msun_per_km3

            return [Dp, Dm]

        dr = 0.001
        r = np.interp(gravitational_mass, M, R)
        p0 = np.interp(gravitational_mass, M, P0)
        eps0 = np.interp(p0, P, EPS)
        m0 = (eps0 * 4 * np.pi / 3 * dr**3) * MeV_per_fm3_to_Msun_per_km3
        x = np.arange(dr, r + dr, dr)
        y0 = [p0, m0]

        p_solv, m_solv = scipy.integrate.odeint(TOVeq, y0=y0, t=x).T
        n_solv = np.interp(p_solv, P, N)

        if np.any(np.isnan(p_solv)):
            cut = np.where(np.isnan(p_solv))[0][0]
            if np.any(np.isnan(m_solv)):
                cut = min(cut, np.where(np.isnan(m_solv))[0][0])
            n_solv = n_solv[:cut]
            m_solv = m_solv[:cut]
            x = x[:cut]

        n_solv *= 1e54  # convert from fm**(-3) to km**(-3)
        integral_factor = scipy.integrate.simpson(
            y=(n_solv) * x**2 / np.sqrt(1 - 2 * geom_msun_km * m_solv / x), x=x
        )
        if np.isnan(integral_factor):
            warnings.warn("baryonic_mass returned nan")
        m_baryonic = 4 * np.pi * integral_factor * particle_mass  #
        return m_baryonic


def maximum_mass_resampling(args):
    """CLI routine: run the postmerger nested-sampling inference and save its posterior.

    Loads the joint posterior samples and prior, counts the available
    tabulated EOS files, builds a `PostmergerInferenceMixIn` +
    `pymultinest.solve.Solver` sampler, runs it, and writes the
    resulting samples to '{args.outdir}/posterior_samples.dat'.

    Parameters
    ----------
    args : argparse.Namespace
        Needs `outdir`, `joint_posterior` (CSV path), `prior` (prior
        file path), `eos_path_macro`, `eos_path_micro`, `use_M_Kepler`,
        `nlive`. If `use_M_Kepler` is True, `prior` must additionally
        define ratio_R and delta.

    Raises
    ------
    Exception
        If `use_M_Kepler` is True but the prior doesn't supply exactly
        the extra ratio_R/delta parameters needed.
    """

    outputfiles = Path(args.outdir, "pm")
    outputfiles.mkdir(parents=True, exist_ok=True)

    posterior_samples = pd.read_csv(args.joint_posterior, header=0, delimiter=" ")
    prior = PriorDict(str(args.prior))
    Neos = len(list(Path(args.eos_path_macro).iterdir()))

    if args.use_M_Kepler:
        n_dims = 7
        if len(prior.keys()) != n_dims - 1:
            raise Exception(
                "If use_M_kepler is True, you need to provide a prior fo ratio_R and delta to be used in the quasi-universal relation."
            )

    else:
        n_dims = 5

    pymulti_kwargs = dict(
        outputfiles_basename=str(outputfiles),
        n_dims=n_dims,
        n_live_points=args.nlive,
        verbose=True,
        resume=True,
        seed=42,
        importance_nested_sampling=False,
        use_MPI=True,
    )

    # postpone import to avoid issues when pymultinest is not installed
    from pymultinest.solve import Solver

    class PostmergerInference(PostmergerInferenceMixIn, Solver):
        pass

    solution = PostmergerInference(
        prior,
        posterior_samples,
        Neos,
        args.eos_path_macro,
        args.eos_path_micro,
        args.use_M_Kepler,
        **pymulti_kwargs,
    )

    samples = solution.samples.T
    posterior = dict()

    posterior["chirp_mass"] = samples[0]
    posterior["eta_star"] = samples[1]
    posterior["EOS"] = samples[2]
    posterior["log10_mdisk"] = samples[3]
    posterior["log10_mej_dyn"] = samples[4]

    if args.use_M_Kepler:
        posterior["ratio_R"] = samples[5]
        posterior["delta"] = samples[6]

    posterior = pd.DataFrame.from_dict(posterior)
    posterior.to_csv(args.outdir + "/posterior_samples.dat", sep=" ", index=False)


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
    maximum_mass_resampling(args)


if __name__ == "__main__":
    main()
