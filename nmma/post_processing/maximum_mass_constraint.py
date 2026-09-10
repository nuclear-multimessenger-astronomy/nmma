from pathlib import Path
import pandas as pd
import warnings
import numpy as np
import scipy.stats
import scipy.integrate

from bilby.gw.prior import PriorDict
from bilby.core.prior import Uniform
import bilby.gw.conversion as conversion
from .parser import maximum_mass_parser
from ..core.parsing import nmma_base_parsing
from ..core.constants import MeV_per_fm3_to_Msun_per_km3, geom_msun_km, particle_mass


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
        return self.priors.rescale(self._search_parameter_keys, x)

    def LogLikelihood(self, x):
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
        "get baryonic mass in Msun, based on M_grav and EoS"

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
    if args is None:
        args = nmma_base_parsing(maximum_mass_parser)
    maximum_mass_resampling(args)


if __name__ == "__main__":
    main()
