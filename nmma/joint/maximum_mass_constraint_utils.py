import numpy as np
import scipy.stats
import scipy.interpolate
import scipy.integrate

from bilby.gw.prior import PriorDict
from bilby.core.prior import Uniform
import bilby.gw.conversion as conversion

import astropy.units as u
import astropy.constants as constants

import os
import sys
import contextlib


def fileno(file_or_fd):
    fd = getattr(file_or_fd, "fileno", lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), "wb") as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, "wb") as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def baryonic_mass(gravitational_mass, EOS, eos_path_macro, eos_path_micro):

    Msun_to_MeV = (
        (1 * u.Msun * constants.c**2).to(u.MeV).value
    )  # 1.1154E+60 MeV are one solar mass
    MeV_per_fm3_to_Msun_per_km3 = (
        1 / Msun_to_MeV * (1.0 * u.fm ** (-3)).to(u.km ** (-3)).value
    )  # 1 MeV/fm**3 is 8.9653E-7 Msun/km**3
    G = (
        (constants.G / constants.c**2).to(u.km / u.Msun).value
    )  # G/c^2 is 1.4766 km/Msun

    R, M, L, P0 = np.loadtxt(eos_path_macro + f"/{EOS}.dat", unpack=True, skiprows=0)
    N, EPS, P, CS2 = np.loadtxt(eos_path_micro + f"/{EOS}.dat", unpack=True, skiprows=0)
    eps_of_p = scipy.interpolate.interp1d(
        P, EPS, kind="linear", bounds_error=False, fill_value=0
    )

    def TOVeq(y, x):
        p, m = y
        eps = eps_of_p(p)

        Dp = (
            -G
            * m
            * eps
            / x**2
            * (1 + p / eps)
            * (1 + 4 * np.pi * (x**3 * p) * MeV_per_fm3_to_Msun_per_km3 / m)
            * (1 - 2 * G * m / x) ** (-1)
        )
        Dm = (4 * np.pi * x**2 * eps) * MeV_per_fm3_to_Msun_per_km3

        return [Dp, Dm]

    dr = 0.001
    r = np.interp(gravitational_mass, M, R)
    p0 = np.interp(gravitational_mass, M, P0)
    eps0 = eps_of_p(p0)
    m0 = (eps0 * 4 * np.pi / 3 * dr**3) * MeV_per_fm3_to_Msun_per_km3
    x = np.arange(dr, r + dr, dr)
    y0 = [p0, m0]

    with stdout_redirected():
        p_solv, m_solv = scipy.integrate.odeint(TOVeq, y0=y0, t=x).T
    n_solv = np.interp(p_solv, P, N)

    if np.any(np.isnan(p_solv)):
        cut = np.where(np.isnan(p_solv))[0][0]
        if np.any(np.isnan(m_solv)):
            cut = min(cut, np.where(np.isnan(m_solv))[0][0])
        n_solv = n_solv[:cut]
        m_solv = m_solv[:cut]
        x = x[:cut]

    n_solv = n_solv * (1.0 * u.fm ** (-3)).to(
        u.km ** (-3)
    )  # convert from fm**(-3) to km**(-3)
    particle_mass = constants.m_p.to(u.Msun).value  # get proton mass in Msun
    m_baryonic = (
        particle_mass
        * 4
        * np.pi
        * scipy.integrate.simpson(
            y=(n_solv) * x**2 / np.sqrt(1 - 2 * G * m_solv / x), x=x
        )
    )  # get baryonic mass in Msun

    if np.isnan(m_baryonic):
        import warnings

        warnings.warn("baryonic_mass returned nan")

    return m_baryonic


def baryonic_Kepler_mass(mTOV, R_14, ratio_R, delta):
    """
    see https://arxiv.org/abs/2307.03225 and https://arxiv.org/abs/1905.03784
    """
    m_max = ratio_R * mTOV
    m_max_b = m_max + 0.78 / R_14 * m_max**2
    m_max_b *= 1 + delta

    return m_max_b


from pymultinest.solve import Solver


class PostmergerInference(Solver):
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
        self.eos_path_macro = eos_path_macro
        self.eos_path_micro = eos_path_micro

        chirp_mass = self.posterior_samples.chirp_mass.to_numpy()
        eta_star = self.posterior_samples.eta_star.to_numpy()
        EOS = self.posterior_samples.EOS.to_numpy()
        log10_mej_dyn = self.posterior_samples.log10_mej_dyn.to_numpy()
        log10_mdisk = self.posterior_samples.log10_mdisk.to_numpy()

        self.KDE = scipy.stats.gaussian_kde(
            (chirp_mass, eta_star, EOS, log10_mdisk, log10_mej_dyn)
        )

        Solver.__init__(self, **kwargs)

    def Prior(self, x):
        return self.priors.rescale(self._search_parameter_keys, x)

    def LogLikelihood(self, x):

        if self.use_M_max:
            chirp_mass, eta_star, EOS, log10_mdisk, log10_mej_dyn, ratio_R, delta = x

        else:
            chirp_mass, eta_star, EOS, log10_mdisk, log10_mej_dyn = x

        logprior = self.KDE.logpdf(
            (chirp_mass, eta_star, EOS, log10_mdisk, log10_mej_dyn)
        )  # use the joint GW+EM posterior as "prior" here

        EOS = int(EOS) + 1
        R, M = np.loadtxt(
            self.eos_path_macro + f"/{EOS}.dat", unpack=True, usecols=[0, 1], skiprows=0
        )
        mTOV = M.max()
        R_14 = np.interp(1.4, M, R)

        q = conversion.symmetric_mass_ratio_to_mass_ratio(0.25 - np.exp(eta_star))
        mass_1, mass_2 = conversion.chirp_mass_and_mass_ratio_to_component_masses(
            chirp_mass, q
        )
        mdisk = 10**log10_mdisk
        mej_dyn = 10**log10_mej_dyn

        m_rem_b = (
            baryonic_mass(mass_1, EOS, self.eos_path_macro, self.eos_path_micro)
            + baryonic_mass(mass_2, EOS, self.eos_path_macro, self.eos_path_micro)
            - mdisk
            - mej_dyn
        )  # calculate the baryonic remnant mass

        if self.use_M_max:
            m_threshold = baryonic_Kepler_mass(
                mTOV, R_14, ratio_R, delta
            )  # if the Kepler limit is the threshold, use the quasiuniversal relation

        else:
            m_threshold = baryonic_mass(
                mTOV, EOS, self.eos_path_macro, self.eos_path_micro
            )  # if the TOV mass is the limit just determine its baryonic mass

        if m_threshold > m_rem_b:
            loglikelihood = np.nan_to_num(-np.inf)
        else:
            loglikelihood = 0

        return logprior + loglikelihood
