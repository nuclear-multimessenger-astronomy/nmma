import numpy as np
import pandas as pd
from astropy import cosmology as cosmo
from astropy import units
from bilby.gw.conversion import (
    chirp_mass_and_mass_ratio_to_total_mass,
    component_masses_to_chirp_mass,
    component_masses_to_symmetric_mass_ratio,
    convert_to_lal_binary_black_hole_parameters,
    convert_to_lal_binary_neutron_star_parameters,
    generate_mass_parameters,
    lambda_1_lambda_2_to_lambda_tilde,
)
from scipy.integrate import simpson
from scipy.special import erf

from .constants import geom_msun_km, get_cosmology, msun_s, msun_to_ergs, set_cosmology


def val_to_scalar(val):
    """
    Convert single-value quantities to scalars for easier handling

    Parameters
    ----------
    val : scalar or array_like

    Returns
    -------
    ``val`` if it is already a scalar, ``val.item()`` if it has size 1,
    and ``numpy.asarray(val)`` otherwise.
    """
    if np.isscalar(val):
        return val
    else:
        val = np.asarray(val)
        if val.size == 1:
            return val.item()
        return val


# =================== distance conversions # ===================
def distance_modulus_nmma(d_lum=1e-5):
    # mag_app = mag_abs + 5* log10(dist/10pc) | NMMA-dist is in Mpc
    #         = mag_abs + 5 * (log10(Mpc/10pc)+ log10(params["luminosity_distance"]))
    # therefore: distance_modulus = mag_app - mag_abs =
    """
    Distance modulus for a luminosity distance in Mpc.

    Parameters
    ----------
    d_lum : float or array_like, default=1e-5
        Luminosity distance in Mpc.

    Returns
    -------
    The distance modulus.
    """
    return 5.0 * (5 + np.log10(d_lum))


def luminosity_distance_to_redshift(distance, cosmology=None):
    """
    Convert luminosity distance to redshift.

    Parameters
    ----------
    distance : float, array_like or pandas.Series
        Luminosity distance in Mpc. More than 50 values are interpolated
        on the grids from :func:`get_cosmo_grids`.
    cosmology : default=None
        If None, :func:`nmma.core.constants.get_cosmology` is used.

    Returns
    -------
    The redshift.
    """
    if cosmology is None:
        cosmology = get_cosmology()
    if isinstance(distance, pd.Series):
        distance = distance.values

    if hasattr(distance, "__len__") and len(distance) > 50:
        d_min, d_max = distance.min(), distance.max()
        dist_grid, z_grid = get_cosmo_grids(d_min, d_max, cosmology)
        return np.interp(distance, dist_grid, z_grid).value
    else:
        return cosmo.z_at_value(
            cosmology.luminosity_distance, distance * units.Mpc
        ).value


def get_cosmo_grids(distance_min, distance_max, cosmology):
    # luminosity_distance_to_redshift gets really slow if too many distances are put in at once
    """
    Build matching luminosity distance and redshift grids.

    Parameters
    ----------
    distance_min, distance_max : float
        Grid endpoints in Mpc.
    cosmology
        Cosmology used for the conversion.

    Returns
    -------
    numpy.ndarray
        The distance grid.
    numpy.ndarray
        The redshift grid, 50 points from ``numpy.geomspace``.
    """
    zmin = cosmo.z_at_value(cosmology.luminosity_distance, distance_min * units.Mpc)
    zmax = cosmo.z_at_value(cosmology.luminosity_distance, distance_max * units.Mpc)
    z_grid = np.geomspace(zmin, zmax, 50)
    dist_grid = cosmology.luminosity_distance(z_grid).value
    return dist_grid, z_grid


def get_redshift(parameters):
    """
    Return the redshift held in ``parameters``.

    Uses ``redshift`` if present, otherwise converts
    ``luminosity_distance``, otherwise returns zeros shaped like the first
    value in ``parameters``.

    Parameters
    ----------
    parameters : dict

    Returns
    -------
    The redshift.
    """
    if "redshift" in parameters:
        return parameters["redshift"]
    elif "luminosity_distance" in parameters:
        return luminosity_distance_to_redshift(parameters["luminosity_distance"])
    else:
        # zeros like the first input of parameters, independent of size and keys
        return np.zeros_like(next(iter(parameters.values())))


def cosmology_to_distance(parameters):
    """
    Fill in ``redshift`` or ``luminosity_distance`` from the other.

    Clones the current cosmology with ``Hubble_constant`` and
    ``Omega_matter`` from ``parameters`` where present. Array-valued
    cosmology parameters are handled one entry at a time.

    Parameters
    ----------
    parameters : dict

    Returns
    -------
    dict
        ``parameters``, updated.
    """
    cosmology = get_cosmology()
    cosmo_parameters = {}
    if "Hubble_constant" in parameters:
        cosmo_parameters["H0"] = parameters["Hubble_constant"]
    if "Omega_matter" in parameters:
        cosmo_parameters["Om0"] = parameters["Omega_matter"]
    # Maybe extend for an even wilder cosmology?
    try:
        alt_cosmo = cosmology.clone(**cosmo_parameters)
        if "luminosity_distance" in parameters:
            # if luminosity distance is available, we assume it is in Mpc
            parameters["redshift"] = luminosity_distance_to_redshift(
                parameters["luminosity_distance"], cosmology=alt_cosmo
            )
        elif "redshift" in parameters:
            parameters["luminosity_distance"] = alt_cosmo.luminosity_distance(
                parameters["redshift"]
            ).value
        else:
            raise KeyError(
                "Either redshift or luminosity_distance must be in parameters"
            )

    except ValueError:
        # if H0 is an array, .clone raises a ValueError
        # in that case we turn a dict with len-n values into a len-n list of dicts with single values
        cosmo_dicts = [
            dict(zip(cosmo_parameters.keys(), vals))
            for vals in zip(*cosmo_parameters.values())
        ]
        alt_cosmos = [cosmology.clone(**cosmo_dict) for cosmo_dict in cosmo_dicts]

        if "luminosity_distance" in parameters:
            # if luminosity distance is available, we assume it is in Mpc
            parameters["redshift"] = np.array(
                [
                    luminosity_distance_to_redshift(
                        parameters["luminosity_distance"][i], cosmology=alt_cosmo
                    )
                    for i, alt_cosmo in enumerate(alt_cosmos)
                ]
            )

        elif "redshift" in parameters:
            parameters["luminosity_distance"] = np.array(
                [
                    alt_cosmo.luminosity_distance(parameters["redshift"][i]).value
                    for i, alt_cosmo in enumerate(alt_cosmos)
                ]
            )
    return parameters


def source_frame_masses(converted_parameters):
    """
    Add ``mass_1_source`` and ``mass_2_source`` to the parameters.

    Applies ``generate_mass_parameters`` first, and fills in ``redshift``
    from ``luminosity_distance`` when it is missing. Source-frame masses
    already present are left as they are.

    Parameters
    ----------
    converted_parameters : dict

    Returns
    -------
    dict
        ``converted_parameters``, updated.
    """
    converted_parameters = generate_mass_parameters(converted_parameters)
    if "redshift" not in converted_parameters:
        distance = converted_parameters["luminosity_distance"]
        converted_parameters["redshift"] = luminosity_distance_to_redshift(distance)
    z = converted_parameters["redshift"]

    if "mass_1_source" not in converted_parameters:
        converted_parameters["mass_1_source"] = np.array(
            converted_parameters["mass_1"] / (1 + z)
        )

    if "mass_2_source" not in converted_parameters:
        converted_parameters["mass_2_source"] = np.array(
            converted_parameters["mass_2"] / (1 + z)
        )

    return converted_parameters


def observation_angle_conversion(parameters):
    """
    Add ``KNtheta`` and ``inclination_EM`` to the parameters.

    ``KNtheta`` is in degrees and ``inclination_EM`` in radians; each is
    filled from the other, or from ``theta_jn``/``cos_theta_jn`` after
    ``numpy.minimum(theta_jn, pi - theta_jn)``.

    Parameters
    ----------
    parameters : dict

    Returns
    -------
    dict
        ``parameters``, updated.
    """
    theta_jn = parameters.get(
        "theta_jn", np.arccos(parameters.get("cos_theta_jn", 1.0))
    )
    theta_jn = np.minimum(
        theta_jn, np.pi - theta_jn
    )  # default effective 0 if neither is given
    if "KNtheta" not in parameters:
        parameters["KNtheta"] = (
            parameters.get("inclination_EM", theta_jn) * 180.0 / np.pi
        )
    if "inclination_EM" not in parameters:
        parameters["inclination_EM"] = parameters["KNtheta"] / 180.0 * np.pi
    return parameters


# =================== mass conversions  ===================


def bbh_source_frame(params):
    """
    Convert parameters to BBH parameters using bilby function.

    Parameters
    ----------
    params : dict

    Returns
    -------
    dict
        :func:`source_frame_masses` of the output of
        ``convert_to_lal_binary_black_hole_parameters``.
    """
    params, _ = convert_to_lal_binary_black_hole_parameters(params)
    return source_frame_masses(params)


def bns_source_frame(params):
    """
    Convert parameters to BNS parameters using bilby function.

    Parameters
    ----------
    params : dict

    Returns
    -------
    dict
        :func:`source_frame_masses` of the output of
        ``convert_to_lal_binary_neutron_star_parameters``.
    """
    params, _ = convert_to_lal_binary_neutron_star_parameters(params)
    return source_frame_masses(params)


def mass_ratio_to_eta(q):
    """
    Convert mass ratio ``q`` to ``q / (1 + q) ** 2``.

    Parameters
    ----------
    q : float or array_like

    Returns
    -------
    The symmetric mass ratio.
    """
    return q / (1 + q) ** 2


def component_masses_to_mass_quantities(m1, m2):
    """
    Convert component masses to chirp mass, symmetric mass ratio and
    mass ratio.

    Parameters
    ----------
    m1, m2 : float or array_like

    Returns
    -------
    tuple
        ``(mchirp, eta, q)``, with ``q = m2 / m1``.
    """
    eta = m1 * m2 / ((m1 + m2) * (m1 + m2))
    mchirp = ((m1 * m2) ** (3.0 / 5.0)) * ((m1 + m2) ** (-1.0 / 5.0))
    q = m2 / m1

    return (mchirp, eta, q)


def chirp_mass_and_eta_to_component_masses(mc, eta):
    """
    Utility function for converting mchirp,eta to component masses. The
    masses are defined so that m1>m2. The rvalue is a tuple (m1,m2).

    Parameters
    ----------
    mc, eta : float or array_like

    Returns
    -------
    tuple
        ``(m1, m2)``.
    """
    M = mc / np.power(eta, 3.0 / 5.0)
    q = (1 - np.sqrt(1.0 - 4.0 * eta) - 2 * eta) / (2.0 * eta)

    m1 = M / (1.0 + q)
    m2 = M * q / (1.0 + q)

    return (m1, m2)


def tidal_deformabilities_and_mass_ratio_to_eff_tidal_deformabilities(
    lambda1, lambda2, q
):
    """
    Convert component tidal deformabilities and mass ratio to the
    effective tidal deformabilities.

    Parameters
    ----------
    lambda1, lambda2, q : float or array_like

    Returns
    -------
    tuple
        ``(lambdaT, dlambdaT)``.
    """
    eta = q / np.power(1.0 + q, 2.0)
    eta2 = eta * eta
    eta3 = eta2 * eta
    root14eta = np.sqrt(1.0 - 4 * eta)

    lambdaT = (8.0 / 13.0) * (
        (1.0 + 7 * eta - 31 * eta2) * (lambda1 + lambda2)
        + root14eta * (1.0 + 9 * eta - 11.0 * eta2) * (lambda1 - lambda2)
    )
    dlambdaT = 0.5 * (
        root14eta
        * (1.0 - 13272.0 * eta / 1319.0 + 8944.0 * eta2 / 1319.0)
        * (lambda1 + lambda2)
        + (
            1.0
            - 15910.0 * eta / 1319.0
            + 32850.0 * eta2 / 1319.0
            + 3380.0 * eta3 / 1319.0
        )
        * (lambda1 - lambda2)
    )

    return lambdaT, dlambdaT


def reweight_to_flat_mass_prior(df):
    """
    Resample ``df`` weighted by ``mass_1 ** 2 / chirp_mass``.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``chirp_mass`` and ``mass_ratio``.

    Returns
    -------
    pandas.DataFrame
        A sample of 30% of the rows.
    """
    total_mass = chirp_mass_and_mass_ratio_to_total_mass(df.chirp_mass, df.mass_ratio)
    m1 = total_mass / (1.0 + df.mass_ratio)
    jacobian = m1 * m1 / df.chirp_mass
    df_new = df.sample(frac=0.3, weights=jacobian)
    return df_new


def convert_mtot_mni(params):
    """
    Add ``mni``, ``mtot``, ``mrp``, ``mni_c`` and ``mrp_c`` to ``params``.

    Each of ``mni``, ``mtot`` and ``mrp`` is filled from its ``log10_``
    counterpart when absent; ``mrp_c`` also reads ``xmix``.

    Parameters
    ----------
    params : dict

    Returns
    -------
    dict
        ``params``, updated.
    """
    for par in ["mni", "mtot", "mrp"]:
        if par not in params:
            params[par] = 10 ** params[f"log10_{par}"]

    params["mni_c"] = params["mni"] / params["mtot"]
    params["mrp_c"] = params["xmix"] * (params["mtot"] - params["mni"]) - params["mrp"]
    return params


# =================== pulsar timing conversions ===================
def binary_mass_function(m_obs, m_comp, sin_i):
    """
    Binary mass function ``(m_comp * sin_i) ** 3 / (m_obs + m_comp) ** 2``.

    Parameters
    ----------
    m_obs, m_comp, sin_i : float or array_like

    Returns
    -------
    The binary mass function.
    """
    return (m_comp * sin_i) ** 3 / (m_obs + m_comp) ** 2


def shapiro_delay(m_comp, sin_i):
    """
    see https://arxiv.org/pdf/1007.0933.pdf

    Parameters
    ----------
    m_comp, sin_i : float or array_like

    Returns
    -------
    The delay in microseconds.
    """
    shapiro_range = msun_s * 1.0e6 * m_comp  # in microseconds
    orthometric_ratio = sin_i / (1 + np.sqrt(1 - sin_i**2))
    return shapiro_range * orthometric_ratio**3


def einstein_delay_orbital_factor(orbital_period, eccentricity):
    """
    see, e.g., 10.1007/978-3-662-62110-3_1, p.12

    Parameters
    ----------
    orbital_period, eccentricity : float or array_like

    Returns
    -------
    The factor taken by :func:`simplified_einstein_delay` as
    ``einstein_factor``.
    """
    return msun_s ** (2 / 3) * eccentricity * (orbital_period / 2 / np.pi) ** (1 / 3)


def simplified_einstein_delay(m_psr, m_comp, einstein_factor):
    """
    see, e.g., 10.1007/978-3-662-62110-3_1, p.12

    Parameters
    ----------
    m_psr, m_comp : float or array_like
    einstein_factor : float or array_like
        As returned by :func:`einstein_delay_orbital_factor`.

    Returns
    -------
    The Einstein delay.
    """
    return einstein_factor * m_comp * (m_psr + 2 * m_comp) / (m_psr + m_comp) ** (4 / 3)


def einstein_delay(m_psr, m_comp, orbital_period, eccentricity):
    """
    see, e.g., 10.1007/978-3-662-62110-3_1, p.12

    Parameters
    ----------
    m_psr, m_comp, orbital_period, eccentricity : float or array_like

    Returns
    -------
    :func:`simplified_einstein_delay` evaluated with the factor from
    :func:`einstein_delay_orbital_factor`.
    """
    einstein_delay_factor = einstein_delay_orbital_factor(orbital_period, eccentricity)
    return simplified_einstein_delay(m_psr, m_comp, einstein_delay_factor)


def mass_parameters_to_sini(total_mass, mass_function, m_comp):
    """
    Invert the binary mass function to get sin(i) for a given total mass and mass function

    Parameters
    ----------
    total_mass, mass_function, m_comp : float or array_like

    Returns
    -------
    ``sin(i)``.
    """
    return np.cbrt(mass_function * total_mass**2) / m_comp


# =================== EOS-related conversions ===================


def EOS_to_ns_parameters(radii, masses, lambdas):
    """
    Extract ``TOV_mass``, ``TOV_radius``, ``R_14`` and ``R_16`` from a
    tabulated EOS.

    Parameters
    ----------
    radii, masses, lambdas : array_like
        Tabulated EOS columns. ``lambdas`` is accepted but not used.

    Returns
    -------
    tuple
        ``(TOV_mass, TOV_radius, R_14, R_16)``. The radii at 1.4 and 1.6
        are interpolated, and are 0 outside the tabulated mass range.
    """
    TOV_mass = masses.max(axis=-1)
    TOV_radius = radii[np.argmax(masses)]
    R_14, R_16 = np.interp(x=[1.4, 1.6], xp=masses, fp=radii, left=0, right=0)

    return TOV_mass, TOV_radius, R_14, R_16


def EOS_to_system_parameters(radii, masses, lambdas, m1_source, m2_source):
    """
    Interpolate a tabulated EOS at the two component masses.

    Parameters
    ----------
    radii, masses, lambdas : array_like
        Tabulated EOS columns.
    m1_source, m2_source : float or array_like
        Source-frame component masses.

    Returns
    -------
    tuple
        ``(lambda_1, lambda_2, radius_1, radius_2)``. The deformabilities
        are interpolated in ``log``; both they and the radii come out as 0
        outside the tabulated mass range.
    """
    (log_lambda_1, log_lambda_2) = np.interp(
        x=[m1_source, m2_source],
        xp=masses,
        fp=np.log(lambdas),
        left=-np.inf,
        right=-np.inf,
    )
    lambda_1 = np.exp(log_lambda_1)
    lambda_2 = np.exp(log_lambda_2)
    (radius_1, radius_2) = np.interp(
        x=[m1_source, m2_source], xp=masses, fp=radii, left=0, right=0
    )

    return lambda_1, lambda_2, radius_1, radius_2


def radii_from_qur(parameters):
    """
    Add ``radius_1``, ``radius_2`` and ``R_16`` to ``parameters``.

    Reads ``mass_1_source``, ``mass_2_source``, ``lambda_1`` and
    ``lambda_2``, and converts each deformability with
    :func:`lambda_to_compactness`.

    Parameters
    ----------
    parameters : dict

    Returns
    -------
    dict
        ``parameters``, updated.
    """
    mass_1_source = parameters["mass_1_source"]
    mass_2_source = parameters["mass_2_source"]
    lambda_1 = parameters["lambda_1"]
    lambda_2 = parameters["lambda_2"]

    compactness_1 = lambda_to_compactness(lambda_1)
    parameters["radius_1"] = mass_and_compactness_to_radius(
        mass_1_source, compactness_1
    )

    compactness_2 = lambda_to_compactness(lambda_2)
    parameters["radius_2"] = mass_and_compactness_to_radius(
        mass_2_source, compactness_2
    )

    chirp_mass_source = component_masses_to_chirp_mass(mass_1_source, mass_2_source)
    lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(
        lambda_1, lambda_2, mass_1_source, mass_2_source
    )

    parameters["R_16"] = (
        chirp_mass_source * np.power(lambda_tilde / 0.0042, 1.0 / 6.0) * geom_msun_km
    )
    return parameters


def lambda_to_compactness(lambda_i):
    """
    Function to link tidal deformability to compactness based on quasi-universal relation

    Parameters
    ----------
    lambda_i : float or array_like

    Returns
    -------
    The compactness.
    """
    loglam = np.log(lambda_i)
    return 0.371 - 0.0391 * loglam + 0.001056 * loglam * loglam


def mass_and_compactness_to_radius(mass, comp):
    # returns 0 if compactness is greater than 0.5, i.e. black hole
    """
    Convert mass and compactness to radius, returning 0 where ``comp >= 0.5``.

    Parameters
    ----------
    mass, comp : float or array_like

    Returns
    -------
    The radius.
    """
    return np.where(comp < 0.5, mass / comp * geom_msun_km, 0.0)


# =================== GRB-related conversions ===================


def gaussian_jet_energy_to_central_isotropic_energy_equivalent(
    Ejet, thetaCore, alphaWing
):
    """
    Takes the total energy of a gaussian jet as well as the angular parameters and returns the isotropic-energy equivalent
    on axis. This means it is assumed that the true jet energy follows some angular structure dEjet / dOmega = epsilon_c * exp(-1/2 theta^2/thetac^2).
    Then the distribution of the isotropic energy equivalent is simply related according to E_iso(theta) = 4pi dEjet / dOmega.

    :param Ejet: Total jet energy in ergs
    :param thetaCore: Core angle in rad
    :param alphaWing: Ratio of the wing angle and core angle.
    :return: The on-axis isotropic-energy equivalent.
    """

    # this is the analytical expression for int_{0}^{alphaWing*thetaCore} sin(x) *exp(-1/2 (x/thetac)^2) dx
    prefactor = np.sqrt(np.pi) * 1.0j * thetaCore * np.exp(-(thetaCore**2) / 2) / 2**1.5
    first_term = erf(0.5 * (np.sqrt(2) * 1.0j * thetaCore + np.sqrt(2) * alphaWing))
    second_term = erf(0.5 * (np.sqrt(2) * 1.0j * thetaCore - np.sqrt(2) * alphaWing))
    third_term = 2 * erf(1.0j * thetaCore / np.sqrt(2))
    integral_factor = prefactor * (first_term + second_term - third_term)
    integral_factor = (
        integral_factor.real
    )  # this imaginary part is always 0 in this expression

    epsilon_c = Ejet / (2 * np.pi * integral_factor)
    Eiso_c = 4 * np.pi * epsilon_c

    return Eiso_c


def powerlaw_jet_energy_to_central_isotropic_energy_equivalent(
    Ejet, thetaCore, alphaWing, b
):
    """
    Takes the total energy of a powerlaw jet as well as the angular parameters and returns the isotropic-energy equivalent
    on axis. This means it is assumed that the true jet energy follows some angular structure dEjet / dOmega = epsilon_c * (1+1/b * (theta/thetaCore)^2)^(-b/2).
    Then the distribution of the isotropic energy equivalent is simply related according to E_iso(theta) = 4pi dEjet / dOmega.

    :param Ejet: Total jet energy in ergs
    :param thetaCore: Core angle in rad
    :param alphaWing: Ratio of the wing angle and core angle.
    :param b: Power law tail of the jet.
    :return: The on-axis isotropic-energy equivalent.
    """
    x = np.linspace(0, alphaWing * thetaCore, 100)
    y = np.sin(x) * (1 + 1 / b * (x / thetaCore) ** 2) ** (-b / 2)
    integral_factor = simpson(x=x, y=y)

    epsilon_c = Ejet / (2 * np.pi * integral_factor)
    Eiso_c = 4 * np.pi * epsilon_c

    return Eiso_c


class EjectaFitting:
    """
    Base class for the ejecta fitting conversions.

    Attributes
    ----------
    mass_fitting_keys : list of str
        The keys :meth:`__call__` writes, in the order
        :meth:`ejecta_parameter_conversion` returns them.
    """

    mass_fitting_keys = ["log10_mej_dyn", "log10_mej_wind", "log10_mej", "log10_E0"]

    def __call__(self, parameters):
        """
        Add the fitted ejecta parameters to ``parameters``.

        Keys already present in ``parameters`` are kept.

        Parameters
        ----------
        parameters : dict

        Returns
        -------
        dict
            ``parameters``, updated.
        """
        conv_parameters = self.ejecta_parameter_conversion(parameters)
        for key, val in zip(self.mass_fitting_keys, conv_parameters):
            # We always prefer explicitly sampled ejecta parameters
            parameters[key] = parameters.get(key, val)
        return parameters

    def ejecta_parameter_conversion(self, parameters):
        """
        Return ``-inf`` for each of :attr:`mass_fitting_keys`.

        Parameters
        ----------
        parameters : dict

        Returns
        -------
        list
        """
        return [-np.inf for _ in self.mass_fitting_keys]


class NSBHEjectaFitting(EjectaFitting):
    """Ejecta fitting for an NSBH system."""

    def chibh2risco(self, chi_bh):
        """see, e.g., https://arxiv.org/pdf/2011.08948.pdf, eq. 2-4.
        This expression gives the innermost stable circular orbit (ISCO) in units of the black hole mass as a function of the dimensionless spin parameter chi_bh.

        Parameters
        ----------
        chi_bh : float or array_like

        Returns
        -------
        The ISCO radius in units of the black hole mass.
        """
        Z1 = 1.0 + (1.0 - chi_bh**2) ** (1.0 / 3) * (
            (1 + chi_bh) ** (1.0 / 3) + (1 - chi_bh) ** (1.0 / 3)
        )
        Z2 = np.sqrt(3.0 * chi_bh**2 + Z1**2.0)

        return 3.0 + Z2 - np.sign(chi_bh) * np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))

    def baryon_mass_NS(self, source_mass, compactness):
        """
        equation (7) in https://arxiv.org/abs/2002.07728

        Parameters
        ----------
        source_mass, compactness : float or array_like

        Returns
        -------
        The baryon mass.
        """

        return source_mass * (1.0 + 0.6 * compactness / (1.0 - 0.5 * compactness))

    def remnant_disk_mass_fitting(
        self,
        mass_1_source,
        mass_2_source,
        compactness_2,
        chi_bh,
        a=0.40642158,
        b=0.13885773,
        c=0.25512517,
        d=0.761250847,
    ):
        """
        equation (4) in https://arxiv.org/pdf/1807.00011

        Parameters
        ----------
        mass_1_source, mass_2_source, compactness_2, chi_bh : float or array_like
        a, b, c, d : float, optional
            Fitting coefficients. Defaults:
            ``a=0.40642158``, ``b=0.13885773``, ``c=0.25512517``, ``d=0.761250847``.

        Returns
        -------
        The remnant disk mass.
        """

        mass_ratio = mass_2_source / mass_1_source
        symm_mass_ratio = mass_ratio / (1.0 + mass_ratio) ** 2

        #  use the BH spin to find the normalized risco
        risco = self.chibh2risco(chi_bh)
        baryon_mass_2 = self.baryon_mass_NS(mass_2_source, compactness_2)

        remnant_mass = (
            a * np.power(symm_mass_ratio, -1.0 / 3.0) * (1.0 - 2.0 * compactness_2)
        )
        remnant_mass += -b * risco / symm_mass_ratio * compactness_2 + c

        remnant_mass = np.maximum(remnant_mass, 0.0)

        remnant_mass = np.power(remnant_mass, 1.0 + d)

        remnant_mass *= baryon_mass_2

        return remnant_mass

    def dynamic_mass_fitting(
        self,
        mass_1_source,
        mass_2_source,
        compactness_2,
        chi_bh,
        a1=7.11595154e-03,
        a2=1.43636803e-03,
        a4=-2.76202990e-02,
        n1=-8.63604211e-01,
        n2=-1.68399507,
    ):
        """
        equation (9) in https://arxiv.org/abs/2002.07728

        Parameters
        ----------
        mass_1_source, mass_2_source, compactness_2, chi_bh : float or array_like
        a1, a2, a4, n1, n2 : float, optional
            Fitting coefficients. Defaults:
            ``a1=7.11595154e-03``, ``a2=1.43636803e-03``,
            ``a4=-2.76202990e-02``, ``n1=-8.63604211e-01``,
            ``n2=-1.68399507``.

        Returns
        -------
        The dynamical ejecta mass.
        """

        mass_ratio = mass_2_source / mass_1_source

        #  use the BH spin to find the normalized risco
        risco = self.chibh2risco(chi_bh)
        baryon_mass_2 = self.baryon_mass_NS(mass_2_source, compactness_2)

        mdyn = a1 * mass_ratio**n1 * (1.0 - 2.0 * compactness_2) / compactness_2
        mdyn += -a2 * mass_ratio**n2 * risco + a4
        mdyn *= baryon_mass_2

        mdyn = np.maximum(0.0, mdyn)

        return mdyn

    def nsbh_parameter_conversion(self, converted_parameters):
        """
        Fit the ejecta parameters for an NSBH system.

        Parameters
        ----------
        converted_parameters : dict
            Reads ``mass_1_source``, ``mass_2_source``, ``radius_2``, ``alpha``
            and ``ratio_zeta``, plus either ``chi_1`` or ``a_1`` with
            ``cos_tilt_1``/``tilt_1``.

        Returns
        -------
        numpy.ndarray
            The four values of :attr:`mass_fitting_keys`, stacked. The
            last, ``log10_E0``, is ``-inf``.
        """
        mass_1_source = converted_parameters["mass_1_source"]
        mass_2_source = converted_parameters["mass_2_source"]

        radius_2 = converted_parameters["radius_2"]
        compactness_2 = mass_2_source * geom_msun_km / radius_2
        try:
            chi_1 = converted_parameters["chi_1"]
        except KeyError:
            cos_tilt_1 = converted_parameters.get(
                "cos_tilt_1", np.cos(converted_parameters["tilt_1"])
            )
            chi_1 = converted_parameters["a_1"] * cos_tilt_1

        mdyn_fit = self.dynamic_mass_fitting(
            mass_1_source, mass_2_source, compactness_2, chi_1
        )
        remnant_disk_fit = self.remnant_disk_mass_fitting(
            mass_1_source, mass_2_source, compactness_2, chi_1
        )
        mdisk_fit = remnant_disk_fit - mdyn_fit
        mej_dyn = mdyn_fit + converted_parameters["alpha"]

        # prevent the output message from being flooded by these warning messages
        old = np.seterr()
        np.seterr(invalid="ignore")
        np.seterr(divide="ignore")

        log_mej_wind = np.full_like(mdisk_fit, -np.inf)
        log_mej_dyn = np.full_like(mdisk_fit, -np.inf)
        disk_mask = mdisk_fit > 0.0

        log_mej_dyn[disk_mask] = np.log10(mej_dyn[disk_mask])
        log_mej_wind[disk_mask] = (
            np.log10(mdisk_fit[disk_mask])
            + np.log10(converted_parameters["ratio_zeta"])[disk_mask]
        )

        total_ejeta_mass = 10**log_mej_dyn + 10**log_mej_wind

        log10_mej = np.log10(total_ejeta_mass)
        # FIXME: NSBH might produce a GRB, too. Why not provide the same expression for BNS?

        np.seterr(**old)
        return np.stack(
            (log_mej_dyn, log_mej_wind, log10_mej, np.full_like(log_mej_wind, -np.inf))
        )

    def ejecta_parameter_conversion(self, parameters):
        """
        Call :meth:`nsbh_parameter_conversion`.

        Parameters
        ----------
        parameters : dict

        Returns
        -------
        As :meth:`nsbh_parameter_conversion`.
        """
        return self.nsbh_parameter_conversion(parameters)


class BNSEjectaFitting(EjectaFitting):
    """Ejecta fitting for a BNS system."""

    def log10_disk_mass_fitting(
        self,
        total_mass,
        mass_ratio,
        MTOV,
        R16,
        a0=-1.725,
        delta_a=-2.337,
        b0=-0.564,
        delta_b=-0.437,
        c=0.958,
        d=0.057,
        beta=5.879,
        q_trans=0.886,
    ):
        """
        See https://arxiv.org/pdf/2205.08513 Eq. (22)
        The coefficients a0, delta_a etc. have been updated since then,
        the ones here are the correct ones.
        The threshold mass is from https://arxiv.org/pdf/1908.05442.pdf.

        Parameters
        ----------
        total_mass, mass_ratio, MTOV, R16 : float or array_like
        a0, delta_a, b0, delta_b, c, d, beta, q_trans : float, optional
            Fitting coefficients. Defaults:
            ``a0=-1.725``, ``delta_a=-2.337``, ``b0=-0.564``,
            ``delta_b=-0.437``, ``c=0.958``, ``d=0.057``, ``beta=5.879``,
            ``q_trans=0.886``.

        Returns
        -------
        ``log10`` of the disk mass.
        """
        k = -3.606 * MTOV / R16 + 2.38
        threshold_mass = k * MTOV

        xi = 0.5 * np.tanh(beta * (mass_ratio - q_trans))

        a = a0 + delta_a * xi
        b = b0 + delta_b * xi

        log10_mdisk = a * (1 + b * np.tanh((c - total_mass / threshold_mass) / d))
        log10_mdisk = np.maximum(-3.0, log10_mdisk)

        return log10_mdisk

    def log10_dynamic_mass_fitting_CoDiMaMe(
        self,
        mass_1,
        mass_2,
        compactness_1,
        compactness_2,
        a=-0.0719,
        b=0.2116,
        d=-2.42,
        n=-2.905,
    ):
        """
        See https://arxiv.org/pdf/1812.04803.pdf

        Parameters
        ----------
        mass_1, mass_2, compactness_1, compactness_2 : float or array_like
        a, b, d, n : float, optional
            Fitting coefficients. Defaults:
            ``a=-0.0719``, ``b=0.2116``, ``d=-2.42``, ``n=-2.905``.

        Returns
        -------
        ``log10`` of the dynamical ejecta mass.
        """

        log10_mdyn = (
            a * (1 - 2 * compactness_1) * mass_1 / compactness_1
            + b * mass_2 * np.power(mass_1 / mass_2, n)
            + d / 2
        )

        log10_mdyn += (
            a * (1 - 2 * compactness_2) * mass_2 / compactness_2
            + b * mass_1 * np.power(mass_2 / mass_1, n)
            + d / 2
        )

        return log10_mdyn

    def dynamic_mass_fitting_KrFo(
        self,
        mass_1,
        mass_2,
        compactness_1,
        compactness_2,
        a=-9.3335,
        b=114.17,
        c=-337.56,
        n=1.5465,
    ):
        """
        See https://arxiv.org/pdf/2002.07728.pdf

        Parameters
        ----------
        mass_1, mass_2, compactness_1, compactness_2 : float or array_like
        a, b, c, n : float, optional
            Fitting coefficients. Defaults:
            ``a=-9.3335``, ``b=114.17``, ``c=-337.56``, ``n=1.5465``.

        Returns
        -------
        The dynamical ejecta mass.
        """

        mdyn = mass_1 * (
            a / compactness_1 + b * np.power(mass_2 / mass_1, n) + c * compactness_1
        )
        mdyn += mass_2 * (
            a / compactness_2 + b * np.power(mass_1 / mass_2, n) + c * compactness_2
        )
        mdyn *= 1e-3

        mdyn = np.maximum(0.0, mdyn)

        return mdyn

    def dynamic_vel_fitting_Radice2018(
        self, mass_1, mass_2, compactness_1, compactness_2, a=-0.287, b=0.494, c=-3.000
    ):
        """
        See https://arxiv.org/pdf/1809.11161 Eq. (22)

        Parameters
        ----------
        mass_1, mass_2, compactness_1, compactness_2 : float or array_like
        a, b, c : float, optional
            Fitting coefficients. Defaults:
            ``a=-0.287``, ``b=0.494``, ``c=-3.000``.

        Returns
        -------
        The dynamical ejecta velocity.
        """

        vej_dyn = a * mass_1 / mass_2 * (1 + c * compactness_1)
        vej_dyn += a * mass_2 / mass_1 * (1 + c * compactness_2)
        vej_dyn += b

        return vej_dyn

    def dynamic_mass_fitting_prompt_collapse(
        self,
        mass_1,
        mass_2,
        lambda_1,
        lambda_2,
        a=1.25e-4,
        b=9.82e-1,
        c=-2.44,
    ):
        """
        See https://arxiv.org/pdf/2411.02342, Eq. (9)

        Parameters
        ----------
        mass_1, mass_2, lambda_1, lambda_2 : float or array_like
        a, b, c : float, optional
            Fitting coefficients. Defaults:
            ``a=1.25e-4``, ``b=9.82e-1``, ``c=-2.44``.

        Returns
        -------
        The dynamical ejecta mass.
        """
        q = mass_2 / mass_1
        lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(
            lambda_1, lambda_2, mass_1, mass_2
        )
        mdyn = (
            a * lambda_tilde * (q ** (-1) - b) * np.exp(c / q)
        )  # this is always positive

        return mdyn

    def dynamic_vel_fitting_prompt_collapse(
        self, mass_1, mass_2, compactness_1, compactness_2, a=-0.395, b=0.798, c=-1.627
    ):
        """
        See https://arxiv.org/pdf/2411.02342, Eq. (10)

        Parameters
        ----------
        mass_1, mass_2, compactness_1, compactness_2 : float or array_like
        a, b, c : float, optional
            Fitting coefficients. Defaults:
            ``a=-0.395``, ``b=0.798``, ``c=-1.627``.

        Returns
        -------
        The dynamical ejecta velocity.
        """
        vdyn = a * mass_1 / mass_2 * (1 + c * compactness_1)
        vdyn += a * mass_2 / mass_1 * (1 + c * compactness_2)
        vdyn += b

        return vdyn

    def log10_disk_mass_fitting_prompt_collapse(
        self, mass_1, mass_2, lambda_1, lambda_2, a=7.70, b=-13.4, c=8.16e-3
    ):
        """
        See https://arxiv.org/pdf/2411.02342, Eq. (11)
        Typo for b, b=-13.4 confirmed through author correspondence

        Parameters
        ----------
        mass_1, mass_2, lambda_1, lambda_2 : float or array_like
        a, b, c : float, optional
            Fitting coefficients. Defaults:
            ``a=7.70``, ``b=-13.4``, ``c=8.16e-3``.

        Returns
        -------
        ``log10`` of the disk mass.
        """
        q = mass_2 / mass_1
        lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(
            lambda_1, lambda_2, mass_1, mass_2
        )
        log10_mdisk = a + b * q + c * lambda_tilde * q**2

        log10_mdisk = np.minimum(log10_mdisk, -1)

        return log10_mdisk

    def chiBH_fitting(
        self, mass_1, mass_2, lambda_1, lambda_2, a=0.537, b=-0.185, c=-0.514
    ):
        """
        See https://arxiv.org/pdf/1812.04803, Eq. (D7)
        nu needs to be divided by 0.25 and lambda_tilde by 400, confirmed through author correspondence

        Parameters
        ----------
        mass_1, mass_2, lambda_1, lambda_2 : float or array_like
        a, b, c : float, optional
            Fitting coefficients. Defaults:
            ``a=0.537``, ``b=-0.185``, ``c=-0.514``.

        Returns
        -------
        The black hole spin.
        """

        lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(
            lambda_1, lambda_2, mass_1, mass_2
        )
        M = mass_1 + mass_2
        nu = component_masses_to_symmetric_mass_ratio(mass_1, mass_2)

        chi_BH = np.tanh(a * (nu / 0.25) ** 2 * (M + b * lambda_tilde / 400) + c)

        return chi_BH

    def bns_ejecta_conversion(self, converted_parameters):
        # prevent the output message flooded by these warning messages
        """
        Fit the ejecta masses for a BNS system.

        Parameters
        ----------
        converted_parameters : dict
            Reads ``mass_1_source``, ``mass_2_source``, ``radius_1``,
            ``radius_2``, ``TOV_mass``, ``R_16``, ``alpha`` and ``ratio_zeta``.

        Returns
        -------
        tuple
            ``(log10_mej_dyn, log10_mej_wind, log10_mej_total,
            log10_mdisk_fit)``.
        """
        old = np.seterr()
        np.seterr(invalid="ignore")
        np.seterr(divide="ignore")

        mass_1_source = converted_parameters["mass_1_source"]
        mass_2_source = converted_parameters["mass_2_source"]

        total_mass = mass_1_source + mass_2_source
        mass_ratio = mass_2_source / mass_1_source

        radius_1 = converted_parameters["radius_1"]
        radius_2 = converted_parameters["radius_2"]

        compactness_1 = mass_1_source * geom_msun_km / radius_1
        compactness_2 = mass_2_source * geom_msun_km / radius_2
        # FIXME: switch to prompt collapse fitting for appropriate thresholds
        mdyn_fit = self.dynamic_mass_fitting_KrFo(
            mass_1_source, mass_2_source, compactness_1, compactness_2
        )

        log10_mdisk_fit = self.log10_disk_mass_fitting(
            total_mass,
            mass_ratio,
            converted_parameters["TOV_mass"],
            converted_parameters["R_16"] / geom_msun_km,
        )

        log10_mej_dyn = np.log10(mdyn_fit + converted_parameters["alpha"])

        log10_mej_wind = np.log10(converted_parameters["ratio_zeta"]) + log10_mdisk_fit

        # FIXME Weizmann: dynamic_mass_fitting_KrFo/log10_disk_mass_fitting
        # clip negative fit values via np.maximum(0, .), which silently turns
        # a -inf signal (radius<=0 -> compactness=inf, i.e. not a real NS)
        # into a plain finite 0.0 before np.isfinite() downstream can catch
        # it. Force -inf explicitly here, before that clipping had a chance
        # to hide it, whenever either component isn't a real NS under this
        # EOS (mirrors the old compactness == 0.5 check from nmma 0.2.3,
        # without relying on radius_1/radius_2 being an actual Schwarzschild
        # radius rather than the 0 sentinel used here).
        not_ns = (radius_1 <= 0) | (radius_2 <= 0)
        log10_mej_dyn = np.where(not_ns, -np.inf, log10_mej_dyn)
        log10_mej_wind = np.where(not_ns, -np.inf, log10_mej_wind)

        # total eject mass
        total_ejeta_mass = 10**log10_mej_dyn + 10**log10_mej_wind
        # FIXME Weizmann: np.seterr(**old) used to run before this log10
        # call, so log10(0) (from a -inf/-inf pair, e.g. once not_ns forces
        # both to -inf) raised an unsuppressed RuntimeWarning: divide by
        # zero. Compute log10_mej_total first, restore seterr after.
        log10_mej_total = np.log10(total_ejeta_mass)

        np.seterr(**old)
        return log10_mej_dyn, log10_mej_wind, log10_mej_total, log10_mdisk_fit

    def grb_energy_conversion(self, converted_parameters, log10_mdisk_fit):
        # GRB afterglow energy
        """
        Fit ``log10_E0`` from the disk mass.

        A power-law jet is used when ``b`` is present, otherwise a gaussian
        jet; if none of ``thetaWing``, ``alphaWing`` and ``b`` is present,
        neither is used.

        Parameters
        ----------
        converted_parameters : dict
            Reads ``ratio_zeta``, ``ratio_epsilon``, ``thetaCore``,
            ``thetaWing``, ``alphaWing`` and ``b``.
        log10_mdisk_fit : array_like
            As returned by :meth:`bns_ejecta_conversion`.

        Returns
        -------
        ``log10_E0``.
        """
        log10_Ejet = np.log10(converted_parameters.get("ratio_epsilon", 2e-4))
        log10_Ejet += np.log10(1.0 - converted_parameters["ratio_zeta"])
        log10_Ejet += log10_mdisk_fit + np.log10(msun_to_ergs)

        thetaCore = converted_parameters.get(
            "thetaCore", 0.105
        )  # default about 6 degree, see arxiv:2210.05695

        if not any(
            key in converted_parameters for key in ["thetaWing", "alphaWing", "b"]
        ):
            return log10_Ejet - np.log10(np.sin(thetaCore / 2) ** 2)

        if "alphaWing" in converted_parameters:
            alphaWing = converted_parameters["alphaWing"]
        else:
            alphaWing = (
                converted_parameters["thetaWing"] / converted_parameters["thetaCore"]
            )

        if "b" in converted_parameters:  # power law jet
            jet_func = powerlaw_jet_energy_to_central_isotropic_energy_equivalent
            data = np.column_stack(
                (10**log10_Ejet, thetaCore, alphaWing, converted_parameters["b"])
            )

        else:
            jet_func = gaussian_jet_energy_to_central_isotropic_energy_equivalent
            data = np.column_stack((10**log10_Ejet, thetaCore, alphaWing))

        out = np.log10([jet_func(*row) for row in data])
        return np.squeeze(out)

    def bns_parameter_conversion(self, parameters):
        """
        Fit the ejecta parameters for a BNS system.

        Parameters
        ----------
        parameters : dict

        Returns
        -------
        numpy.ndarray
            The four values of :attr:`mass_fitting_keys`, with non-finite
            entries replaced by ``-inf``. ``log10_E0`` is taken from
            ``parameters`` when present.
        """
        (
            log10_mej_dyn,
            log10_mej_wind,
            log10_mej_total,
            log10_mdisk_fit,
        ) = self.bns_ejecta_conversion(parameters)

        if "log10_E0" in parameters:
            log10_E0 = parameters["log10_E0"]
        else:
            log10_E0 = self.grb_energy_conversion(parameters, log10_mdisk_fit)

        converted_ejecta = (log10_mej_dyn, log10_mej_wind, log10_mej_total, log10_E0)

        return np.where(np.isfinite(converted_ejecta), converted_ejecta, -np.inf)

    def ejecta_parameter_conversion(self, parameters):
        """
        Call :meth:`bns_parameter_conversion`.

        Parameters
        ----------
        parameters : dict

        Returns
        -------
        As :meth:`bns_parameter_conversion`.
        """
        return self.bns_parameter_conversion(parameters)


class KilonovaEjectaFitting(BNSEjectaFitting, NSBHEjectaFitting):
    """Ejecta fitting that selects the BNS or NSBH conversion per system."""

    def ejecta_parameter_conversion(self, parameters):
        # FIXME Weizmann: routing used to check radius_1>0 alone for BNS,
        # not radius_2>0 too. mass_1 >= mass_2 by convention, so in the
        # common case radius_1>0 already implies radius_2>0 (a lighter mass
        # is inside the EOS's mass range whenever a heavier one is), but
        # not always: mass_2 can fall below the EOS table's tabulated
        # minimum mass even while mass_1 is a valid NS mass. That row was
        # still routed to bns_parameter_conversion, which computed
        # compactness_2 = mass_2*geom_msun_km/radius_2 = inf (radius_2==0),
        # and that inf got silently clipped to a finite mej by
        # np.maximum(0, .) inside the mass-fitting formulas before
        # np.isfinite() downstream had any chance to catch it, a
        # physically wrong, finite ejecta mass got published for a system
        # that isn't actually a BNS under this EOS. Requiring radius_2>0
        # too keeps bns_parameter_conversion from ever seeing that case;
        # it now correctly falls through to nsbh/BBH instead.
        """
        Fit the ejecta parameters, choosing the conversion per system.

        ``radius_1`` and ``radius_2`` both positive selects
        :meth:`bns_parameter_conversion`, ``radius_2`` positive alone selects
        :meth:`nsbh_parameter_conversion`, and otherwise all four values are
        ``-inf``.

        Parameters
        ----------
        parameters : dict

        Returns
        -------
        numpy.ndarray
            Four values, ordered as :attr:`mass_fitting_keys`.
        """
        try:
            # both objects are NS
            if (parameters["radius_1"] > 0.0) and (parameters["radius_2"] > 0.0):
                return self.bns_parameter_conversion(parameters)
            # heavier object is BH, but lighter object is NS
            elif parameters["radius_2"] > 0.0:
                return self.nsbh_parameter_conversion(parameters)
            # both objects are BHs (or lighter mass is below the EOS's
            # tabulated range)
            else:
                return np.full(4, -np.inf)
        except ValueError:
            # ValueError occurs when trying to obtain truth values of arrays
            # -> evaluate many points at once and chose conditional ejecta_fitting
            is_bns = (parameters["radius_1"] > 0.0) & (parameters["radius_2"] > 0.0)
            return np.where(
                is_bns,  # both objects are NS
                self.bns_parameter_conversion(parameters),
                np.where(
                    parameters["radius_2"] > 0.0,  # elif component 2 is a NS
                    self.nsbh_parameter_conversion(parameters),
                    # else assume BBH (i.e., no ejecta)
                    np.full((4,) + parameters["mass_1_source"].shape, -np.inf),
                ),
            )


class MultimessengerConversion:
    """
    An ordered chain of parameter conversions.

    Parameters
    ----------
    *conversions
        Callables applied in order by :meth:`core_conversion`.
    """

    def __init__(self, *conversions):
        self._conversions = conversions

    @classmethod
    def from_args(cls, args):
        # FIXME: implement argument parsing to select conversions
        """
        Not implemented.

        Parameters
        ----------
        args

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("from_args not yet implemented")

    @classmethod
    def from_dict(cls, instruction_dict):
        """
        Build a conversion chain from an instruction dict.

        The keys read are ``cosmo``, ``gw``, ``eos``, ``ejecta``, ``em`` and
        ``custom``. ``cosmo`` also calls
        :func:`nmma.core.constants.set_cosmology`.

        Parameters
        ----------
        instruction_dict : dict

        Returns
        -------
        MultimessengerConversion
        """
        conversions = []

        # NOTE: Order matters!!!
        if "cosmo" in instruction_dict:
            set_cosmology(instruction_dict["cosmo"])
            conversions.append(cosmology_to_distance)

        if "gw" in instruction_dict:
            conversions.append(instruction_dict["gw"])

        if "eos" in instruction_dict:
            conversions.append(instruction_dict["eos"])

        if "ejecta" in instruction_dict:
            conversions.append(KilonovaEjectaFitting())

        if "em" in instruction_dict:
            conversions.append(instruction_dict["em"])

        if "custom" in instruction_dict:
            conversions.append(instruction_dict["custom"])

        return cls(*conversions)

    @classmethod
    def basic_cbc(cls, eos_conversion, em_conversion):
        """
        Build a chain of :func:`bbh_source_frame`, ``eos_conversion``,
        :class:`KilonovaEjectaFitting` and ``em_conversion``.

        Parameters
        ----------
        eos_conversion, em_conversion : callable

        Returns
        -------
        MultimessengerConversion
        """
        return cls(
            bbh_source_frame, eos_conversion, KilonovaEjectaFitting(), em_conversion
        )

    def convert_to_multimessenger_parameters(self, parameters, add_new_keys=False):
        """
        Run the conversion chain over ``parameters``.

        Values pass through :func:`val_to_scalar` before and after
        :meth:`core_conversion`.

        Parameters
        ----------
        parameters : dict
        add_new_keys : bool, default=False
            If True, also return the keys that were not in ``parameters``.

        Returns
        -------
        The converted parameters, or ``(converted_parameters, added_keys)``
        when ``add_new_keys`` is True.
        """
        original_keys = list(parameters.keys())
        converted_parameters = {k: val_to_scalar(v) for k, v in parameters.items()}

        converted_parameters = self.core_conversion(converted_parameters)

        converted_parameters = {
            k: val_to_scalar(v) for k, v in converted_parameters.items()
        }

        if add_new_keys:
            added_keys = [
                k for k in converted_parameters.keys() if k not in original_keys
            ]
            return converted_parameters, added_keys
        else:
            return converted_parameters

    def core_conversion(self, parameters):
        """
        Apply each conversion in turn.

        Parameters
        ----------
        parameters : dict

        Returns
        -------
        The output of the last conversion.
        """
        for conv in self._conversions:
            parameters = conv(parameters)
        return parameters

    def identity_conversion(self, parameters):
        """
        Return ``parameters`` unchanged.

        Parameters
        ----------
        parameters

        Returns
        -------
        ``parameters``.
        """
        return parameters


# fmt: off
label_mapping = {
    # Cosmology parameters #
    "Hubble_constant"   : r"$H_0{\rm\,[km\,s^{-1}\,Mpc^{-1}]}$",
    "Omega_matter"      : r"$\Omega_{m}$",
    "redshift"          : r"$z$",
    # System parameters #
    "inclination_EM"    : r"$\theta_{obs}$",
    "theta_jn"          : r"$\theta_{JN}$",
    "cos_theta_jn"      : r"$\cos{\theta_{JN}}$",
    "luminosity_distance": r"$d_L\,{\rm [Mpc]}$",
    # GW parameters #
    "chirp_mass"        : r"$\mathcal{M}_c\,{\rm [M_{\odot}]}$",
    "mass_ratio"        : r"$q$",
    "chi_eff"           : r"$\chi_{\rm{eff}}$",
    "mass_1_source"     : r"$m_{1,s}\,{\rm [M_{\odot}]}$",
    "mass_2_source"     : r"$m_{2,s}\,{\rm [M_{\odot}]}$",
    # KN parameters #
    "log10_mej"         : r"$\log_{10}(M_{\rm{ej}}\,{\rm [M_{\odot}]})$",
    "log10_mej_dyn"     : r"$\log_{10}(M_{\rm{dyn}}\,{\rm [M_{\odot}]})$",
    "log10_mej_wind"    : r"$\log_{10}(M_{\rm{wind}}\,{\rm [M_{\odot}]})$",
    "ratio_zeta"        : r"$\zeta$",
    "alpha"             : r"$\alpha$",
    "KNtheta"           : r"$\theta_{\rm obs}\,[^\circ]$",
    "KNphi"             : r"$\phi_{KN}\,[^\circ]$",
    # Bu parameters #
    "vej_dyn"           : r"$v_{\rm{dyn}}\,{\rm [c]}$",
    "vej_wind"          : r"$v_{\rm{wind}}\,{\rm [c]}$",
    "v_ej_dyn"          : r"$v_{\rm{dyn}}\,{\rm [c]}$",
    "v_ej_wind"         : r"$v_{\rm{wind}}\,{\rm [c]}$",
    "Ye_dyn"            : r"$Y_{e,{\rm{dyn}}}$",
    "kappa_Ye"          : r"$\kappa_{\rm{Y_e}}$",
    "kappa_v"           : r"$\kappa_{v}$",
    # GRB parameters #
    "log10_E0"          : r"$\log_{10}(E_{\rm iso,0}\,{\rm [erg]})$",
    "ratio_epsilon"     : r"$\epsilon$",
    "thetaCore"         : r"$\theta_{c}$",
    "thetaWing"         : r"$\theta_{w}$",
    "alphaWing"         : r"$\alpha_{w}$",
    "log10_n0"          : r"$\log_{10}(n_{0}\,{\rm [cm^{-3}]})$",
    "p"                 : r"$p$",
    "log10_epsilon_e"   : r"$\log_{10}(\epsilon_{e})$",
    "log10_epsilon_B"   : r"$\log_{10}(\epsilon_{B})$",
    # Ejecta parameters #
    "mni"               : r"$M_{\rm{Ni}}\,{\rm [M_{\odot}]}$",
    "mtot"              : r"$M_{\rm{tot}}\,{\rm [M_{\odot}]}$",
    "mrp"               : r"$M_{\rm{rp}}\,{\rm [M_{\odot}]}$",
    "mni_c"             : r"$M_{\rm{Ni}}/M_{\rm{tot}}$",
    "mrp_c"             : r"$M_{\rm{rp,c}}\,{\rm [M_{\odot}]}$",
    # EOS parameters #
    "L_sym"             : r"$L_{\rm sym}\,{\rm [MeV]}$",
    "K_sym"             : r"$K_{\rm sym}\,{\rm [MeV]}$",
    "K_sat"             : r"$K_{\rm sat}\,{\rm [MeV]}$",
    "3n_sat"            : r"$c^2_{3n_{\rm sat}}\,{\rm [c^2]}$",
    "5n_sat"            : r"$c^2_{5n_{\rm sat}}\,{\rm [c^2]}$",
    "TOV_mass"          : r"$M_{\rm{TOV}}\,{\rm [M_{\odot}]}$",
    "R_14"              : r"$R_{1.4}\,{\rm[km]}$",
    "lambda_tilde"      : r"$\tilde{\Lambda}$",
}
# fmt: on
