import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.integrate import simpson
from astropy import units
from astropy import cosmology as cosmo
from .constants import geom_msun_km, msun_to_ergs, msun_s, get_cosmology, set_cosmology

from bilby.gw.conversion import (
    component_masses_to_chirp_mass,
    component_masses_to_symmetric_mass_ratio,
    lambda_1_lambda_2_to_lambda_tilde,
    convert_to_lal_binary_black_hole_parameters,
    convert_to_lal_binary_neutron_star_parameters,
    generate_mass_parameters,
    chirp_mass_and_mass_ratio_to_total_mass
)

def val_to_scalar(val):
    """Convert single-value quantities to scalars for easier handling"""
    if np.isscalar(val):
        return val
    else:
        val = np.asarray(val)
        if val.size == 1:
            return val.item()
        return val
    
########################## distance conversions ####################################
def distance_modulus_nmma(d_lum = 1e-5):
        # mag_app = mag_abs + 5* log10(dist/10pc) | NMMA-dist is in Mpc
        #         = mag_abs + 5 * (log10(Mpc/10pc)+ log10(params["luminosity_distance"]))  
        # therefore: distance_modulus = mag_app - mag_abs =
        return  5.0 * (5+ np.log10(d_lum))

def luminosity_distance_to_redshift(distance, cosmology = None):
    if cosmology is None:
        cosmology = get_cosmology()
    if isinstance(distance, pd.Series):
        distance = distance.values

    if hasattr(distance, '__len__') and len(distance)>50: 
        d_min, d_max = distance.min(), distance.max()
        dist_grid, z_grid = get_cosmo_grids(d_min, d_max, cosmology)
        return np.interp(distance, dist_grid, z_grid).value
    else:
        return cosmo.z_at_value(cosmology.luminosity_distance, distance *units.Mpc).value
        
def get_cosmo_grids(distance_min, distance_max, cosmology):
    #luminosity_distance_to_redshift gets really slow if too many distances are put in at once
    zmin = cosmo.z_at_value(cosmology.luminosity_distance, distance_min * units.Mpc)
    zmax = cosmo.z_at_value(cosmology.luminosity_distance, distance_max * units.Mpc)
    z_grid = np.geomspace(zmin, zmax, 50)
    dist_grid = cosmology.luminosity_distance(z_grid).value
    return dist_grid, z_grid

def get_redshift(parameters):
    if "redshift" in parameters:
        return parameters["redshift"]
    elif "luminosity_distance" in parameters:
            return luminosity_distance_to_redshift(parameters["luminosity_distance"])
    else:
        ## zeros like the first input of parameters, independent of size and keys
        return np.zeros_like(next(iter(parameters.values()))) 

def cosmology_to_distance(parameters):
    cosmology= get_cosmology()
    cosmo_parameters = {}
    if "Hubble_constant" in parameters:
        cosmo_parameters["H0"] = parameters["Hubble_constant"]
    if "Omega_matter" in parameters:
        cosmo_parameters["Om0"] = parameters["Omega_matter"]
    ## Maybe extend for an even wilder cosmology?
    try:
        alt_cosmo = cosmology.clone(**cosmo_parameters)
        if "luminosity_distance" in parameters:
            # if luminosity distance is available, we assume it is in Mpc
            parameters["redshift"] = luminosity_distance_to_redshift(
                parameters["luminosity_distance"], cosmology=alt_cosmo)
        elif "redshift" in parameters:
            parameters["luminosity_distance"] = alt_cosmo.luminosity_distance(parameters["redshift"]).value
        else:
            raise KeyError("Either redshift or luminosity_distance must be in parameters")

    except ValueError:
        # if H0 is an array, .clone raises a ValueError
        # in that case we turn a dict with len-n values into a len-n list of dicts with single values
        cosmo_dicts = [dict(zip(cosmo_parameters.keys(), vals)) for vals in zip(*cosmo_parameters.values())]
        alt_cosmos = [cosmology.clone(**cosmo_dict) for cosmo_dict in cosmo_dicts]

        if 'luminosity_distance' in parameters:
            # if luminosity distance is available, we assume it is in Mpc
            parameters["redshift"] = np.array(
                [luminosity_distance_to_redshift(
                    parameters["luminosity_distance"][i], cosmology=alt_cosmo) 
                for i, alt_cosmo in enumerate(alt_cosmos)])
            
        elif "redshift" in parameters:
            parameters["luminosity_distance"] = np.array(
                [alt_cosmo.luminosity_distance(parameters["redshift"][i]).value 
                for i, alt_cosmo in enumerate(alt_cosmos)])
    return parameters

def source_frame_masses(converted_parameters):
    converted_parameters = generate_mass_parameters(converted_parameters)
    if "redshift" not in converted_parameters:
        distance = converted_parameters["luminosity_distance"]
        converted_parameters["redshift"] = luminosity_distance_to_redshift(distance)
    z = converted_parameters["redshift"]

    if "mass_1_source" not in converted_parameters:
        converted_parameters["mass_1_source"] = np.array(converted_parameters["mass_1"] / (1 + z))

    if "mass_2_source" not in converted_parameters:
        converted_parameters["mass_2_source"] = np.array(converted_parameters["mass_2"] / (1 + z))

    return converted_parameters

def observation_angle_conversion(parameters):
    theta_jn = parameters.get('theta_jn', np.arccos(parameters.get("cos_theta_jn", 1.)))
    theta_jn = np.minimum(theta_jn, np.pi - theta_jn)  #default effective 0 if neither is given
    if "KNtheta" not in parameters:
        parameters["KNtheta"] =  parameters.get("inclination_EM", theta_jn ) * 180.0 / np.pi
    if "inclination_EM" not in parameters:
        parameters["inclination_EM"] = parameters["KNtheta"] / 180.0 * np.pi
    return parameters


############################## mass conversions ####################################

def bbh_source_frame(params):
    """Convert parameters to BBH parameters using bilby function."""
    params, _ = convert_to_lal_binary_black_hole_parameters(params)
    return source_frame_masses(params)

def bns_source_frame(params):
    """Convert parameters to BNS parameters using bilby function."""
    params, _ = convert_to_lal_binary_neutron_star_parameters(params)
    return source_frame_masses(params)

def mass_ratio_to_eta(q):
    return q / (1 + q) ** 2

def component_masses_to_mass_quantities(m1, m2):
    eta = m1 * m2 / ((m1 + m2) * (m1 + m2))
    mchirp = ((m1 * m2) ** (3.0 / 5.0)) * ((m1 + m2) ** (-1.0 / 5.0))
    q = m2 / m1

    return (mchirp, eta, q)

def chirp_mass_and_eta_to_component_masses(mc, eta):
    """
    Utility function for converting mchirp,eta to component masses. The
    masses are defined so that m1>m2. The rvalue is a tuple (m1,m2).
    """
    M = mc / np.power(eta, 3. / 5.)
    q = (1 - np.sqrt(1. - 4. * eta) - 2 * eta) / (2. * eta)

    m1 = M / (1. + q)
    m2 = M * q / (1. + q)

    return (m1, m2)

def tidal_deformabilities_and_mass_ratio_to_eff_tidal_deformabilities(lambda1, lambda2, q):
    eta = q / np.power(1. + q, 2.)
    eta2 = eta * eta
    eta3 = eta2 * eta
    root14eta = np.sqrt(1. - 4 * eta)

    lambdaT = (8. / 13.) * ((1. + 7 * eta - 31 * eta2) * (lambda1 + lambda2) + root14eta * (1. + 9 * eta - 11. * eta2) * (lambda1 - lambda2))
    dlambdaT = 0.5 * (root14eta * (1. - 13272. * eta / 1319. + 8944. * eta2 / 1319.) * (lambda1 + lambda2) + (1. - 15910. * eta / 1319. + 32850. * eta2 / 1319. + 3380. * eta3 / 1319.) * (lambda1 - lambda2))

    return lambdaT, dlambdaT


def reweight_to_flat_mass_prior(df):
    total_mass = chirp_mass_and_mass_ratio_to_total_mass(df.chirp_mass, df.mass_ratio)
    m1 = total_mass / (1. + df.mass_ratio)
    jacobian = m1 * m1 / df.chirp_mass
    df_new = df.sample(frac=0.3, weights=jacobian)
    return df_new


def convert_mtot_mni(params):

    for par in ["mni", "mtot", "mrp"]:
        if par not in params:
            params[par] = 10**params[f"log10_{par}"]

    params["mni_c"] = params["mni"] / params["mtot"]
    params["mrp_c"] = (params["xmix"]*(params["mtot"]-params["mni"])-params["mrp"])
    return params

############################## pulsar timing conversions ####################################
def binary_mass_function(m_obs, m_comp, sin_i):
    return (m_comp * sin_i)**3 / (m_obs + m_comp)**2

def shapiro_delay(m_comp, sin_i):
    "see https://arxiv.org/pdf/1007.0933.pdf"
    shapiro_range = msun_s*1.e6 * m_comp # in microseconds
    orthometric_ratio = sin_i/(1+np.sqrt(1-sin_i**2))
    return shapiro_range * orthometric_ratio**3

def einstein_delay_orbital_factor(orbital_period, eccentricity):
    "see, e.g., 10.1007/978-3-662-62110-3_1, p.12 "
    return msun_s**(2/3) * eccentricity * (orbital_period /2/np.pi)**(1/3)
def simplified_einstein_delay(m_psr, m_comp, einstein_factor):
    "see, e.g., 10.1007/978-3-662-62110-3_1, p.12 "
    return einstein_factor *m_comp * (m_psr + 2*m_comp) / (m_psr + m_comp)**(4/3)

def einstein_delay(m_psr, m_comp, orbital_period, eccentricity):
    "see, e.g., 10.1007/978-3-662-62110-3_1, p.12 "
    einstein_delay_factor = einstein_delay_orbital_factor(orbital_period, eccentricity)
    return simplified_einstein_delay(m_psr, m_comp, einstein_delay_factor)

def mass_parameters_to_sini(total_mass, mass_function, m_comp):
    "Invert the binary mass function to get sin(i) for a given total mass and mass function"
    return np.cbrt(mass_function * total_mass**2)/m_comp

############################## EOS-related conversions ####################################

def EOS_to_ns_parameters(radii, masses, lambdas):
    TOV_mass = masses.max(axis=-1)
    TOV_radius = radii[np.argmax(masses)]
    R_14, R_16 = np.interp(x=[1.4, 1.6], xp=masses, fp=radii, left=0, right=0)

    return TOV_mass, TOV_radius, R_14, R_16

def EOS_to_system_parameters(radii, masses, lambdas, m1_source, m2_source):
    (log_lambda_1, log_lambda_2) = np.interp(x=[m1_source, m2_source],
            xp= masses, fp=np.log(lambdas), left=-np.inf, right=-np.inf)
    lambda_1 = np.exp(log_lambda_1)
    lambda_2 = np.exp(log_lambda_2)
    (radius_1, radius_2) = np.interp( x=[m1_source, m2_source],
            xp=masses, fp= radii, left =0, right=0)

    return lambda_1, lambda_2, radius_1, radius_2

def radii_from_qur(parameters):
    mass_1_source = parameters["mass_1_source"]
    mass_2_source = parameters["mass_2_source"]    
    lambda_1 = parameters["lambda_1"]
    lambda_2 = parameters["lambda_2"]

    compactness_1 = lambda_to_compactness(lambda_1)
    parameters["radius_1"] = mass_and_compactness_to_radius(mass_1_source, compactness_1)
    
    compactness_2 = lambda_to_compactness(lambda_2)
    parameters["radius_2"] = mass_and_compactness_to_radius(mass_2_source, compactness_2)

    chirp_mass_source = component_masses_to_chirp_mass(mass_1_source, mass_2_source)
    lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(
        lambda_1, lambda_2, mass_1_source, mass_2_source
    )

    parameters["R_16"] = (
        chirp_mass_source
        * np.power(lambda_tilde / 0.0042, 1.0 / 6.0)
        * geom_msun_km
    )
    return parameters

def lambda_to_compactness(lambda_i):
    "Function to link tidal deformability to compactness based on quasi-universal relation"
    loglam= np.log(lambda_i)
    return 0.371 - 0.0391 * loglam + 0.001056 * loglam * loglam

def mass_and_compactness_to_radius(mass, comp):
    ### returns 0 if compactness is greater than 0.5, i.e. black hole
    return np.where(comp<0.5, mass / comp * geom_msun_km, 0.0)

############################## GRB-related conversions ####################################

def gaussian_jet_energy_to_central_isotropic_energy_equivalent(Ejet, thetaCore, alphaWing):
    """
    Takes the total energy of a gaussian jet as well as the angular parameters and returns the isotropic-energy equivalent 
    on axis. This means it is assumed that the true jet energy follows some angular structure dEjet / dOmega = epsilon_c * exp(-1/2 theta^2/thetac^2).
    Then the distribution of the isotropic energy equivalent is simply related according to E_iso(theta) = 4pi dEjet / dOmega.
    
    :param Ejet: Total jet energy in ergs
    :param thetaCore: Core angle in rad
    :param alphaWing: Ratio of the wing angle and core angle.
    """
    
    # this is the analytical expression for int_{0}^{alphaWing*thetaCore} sin(x) *exp(-1/2 (x/thetac)^2) dx
    prefactor = np.sqrt(np.pi) * 1.j*thetaCore *np.exp(-thetaCore**2/2) / 2**1.5
    first_term = erf(0.5*(np.sqrt(2)*1.j*thetaCore + np.sqrt(2)*alphaWing))
    second_term = erf(0.5*(np.sqrt(2)*1.j*thetaCore - np.sqrt(2)*alphaWing))
    third_term = 2*erf(1.j*thetaCore/np.sqrt(2))
    integral_factor = prefactor * (first_term + second_term - third_term)
    integral_factor = integral_factor.real # this imaginary part is always 0 in this expression

    epsilon_c = Ejet / (2*np.pi* integral_factor)
    Eiso_c = 4*np.pi*epsilon_c

    return Eiso_c

def powerlaw_jet_energy_to_central_isotropic_energy_equivalent(Ejet, thetaCore, alphaWing, b):
    """
    Takes the total energy of a powerlaw jet as well as the angular parameters and returns the isotropic-energy equivalent 
    on axis. This means it is assumed that the true jet energy follows some angular structure dEjet / dOmega = epsilon_c * (1+1/b * (theta/thetaCore)^2)^(-b/2).
    Then the distribution of the isotropic energy equivalent is simply related according to E_iso(theta) = 4pi dEjet / dOmega.
    
    :param Ejet: Total jet energy in ergs
    :param thetaCore: Core angle in rad
    :param alphaWing: Ratio of the wing angle and core angle.
    :param b: Power law tail of the jet.
    """
    x = np.linspace(0, alphaWing*thetaCore, 100)
    y = np.sin(x)*(1+1/b*(x/thetaCore)**2)**(-b/2)
    integral_factor = simpson(x=x, y=y)

    epsilon_c = Ejet / (2*np.pi* integral_factor)
    Eiso_c = 4*np.pi*epsilon_c

    return Eiso_c
        
class EjectaFitting:
    mass_fitting_keys =["log10_mej_dyn", "log10_mej_wind", "log10_mej", "log10_E0"]
    
    def __call__(self, parameters):
        conv_parameters = self.ejecta_parameter_conversion(parameters)
        for key, val in zip(self.mass_fitting_keys,
                conv_parameters):
            # We always prefer explicitly sampled ejecta parameters
            parameters[key] = parameters.get(key, val)
        return parameters
    
    def ejecta_parameter_conversion(self, parameters):
        return [ -np.inf for _ in self.mass_fitting_keys ]

class NSBHEjectaFitting(EjectaFitting):
    def chibh2risco(self, chi_bh):

        Z1 = 1.0 + (1.0 - chi_bh ** 2) ** (1.0 / 3) * (
            (1 + chi_bh) ** (1.0 / 3) + (1 - chi_bh) ** (1.0 / 3)
        )
        Z2 = np.sqrt(3.0 * chi_bh ** 2 + Z1 ** 2.0)

        return 3.0 + Z2 - np.sign(chi_bh) * np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))

    def baryon_mass_NS(self, source_mass, compactness):
        """
        equation (7) in https://arxiv.org/abs/2002.07728
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
        '''
        equation (4) in https://arxiv.org/pdf/1807.00011
        '''

        mass_ratio_invert = mass_1_source / mass_2_source
        symm_mass_ratio = mass_ratio_invert / (1.0 + mass_ratio_invert)**2

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
        n1=8.63604211e-01,
        n2=1.68399507,
    ):

        """
        equation (9) in https://arxiv.org/abs/2002.07728
        """

        mass_ratio_invert = mass_1_source / mass_2_source

        #  use the BH spin to find the normalized risco
        risco = self.chibh2risco(chi_bh)
        baryon_mass_2 = self.baryon_mass_NS(mass_2_source, compactness_2)

        mdyn = (
            a1
            * np.power(mass_ratio_invert, n1)
            * (1.0 - 2.0 * compactness_2)
            / compactness_2
        )
        mdyn += -a2 * np.power(mass_ratio_invert, n2) * risco + a4
        mdyn *= baryon_mass_2

        mdyn = np.maximum(0.0, mdyn)

        return mdyn

    def nsbh_parameter_conversion(self, converted_parameters):

        mass_1_source = converted_parameters["mass_1_source"]
        mass_2_source = converted_parameters["mass_2_source"]

        radius_2 = converted_parameters["radius_2"]
        compactness_2 = mass_2_source * geom_msun_km / radius_2
        try:
            chi_1 = converted_parameters["chi_1"]
        except KeyError:
            cos_tilt_1 = converted_parameters.get("cos_tilt_1", np.cos(converted_parameters["tilt_1"]))
            chi_1 = converted_parameters["a_1"] * cos_tilt_1


        mdyn_fit = self.dynamic_mass_fitting(
            mass_1_source, mass_2_source, compactness_2, chi_1
        )
        remnant_disk_fit = self.remnant_disk_mass_fitting(
            mass_1_source, mass_2_source, compactness_2, chi_1
        )
        mdisk_fit = remnant_disk_fit - mdyn_fit
        mej_dyn = mdyn_fit + converted_parameters['alpha']

        # prevent the output message from being flooded by these warning messages
        old = np.seterr()
        np.seterr(invalid='ignore')
        np.seterr(divide='ignore')

        log_mej_wind = np.full_like(mdisk_fit, -np.inf)
        log_mej_dyn  = np.full_like(mdisk_fit, -np.inf)
        disk_mask = mdisk_fit > 0.

        log_mej_dyn[disk_mask] = np.log10(mej_dyn[disk_mask])
        log_mej_wind[disk_mask] = np.log10(mdisk_fit[disk_mask]) + np.log10(converted_parameters["ratio_zeta"])[disk_mask]
        

        total_ejeta_mass = 10**log_mej_dyn + 10**log_mej_wind

        log10_mej = np.log10(total_ejeta_mass)
        # FIXME: NSBH might produce a GRB, too. Why not provide the same expression for BNS?

        np.seterr(**old)
        return np.stack((log_mej_dyn, log_mej_wind, log10_mej, np.full_like(log_mej_wind, -np.inf)))
    
    def ejecta_parameter_conversion(self, parameters):
        return self.nsbh_parameter_conversion(parameters)

class BNSEjectaFitting(EjectaFitting):
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
            self,
            mass_1,
            mass_2,
            compactness_1,
            compactness_2,
            a=-0.287,
            b=0.494,
            c=-3.000
    ):
        """
        See https://arxiv.org/pdf/1809.11161 Eq. (22)
        """

        vej_dyn = a* mass_1/mass_2 * (1+c *compactness_1)
        vej_dyn += a* mass_2/mass_1 * (1+c* compactness_2)
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
        """
        q = mass_2 / mass_1
        lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2)
        mdyn = a*lambda_tilde*(q**(-1) -b) * np.exp(c/q) # this is always positive

        return mdyn
    
    def dynamic_vel_fitting_prompt_collapse(
            self,
            mass_1,
            mass_2,
            compactness_1,
            compactness_2,
            a=-0.395,
            b=0.798,
            c=-1.627):
        """
        See https://arxiv.org/pdf/2411.02342, Eq. (10)
        """        
        vdyn = a * mass_1/mass_2 *(1 + c*compactness_1)
        vdyn += a * mass_2/mass_1 * (1 + c* compactness_2)
        vdyn += b

        return vdyn
    
    def log10_disk_mass_fitting_prompt_collapse(
            self,
            mass_1,
            mass_2,
            lambda_1,
            lambda_2,
            a=7.70,
            b=-13.4,
            c=8.16e-3):
        """
        See https://arxiv.org/pdf/2411.02342, Eq. (11)
        Typo for b, b=-13.4 confirmed through author correspondence
        """
        q = mass_2 / mass_1
        lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2)
        log10_mdisk = a + b * q + c * lambda_tilde * q**2

        log10_mdisk = np.minimum(log10_mdisk, -1)

        return log10_mdisk
    
    def chiBH_fitting(
            self,
            mass_1,
            mass_2,
            lambda_1,
            lambda_2,
            a = 0.537,
            b = -0.185,
            c = -0.514
    ):
        """
        See https://arxiv.org/pdf/1812.04803, Eq. (D7)
        nu needs to be divided by 0.25 and lambda_tilde by 400, confirmed through author correspondence
        """

        lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2)
        M = mass_1 + mass_2
        nu = component_masses_to_symmetric_mass_ratio(mass_1, mass_2)
        
        chi_BH = np.tanh(a*(nu/0.25)**2*(M+b*lambda_tilde / 400)+ c)

        return chi_BH

    def bns_ejecta_conversion(self, converted_parameters):

        # prevent the output message flooded by these warning messages
        old = np.seterr()
        np.seterr(invalid='ignore')
        np.seterr(divide='ignore')

        mass_1_source = converted_parameters["mass_1_source"]
        mass_2_source = converted_parameters["mass_2_source"]

        total_mass = mass_1_source + mass_2_source
        mass_ratio = mass_2_source / mass_1_source

        radius_1 = converted_parameters["radius_1"]
        radius_2 = converted_parameters["radius_2"]

        compactness_1 = mass_1_source * geom_msun_km / radius_1
        compactness_2 = mass_2_source * geom_msun_km / radius_2
        #FIXME: switch to prompt collapse fitting for appropriate thresholds
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
        # total eject mass
        total_ejeta_mass = 10**log10_mej_dyn + 10**log10_mej_wind

        np.seterr(**old)
        return log10_mej_dyn, log10_mej_wind, np.log10(total_ejeta_mass), log10_mdisk_fit
    
    def grb_energy_conversion(self, converted_parameters, log10_mdisk_fit):

        # GRB afterglow energy
        log10_Ejet = np.log10(converted_parameters.get("ratio_epsilon", 2e-4))
        log10_Ejet += np.log10(1.0 - converted_parameters["ratio_zeta"])
        log10_Ejet += log10_mdisk_fit + np.log10(msun_to_ergs)

        thetaCore = converted_parameters.get("thetaCore", 0.105) ## default about 6 degree, see arxiv:2210.05695
        
        if not any(key in converted_parameters for key in ["thetaWing", "alphaWing", "b"]):
            return log10_Ejet - np.log10(np.sin(thetaCore/2)**2)

        if "alphaWing" in converted_parameters:
            alphaWing = converted_parameters['alphaWing'] 
        else:
            alphaWing = converted_parameters["thetaWing"] / converted_parameters["thetaCore"]

        
        if "b" in converted_parameters: # power law jet
            jet_func = powerlaw_jet_energy_to_central_isotropic_energy_equivalent
            data = np.column_stack((10**log10_Ejet, thetaCore, alphaWing, converted_parameters["b"]))
            
        else:
            jet_func = gaussian_jet_energy_to_central_isotropic_energy_equivalent
            data = np.column_stack((10**log10_Ejet, thetaCore, alphaWing))
                
        return np.log10([jet_func(*row) for row in data])
    
        

    def bns_parameter_conversion(self, parameters):
        log10_mej_dyn, log10_mej_wind, log10_mej_total, log10_mdisk_fit = self.bns_ejecta_conversion(parameters)

        if "log10_E0" in parameters:
            log10_E0 = parameters["log10_E0"]
        else:
            log10_E0 = self.grb_energy_conversion(parameters, log10_mdisk_fit)
        
        converted_ejecta = (log10_mej_dyn, log10_mej_wind, log10_mej_total, log10_E0)

        return  np.where(np.isfinite(converted_ejecta), converted_ejecta, -np.inf)

    def ejecta_parameter_conversion(self, parameters):
        return self.bns_parameter_conversion(parameters)

class KilonovaEjectaFitting(BNSEjectaFitting, NSBHEjectaFitting):
    def ejecta_parameter_conversion(self, parameters):
        try:
            #heavier object is a NS
            if parameters['radius_1'] > 0.:
                return self.bns_parameter_conversion(parameters)
            # heavier object is BH, but lighter object is NS
            elif parameters['radius_2']>0.:
                return self.nsbh_parameter_conversion(parameters)
            # both objects are BHs
            else:
                return np.full(4, -np.inf)
        except ValueError:
            #ValueError occurs when trying to obtain truth values of arrays 
            # -> evaluate many points at once and chose conditional ejecta_fitting
            return  np.where(parameters["radius_1"]>0.,  #heavier object is a NS
                        self.bns_parameter_conversion(parameters),
                    np.where(parameters["radius_2"]>0., ## elif component 2 is a NS
                        self.nsbh_parameter_conversion(parameters),
                    ### else assume BBH (i.e., no ejecta)
                        np.full((4,)+ parameters["mass_1_source"].shape, -np.inf)
                        )
                    )
              
class MultimessengerConversion:
    def __init__(self, *conversions):
        self._conversions = conversions
    
    @classmethod
    def from_args(cls, args):
        # FIXME: implement argument parsing to select conversions
        raise NotImplementedError("from_args not yet implemented")

    @classmethod
    def from_dict(cls, instruction_dict):
        conversions = []

        # NOTE: Order matters!!!
        if 'cosmo' in instruction_dict:
            set_cosmology(instruction_dict['cosmo'])
            conversions.append(cosmology_to_distance) 

        if 'gw' in instruction_dict:
            conversions.append(instruction_dict['gw'])

        if 'eos' in instruction_dict:
            conversions.append(instruction_dict['eos'])

        if 'ejecta' in instruction_dict:    
            conversions.append(KilonovaEjectaFitting())

        if 'em' in instruction_dict:
            conversions.append(instruction_dict['em'])

        return cls(*conversions)
    
    @classmethod
    def basic_cbc(cls, eos_conversion, em_conversion):
        return cls(bbh_source_frame, eos_conversion, KilonovaEjectaFitting(), em_conversion)
    
    def convert_to_multimessenger_parameters(self, parameters, add_new_keys=False):
        original_keys = list(parameters.keys())
        converted_parameters = {k: val_to_scalar(v) for k, v in parameters.items()}

        converted_parameters = self.core_conversion(converted_parameters)

        converted_parameters = {k: val_to_scalar(v) for k, v in converted_parameters.items()}
        
        if add_new_keys:
            added_keys = [k for k in converted_parameters.keys() if k not in original_keys]
            return converted_parameters, added_keys
        else:
            return converted_parameters
    
    def core_conversion(self, parameters):
        for conv in self._conversions:
            parameters = conv(parameters)
        return parameters
    
    def identity_conversion(self, parameters):
        return parameters

        
        
label_mapping = {
    ## Cosmology parameters ##
    'Hubble_constant'       : r'$H_0{\rm [km\,s^{-1}\,Mpc^{-1}]}$',
    'Omega_matter'          : r'$\Omega_{m}$',
    'redshift'              : r'$z$',
    ## System parameters ##
    'inclination_EM'        : r'$\theta_{obs}$',
    'theta_jn'              : r'$\theta_{JN}$',
    'cos_theta_jn'          : r'$\cos{\theta_{JN}}$',
    'luminosity_distance'   : r'$d_L{\rm [Mpc]}$', 
    ## GW parameters ##
    'chirp_mass'            :r'$\mathcal{M}_c{\rm [M_{\odot}]}$',
    'mass_ratio'            : r'$q$', 
    'chi_eff'               : r'$\chi_{\rm{eff}}$', 
    'mass_1_source'         : r'$m_{1,s}{\rm [M_{\odot}]}$', 
    'mass_2_source'         : r'$m_{2,s}{\rm [M_{\odot}]}$', 
    ## KN parameters ##
    'log10_mej'             : r'$\log_{10}(M_{\rm{ej}}{\rm [M_{\odot}]})$',
    'log10_mej_dyn'         : r'$\log_{10}(M_{\rm{ej,dyn}}{\rm [M_{\odot}]})$',
    'log10_mej_wind'        : r'$\log_{10}(M_{\rm{ej,wind}}{\rm [M_{\odot}]})$',
    'log10_E0'              : r'$\log_{10}(E_0{\rm [erg]})$',
    'ratio_zeta'            : r'$\zeta$',
    'alpha'                 : r'$\alpha$',
    'KNtheta'               : r'$\theta_{KN} [^\circ]$',
    'KNphi'                 : r'$\phi_{KN} [^\circ]$',
    'kappa_Ye'              : r'$\kappa_{\rm{Y_e}}$',
    'kappa_v'               : r'$\kappa_{v}$', 
    ## GRB parameters ##
    'log10_E0'              : r'$\log_{10}(E_0{\rm [erg]})$',
    'ratio_epsilon'         : r'$\epsilon$',
    'thetaCore'             : r'$\theta_{c}$',
    'thetaWing'             : r'$\theta_{w}$',
    'alphaWing'             : r'$\alpha_{w}$',
    'log10_n0'              : r'$\log_{10}(n_{0}{\rm [cm^{-3}]})$',
    'p'                     : r'$p$',
    'log10_epsilon_e'       : r'$\log_{10}(\epsilon_{e})$',
    'log10_epsilon_B'       : r'$\log_{10}(\epsilon_{B})$',
    ## Ejecta parameters ##
    'mni'                   : r'$M_{\rm{Ni}}{\rm [M_{\odot}]}$',
    'mtot'                  : r'$M_{\rm{tot}}{\rm [M_{\odot}]}$',
    'mrp'                   : r'$M_{\rm{rp}}{\rm [M_{\odot}]}$',
    'mni_c'                 : r'$M_{\rm{Ni}}/M_{\rm{tot}}$',
    'mrp_c'                 : r'$M_{\rm{rp,c}}{\rm [M_{\odot}]}$',
    ### EOS parameters ###
    'L_sym'                 : r'$L_{\rm sym}{\rm [MeV]}$',
    'K_sym'                 : r'$K_{\rm sym}{\rm [MeV]}$',
    'K_sat'                 : r'$K_{\rm sat}{\rm [MeV]}$',
    '3n_sat'                : r'$c^2_{3n_{\rm sat}}{\rm [c^2]}$',
    '5n_sat'                : r'$c^2_{5n_{\rm sat}}{\rm [c^2]}$',
    'TOV_mass'              : r'$M_{\rm{TOV}}{\rm [M_{\odot}]}$',
    'R_14'                  : r'$R_{1.4}{\rm[km]}$',
    'lambda_tilde'          : r'$\tilde{\Lambda}$', 
}