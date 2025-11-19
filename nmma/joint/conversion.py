from __future__ import division

import numpy as np
import pandas as pd
from astropy import units
from astropy import cosmology as cosmo
from nmma.joint.constants import geom_msun_km, msun_to_ergs, default_cosmology

from bilby.gw.conversion import (
    component_masses_to_chirp_mass,
    lambda_1_lambda_2_to_lambda_tilde,
    convert_to_lal_binary_black_hole_parameters,
    generate_mass_parameters,
    chirp_mass_and_mass_ratio_to_total_mass
)

from ..eos.eos_processing import setup_eos_generator

########################## distance conversions ####################################
def distance_modulus_nmma(d_lum = 1e-5):
        # mag_app = mag_abs + 5* log10(dist/10pc) | NMMA-dist is in Mpc
        #         = mag_abs + 5 * (log10(Mpc/10pc)+ log10(params["luminosity_distance"]))  
        # therefore: distance_modulus = mag_app - mag_abs =
        return  5.0 * (5+ np.log10(d_lum))

def luminosity_distance_to_redshift(distance, cosmology = default_cosmology):

    if isinstance(distance, pd.Series):
        distance = distance.values

    if hasattr(distance, '__len__') and len(distance)>50: #luminosity_distance_to_redshift gets really slow if too many distances are put in at once
        # zmax since solutions can be non-unique at high distances!
        zmin = cosmo.z_at_value(cosmology.luminosity_distance, distance.min() * units.Mpc)
        zmax = cosmo.z_at_value(cosmology.luminosity_distance, distance.max() * units.Mpc)
        zgrid = np.geomspace(zmin, zmax, 50)
        distance_grid = cosmology.luminosity_distance(zgrid).value
        return np.interp(distance, distance_grid, zgrid).value
    else:
        return cosmo.z_at_value(cosmology.luminosity_distance, distance *units.Mpc).value
        
def get_redshift(parameters):
    if "redshift" in parameters:
        return parameters["redshift"]
    elif "luminosity_distance" in parameters:
            return luminosity_distance_to_redshift(parameters["luminosity_distance"])
    else:
        ## zeros like the first input of parameters, independent of size and keys
        return np.zeros_like(next(iter(parameters.values()))) 

def cosmology_to_distance(parameters, cosmology= default_cosmology):
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
    if "redshift" not in converted_parameters.keys():
        distance = converted_parameters["luminosity_distance"]
        converted_parameters["redshift"] = luminosity_distance_to_redshift(distance)

    if "mass_1_source" not in converted_parameters.keys():
        z = converted_parameters["redshift"]
        converted_parameters["mass_1_source"] = np.array(converted_parameters["mass_1"] / (1 + z))

    if "mass_2_source" not in converted_parameters.keys():
        z = converted_parameters["redshift"]
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

############################## EOS-related conversions ####################################
class EoSConverter:
    def __init__(self, args, method=None):
        # Case 1: eos is generated from emulator on the fly
        if method == "emulated":
            self.tov_emulator = setup_eos_generator(args)
            self.macro_conversion = self.tov_emulator.generate_macro_eos
        
        elif method == "tabulated":
            #case 2a: we use a single eos
            if getattr(args, 'eos_file', None):
                self.eos_data = [np.loadtxt(args.eos_file, usecols = [0,1,2]).T]
                self.macro_conversion = self.single_eos_from_ram

            # Case 2b: precomputed eos data is loaded to ram
            elif args.eos_to_ram:
                self.eos_data = [np.loadtxt(f"{args.eos_data}/{j+1}.dat", usecols = [0,1,2]).T 
                                for j in range(args.Neos)]
                self.macro_conversion = self.eos_from_ram

            # Case 2c: eos are loaded directly from file
            else:
                self.eos_data = args.eos_data
                self.macro_conversion = self.eos_direct_load

        
    def full_eos_conversion(self, parameters):
        self.macro_conversion(parameters)
        return self.macro_props_from_eos(parameters)    

    def eos_direct_load(self, converted_parameters):
        EOSID = np.atleast_1d(converted_parameters["EOS"]).astype(int)
        return [np.loadtxt(f"{self.args.eos_data}/{j+1}.dat", usecols = [0,1,2]).T for j in EOSID]

    def eos_from_ram(self, converted_parameters):
        EOSID = np.atleast_1d(converted_parameters["EOS"]).astype(int)
        return [self.eos_data[i] for i in EOSID]
    
    def single_eos_from_ram(self, _):
        return self.eos_data
      


    def compute_macro_parameters(self, parameters):
        eos_data = self.macro_conversion(parameters)
        if len(eos_data) ==1:
            radii, masses, lambdas = eos_data[0]
        else:
            radii, masses, lambdas = map(list, zip(*eos_data))
        
        self.macro_parameters = {'radii': radii, 'masses': masses, 'lambdas': lambdas}

    
    def macro_props_from_eos(self, converted_parameters):
        eos_keys = ["TOV_mass", "TOV_radius", "lambda_1", "lambda_2",
                    "radius_1", "radius_2", "R_14", "R_16"]
        
        m1_source = converted_parameters["mass_1_source"]
        m2_source = converted_parameters["mass_2_source"]
        radii, masses, lambdas = self.macro_parameters.values()

        if isinstance(radii, np.ndarray): # single eos case
            for key, val_array in zip(eos_keys, 
            self.EOS2Parameters(radii, masses, lambdas, m1_source, m2_source)
            ):
                converted_parameters[key] = val_array
        else:
            ### assuming TOV mass and radius are the last entries of the respective arrays
            TOV_mass_list, TOV_radius_list, R_14_list, R_16_list = [], [], [], []
            lambda_1_list, lambda_2_list, radius_1_list, radius_2_list = [], [], [], []
            for i, rad in enumerate(radii):
                (TOV_mass, TOV_radius, lambda_1, lambda_2, radius_1,
                    radius_2, R_14, R_16
                ) = self.EOS2Parameters(rad, masses[i], lambdas[i], m1_source[i],  m2_source[i] )
                    
                TOV_radius_list.append(TOV_radius)
                TOV_mass_list.append(TOV_mass)
                lambda_1_list.append(lambda_1)
                lambda_2_list.append(lambda_2)
                radius_1_list.append(radius_1)
                radius_2_list.append(radius_2)
                R_14_list.append(R_14)
                R_16_list.append(R_16)
        
            for key, _list in zip(eos_keys, [
                TOV_mass_list, TOV_radius_list, lambda_1_list,
                lambda_2_list, radius_1_list, radius_2_list, R_14_list, R_16_list
            ]):
                converted_parameters[key] = np.array(_list)

        return converted_parameters


    def EOS2Parameters(self, radii, masses, lambdas, m1_source, m2_source
    ):
        ### FIXME: Under what circumstance would these not simply be mass_val[-1], radius_val[-1]?
        TOV_mass = masses.max(axis=-1)
        TOV_radius = radii[np.argmax(masses)]

        (log_lambda_1, log_lambda_2) = np.interp(x=[m1_source, m2_source],
                xp= masses, fp=np.log(lambdas), left=0, right=0)
        lambda_1 = np.exp(log_lambda_1)
        lambda_2 = np.exp(log_lambda_2)
        try:
            (radius_1, radius_2, R_14, R_16) = np.interp(
                    x=[m1_source, m2_source, 1.4, 1.6],
                    xp=masses, fp= radii, left =0, right=0)

            return TOV_mass, TOV_radius, lambda_1, lambda_2, radius_1, radius_2, R_14, R_16
        ## radius interpolation will raise an error if dealing with multiple sources at once
        # In that case we return all values as corresponding arrays
        except ValueError:
            ref = np.ones_like(radius_1)
            (radius_1, radius_2, R_14, R_16) = np.interp(
                    x=[m1_source, m2_source, 1.4*ref, 1.6*ref],
                    xp=masses, fp= radii, left =0, right=0)

            return ref*TOV_mass, ref*TOV_radius, lambda_1, lambda_2, radius_1, radius_2, R_14, R_16

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

class BBHEjectaFitting:
    def __init__(self):
        self.mass_fitting_keys =["log10_mej_dyn", "log10_mej_wind", "log10_mej", "log10_E0"]
    def vals_only_ejecta_parameter_conversion(self, converted_parameters):
        return  np.full((4,)+ converted_parameters["mass_1_source"].shape, -np.inf)
        

    def ejecta_parameter_conversion(self, converted_parameters):
        for key, val in zip(self.mass_fitting_keys, self.vals_only_ejecta_parameter_conversion(converted_parameters)):
            converted_parameters[key] = val
        
        return converted_parameters
    

class NSBHEjectaFitting:
    def __init__(self):
        self.proper_mass_fitting_keys =["log10_mej_dyn", "log10_mej_wind"]
        self.uniform_mass_fitting_keys =["log10_mej_dyn", "log10_mej_wind", "log10_mej", "log10_E0"]

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

    def vals_only_ejecta_parameter_conversion(self, converted_parameters, uniform_output =False):

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
        

        if uniform_output:
            total_ejeta_mass = 10**log_mej_dyn + 10**log_mej_wind

            log10_mej = np.log10(total_ejeta_mass)

            np.seterr(**old)
            return np.stack((log_mej_dyn, log_mej_wind, log10_mej, np.full_like(log_mej_wind, -np.inf)))
        else:

            np.seterr(**old)
            return log_mej_dyn, log_mej_wind 
    
    def ejecta_parameter_conversion(self, converted_parameters, uniform_output=False):
        if uniform_output:
            add_keys = self.uniform_mass_fitting_keys
        else: 
            add_keys = self.proper_mass_fitting_keys

        for key, val in zip(add_keys, 
                self.vals_only_ejecta_parameter_conversion(converted_parameters, uniform_output)):
            converted_parameters[key] = val
        return converted_parameters


class BNSEjectaFitting:
    def __init__(self):
        self.mass_fitting_keys =["log10_mej_dyn", "log10_mej_wind", "log10_mej", "log10_E0"]

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

    def vals_only_ejecta_parameter_conversion(self, converted_parameters):

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
        # GRB afterglow energy
        log10_E0_MSUN = (
            np.log10(converted_parameters["ratio_epsilon"])
            + np.log10(1.0 - converted_parameters["ratio_zeta"])
            + log10_mdisk_fit
        )
        
        np.seterr(**old)
        converted_ejecta = np.stack((log10_mej_dyn, log10_mej_wind, np.log10(total_ejeta_mass), log10_E0_MSUN + np.log10(msun_to_ergs) ))

        return  np.where(np.isfinite(converted_ejecta), converted_ejecta, -np.inf)


    def ejecta_parameter_conversion(self, converted_parameters): 
        for key, val in zip(self.mass_fitting_keys, 
                self.vals_only_ejecta_parameter_conversion(converted_parameters)):
            converted_parameters[key] = val

        return converted_parameters


class MultimessengerConversion:
    def __init__(self, args, messengers, ana_modifiers =[], internal_conversions = {}):
        self.messengers     = messengers
        self.modifiers      = ana_modifiers
        self.args           = args

        self.BNSejectaFitting   = BNSEjectaFitting()
        self.NSBHejectaFitting  = NSBHEjectaFitting()
        self.BBHejectaFitting   = BBHEjectaFitting()

        run_cosmo = getattr(args, 'cosmology', None)
        if run_cosmo is None:
            self.cosmology = default_cosmology
        else:
            self.cosmology = run_cosmo

        
        self.eos_conversion= internal_conversions.get('eos', self.setup_eos_conversion(args))
        self.gw_conversion = internal_conversions.get('gw', convert_to_lal_binary_black_hole_parameters)
        self.em_conversion = internal_conversions.get('em', observation_angle_conversion)
        
    def setup_eos_conversion(self, args):
        "This should only be called if we want to get ns-properties "
        "but do not evaluate a corresponding EoS-likelihood. "
        # 
        # Case 1: eos is generated from emulator on the fly
        if 'eos' in self.messengers:
            self.eos_converter = EoSConverter(args, method='emulated')
            return self.eos_converter.full_eos_conversion
        
        # Case 2: eos is loaded from tabulated data
        elif 'tabulated_eos' in self.modifiers:
            self.eos_converter = EoSConverter(args, method='tabulated')
            return self.eos_converter.full_eos_conversion
        
        # Case 3: no eos sampling, use quasi-universal relations instead
        else:
            return radii_from_qur 
        

    def ejecta_parameter_conversion(self, parameters):
        try:
            #heavier object is a NS
            if parameters['radius_1'] > 0.:
                ejecta_parameters = self.BNSejectaFitting.vals_only_ejecta_parameter_conversion(parameters)
            # heavier object is BH, but lighter object is NS
            elif parameters['radius_2']>0.:
                ejecta_parameters = self.NSBHejectaFitting.vals_only_ejecta_parameter_conversion(parameters, True)
            # both objects are BHs
            else:
                ejecta_parameters = self.BBHejectaFitting.vals_only_ejecta_parameter_conversion(parameters)
        except ValueError:
            #ValueError occurs when trying to obtain truth values of arrays 
            # -> evaluate many points at once and chose conditional ejecta_fitting
            ejecta_parameters = np.where(parameters["radius_1"]>0.,   #heavier object is a NS
                self.BNSejectaFitting.vals_only_ejecta_parameter_conversion(parameters),
                np.where(parameters["radius_2"]>0., ## elif component 2 is a NS
                    self.NSBHejectaFitting.vals_only_ejecta_parameter_conversion(parameters, True),
                    ### else assume BBH (i.e., no ejecta)
                    self.BBHejectaFitting.vals_only_ejecta_parameter_conversion(parameters),
                    )
                )
            
        for key, val in zip(self.BNSejectaFitting.mass_fitting_keys,
                ejecta_parameters):
            parameters[key] = val
        return parameters

    def convert_to_multimessenger_parameters(self, parameters):
        original_keys = list(parameters.keys())
        converted_parameters = self.clean_datatypes(parameters)

        if "Hubble" in self.modifiers:
            converted_parameters = cosmology_to_distance(
            converted_parameters, cosmology=self.cosmology
        )
        if "gw" in self.messengers:            
            converted_parameters, _ = self.gw_conversion(converted_parameters)

        converted_parameters = source_frame_masses(converted_parameters)

        ####EOS/ tidal treatment
        converted_parameters = self.eos_conversion(converted_parameters)

            
        if "em" in self.messengers:
            converted_parameters = self.ejecta_parameter_conversion(converted_parameters)
            converted_parameters, _ = self.em_conversion(converted_parameters)

        added_keys = [key for key in converted_parameters.keys()
                      if key not in original_keys]

        converted_parameters = self.clean_datatypes(converted_parameters)
        return converted_parameters, added_keys
    
    def identity_conversion(self, parameters):
        return parameters, []

    def clean_datatypes(self, parameters):
        """Homogenise data types, preferentially to scalars instead of arrays"""
        try:
            return {k: np.ravel(v).item() for k, v in parameters.copy().items()}
        except ValueError:
            return {k: np.ravel(v) for k, v in parameters.copy().items()}
        
