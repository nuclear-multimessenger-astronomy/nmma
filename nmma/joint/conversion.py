from __future__ import division
import sys

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.constants

import lal

from bilby.gw.conversion import (
    component_masses_to_chirp_mass,
    lambda_1_lambda_2_to_lambda_tilde,
    convert_to_lal_binary_black_hole_parameters,
    luminosity_distance_to_redshift,
    generate_mass_parameters,
    generate_tidal_parameters,
    _generate_all_cbc_parameters,
)


def source_frame_masses(converted_parameters, added_keys):

    if "redshift" not in converted_parameters.keys():
        distance = converted_parameters["luminosity_distance"]
        converted_parameters["redshift"] = luminosity_distance_to_redshift(distance)
        added_keys = added_keys + ["redshift"]

    if "mass_1_source" not in converted_parameters.keys():
        z = converted_parameters["redshift"]
        converted_parameters["mass_1_source"] = converted_parameters["mass_1"] / (1 + z)
        added_keys = added_keys + ["mass_1_source"]

    if "mass_2_source" not in converted_parameters.keys():
        z = converted_parameters["redshift"]
        converted_parameters["mass_2_source"] = converted_parameters["mass_2"] / (1 + z)
        added_keys = added_keys + ["mass_2_source"]

    return converted_parameters, added_keys


def EOS2Parameters(
    interp_mass_radius, interp_mass_lambda, mass_1_source, mass_2_source
):

    TOV_mass = interp_mass_radius.x[-1]
    TOV_radius = interp_mass_radius.y[-1]

    minimum_mass = interp_mass_radius.x[0]

    if mass_1_source < minimum_mass or mass_1_source > TOV_mass:
        lambda_1 = np.array([0.0])
        radius_1 = np.array([2.0 * mass_1_source * lal.MRSUN_SI / 1e3])
    else:
        lambda_1 = np.array([interp_mass_lambda(mass_1_source)])
        radius_1 = np.array([interp_mass_radius(mass_1_source)])

    if mass_2_source < minimum_mass or mass_2_source > TOV_mass:
        lambda_2 = np.array([0.0])
        radius_2 = np.array([2.0 * mass_2_source * lal.MRSUN_SI / 1e3])
    else:
        lambda_2 = np.array([interp_mass_lambda(mass_2_source)])
        radius_2 = np.array([interp_mass_radius(mass_2_source)])

    return TOV_mass, TOV_radius, lambda_1, lambda_2, radius_1, radius_2


class NSBHEjectaFitting(object):
    def __init__(self):
        pass

    def chieff2risco(self, chi_eff):

        Z1 = 1.0 + (1.0 - chi_eff ** 2) ** (1.0 / 3) * (
            (1 + chi_eff) ** (1.0 / 3) + (1 - chi_eff) ** (1.0 / 3)
        )
        Z2 = np.sqrt(3.0 * chi_eff ** 2 + Z1 ** 2.0)

        return 3.0 + Z2 - np.sign(chi_eff) * np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))

    def remnant_disk_mass_fitting(
        self,
        mass_1_source,
        mass_2_source,
        compactness_2,
        chi_eff,
        a=0.40642158,
        b=0.13885773,
        c=0.25512517,
        d=0.761250847,
    ):

        mass_ratio_invert = mass_1_source / mass_2_source
        symm_mass_ratio = mass_ratio_invert / np.power(1.0 + mass_ratio_invert, 2.0)

        risco = self.chieff2risco(chi_eff)
        bayon_mass_2 = (
            mass_2_source * (1.0 + 0.6 * compactness_2) / (1.0 - 0.5 * compactness_2)
        )

        remant_mass = (
            a * np.power(symm_mass_ratio, -1.0 / 3.0) * (1.0 - 2.0 * compactness_2)
        )
        remant_mass += -b * risco / symm_mass_ratio * compactness_2 + c

        remant_mass = np.maximum(remant_mass, 0.0)

        remant_mass = np.power(remant_mass, 1.0 + d)

        remant_mass *= bayon_mass_2

        return remant_mass

    def dynamic_mass_fitting(
        self,
        mass_1_source,
        mass_2_source,
        compactness_2,
        chi_eff,
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

        risco = self.chieff2risco(chi_eff)
        bayon_mass_2 = (
            mass_2_source * (1.0 + 0.6 * compactness_2) / (1.0 - 0.5 * compactness_2)
        )

        mdyn = (
            a1
            * np.power(mass_ratio_invert, n1)
            * (1.0 - 2.0 * compactness_2)
            / compactness_2
        )
        mdyn += -a2 * np.power(mass_ratio_invert, n2) * risco + a4
        mdyn *= bayon_mass_2

        mdyn = np.maximum(0.0, mdyn)

        return mdyn

    def ejecta_parameter_conversion(self, converted_parameters, added_keys):

        mass_1_source = converted_parameters["mass_1_source"]
        mass_2_source = converted_parameters["mass_2_source"]
        total_mass_source = mass_1_source + mass_1_source

        radius_2 = converted_parameters["radius_2"]
        compactness_2 = mass_2_source * lal.MRSUN_SI / (radius_2 * 1e3)

        if "cos_tilt_1" not in converted_parameters:
            converted_parameters["cos_tilt_1"] = np.cos(converted_parameters["tilt_1"])
        if "cos_tilt_2" not in converted_parameters:
            converted_parameters["cos_tilt_2"] = np.cos(converted_parameters["tilt_2"])

        chi_1 = converted_parameters["a_1"] * converted_parameters["cos_tilt_1"]
        chi_2 = converted_parameters["a_2"] * converted_parameters["cos_tilt_2"]
        chi_eff = (mass_1_source * chi_1 + mass_2_source * chi_2) / total_mass_source

        mdyn_fit = self.dynamic_mass_fitting(
            mass_1_source, mass_2_source, compactness_2, chi_eff
        )
        remnant_disk_fit = self.remnant_disk_mass_fitting(
            mass_1_source, mass_2_source, compactness_2, chi_eff
        )
        mdisk_fit = remnant_disk_fit - mdyn_fit

        log_mdyn_fit = np.log(mdyn_fit)
        log_alpha = converted_parameters["log10_alpha"] * np.log(10.0)
        log_mej_dyn = np.logaddexp(log_mdyn_fit, log_alpha)
        log10_mej_dyn = log_mej_dyn / np.log(10.0)

        converted_parameters["log10_mej_dyn"] = log10_mej_dyn
        converted_parameters["log10_mej_wind"] = np.log10(
            converted_parameters["ratio_zeta"]
        ) + np.log10(mdisk_fit)

        if isinstance(compactness_2, (list, tuple, pd.core.series.Series, np.ndarray)):
            BH_index = np.where(compactness_2 == 0.5)[0]
            negative_mdisk_index = np.where(mdisk_fit <= 0.0)[0]
            converted_parameters["log10_mej_dyn"][BH_index] = -np.inf
            converted_parameters["log10_mej_dyn"][negative_mdisk_index] = -np.inf
            converted_parameters["log10_mej_wind"][BH_index] = -np.inf
            converted_parameters["log10_mej_wind"][negative_mdisk_index] = -np.inf

        else:
            if compactness_2 == 0.5 or mdisk_fit < 0.0:
                converted_parameters["log10_mej_dyn"] = -np.inf
                converted_parameters["log10_mej_wind"] = -np.inf

        added_keys = added_keys + ["log10_mej_dyn", "log10_mej_wind"]

        return converted_parameters, added_keys


class BNSEjectaFitting(object):
    def __init__(self):
        pass

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

    def ejecta_parameter_conversion(self, converted_parameters, added_keys):

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

        compactness_1 = mass_1_source * lal.MRSUN_SI / (radius_1 * 1e3)
        compactness_2 = mass_2_source * lal.MRSUN_SI / (radius_2 * 1e3)

        mdyn_fit = self.dynamic_mass_fitting_KrFo(
            mass_1_source, mass_2_source, compactness_1, compactness_2
        )

        log10_mdisk_fit = self.log10_disk_mass_fitting(
            total_mass,
            mass_ratio,
            converted_parameters["TOV_mass"],
            converted_parameters["R_16"] * 1e3 / lal.MRSUN_SI,
        )

        mej_dyn = mdyn_fit + converted_parameters["alpha"]
        log10_mej_dyn = np.log10(mej_dyn)

        converted_parameters["log10_mej_dyn"] = log10_mej_dyn
        log10_mej_wind = np.log10(converted_parameters["ratio_zeta"]) + log10_mdisk_fit
        converted_parameters["log10_mej_wind"] = log10_mej_wind
        # GRB afterglow energy
        log10_E0_MSUN = (
            np.log10(converted_parameters["ratio_epsilon"])
            + np.log10(1.0 - converted_parameters["ratio_zeta"])
            + log10_mdisk_fit
        )
        log10_E0_erg = log10_E0_MSUN + np.log10(
            lal.MSUN_SI * scipy.constants.c * scipy.constants.c * 1e7
        )
        converted_parameters["log10_E0"] = log10_E0_erg

        if (
            isinstance(compactness_1, (list, tuple, pd.core.series.Series, np.ndarray))
            and len(compactness_1) > 1
        ):

            mdyn_nan_index = np.where((~np.isfinite(log10_mej_dyn)))[0]
            if not isinstance(converted_parameters, pd.DataFrame):
                converted_parameters["log10_mej_dyn"][mdyn_nan_index] = -np.inf
            else:
                converted_parameters.loc[mdyn_nan_index, "log10_mej_dyn"] = -np.inf

            mwind_nan_index = np.where((~np.isfinite(log10_mej_wind)))[0]
            if not isinstance(converted_parameters, pd.DataFrame):
                converted_parameters["log10_mej_wind"][mwind_nan_index] = -np.inf
            else:
                converted_parameters.loc[mwind_nan_index, "log10_mej_wind"] = -np.inf

            E0_nan_index = np.where((~np.isfinite(log10_E0_erg)))[0]
            if not isinstance(converted_parameters, pd.DataFrame):
                converted_parameters["log10_E0"][E0_nan_index] = -np.inf
            else:
                converted_parameters.loc[E0_nan_index, "log10_E0"] = -np.inf

            BH_index = np.where((compactness_1 == 0.5) + (compactness_2 == 0.5))[0]
            if not isinstance(converted_parameters, pd.DataFrame):
                converted_parameters["log10_mej_dyn"][BH_index] = -np.inf
                converted_parameters["log10_mej_wind"][BH_index] = -np.inf
                converted_parameters["log10_E0"][BH_index] = -np.inf
            else:
                converted_parameters.loc[BH_index, "log10_mej_dyn"] = -np.inf
                converted_parameters.loc[BH_index, "log10_mej_wind"] = -np.inf
                converted_parameters.loc[BH_index, "log10_E0"] = -np.inf

        else:
            if not np.isfinite(log10_mej_dyn):
                converted_parameters["log10_mej_dyn"] = -np.inf

            if not np.isfinite(log10_mej_wind):
                converted_parameters["log10_mej_wind"] = -np.inf

            if not np.isfinite(log10_E0_erg):
                converted_parameters["log10_E0"] = -np.inf

            if compactness_1 == 0.5 or compactness_2 == 0.5:
                converted_parameters["log10_mej_dyn"] = -np.inf
                converted_parameters["log10_mej_wind"] = -np.inf
                converted_parameters["log10_E0"] = -np.inf

        added_keys = added_keys + ["log10_mej_dyn", "log10_mej_wind", "log10_E0"]

        np.seterr(**old)

        return converted_parameters, added_keys


class MultimessengerConversion(object):
    def __init__(self, eos_data_path, Neos, binary_type):
        self.eos_data_path = eos_data_path
        self.Neos = Neos
        self.eos_interp_dict = {}
        self.binary_type = binary_type

        for EOSID in range(1, self.Neos + 1):
            radius, mass, Lambda = np.loadtxt(
                "{0}/{1}.dat".format(self.eos_data_path, EOSID),
                unpack=True,
                usecols=[0, 1, 2],
            )

            interp_mass_lambda = scipy.interpolate.interp1d(mass, Lambda)
            interp_mass_radius = scipy.interpolate.interp1d(mass, radius)

            self.eos_interp_dict[EOSID] = [interp_mass_lambda, interp_mass_radius]

        if self.binary_type == "BNS":
            ejectaFitting = BNSEjectaFitting()

        elif self.binary_type == "NSBH":
            ejectaFitting = NSBHEjectaFitting()

        else:
            print("Unknown binary type, exiting")
            sys.exit()

        self.ejecta_parameter_conversion = ejectaFitting.ejecta_parameter_conversion

    def convert_to_multimessenger_parameters(self, parameters):
        converted_parameters = parameters.copy()
        original_keys = list(converted_parameters.keys())
        converted_parameters, added_keys = convert_to_lal_binary_black_hole_parameters(
            converted_parameters
        )

        converted_parameters, added_keys = source_frame_masses(
            converted_parameters, added_keys
        )
        mass_1_source = converted_parameters["mass_1_source"]
        mass_2_source = converted_parameters["mass_2_source"]

        if "EOS" in converted_parameters:
            if isinstance(
                converted_parameters["EOS"],
                (list, tuple, pd.core.series.Series, np.ndarray),
            ):
                TOV_radius_list = []
                TOV_mass_list = []
                lambda_1_list = []
                lambda_2_list = []
                radius_1_list = []
                radius_2_list = []
                R_14_list = []
                R_16_list = []
                EOSID = np.array(converted_parameters["EOS"]).astype(int) + 1

                for i in range(0, len(EOSID)):

                    interp_mass_lambda, interp_mass_radius = self.eos_interp_dict[
                        EOSID[i]
                    ]

                    (
                        TOV_mass,
                        TOV_radius,
                        lambda_1,
                        lambda_2,
                        radius_1,
                        radius_2,
                    ) = EOS2Parameters(
                        interp_mass_radius,
                        interp_mass_lambda,
                        mass_1_source[i],
                        mass_2_source[i],
                    )

                    TOV_radius_list.append(TOV_radius)
                    TOV_mass_list.append(TOV_mass)
                    lambda_1_list.append(lambda_1[0])
                    lambda_2_list.append(lambda_2[0])
                    radius_1_list.append(radius_1[0])
                    radius_2_list.append(radius_2[0])
                    R_14_list.append(interp_mass_radius(1.4))
                    R_16_list.append(interp_mass_radius(1.6))

                converted_parameters["TOV_mass"] = np.array(TOV_mass_list)
                converted_parameters["TOV_radius"] = np.array(TOV_radius_list)

                converted_parameters["radius_1"] = np.array(radius_1_list)
                converted_parameters["radius_2"] = np.array(radius_2_list)

                converted_parameters["lambda_1"] = np.array(lambda_1_list)
                converted_parameters["lambda_2"] = np.array(lambda_2_list)

                converted_parameters["R_14"] = np.array(R_14_list)
                converted_parameters["R_16"] = np.array(R_16_list)

            else:
                EOSID = int(converted_parameters["EOS"]) + 1
                interp_mass_lambda, interp_mass_radius = self.eos_interp_dict[EOSID]

                (
                    TOV_mass,
                    TOV_radius,
                    lambda_1,
                    lambda_2,
                    radius_1,
                    radius_2,
                ) = EOS2Parameters(
                    interp_mass_radius, interp_mass_lambda, mass_1_source, mass_2_source
                )

                converted_parameters["TOV_radius"] = TOV_radius
                converted_parameters["TOV_mass"] = TOV_mass

                converted_parameters["radius_1"] = radius_1
                converted_parameters["radius_2"] = radius_2

                converted_parameters["lambda_1"] = lambda_1
                converted_parameters["lambda_2"] = lambda_2

                converted_parameters["R_14"] = interp_mass_radius(1.4)
                converted_parameters["R_16"] = interp_mass_radius(1.6)

            added_keys = added_keys + [
                "lambda_1",
                "lambda_2",
                "TOV_mass",
                "TOV_radius",
                "radius_1",
                "radius_2",
                "R_14",
                "R_16",
            ]

            converted_parameters, added_keys = self.ejecta_parameter_conversion(
                converted_parameters, added_keys
            )

            theta_jn = converted_parameters["theta_jn"]
            converted_parameters["KNtheta"] = (
                180 / np.pi * np.minimum(theta_jn, np.pi - theta_jn)
            )
            converted_parameters["inclination_EM"] = (
                converted_parameters["KNtheta"] * np.pi / 180.0
            )

            added_keys = added_keys + ["KNtheta", "inclination_EM"]

        added_keys = [
            key for key in converted_parameters.keys() if key not in original_keys
        ]

        return converted_parameters, added_keys

    def generate_all_parameters(self, sample, likelihood=None, priors=None, npool=1):
        waveform_defaults = {
            "reference_frequency": 50.0,
            "waveform_approximant": "TaylorF2",
            "minimum_frequency": 20.0,
        }
        output_sample = _generate_all_cbc_parameters(
            sample,
            defaults=waveform_defaults,
            base_conversion=self.convert_to_multimessenger_parameters,
            likelihood=likelihood,
            priors=priors,
            npool=npool,
        )
        output_sample = generate_tidal_parameters(output_sample)
        return output_sample

    def priors_conversion_function(self, sample):
        out_sample = sample.copy()
        out_sample, _ = self.convert_to_multimessenger_parameters(out_sample)
        out_sample = generate_mass_parameters(out_sample)
        out_sample = generate_tidal_parameters(out_sample)
        return out_sample


class MultimessengerConversionWithLambdas(object):
    def __init__(self, binary_type):
        self.binary_type = binary_type

        if self.binary_type == "BNS":
            ejectaFitting = BNSEjectaFitting()

        elif self.binary_type == "NSBH":
            ejectaFitting = NSBHEjectaFitting()

        else:
            print("Unknown binary type, exiting")
            sys.exit()

        self.ejecta_parameter_conversion = ejectaFitting.ejecta_parameter_conversion

    def convert_to_multimessenger_parameters(self, parameters):
        converted_parameters = parameters.copy()
        original_keys = list(converted_parameters.keys())
        converted_parameters, added_keys = convert_to_lal_binary_black_hole_parameters(
            converted_parameters
        )

        converted_parameters, added_keys = source_frame_masses(
            converted_parameters, added_keys
        )
        mass_1_source = converted_parameters["mass_1_source"]
        mass_2_source = converted_parameters["mass_2_source"]

        lambda_1 = converted_parameters["lambda_1"]
        lambda_2 = converted_parameters["lambda_2"]

        log_lambda_1 = np.log(lambda_1)
        log_lambda_2 = np.log(lambda_2)

        compactness_1 = (
            0.371 - 0.0391 * log_lambda_1 + 0.001056 * log_lambda_1 * log_lambda_1
        )
        compactness_2 = (
            0.371 - 0.0391 * log_lambda_2 + 0.001056 * log_lambda_2 * log_lambda_2
        )

        converted_parameters["radius_1"] = (
            mass_1_source / compactness_1 * lal.MRSUN_SI / 1e3
        )
        converted_parameters["radius_2"] = (
            mass_2_source / compactness_2 * lal.MRSUN_SI / 1e3
        )

        chirp_mass_source = component_masses_to_chirp_mass(mass_1_source, mass_2_source)
        lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(
            lambda_1, lambda_2, mass_1_source, mass_2_source
        )

        converted_parameters["R_16"] = (
            chirp_mass_source
            * np.power(lambda_tilde / 0.0042, 1.0 / 6.0)
            * lal.MRSUN_SI
            / 1e3
        )

        added_keys = added_keys + ["radius_1", "radius_2", "R_16"]

        converted_parameters, added_keys = self.ejecta_parameter_conversion(
            converted_parameters, added_keys
        )

        theta_jn = converted_parameters["theta_jn"]
        converted_parameters["KNtheta"] = (
            180 / np.pi * np.minimum(theta_jn, np.pi - theta_jn)
        )
        converted_parameters["inclination_EM"] = (
            converted_parameters["KNtheta"] * np.pi / 180.0
        )

        added_keys = added_keys + ["KNtheta", "inclination_EM"]

        added_keys = [
            key for key in converted_parameters.keys() if key not in original_keys
        ]

        return converted_parameters, added_keys

    def generate_all_parameters(self, sample, likelihood=None, priors=None, npool=1):
        waveform_defaults = {
            "reference_frequency": 50.0,
            "waveform_approximant": "TaylorF2",
            "minimum_frequency": 20.0,
        }
        output_sample = _generate_all_cbc_parameters(
            sample,
            defaults=waveform_defaults,
            base_conversion=self.convert_to_multimessenger_parameters,
            likelihood=likelihood,
            priors=priors,
            npool=npool,
        )
        output_sample = generate_tidal_parameters(output_sample)
        return output_sample

    def priors_conversion_function(self, sample):
        out_sample = sample.copy()
        out_sample, _ = self.convert_to_multimessenger_parameters(out_sample)
        out_sample = generate_mass_parameters(out_sample)
        out_sample = generate_tidal_parameters(out_sample)
        return out_sample
