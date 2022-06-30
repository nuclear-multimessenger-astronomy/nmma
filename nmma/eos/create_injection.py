import argparse
import json
import sys
import pandas as pd

import numpy as np
import scipy.interpolate

import lalsimulation as lalsim
from gwpy.table import Table

try:
    import ligo.lw  # noqa F401
except ImportError:
    raise ImportError("You do not have ligo.lw install: $ pip install python-ligo-lw")

from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters

from bilby_pipe.create_injections import InjectionCreator

from ..joint.conversion import (
    source_frame_masses,
    EOS2Parameters,
    NSBHEjectaFitting,
    BNSEjectaFitting,
)


def file_to_dataframe(
    injection_file, reference_frequency, aligned_spin=False, trigger_time=0.0
):
    if injection_file.endswith(".xml") or injection_file.endswith(".xml.gz"):
        table = Table.read(injection_file, format="ligolw", tablename="sim_inspiral")
    elif injection_file.endswith(".dat"):
        table = Table.read(injection_file, format="csv", delimiter="\t")
    else:
        raise ValueError("Only understand xml and dat")

    injection_values = {
        "mass_1": [],
        "mass_2": [],
        "luminosity_distance": [],
        "psi": [],
        "phase": [],
        "geocent_time": [],
        "ra": [],
        "dec": [],
        "theta_jn": [],
        "a_1": [],
        "a_2": [],
        "tilt_1": [],
        "tilt_2": [],
        "phi_12": [],
        "phi_jl": [],
    }
    for row in table:
        injection_values["mass_1"].append(max(float(row["mass1"]), float(row["mass2"])))
        injection_values["mass_2"].append(min(float(row["mass1"]), float(row["mass2"])))
        injection_values["luminosity_distance"].append(float(row["distance"]))

        if "polarization" in row:
            injection_values["psi"].append(float(row["polarization"]))
        else:
            injection_values["psi"].append(0.0)

        if "coa_phase" in row:
            coa_phase = float(row["coa_phase"])
            injection_values["phase"].append(float(row["coa_phase"]))
        else:
            coa_phase = 0.0
            injection_values["phase"].append(0.0)

        if "geocent_end_time" in row:
            injection_values["geocent_time"].append(float(row["geocent_end_time"]))
        else:
            injection_values["geocent_time"].append(trigger_time)

        injection_values["ra"].append(float(row["longitude"]))
        injection_values["dec"].append(float(row["latitude"]))

        if aligned_spin:

            args_list = [
                float(arg)
                for arg in [
                    row["inclination"],
                    0.0,
                    0.0,
                    row["spin1z"],
                    0.0,
                    0.0,
                    row["spin2z"],
                    row["mass1"],
                    row["mass2"],
                    reference_frequency,
                    coa_phase,
                ]
            ]

        else:

            args_list = [
                float(arg)
                for arg in [
                    row["inclination"],
                    row["spin1x"],
                    row["spin1y"],
                    row["spin1z"],
                    row["spin2x"],
                    row["spin2y"],
                    row["spin2z"],
                    row["mass1"],
                    row["mass2"],
                    reference_frequency,
                    row["coa_phase"],
                ]
            ]
        (
            theta_jn,
            phi_jl,
            tilt_1,
            tilt_2,
            phi_12,
            a_1,
            a_2,
        ) = lalsim.SimInspiralTransformPrecessingWvf2PE(*args_list)
        injection_values["theta_jn"].append(theta_jn)
        injection_values["phi_jl"].append(phi_jl)
        injection_values["tilt_1"].append(tilt_1)
        injection_values["tilt_2"].append(tilt_2)
        injection_values["phi_12"].append(phi_12)
        injection_values["a_1"].append(a_1)
        injection_values["a_2"].append(a_2)

    injection_values = pd.DataFrame.from_dict(injection_values)
    return injection_values


def main():

    parser = argparse.ArgumentParser(
        description="Process a bilby injection file for nmma consumption"
    )
    parser.add_argument(
        "--prior-file",
        type=str,
        required=True,
        help="The prior file from which to generate injections",
    )
    parser.add_argument(
        "--injection-file",
        type=str,
        required=False,
        help="The xml injection file or bilby injection json file to be used (optional)",
    )
    parser.add_argument(
        "--reference-frequency",
        type=str,
        required=False,
        default=20,
        help="The reference frequency in the provided injection file (default: 20)",
    )
    parser.add_argument(
        "--aligned-spin",
        action="store_true",
        help="Whether the spin is aligned in the provide injection file",
    )
    parser.add("-f", "--filename", type=str, default="injection")
    parser.add_arg(
        "-e",
        "--extension",
        type=str,
        default="dat",
        choices=["json", "dat"],
        help="Prior file format",
    )
    parser.add_arg(
        "-n",
        "--n-injection",
        type=int,
        default=None,
        help="The number of injections to generate: not required if --gps-file or injection file is also given",
        required=False,
    )
    parser.add_arg(
        "-t",
        "--trigger-time",
        type=int,
        default=0,
        help=(
            "The trigger time to use for setting a geocent_time prior "
            "(default=0). Ignored if a geocent_time prior exists in the "
            "prior_file or --gps-file is given."
        ),
    )
    parser.add_arg(
        "-g",
        "--gps-file",
        type=str,
        default=None,
        help=(
            "A list of gps start times to use for setting a geocent_time prior"
            ". Note, the trigger time is obtained from "
            " start_time + duration - post_trigger_duration."
        ),
    )
    parser.add(
        "--deltaT",
        type=float,
        default=0.2,
        help=(
            "The symmetric width (in s) around the trigger time to"
            " search over the coalesence time. Ignored if a geocent_time prior"
            " exists in the prior_file"
        ),
    )
    parser.add_arg(
        "--post-trigger-duration",
        type=float,
        default=2,
        help=(
            "The post trigger duration (default=2s), used only in conjunction "
            "with --gps-file"
        ),
    )
    parser.add_arg(
        "--duration",
        type=float,
        default=4,
        help=(
            "The segment duration (default=4s), used only in conjunction with "
            "--gps-file"
        ),
    )
    parser.add(
        "-s",
        "--generation-seed",
        default=42,
        type=int,
        help="Random seed used during data generation (default: 42)",
    )
    parser.add_argument(
        "--grb-resolution",
        type=float,
        default=5,
        help="Upper bound on the ratio between thetaWing and thetaCore (default: 5)",
    )
    parser.add_argument(
        "--eos-file",
        type=str,
        required=False,
        help="EOS file in (radius [km], mass [solar mass], lambda)",
    )
    parser.add_argument(
        "--binary-type", type=str, required=False, help="Either BNS or NSBH"
    )
    parser.add_argument(
        "--eject",
        action="store_true",
        help="Whether to create injection files with eject properties",
    )
    parser.add_argument("-d", "--detections-file", type=str)
    parser.add_argument("-i", "--indices-file", type=str)
    parser.add_argument(
        "--original-parameters",
        action="store_true",
        help="Whether to only sample prior parameters in injection file",
    )
    args = parser.parse_args()

    if not args.original_parameters:
        # check the binary type
        assert args.binary_type in ["BNS", "NSBH"], "Unknown binary type"

    # check injection file format
    if args.injection_file:
        assert (
            args.injection_file.endswith(".json")
            or args.injection_file.endswith(".xml")
            or args.injection_file.endswith(".xml.gz")
            or args.injection_file.endswith(".dat")
        ), "Unknown injection file format"

    if not args.original_parameters:
        # load the EOS
        radii, masses, lambdas = np.loadtxt(
            args.eos_file, usecols=[0, 1, 2], unpack=True
        )
        interp_mass_radius = scipy.interpolate.interp1d(masses, radii)
        interp_mass_lambda = scipy.interpolate.interp1d(masses, lambdas)

    # load the injection json file
    if args.injection_file:
        if args.injection_file.endswith(".json"):
            with open(args.injection_file, "rb") as f:
                injection_data = json.load(f)
                datadict = injection_data["injections"]["content"]
                dataframe_from_inj = pd.DataFrame.from_dict(datadict)
        elif (
            args.injection_file.endswith(".xml")
            or args.injection_file.endswith(".xml.gz")
            or args.injection_file.endswith(".dat")
        ):
            dataframe_from_inj = file_to_dataframe(
                args.injection_file,
                args.reference_frequency,
                aligned_spin=args.aligned_spin,
                trigger_time=args.trigger_time,
            )
    else:
        dataframe_from_inj = pd.DataFrame()
        print(
            "No injection files provided, "
            "will generate injection based on the prior file provided only"
        )

    if len(dataframe_from_inj) > 0:
        args.n_injection = len(dataframe_from_inj)

    # create the injection dataframe from the prior_file
    injection_creator = InjectionCreator(
        prior_file=args.prior_file,
        prior_dict=None,
        n_injection=args.n_injection,
        default_prior="PriorDict",
        trigger_time=args.trigger_time,
        deltaT=args.deltaT,
        gps_file=args.gps_file,
        duration=args.duration,
        post_trigger_duration=args.post_trigger_duration,
        generation_seed=args.generation_seed,
    )
    dataframe_from_prior = injection_creator.get_injection_dataframe()

    inj_columns = set(dataframe_from_inj.columns.tolist())
    prior_columns = set(dataframe_from_prior.columns.tolist())

    columns_to_remove = list(inj_columns.intersection(prior_columns))
    dataframe_from_prior.drop(columns=columns_to_remove, inplace=True)

    # combine the dataframes
    dataframe = pd.DataFrame.merge(
        dataframe_from_inj,
        dataframe_from_prior,
        how="outer",
        left_index=True,
        right_index=True,
    )

    if args.detections_file is not None:
        dets = np.loadtxt(args.detections_file)
        dataframe = dataframe.iloc[dets]
        dataframe = dataframe.reset_index(drop=True)

    if args.original_parameters:
        # dump the whole thing back into a json injection file
        injection_creator.write_injection_dataframe(
            dataframe, args.filename, args.extension
        )
        sys.exit(0)

    # convert to all necessary parameters
    dataframe, _ = convert_to_lal_binary_black_hole_parameters(dataframe)
    Ninj = len(dataframe)

    # estimate the lambdas and radii with an eos given
    dataframe, _ = source_frame_masses(dataframe, [])
    TOV_mass = []
    TOV_radius = []
    lambda_1 = []
    lambda_2 = []
    radius_1 = []
    radius_2 = []

    for injIdx in range(0, Ninj):
        mMax, rMax, lam1, lam2, r1, r2 = EOS2Parameters(
            interp_mass_radius,
            interp_mass_lambda,
            dataframe["mass_1_source"][injIdx],
            dataframe["mass_2_source"][injIdx],
        )

        TOV_mass.append(mMax)
        TOV_radius.append(rMax)
        lambda_1.append(lam1.item())
        lambda_2.append(lam2.item())
        radius_1.append(r1.item())
        radius_2.append(r2.item())

    dataframe["TOV_mass"] = np.array(TOV_mass)
    dataframe["TOV_radius"] = np.array(TOV_radius)
    dataframe["lambda_1"] = np.array(lambda_1)
    dataframe["lambda_2"] = np.array(lambda_2)
    dataframe["radius_1"] = np.array(radius_1)
    dataframe["radius_2"] = np.array(radius_2)
    dataframe["R_16"] = np.ones(len(dataframe)) * interp_mass_radius(1.6)
    dataframe["R_14"] = np.ones(len(dataframe)) * interp_mass_radius(1.4)

    if args.eject:
        if args.binary_type == "BNS":
            ejectaFitting = BNSEjectaFitting()

        elif args.binary_type == "NSBH":
            ejectaFitting = NSBHEjectaFitting()

        else:
            print("Unknown binary type, exiting")
            sys.exit()

        dataframe, _ = ejectaFitting.ejecta_parameter_conversion(dataframe, [])
        theta_jn = dataframe["theta_jn"]
        dataframe["inclination_EM"] = np.minimum(theta_jn, np.pi - theta_jn)
        dataframe["KNtheta"] = 180.0 / np.pi * dataframe["inclination_EM"]

        log10_mej_dyn = dataframe["log10_mej_dyn"]
        log10_mej_wind = dataframe["log10_mej_wind"]

        if "thetaWing" in dataframe and "thetaCore" in dataframe:
            print("Checking GRB resolution")
            grb_res = dataframe["thetaWing"] / dataframe["thetaCore"]
            index_taken = np.where(
                np.isfinite(log10_mej_dyn)
                * np.isfinite(log10_mej_wind)
                * (grb_res < args.grb_resolution)
            )[0]
        else:
            index_taken = np.where(
                np.isfinite(log10_mej_dyn) * np.isfinite(log10_mej_wind)
            )[0]

        dataframe = dataframe.take(index_taken)

        print("{0} injections left".format(len(index_taken)))

    if args.indices_file:
        if args.detections_file is not None:
            idxs = dets[index_taken]
        else:
            idxs = index_taken
        np.savetxt(args.indices_file, idxs, fmt="%d")

    # dump the whole thing back into a json injection file
    injection_creator.write_injection_dataframe(
        dataframe, args.filename, args.extension
    )
