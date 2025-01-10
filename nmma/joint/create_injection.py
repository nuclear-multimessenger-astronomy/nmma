import argparse
import json
import pandas as pd

import numpy as np

from  lalsimulation import SimInspiralTransformPrecessingWvf2PE as lalsim_conversion
from gwpy.table import Table

try:
    import ligo.lw  # noqa F401
except ImportError:
    raise ImportError("You do not have ligo.lw installed: $ pip install python-ligo-lw")

from bilby_pipe.create_injections import InjectionCreator

from .conversion import  MultimessengerConversion


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
        "simulation_id": [],
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
        coa_phase = row.get("coa_phase", 0)
        if aligned_spin:
            spin_args = [0.0, 0.0, row["spin1z"], 0.0, 0.0, row["spin2z"]]  
        else:
            spin_args = [row["spin1x"], row["spin1y"], row["spin1z"], row["spin2x"], row["spin2y"], row["spin2z"]]
        precession_args = [row["inclination"], *spin_args,
            row["mass1"], row["mass2"], reference_frequency, coa_phase]
        conversion_args = [float(arg) for arg in precession_args]
        conversion_keys = ["theta_jn","phi_jl", "tilt_1" ,"tilt_2", "phi_12", "a_1","a_2"]
        
        for key, val in zip(conversion_keys, lalsim_conversion(*conversion_args) ):
            injection_values[key].append(val)
 
        injection_values["simulation_id"].append(int(row["simulation_id"]))
        injection_values["luminosity_distance"].append(float(row["distance"]))
        injection_values["psi"].append(float(row.get("polarization", 0)))
        injection_values["ra"].append(float(row["longitude"]))
        injection_values["dec"].append(float(row["latitude"]))

        injection_values["mass_1"].append(max(float(row["mass1"]), float(row["mass2"])))
        injection_values["mass_2"].append(min(float(row["mass1"]), float(row["mass2"])))
        injection_values["phase"].append(float(coa_phase))
        geocent_time = float(row.get("geocent_end_time", trigger_time))
        geocent_time_ns = float(row.get("geocent_end_time_ns", 0)) * 1e-9
        injection_values["geocent_time"].append(geocent_time + geocent_time_ns)

    injection_values = pd.DataFrame.from_dict(injection_values)
    return injection_values


def get_parser():

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
    parser.add_argument("-f", "--filename", type=str, default="injection")
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        default="dat",
        choices=["json", "dat"],
        help="Prior file format",
    )
    parser.add_argument(
        "-n",
        "--n-injection",
        type=int,
        default=None,
        help="The number of injections to generate: not required if --gps-file or injection file is also given",
        required=False,
    )
    parser.add_argument(
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
    parser.add_argument(
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
    parser.add_argument(
        "--deltaT",
        type=float,
        default=0.2,
        help=(
            "The symmetric width (in s) around the trigger time to"
            " search over the coalesence time. Ignored if a geocent_time prior"
            " exists in the prior_file"
        ),
    )
    parser.add_argument(
        "--post-trigger-duration",
        type=float,
        default=2,
        help=(
            "The post trigger duration (default=2s), used only in conjunction "
            "with --gps-file"
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4,
        help=(
            "The segment duration (default=4s), used only in conjunction with "
            "--gps-file"
        ),
    )
    parser.add_argument(
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
    parser.add_argument(
        "-r",
        "--repeated-simulations",
        default=0,
        type=int,
        help="Number of repeated simulations, fixing other parameters (default: 0)",
    )
    return parser


def main(args=None):

    if args is None:
        parser = get_parser()
        args = parser.parse_args()

    seed = args.generation_seed
    np.random.seed(seed)

    # check injection file format
    if args.injection_file:
        assert (
            args.injection_file.endswith(".json")
            or args.injection_file.endswith(".xml")
            or args.injection_file.endswith(".xml.gz")
            or args.injection_file.endswith(".dat")
        ), "Unknown injection file format"

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
        gpstimes=args.gps_file,
        duration=args.duration,
        post_trigger_duration=args.post_trigger_duration,
        generation_seed=args.generation_seed,
    )
    dataframe_from_prior = injection_creator.get_injection_dataframe()
    if args.repeated_simulations > 0:
        repeats = []
        timeshifts = []
        injection_creator.n_injection = args.repeated_simulations
        for index, row in dataframe_from_prior.iterrows():
            timeshift_frame = injection_creator.get_injection_dataframe()
            for ii in range(args.repeated_simulations):
                timeshifts.append(timeshift_frame["timeshift"][ii])
                repeats.append(row)
        dataframe_from_prior = pd.concat(repeats, axis=1).transpose().reset_index()
        dataframe_from_prior.drop(
            labels=["index", "timeshift"], axis="columns", inplace=True
        )
        dataframe_from_prior["timeshift"] = timeshifts

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

    # Move dataframe index column to simulation_id if column does not exist
    if "simulation_id" not in dataframe.columns:
        dataframe = dataframe.reset_index().rename({"index": "simulation_id"}, axis=1)

    if args.original_parameters:
        # dump the whole thing back into a json injection file
        injection_creator.write_injection_dataframe(
            dataframe, args.filename, args.extension
        )
        return
    
    # else:
    args.eos_to_ram = False
    messengers = ['gw']
    if args.eject:
        messengers.append('em')
    param_conversion = MultimessengerConversion(args, messengers, ana_modifiers=['eos'])

    # convert to all necessary parameters
    dataframe, _ = param_conversion.convert_to_multimessenger_parameters(dataframe)

   
    if args.eject:

        log10_mej_dyn = dataframe["log10_mej_dyn"]
        log10_mej_wind = dataframe["log10_mej_wind"]
        index_condition = np.isfinite(log10_mej_dyn) * np.isfinite(log10_mej_wind)
        if "thetaWing" in dataframe and "thetaCore" in dataframe:
            print("Checking GRB resolution")
            grb_res = dataframe["thetaWing"] / dataframe["thetaCore"]
            index_condition *= (grb_res < args.grb_resolution)
        
        index_taken = np.where(index_condition)[0]

        dataframe = dataframe.take(index_taken)

        print(f"{len(index_taken)} injections left")

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
