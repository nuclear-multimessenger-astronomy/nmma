import argparse
import json
import pandas as pd

from tqdm import tqdm

import numpy as np

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
    elif injection_file.endswith(".ecsv"):
        from astropy.table import Table as astro_Table

        table = astro_Table.read(injection_file)
    else:
        raise ValueError("Only understand xml, dat and ecsv")

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
        injection_values["simulation_id"].append(int(row["simulation_id"]))
        injection_values["mass_1"].append(max(float(row["mass1"]), float(row["mass2"])))
        injection_values["mass_2"].append(min(float(row["mass1"]), float(row["mass2"])))
        injection_values["luminosity_distance"].append(float(row["distance"]))

        if "polarization" in row.colnames:
            injection_values["psi"].append(float(row["polarization"]))
        else:
            injection_values["psi"].append(0.0)

        if "coa_phase" in row.colnames:
            coa_phase = float(row["coa_phase"])
            injection_values["phase"].append(float(row["coa_phase"]))
        else:
            coa_phase = 0.0
            injection_values["phase"].append(0.0)

        if "geocent_end_time" in row.colnames:
            if "geocent_end_time_ns" in row.colnames:
                injection_values["geocent_time"].append(
                    float(row["geocent_end_time"])
                    + float(row["geocent_end_time_ns"]) * (10**-9)
                )
            else:
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
    parser.add_argument(
        "--snr-cutoff",
        default=0.0,
        type=float,
        help="The value of SNR cutoff for the injection generation (default: 0)",
    )
    parser.add_argument(
        "--ifos",
        default="H1,L1,V1",
        type=str,
        help="Comma seperated list of ifos to be used for SNR calculation",
    )
    parser.add_argument(
        "--ifos-psd",
        default="",
        type=str,
        help="Comma seperated list of psds files to be used (default: using bilby default)",
    )
    parser.add_argument("-d", "--detections-file", type=str)
    parser.add_argument("-i", "--indices-file", type=str)
    parser.add_argument(
        "--original-parameters",
        action="store_true",
        help="Whether to only sample prior parameters in injection file",
    )
    parser.add(
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

    if not args.original_parameters:
        # check the binary type
        assert args.binary_type in ["BNS", "NSBH"], "Unknown binary type"

    seed = args.generation_seed
    np.random.seed(seed)

    # check injection file format
    if args.injection_file:
        assert (
            args.injection_file.endswith(".json")
            or args.injection_file.endswith(".xml")
            or args.injection_file.endswith(".xml.gz")
            or args.injection_file.endswith(".dat")
            or args.injection_file.endswith(".ecsv")
        ), "Unknown injection file format"

    if not args.original_parameters:
        # load the EOS
        radius_val, mass_val, Lambda_val = np.loadtxt(
            args.eos_file, usecols=[0, 1, 2], unpack=True
        )

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
            or args.injection_file.endswith(".ecsv")
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

    # remove the gps time from the prior dataframe
    # in case that is in the injection table
    inj_columns = set(dataframe_from_inj.columns.tolist())
    if "geocent_time" in inj_columns:
        dataframe_from_prior.drop(
            columns=[
                "geocent_time",
            ],
            inplace=True,
        )
    elif "geocent_time_x" in inj_columns:
        dataframe_from_prior.drop(
            columns=[
                "geocent_time_x",
            ],
            inplace=True,
        )

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
        mMax, rMax, lam1, lam2, r1, r2, R_14, R_16 = EOS2Parameters(
            mass_val,
            radius_val,
            Lambda_val,
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
    dataframe["R_16"] = np.ones(len(dataframe)) * R_16
    dataframe["R_14"] = np.ones(len(dataframe)) * R_14

    if args.eject:
        if args.binary_type == "BNS":
            ejectaFitting = BNSEjectaFitting()
            dataframe = dataframe[dataframe["lambda_1"] != 0.0]
            dataframe = dataframe[dataframe["lambda_2"] != 0.0]
            dataframe.reset_index(drop=True, inplace=True)

        elif args.binary_type == "NSBH":
            ejectaFitting = NSBHEjectaFitting()
            dataframe = dataframe[dataframe["lambda_1"] == 0.0]
            dataframe = dataframe[dataframe["lambda_2"] != 0.0]
            dataframe.reset_index(drop=True, inplace=True)

        else:
            raise ValueError("Unknown binary type")

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

        print("{0} injections left after ejecta filtering".format(len(index_taken)))

    if args.indices_file:
        if args.detections_file is not None:
            idxs = dets[index_taken]
        else:
            idxs = index_taken
        np.savetxt(args.indices_file, idxs, fmt="%d")

    if args.snr_cutoff:
        print(f"Removing injection with SNR less than {args.snr_cutoff}")
        import bilby

        bilby.core.utils.setup_logger(log_level="ERROR")
        IFOs = bilby.gw.detector.InterferometerList(args.ifos.split(","))

        if len(args.ifos_psd) > 0:
            from bilby.gw.detector import PowerSpectralDensity

            psds = args.ifos_psd.split(",")
            for ifo, psd in zip(IFOs, psds):
                ifo.power_spectral_density = PowerSpectralDensity(psd_file=psd)

        wf_model = bilby.gw.waveform_generator.WaveformGenerator(
            duration=256,
            sampling_frequency=2048,
            frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
            waveform_arguments={
                "waveform_approximant": "IMRPhenomPv2_NRTidalv2",
                "reference_frequency": 20,
                "minimum_frequency": 20,  # Ensure consistency with PSD
            },
        )
        # Loop over injections to compute SNRs
        snr_list = []
        for _, inj in tqdm(dataframe.iterrows()):
            # create data just to please the bilby sanity check
            IFOs.set_strain_data_from_power_spectral_densities(
                sampling_frequency=2048,
                duration=256,
                start_time=inj["geocent_time"] - (256 - 2.0),
            )
            # Inject the waveform into the detectors
            snr = 0.0
            for ifo in IFOs:
                ifo.inject_signal(
                    waveform_generator=wf_model,
                    parameters=inj.to_dict(),
                    raise_error=False,
                )
                snr += ifo.meta_data["optimal_SNR"]
            snr_list.append(np.sqrt(snr))
        index_taken = np.where(np.array(snr_list) >= args.snr_cutoff)[0]
        dataframe["optimal_network_snr"] = np.array(snr_list)
        dataframe = dataframe.take(index_taken)

        print("{0} injections left after snr filtering".format(len(index_taken)))

    # re-index the simulation_id
    if not args.injection_file:
        print("Re-indexing the simulation_id")
        dataframe["simulation_id"] = np.arange(len(dataframe))
    # dump the whole thing back into a json injection file
    injection_creator.write_injection_dataframe(
        dataframe, args.filename, args.extension
    )
