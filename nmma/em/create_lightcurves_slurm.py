import os
import argparse
import json

import numpy as np

import bilby


def main():

    parser = argparse.ArgumentParser(
        description="Condor lightcurve files from nmma injection file"
    )
    parser.add_argument(
        "--injection",
        type=str,
        required=True,
        help="The bilby injection json file to be used",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Path to the output directory"
    )
    parser.add_argument(
        "--analysis-file",
        type=str,
        required=True,
        help="The analysis bash script to be replicated",
    )
    parser.add_argument(
        "--lightcurves-per-job",
        type=int,
        required=False,
        help="Number of light curves per job",
        default=100,
    )
    args = parser.parse_args()

    with open(args.injection, "r") as f:
        injection_dict = json.load(f, object_hook=bilby.core.utils.decode_bilby_json)

    logdir = os.path.join(args.outdir, "logs")
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    number_jobs = int(
        np.ceil(len(injection_dict["injections"]) / args.lightcurves_per_job)
    )

    for ii in range(number_jobs):
        with open(args.analysis_file, "r") as file:
            analysis = file.read()

        injection_min, injection_max = ii * number_jobs, (ii + 1) * number_jobs

        analysis = analysis.replace(
            "INJRANGE", "%d,%d" % (injection_min, injection_max)
        )
        analysis_file = os.path.join(args.outdir, "inference_%d.sh" % ii)

        fid = open(analysis_file, "w")
        fid.write(analysis)
        fid.close()

    fid.close()


if __name__ == "__main__":
    main()
