import os
import argparse
import json

import numpy as np

import bilby

from subprocess import check_output


def main():

    parser = argparse.ArgumentParser(
        description="Condor lightcurve files from nmma injection file"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the kilonova model to be used"
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
    parser.add_argument("--label", type=str, required=True, help="Label for the run")
    parser.add_argument(
        "--condor-dag-file",
        type=str,
        required=True,
        help="The condor dag file to be created",
    )
    parser.add_argument(
        "--condor-sub-file",
        type=str,
        required=True,
        help="The condor sub file to be created",
    )
    parser.add_argument(
        "--bash-file", type=str, required=True, help="The bash file to be created"
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

    lc_generation = (
        check_output(["which", "light_curve_generation"]).decode().replace("\n", "")
    )

    number_jobs = int(
        np.ceil(len(injection_dict["injections"]) / args.lightcurves_per_job)
    )

    job_number = 0
    fid = open(args.condor_dag_file, "w")
    fid1 = open(args.bash_file, "w")
    for ii in range(number_jobs):
        # with open(args.analysis_file, 'r') as file:
        #    analysis = file.read()

        injection_min, injection_max = ii * number_jobs, (ii + 1) * number_jobs

        fid.write("JOB %d %s\n" % (job_number, args.condor_sub_file))
        fid.write("RETRY %d 3\n" % (job_number))
        fid.write(
            'VARS %d jobNumber="%d" INJRANGE="%d,%d"\n'
            % (job_number, job_number, injection_min, injection_max)
        )
        fid.write("\n\n")
        job_number = job_number + 1

        fid1.write(
            "%s --model %s --svd-path /home/%s/gwemlightcurves/output/svdmodels --outdir %s --label %s --injection-detection-limit 24.1,25.0,25.0,25.3,24.5,23.0,23.2,22.6,22.6 --injection %s --injection-range %d,%d --filters u,g,r,i,z,y,J,H,K --plot\n"
            % (
                lc_generation,
                args.model,
                os.environ["USER"],
                args.outdir,
                args.label,
                args.injection,
                injection_min,
                injection_max,
            )
        )

    fid.close()

    fid = open(args.condor_sub_file, "w")
    fid.write("executable = %s\n" % lc_generation)
    fid.write(f"output = {logdir}/out.$(jobNumber)\n")
    fid.write(f"error = {logdir}/err.$(jobNumber)\n")
    fid.write(
        "arguments = --model %s --svd-path /home/%s/gwemlightcurves/output/svdmodels --outdir %s --label %s --injection-detection-limit 24.1,25.0,25.0,25.3,24.5,23.0,23.2,22.6,22.6 --injection %s --injection-range $(INJRANGE) --filters u,g,r,i,z,y,J,H,K --plot\n"
        % (args.model, os.environ["USER"], args.outdir, args.label, args.injection)
    )
    fid.write('requirements = OpSys == "LINUX"\n')
    fid.write("request_memory = 8192\n")
    fid.write("request_disk = 500 MB\n")
    fid.write("request_cpus = 1\n")
    fid.write("accounting_group = ligo.dev.o2.burst.allsky.stamp\n")
    fid.write("notification = nevers\n")
    fid.write("getenv = true\n")
    fid.write("log = /local/%s/light_curve_generation.log\n" % os.environ["USER"])
    fid.write("+MaxHours = 24\n")
    fid.write("universe = vanilla\n")
    fid.write("queue 1\n")


if __name__ == "__main__":
    main()
