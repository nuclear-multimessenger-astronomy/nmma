import numpy as np
import argparse
import os
import glob
from subprocess import check_output


def main():

    parser = argparse.ArgumentParser(
        description="Inference on binary source parameters with kilonova ejecta posterior and GW source posterior given."
    )

    parser.add_argument(
        "--outdir", type=str, required=True, help="The running directory"
    )
    parser.add_argument(
        "--GWsamples",
        type=str,
        required=True,
        help="Data from the posterior of the GW injection",
    )
    parser.add_argument(
        "--EMsamples",
        type=str,
        required=True,
        help="Data from the posterior of the EM injection using model of Bu2019lm ",
    )
    parser.add_argument("--EOSpath", type=str, required=True, help="The EOS data")
    parser.add_argument("--Neos", type=int, required=True, help="Number of")
    parser.add_argument(
        "--nlive", type=int, required=False, default=1024, help="live number"
    )
    parser.add_argument(
        "--GWprior", type=str, required=True, help="Prior file used for the GW analysis"
    )
    parser.add_argument(
        "--EMprior",
        type=str,
        required=True,
        help="Prior file used for the EM eos analysis",
    )
    parser.add_argument("-d", "--detections-file", type=str)
    parser.add_argument(
        "--total-ejecta-mass",
        action="store_true",
        help="To run with total ejecta mass, if not activated, the two ejecta are consider seperately",
    )
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
    args = parser.parse_args()

    gwsamples = glob.glob(os.path.join(args.GWsamples, "*.dat"))

    if args.detections_file is not None:
        events = np.loadtxt(args.detections_file, usecols=[0]).astype(int)
    else:
        events = np.arange(len(gwsamples))

    logdir = os.path.join(args.outdir, "logs")
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    gwem_resampling = (
        check_output(["which", "gwem_resampling"]).decode().replace("\n", "")
    )

    job_number = 0

    fid = open(args.condor_dag_file, "w")
    fid1 = open(args.bash_file, "w")
    for ii in events:

        gwsamples = os.path.join(
            args.GWsamples, "inj_PhD_posterior_samples_%d.dat" % ii
        )
        if not os.path.isfile(gwsamples):
            print(f"missing {gwsamples}... continuing")
            continue

        emsamples = os.path.join(
            args.EMsamples, "%d/injection_Bu2019lm_posterior_samples.dat" % ii
        )
        if not os.path.isfile(emsamples):
            print(f"missing {emsamples}... continuing")
            continue

        outdir = os.path.join(args.outdir, "%d" % ii)

        fid.write("JOB %d %s\n" % (job_number, args.condor_sub_file))
        fid.write("RETRY %d 3\n" % (job_number))
        fid.write(
            'VARS %d jobNumber="%d" outdir="%s" GWsamples="%s" EMsamples="%s"\n'
            % (job_number, job_number, outdir, gwsamples, emsamples)
        )
        fid.write("\n\n")
        job_number = job_number + 1

        if args.total_ejecta_mass:
            fid1.write(
                "%s --outdir %s --EMsamples %s --GWsamples %s --EOS %s --nlive 8192 --GWprior %s --EMprior %s --total-ejecta-mass --Neos %d\n"
                % (
                    gwem_resampling,
                    outdir,
                    emsamples,
                    gwsamples,
                    args.EOSpath,
                    args.GWprior,
                    args.EMprior,
                    args.Neos,
                )
            )
        else:
            fid1.write(
                "%s --outdir %s --EMsamples %s --GWsamples %s --EOS %s --nlive 8192 --GWprior %s --EMprior %s --Neos %d\n"
                % (
                    gwem_resampling,
                    outdir,
                    emsamples,
                    gwsamples,
                    args.EOSpath,
                    args.GWprior,
                    args.EMprior,
                    args.Neos,
                )
            )

    fid.close()
    fid1.close()

    fid = open(args.condor_sub_file, "w")
    fid.write("executable = %s\n" % gwem_resampling)
    fid.write(f"output = {logdir}/out.$(jobNumber)\n")
    fid.write(f"error = {logdir}/err.$(jobNumber)\n")
    if args.total_ejecta_mass:
        fid.write(
            "arguments = --outdir $(outdir) --EMsamples $(EMsamples) --GWsamples $(GWsamples) --EOS %s --nlive 8192 --GWprior %s --EMprior %s --total-ejecta-mass --Neos %d\n"
            % (args.EOSpath, args.GWprior, args.EMprior, args.Neos)
        )
    else:
        fid.write(
            "arguments = --outdir $(outdir) --EMsamples $(EMsamples) --GWsamples $(GWsamples) --EOS %s --nlive 8192 --GWprior %s --EMprior %s --Neos %d\n"
            % (args.EOSpath, args.GWprior, args.EMprior, args.Neos)
        )
    fid.write('requirements = OpSys == "LINUX"\n')
    fid.write("request_memory = 8192\n")
    fid.write("request_disk = 500 MB\n")
    fid.write("request_cpus = 1\n")
    fid.write("accounting_group = ligo.dev.o2.burst.allsky.stamp\n")
    fid.write("notification = nevers\n")
    fid.write("getenv = true\n")
    fid.write("log = /local/%s/gwem_resampling.log\n" % os.environ["USER"])
    fid.write("+MaxHours = 24\n")
    fid.write("universe = vanilla\n")
    fid.write("queue 1\n")


if __name__ == "__main__":
    main()
