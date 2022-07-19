import os
import argparse
import json
import pandas as pd

import bilby
from bilby_pipe.create_injections import InjectionCreator

from subprocess import check_output


def main():

    parser = argparse.ArgumentParser(description="Slurm files from nmma injection file")
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the kilonova model to be used"
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
        required=True,
        help="The bilby injection json file to be used",
    )
    parser.add_argument(
        "--nlive",
        type=int,
        required=False,
        default=512,
        help="The number of live points to be used (default: 512)",
    )
    parser.add_argument(
        "--filters",
        type=str,
        help="A comma seperated list of filters to use (e.g. g,r,i). If none is provided, will use all the filters available",
    )
    parser.add_argument(
        "--injection-detection-limit",
        metavar="mAB",
        type=str,
        default="24.1,25.0,25.0,25.3,24.5,23.0,23.2,22.6,22.6",
        help="The highest mAB to be presented in the injection data set, any mAB higher than this will become a non-detection limit. Should be comma delimited list same size as injection set. (default for filters u,g,r,i,z,y,J,H,K is 24.1,25.0,25.0,25.3,24.5,23.0,23.2,22.6,22.6)",
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
    parser.add_argument("-o", "--outdir", type=str, default="outdir")
    parser.add_argument(
        "--photometry-augmentation",
        help="Augment photometry to improve parameter recovery",
        action="store_true",
    )
    parser.add_argument(
        "--photometry-augmentation-N-points",
        help="Number of augmented points to include",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--photometry-augmentation-filters",
        type=str,
        help="A comma seperated list of filters to use for augmentation (e.g. g,r,i). If none is provided, will use all the filters available",
    )
    parser.add_argument(
        "--photometry-augmentation-N-simulations",
        help="Number of augmented simulation runs to perform",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    # load the injection json file
    if args.injection_file:
        if args.injection_file.endswith(".json"):
            with open(args.injection_file, "rb") as f:
                injection_data = json.load(f)
                datadict = injection_data["injections"]["content"]
                dataframe_from_inj = pd.DataFrame.from_dict(datadict)
        else:
            print("Only json supported.")
            exit(1)

    if len(dataframe_from_inj) > 0:
        args.n_injection = len(dataframe_from_inj)

    # create the injection dataframe from the prior_file
    injection_creator = InjectionCreator(
        prior_file=args.prior_file,
        prior_dict=None,
        n_injection=args.n_injection,
        default_prior="PriorDict",
        gps_file=None,
        trigger_time=0,
        generation_seed=0,
    )
    dataframe_from_prior = injection_creator.get_injection_dataframe()

    # combine the dataframes
    dataframe = pd.DataFrame.merge(
        dataframe_from_inj,
        dataframe_from_prior,
        how="outer",
        left_index=True,
        right_index=True,
    )

    lc_analysis = (
        check_output(["which", "light_curve_analysis"]).decode().replace("\n", "")
    )

    logdir = os.path.join(args.outdir, "logs")
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    job_number = 0
    fid = open(args.condor_dag_file, "w")
    fid1 = open(args.bash_file, "w")
    for index, row in dataframe.iterrows():
        # with open(args.analysis_file, 'r') as file:
        #    analysis = file.read()

        outdir = os.path.join(args.outdir, str(index))
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        if args.photometry_augmentation:
            for ii in range(args.photometry_augmentation_N_simulations):
                outdir = os.path.join(args.outdir, str(index), str(ii))
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)

                priors = bilby.gw.prior.PriorDict(args.prior_file)
                priors.to_file(outdir, label="injection")
                priorfile = os.path.join(outdir, "injection.prior")
                injfile = os.path.join(outdir, "lc.csv")

                fid.write("JOB %d %s\n" % (job_number, args.condor_sub_file))
                fid.write("RETRY %d 3\n" % (job_number))
                fid.write(
                    'VARS %d jobNumber="%d" PRIOR="%s" OUTDIR="%s" INJOUT="%s" INJNUM="%s" SEED="%d"\n'
                    % (
                        job_number,
                        job_number,
                        priorfile,
                        outdir,
                        injfile,
                        str(index),
                        ii,
                    )
                )
                fid.write("\n\n")
                job_number = job_number + 1

                fid1.write(
                    f"{lc_analysis} --model {args.model} --svd-path ./svdmodels --outdir {outdir} --label injection_{args.model} --prior {args.prior_file} --tmin 0 --tmax 20 --dt 0.5 --error-budget 1 --nlive {args.nlive} --Ebv-max 0 --injection {args.injection_file} --injection-num {str(index)} --injection-detection-limit {args.injection_detection_limit} --injection-outfile {injfile} --generation-seed 42 --filters {args.filters} --plot --remove-nondetections --photometry-augmentation --photometry-augmentation-N-points {args.photometry_augmentation_N_points} --photometry-augmentation-filters {args.photometry_augmentation_filters} --photometry-augmentation-seed {ii}\n"
                )
        else:
            priors = bilby.gw.prior.PriorDict(args.prior_file)
            priors.to_file(outdir, label="injection")
            priorfile = os.path.join(outdir, "injection.prior")
            injfile = os.path.join(outdir, "lc.csv")

            fid.write("JOB %d %s\n" % (job_number, args.condor_sub_file))
            fid.write("RETRY %d 3\n" % (job_number))
            fid.write(
                'VARS %d jobNumber="%d" PRIOR="%s" OUTDIR="%s" INJOUT="%s" INJNUM="%s"\n'
                % (job_number, job_number, priorfile, outdir, injfile, str(index))
            )
            fid.write("\n\n")
            job_number = job_number + 1

            fid1.write(
                f"{lc_analysis} --model {args.model} --svd-path ./svdmodels --outdir %s --label injection_{args.model} --prior {args.prior_file} --tmin 0 --tmax 20 --dt 0.5 --error-budget 1 --nlive {args.nlive} --Ebv-max 0 --injection {args.injection_file} --injection-num {str(index)} --injection-detection-limit {args.injection_detection_limit} --injection-outfile {injfile} --generation-seed 42 --filters {args.filters} --plot --remove-nondetections\n"
            )

    fid.close()

    fid = open(args.condor_sub_file, "w")
    fid.write("executable = %s\n" % lc_analysis)
    fid.write(f"output = {logdir}/out.$(jobNumber)\n")
    fid.write(f"error = {logdir}/err.$(jobNumber)\n")

    if args.photometry_augmentation:
        fid.write(
            f"arguments = --model {args.model} --svd-path ./svdmodels --outdir $(OUTDIR) --label injection_{args.model} --prior {args.prior_file} --tmin 0 --tmax 20 --dt 0.5 --error-budget 1 --nlive {args.nlive} --Ebv-max 0 --injection {args.injection_file} --injection-num $(INJNUM) --injection-detection-limit {args.injection_detection_limit} --injection-outfile $(INJOUT) --generation-seed 42 --filters {args.filters} --plot --remove-nondetections --photometry-augmentation --photometry-augmentation-N-points {args.photometry_augmentation_N_points} --photometry-augmentation-filters {args.photometry_augmentation_filters} --photometry-augmentation-seed $(SEED)\n"
        )
    else:
        fid.write(
            f"arguments = --model {args.model} --svd-path ./svdmodels --outdir $(OUTDIR) --label injection_{args.model} --prior {args.prior_file} --tmin 0 --tmax 20 --dt 0.5 --error-budget 1 --nlive {args.nlive} --Ebv-max 0 --injection {args.injection_file} --injection-num $(INJNUM) --injection-detection-limit {args.injection_detection_limit} --injection-outfile $(INJOUT) --generation-seed 42 --filters {args.filters} --plot --remove-nondetections\n"
        )
    fid.write('requirements = OpSys == "LINUX"\n')
    fid.write("request_memory = 8192\n")
    fid.write("request_disk = 500 MB\n")
    fid.write("request_cpus = 1\n")
    fid.write("accounting_group = ligo.dev.o2.burst.allsky.stamp\n")
    fid.write("notification = nevers\n")
    fid.write("getenv = true\n")
    fid.write("log = /local/%s/light_curve_analysis.log\n" % os.environ["USER"])
    fid.write("+MaxHours = 24\n")
    fid.write("universe = vanilla\n")
    fid.write("queue 1\n")


if __name__ == "__main__":
    main()
