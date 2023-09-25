#!/usr/bin/env python
import argparse
import pathlib
import os

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Beginning of analysis.py inputs. Keep this up to date as analysis.py changes (along with the two lists below).
    parser.add_argument(
        "--config",
        type=str,
        help="Name of the configuration file containing parameter values.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the kilonova model to be used",
        default="$MOD",
    )
    parser.add_argument(
        "--interpolation-type",
        type=str,
        help="SVD interpolation scheme.",
        default="sklearn_gp",
    )
    parser.add_argument(
        "--svd-path",
        type=str,
        help="Path to the SVD directory, with {model}_mag.pkl and {model}_lbol.pkl",
        default="svdmodels",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Path to the output directory",
        default="outdir",
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Label for the run",
        default="$LABEL",
    )
    parser.add_argument(
        "--trigger-time",
        # type=float,
        help="Trigger time in modified julian day, not required if injection set is provided",
        default="$TT",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the data file in [time(isot) filter magnitude error] format",
        default="$DATA",
    )
    parser.add_argument(
        "--prior", type=str, help="Path to the prior file", default="priors/$MOD.prior"
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=0.05,
        help="Days to start analysing from the trigger time (default: 0)",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=14.0,
        help="Days to stop analysing from the trigger time (default: 14)",
    )
    parser.add_argument(
        "--dt", type=float, default=0.1, help="Time step in day (default: 0.1)"
    )
    parser.add_argument(
        "--log-space-time",
        action="store_true",
        default=False,
        help="Create the sample_times to be uniform in log-space",
    )
    parser.add_argument(
        "--n-tstep",
        type=int,
        default=50,
        help="Number of time steps (used with --log-space-time, default: 50)",
    )
    parser.add_argument(
        "--photometric-error-budget",
        type=float,
        default=0.1,
        help="Photometric error (mag) (default: 0.1)",
    )
    parser.add_argument(
        "--svd-mag-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for mag evaluation (default: 10)",
    )
    parser.add_argument(
        "--svd-lbol-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for lbol evaluation (default: 10)",
    )
    parser.add_argument(
        "--filters",
        type=str,
        help="A comma seperated list of filters to use (e.g. g,r,i). If none is provided, will use all the filters available",
    )
    parser.add_argument(
        "--Ebv-max",
        type=float,
        default=0.5724,
        help="Maximum allowed value for Ebv (default:0.5724)",
    )
    parser.add_argument(
        "--grb-resolution",
        type=float,
        default=5,
        help="The upper bound on the ratio between thetaWing and thetaCore (default: 5)",
    )
    parser.add_argument(
        "--jet-type",
        type=int,
        default=0,
        help="Jet type to used used for GRB afterglow light curve (default: 0)",
    )
    parser.add_argument(
        "--error-budget",
        type=str,
        default="1.0",
        help="Additional systematic error (mag) to be introduced (default: 1)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="pymultinest",
        help="Sampler to be used (default: pymultinest)",
    )
    parser.add_argument(
        "--soft-init",
        action="store_true",
        default=False,
        help="To start the sampler softly (without any checking, default: False)",
    )
    parser.add_argument(
        "--sampler-kwargs",
        default="{}",
        type=str,
        help="Additional kwargs (e.g. {'evidence_tolerance':0.5}) for bilby.run_sampler, put a double quotation marks around the dictionary",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=1,
        help="Number of cores to be used, only needed for dynesty (default: 1)",
    )
    parser.add_argument(
        "--nlive", type=int, default=2048, help="Number of live points (default: 2048)"
    )
    parser.add_argument(
        "--reactive-sampling",
        action="store_true",
        default=False,
        help="To use reactive sampling in ultranest (default: False)",
    )
    parser.add_argument(
        "--seed",
        metavar="seed",
        type=int,
        default=42,
        help="Sampling seed (default: 42)",
    )
    parser.add_argument(
        "--injection", metavar="PATH", type=str, help="Path to the injection json file"
    )
    parser.add_argument(
        "--injection-num",
        metavar="eventnum",
        type=int,
        help="The injection number to be taken from the injection set",
    )
    parser.add_argument(
        "--injection-detection-limit",
        metavar="mAB",
        type=str,
        help="The highest mAB to be presented in the injection data set, any mAB higher than this will become a non-detection limit. Should be comma delimited list same size as injection set.",
    )
    parser.add_argument(
        "--injection-outfile",
        metavar="PATH",
        type=str,
        help="Path to the output injection lightcurve",
    )
    parser.add_argument(
        "--injection-model",
        type=str,
        help="Name of the kilonova model to be used for injection (default: the same as model used for recovery)",
    )
    parser.add_argument(
        "--remove-nondetections",
        action="store_true",
        default=False,
        help="remove non-detections from fitting analysis",
    )
    parser.add_argument(
        "--detection-limit",
        metavar="DICT",
        type=str,
        default=None,
        help="Dictionary for detection limit per filter, e.g., {'r':22, 'g':23}, put a double quotation marks around the dictionary",
    )
    parser.add_argument(
        "--with-grb-injection",
        help="If the injection has grb included",
        action="store_true",
    )
    parser.add_argument(
        "--prompt-collapse",
        help="If the injection simulates prompt collapse and therefore only dynamical",
        action="store_true",
    )
    parser.add_argument(
        "--ztf-sampling", help="Use realistic ZTF sampling", action="store_true"
    )
    parser.add_argument(
        "--ztf-uncertainties",
        help="Use realistic ZTF uncertainties",
        action="store_true",
    )
    parser.add_argument(
        "--ztf-ToO",
        help="Adds realistic ToO observations during the first one or two days. Sampling depends on exposure time specified. Valid values are 180 (<1000sq deg) or 300 (>1000sq deg). Won't work w/o --ztf-sampling",
        type=str,
        choices=["180", "300"],
    )
    parser.add_argument(
        "--train-stats",
        help="Creates a file too.csv to derive statistics",
        action="store_true",
    )
    parser.add_argument(
        "--rubin-ToO",
        help="Adds ToO obeservations based on the strategy presented in arxiv.org/abs/2111.01945.",
        action="store_true",
    )
    parser.add_argument(
        "--rubin-ToO-type",
        help="Type of ToO observation. Won't work w/o --rubin-ToO",
        type=str,
        choices=["BNS", "NSBH"],
    )
    parser.add_argument(
        "--xlim",
        type=str,
        default="0,14",
        help="Start and end time for light curve plot (default: 0-14)",
    )
    parser.add_argument(
        "--ylim",
        type=str,
        default="22,16",
        help="Upper and lower magnitude limit for light curve plot (default: 22-16)",
    )
    parser.add_argument(
        "--generation-seed",
        metavar="seed",
        type=int,
        default=42,
        help="Injection generation seed (default: 42)",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False, help="add best fit plot"
    )
    parser.add_argument(
        "--bilby-zero-likelihood-mode",
        action="store_true",
        default=False,
        help="enable prior run",
    )
    parser.add_argument(
        "--photometry-augmentation",
        help="Augment photometry to improve parameter recovery",
        action="store_true",
    )
    parser.add_argument(
        "--photometry-augmentation-seed",
        metavar="seed",
        type=int,
        default=0,
        help="Optimal generation seed (default: 0)",
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
        "--photometry-augmentation-times",
        type=str,
        help="A comma seperated list of times to use for augmentation in days post trigger time (e.g. 0.1,0.3,0.5). If none is provided, will use random times between tmin and tmax",
    )

    parser.add_argument(
        "--conditional-gaussian-prior-thetaObs",
        action="store_true",
        default=False,
        help="The prior on the inclination is against to a gaussian prior centered at zero with sigma = thetaCore / N_sigma",
    )

    parser.add_argument(
        "--conditional-gaussian-prior-N-sigma",
        default=1,
        type=float,
        help="The input for N_sigma; to be used with conditional-gaussian-prior-thetaObs set to True",
    )

    parser.add_argument(
        "--sample-over-Hubble",
        action="store_true",
        default=False,
        help="To sample over Hubble constant and redshift",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="print out log likelihoods",
    )

    parser.add_argument(
        "--refresh-models-list",
        type=bool,
        default=False,
        help="Refresh the list of models available on Zenodo",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        default=False,
        help="only look for local svdmodels (ignore Zenodo)",
    )

    parser.add_argument(
        "--bestfit",
        help="Save the best fit parameters and magnitudes to JSON",
        action="store_true",
        default=False,
    )
    # End of analysis inputs. Slurm-specific inputs below.

    parser.add_argument(
        "--Ncore",
        default=8,
        type=int,
        help="number of cores for mpiexec",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="lightcurve-analysis",
        help="job name",
    )
    parser.add_argument(
        "--slurm-dir-name",
        type=str,
        default="hpc",
        help="directory name for logs/slurm scripts",
    )
    parser.add_argument(
        "--cluster-name",
        type=str,
        default="Expanse",
        help="Name of HPC cluster",
    )
    parser.add_argument(
        "--partition-type",
        type=str,
        default="shared",
        help="Partition name to request for computing",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes to request for computing",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="Number of GPUs to request",
    )
    parser.add_argument(
        "--memory-GB",
        type=int,
        default=64,
        help="Memory allocation to request for computing",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="24:00:00",
        help="Walltime for instance",
    )
    parser.add_argument(
        "--mail-type",
        type=str,
        default="NONE",
        help="slurm mail type (e.g. NONE, FAIL, ALL)",
    )
    parser.add_argument(
        "--mail-user",
        type=str,
        default="",
        help="contact email address",
    )
    parser.add_argument(
        "--account-name",
        type=str,
        default="umn131",
        help="Name of account with current HPC allocation",
    )
    parser.add_argument(
        "--python-env-name",
        type=str,
        default="nmma_env",
        help="Name of python environment to activate",
    )
    parser.add_argument(
        "--script-name",
        type=str,
        default="slurm.sub",
        help="script name",
    )

    args = parser.parse_args()

    args_vars = vars(args)

    # List of non-boolean arguments for analysis.py
    non_bool_arg_list = [
        "config",
        "model",
        "interpolation_type",
        "svd_path",
        "outdir",
        "label",
        "trigger_time",
        "data",
        "prior",
        "tmin",
        "tmax",
        "dt",
        "n_tstep",
        "photometric_error_budget",
        "svd_mag_ncoeff",
        "filters",
        "Ebv_max",
        "grb_resolution",
        "jet_type",
        "error_budget",
        "sampler",
        "sampler_kwargs",
        "cpus",
        "nlive",
        "seed",
        "injection",
        "injection_num",
        "injection_detection_limit",
        "injection_outfile",
        "injection_model",
        "detection_limit",
        "ztf_ToO",
        "rubin_ToO_type",
        "xlim",
        "ylim",
        "generation_seed",
        "photometry_augmentation_seed",
        "photometry_augmentation_N_points",
        "photometry_augmentation_filters",
        "photometry_augmentation_ties",
        "conditional_gaussian_prior_N_sigma",
    ]

    # List of boolean arguments for analysis.py
    bool_arg_list = [
        "log_space_time",
        "soft_init",
        "reactive_sampling",
        "remove_nondetections",
        "with_grb_injection",
        "prompt_collapse",
        "ztf_sampling",
        "ztf_uncertainties" "train_stats",
        "rubin_ToO",
        "plot",
        "bilby_zero_likelihood_mode",
        "photometry_augmentation" "conditional_gaussian_prior_thetaObs",
        "sample_over_Hubble" "verbose",
        "refresh_models_list",
        "local_only",
        "bestfit",
    ]

    # Manipulate args for easy inclusion in slurm script
    bool_args_to_add = []
    non_bool_args_to_add = []
    for arg in args_vars.keys():
        if arg in bool_arg_list:
            if args_vars[arg]:
                hyphen_arg = arg.replace("_", "-")
                bool_args_to_add.append(f"--{hyphen_arg}")
        elif arg in non_bool_arg_list:
            hyphen_arg = arg.replace("_", "-")
            non_bool_args_to_add.append(f"--{hyphen_arg} {args_vars[arg]}")

    bool_args = " ".join(bool_args_to_add)
    non_bool_args = " ".join(non_bool_args_to_add)

    scriptName = args.script_name
    script_path = BASE_DIR / scriptName

    dirname = f"{args.slurm_dir_name}"
    jobname = f"{args.job_name}"

    dirpath = BASE_DIR / dirname
    os.makedirs(dirpath, exist_ok=True)

    slurmDir = os.path.join(dirpath, "slurm")
    os.makedirs(slurmDir, exist_ok=True)

    logsDir = os.path.join(dirpath, "logs")
    os.makedirs(logsDir, exist_ok=True)

    # Write slurm script based on inputs
    fid = open(os.path.join(slurmDir, scriptName), "w")
    fid.write("#!/bin/bash\n")
    fid.write(f"#SBATCH --job-name={jobname}.job\n")
    fid.write(f"#SBATCH --output=../logs/{jobname}_%A_%a.out\n")
    fid.write(f"#SBATCH --error=../logs/{jobname}_%A_%a.err\n")
    fid.write(f"#SBATCH -p {args.partition_type}\n")
    fid.write(f"#SBATCH --nodes {args.nodes}\n")
    fid.write(f"#SBATCH --ntasks-per-node {args.Ncore}\n")
    fid.write(f"#SBATCH --gpus {args.gpus}\n")
    fid.write(f"#SBATCH --mem {args.memory_GB}G\n")
    fid.write(f"#SBATCH --time={args.time}\n")
    fid.write(f"#SBATCH -A {args.account_name}\n")
    fid.write(f"#SBATCH --mail-type={args.mail_type}\n")
    if args.mail_type != "NONE":
        fid.write(f"#SBATCH --mail-user={args.mail_user}\n")

    if args.cluster_name in ["Expanse", "expanse", "EXPANSE"]:
        fid.write("module purge\n")
        if args.gpus > 0:
            fid.write("module add gpu/0.15.4\n")
            fid.write("module add cuda\n")
        fid.write(f"source activate {args.python_env_name}\n")

    fid.write(
        f"mpiexec -n {args.Ncore} -hosts=$(hostname) lightcurve-analysis {non_bool_args} {bool_args}"
    )

    fid.close()

    print()
    print(
        f'Wrote {slurmDir}/{scriptName} and created "logs" and "slurm" directories within "{args.slurm_dir_name}".'
    )
    print()
    print(
        "Default wildcard inputs are --model ($MOD), --label ($LABEL), --trigger-time ($TT), and --data ($DATA).\nNote that the default prior is priors/$MOD.prior."
    )
    print()
    print(
        f'To queue this script, run e.g. "sbatch --export=MOD=Bu2019lm,LABEL=test,TT=59361.0,DATA=example_files/candidate_data/ZTF21abdpqpq.dat {scriptName}" on your HPC.'
    )
    print()
