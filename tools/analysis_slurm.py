#!/usr/bin/env python
import argparse
import pathlib
import os
from nmma.em.analysis import get_parser

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()

if __name__ == "__main__":

    # Get analysis.py parser
    nmma_parser = get_parser(add_help=False)

    # Get argument names from nmma_parser
    nmma_arg_list = []
    for action in nmma_parser._actions:
        arg_name = action.option_strings[0]
        nmma_arg_list.append(arg_name[2:].replace("-", "_"))

    # Create new parser that inherits analysis.py arguments
    parser = argparse.ArgumentParser(parents=[nmma_parser])

    # Slurm-specific arguments
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

    wildcard_mapper = {
        "model": "$MODEL",
        "label": "$LABEL",
        "trigger_time": "$TT",
        "data": "$DATA",
        "prior": "priors/$MODEL.prior",
    }

    for key in ["model", "label", "trigger_time", "data", "prior"]:
        if args_vars[key] is None:
            args_vars[key] = wildcard_mapper[key]

    # Manipulate args for easy inclusion in slurm script
    args_to_add = []
    for arg in args_vars.keys():
        if arg in nmma_arg_list:
            arg_value = args_vars[arg]
            if type(arg_value) == bool:
                if arg_value:
                    hyphen_arg = arg.replace("_", "-")
                    args_to_add.append(f"--{hyphen_arg}")
            elif (arg_value is not None) & (arg_value != "None"):
                hyphen_arg = arg.replace("_", "-")
                args_to_add.append(f"--{hyphen_arg} {args_vars[arg]}")
            else:
                continue

    all_args = " ".join(args_to_add)

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
        if args.gpus > 0:
            fid.write("module add gpu/0.15.4\n")
            fid.write("module add cuda\n")
        fid.write(f"source activate {args.python_env_name}\n")

    fid.write(
        f"mpiexec -n {args.Ncore} -hosts=$(hostname) lightcurve-analysis {all_args}"
    )

    fid.close()

    print()
    print(
        f'Wrote {slurmDir}/{scriptName} and created "logs" and "slurm" directories within "{args.slurm_dir_name}".'
    )
    print()
    print(
        "Default wildcard inputs are --model ($MODEL), --label ($LABEL), --trigger-time ($TT), and --data ($DATA).\nNote that the default prior is priors/$MODEL.prior."
    )
    print()
    print(
        f'To queue this script, run e.g. "sbatch --export=MODEL=Bu2019lm,LABEL=test,TT=59361.0,DATA=example_files/candidate_data/ZTF21abdpqpq.dat {scriptName}" on your HPC.'
    )
    print()
