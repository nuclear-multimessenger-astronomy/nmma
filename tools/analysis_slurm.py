#!/usr/bin/env python
import argparse
import pathlib
import os
from nmma.joint.base_parsing import nmma_base_parsing
from nmma.em.em_parsing import multi_wavelength_analysis_parser
import numpy as np

BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()



def get_slurm_parser(parser):
    parser = multi_wavelength_analysis_parser(parser)
    slurm_args = parser.add_argument_group(
        title="Slurm arguments",
        description="Arguments for running the lightcurve analysis on a Slurm HPC cluster",
    )
    # Slurm-specific arguments
    slurm_args.add_argument("--Ncore", default=8, type=int,
        help="number of cores for mpiexec")
    slurm_args.add_argument("--job-name", type=str, default="lightcurve-analysis")
    slurm_args.add_argument("--logs-dir-name", type=str, default="slurm_logs",
        help="directory name for slurm logs")
    slurm_args.add_argument("--cluster-name", type=str, default="Expanse",
        help="Name of HPC cluster")
    slurm_args.add_argument("--partition-type", type=str, default="shared",
        help="Partition name to request for computing")
    slurm_args.add_argument("--nodes", type=int, default=1,
        help="Number of nodes to request for computing")
    slurm_args.add_argument("--gpus", type=int, default=0,
        help="Number of GPUs to request")
    slurm_args.add_argument("--memory-GB", type=int, default=64,
        help="Memory allocation to request for computing")
    slurm_args.add_argument("--time", type=str, default="24:00:00",
        help="Walltime for instance")
    slurm_args.add_argument("--mail-type", type=str, default="NONE",
        help="slurm mail type (e.g. NONE, FAIL, ALL)")
    slurm_args.add_argument("--mail-user", type=str, default="",
        help="contact email address")
    slurm_args.add_argument("--python-env-name", type=str, default="nmma_env",
        help="Name of python environment to activate")
    slurm_args.add_argument("--script-name", type=str, default="slurm.sub")

    return parser


def main(args=None):
    parser = nmma_base_parsing(get_slurm_parser, return_parser=True)
    if args is None:
        args = parser.parse_args()

    args_vars = vars(args)

    wildcard_mapper = {
        "em_model": "$MODEL",
        "label": "$LABEL",
        "trigger_time": "$TT",
        "light_curve_data": "$DATA",
        "prior": "priors/$PRIOR.prior",
        "em_tmin": "$TMIN",
        "em_tmax": "$TMAX",
        "em_tstep": "$DT",
        "skip_sampling": "$SKIP_SAMPLING",
    }

    wildcard_keys = [
        "em_model",
        "label",
        "trigger_time",
        "light_curve_data",
        "prior",
        "em_tmin",
        "em_tmax",
        "em_tstep",
    ]

    args.outdir = f"{args.outdir}/$LABEL"
    for key in wildcard_keys:
        val = args_vars[key]
        if isinstance(val, (float, int)) and np.isnan(val):
            args_vars[key] = wildcard_mapper[key]
        elif val in [None, "None"]:
            args_vars[key] = wildcard_mapper[key]

    # Manipulate args for easy inclusion in slurm script
    ignore_args = ['help']
    for g in parser._action_groups:
        if g.title == "Slurm arguments":
            for act in g._group_actions:
                ignore_args.append(act.dest)

    args_to_add = []
    for act in parser._actions:
        if act.dest not in ignore_args:
            args_to_add.append(f"{act.option_strings[0]} {args_vars[act.dest]}")


    all_args = " ".join(args_to_add)

   

    scriptName = args.script_name

    logsDirName = args.logs_dir_name
    jobname = f"{args.job_name}"

    logsDir = os.path.join(BASE_DIR, logsDirName)
    os.makedirs(logsDir, exist_ok=True)

    # Write slurm script based on inputs
    fid = open(os.path.join(BASE_DIR, scriptName), "w")
    fid.write("#!/bin/bash\n")
    fid.write(f"#SBATCH --job-name={jobname}.job\n")
    fid.write(f"#SBATCH --output={logsDirName}/{jobname}_%A_%a.out\n")
    fid.write(f"#SBATCH --error={logsDirName}/{jobname}_%A_%a.err\n")
    fid.write(f"#SBATCH -p {args.partition_type}\n")
    fid.write(f"#SBATCH --nodes {args.nodes}\n")
    fid.write(f"#SBATCH --ntasks-per-node {args.Ncore}\n")
    fid.write(f"#SBATCH --gpus {args.gpus}\n")
    fid.write(f"#SBATCH --mem {args.memory_GB}G\n")
    fid.write(f"#SBATCH --time={args.time}\n")
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
        f'Wrote {scriptName} and created {logsDirName} directory within "{BASE_DIR}".'
    )
    print()
    print(
        "Default wildcard inputs are --em-model ($MODEL), --trigger-time ($TT), and --light-curve-data ($DATA).\nNote that the default prior is priors/$MODEL.prior."
    )
    print()
    print(
        "It is also recommended to set the following keywords to 'None' when running this script to allow them to be customized: --label ($LABEL), --em-tmin ($TMIN), --em-tmax ($TMAX), --em-tstep ($DT), and --skip-sampling ($SKIP_SAMPLING)"
    )
    print()
    print(
        f'To queue this script, run e.g. "sbatch --export=MODEL=Bu2019lm,TT=59361.0,DATA=example_files/candidate_data/ZTF21abdpqpq.dat {scriptName}" on your HPC.'
    )
    print()


if __name__ == "__main__":
    main()
