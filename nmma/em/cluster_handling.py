import os
import numpy as np
import subprocess
from concurrent.futures import ThreadPoolExecutor

from . import em_parsing as emp
from ..core.parsing import nmma_base_parsing, slurm_analysis_parser
from ..core.utils import read_injection_file, load_yaml


def lc_creation():
    args = emp.parsing_and_logging(emp.slurm_lc_parser)
    os.makedirs(os.path.join(args.outdir, "logs"), exist_ok=True)

    injection_df = read_injection_file(args)
    n_jobs = int(np.ceil(len(injection_df) / args.n_per_job))

    for ii in range(n_jobs):
        with open(args.analysis_file, "r") as file:
            analysis = file.read()
        analysis = analysis.replace("INJRANGE", f"{ii*n_jobs:i},{(ii+1)*n_jobs:i}") 

        with open(os.path.join(args.outdir, f"inference_{ii:i}.sh"), "w") as file:
            file.write(analysis)
            

def slurm_analysis(args=None):
    parser = nmma_base_parsing(
        (slurm_analysis_parser, emp.multi_wavelength_analysis_parser),
        return_parser=True)
    if args is None:
        args = parser.parse_args()

    args_vars = vars(args)

    wildcard_mapper = {
        "em_model": "$MODEL",
        "label": "$LABEL",
        "trigger_time": "$TT",
        "light_curve_data": "$DATA",
        "prior_file": "priors/$PRIOR.prior",
        "em_tmin": "$TMIN",
        "em_tmax": "$TMAX",
        "em_tstep": "$DT",
    }
    args.outdir = f"{args.outdir}/$LABEL"

    for key, v in wildcard_mapper.items():
        args_val = args_vars[key]
        if (isinstance(args_val, (float, int)) and np.isnan(args_val)) or args_val in [None, "None"]:
            args_vars[key] = v

    # Manipulate args for easy inclusion in slurm script
    ignore_args = ['help']
    for g in parser._action_groups:
        if g.title == "Slurm arguments":
            for act in g._group_actions:
                ignore_args.append(act.dest)

    all_args = " ".join([
        f"{act.option_strings[0]} {args_vars[act.dest]}"
        for act in parser._actions if act.dest not in ignore_args
    ])
    
    job_name = args.job_name if args.job_name else "lightcurve-analysis"

    # Write slurm script based on inputs
    sbatch_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}.job",
        f"#SBATCH --output={args.logs_dir_name}/{job_name}_%A_%a.out",
        f"#SBATCH --error={args.logs_dir_name}/{job_name}_%A_%a.err",
        f"#SBATCH -p {args.partition_type}",
        f"#SBATCH --nodes {args.nodes}",
        f"#SBATCH --ntasks-per-node {args.Ncore}",
        f"#SBATCH --gpus {args.gpus}",
        f"#SBATCH --mem {args.memory_GB}G",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --mail-type={args.mail_type}",
    ]

    if args.mail_type != "NONE" and args.mail_user:
        sbatch_lines.append(f"#SBATCH --mail-user={args.mail_user}")

    # Cluster-specific setup
    if args.cluster_name.lower() == "expanse":
        if args.gpus > 0:
            sbatch_lines.append("module add gpu/0.15.4")
            sbatch_lines.append("module add cuda")
        sbatch_lines.append(f"source activate {args.python_env_name}")

    # MPI command (all_args already prepared above)
    sbatch_lines.append(f"mpiexec -n {args.Ncore} -hosts=$(hostname) lightcurve-analysis {all_args}")
    
    # Create log directory and  write the script
    os.makedirs(os.path.join(args.base_dir, args.logs_dir_name), exist_ok=True)
    with open(os.path.join(args.base_dir, args.script_name), "w") as f:
        f.write("\n".join(sbatch_lines))
        
    # make the script executable
    # os.chmod(script_path, 0o755)


    print(
        f'Wrote {args.script_name} and created {args.logs_dir_name} directory within "{args.base_dir}". \n',
        "Default wildcard inputs are --em-model ($MODEL), --trigger-time ($TT), and --light-curve-data ($DATA).\n", 
        "Note that the default prior is priors/$MODEL.prior. \n",
        "It is also recommended to set the following keywords to 'None' when running this script to allow them to be customized: --label ($LABEL), --em-tmin ($TMIN), --em-tmax ($TMAX), --em-tstep ($DT), and --skip-sampling ($SKIP_SAMPLING) \n ",
        f'To queue this script, run e.g. "sbatch --export=MODEL=Bu2019lm,TT=59361.0,DATA=example_files/candidate_data/ZTF21abdpqpq.dat {args.script_name}" on your HPC.'
    )

def run_cmd_in_subprocess(cmd):
    subprocess.run(cmd)


def multi_config_analysis(args=None):
    parser = nmma_base_parsing(emp.multi_config_parser, return_parser=True)
    args, _ = parser.parse_known_args(namespace=args)

    main_args = nmma_base_parsing(emp.multi_wavelength_analysis_parser, cli_args=[])

    yaml_dict = load_yaml(args.config)

    total_configs = len(list(yaml_dict.keys()))
    futures = []

    with ThreadPoolExecutor() as executor:
        for analysis_set in yaml_dict.keys():
            params = yaml_dict[analysis_set]

            if "process-per-config" in params or args.process is None:
                processes = params["process-per-config"]
            elif args.parallel and args.process is not None:
                processes = args.process // total_configs
            else:
                processes = args.process

            cmd = ["mpiexec", "-np", str(processes), "lightcurve-analysis"]

            for key, value in params.items():
                key = key.replace("-", "_")

                if key == "process_per_config":
                    continue

                if key not in main_args:
                    raise ValueError(f"{key} not a known argument... please remove")
                key = key.replace("_", "-")

                cmd.append(f"--{key}")
                if value is not True:
                    cmd.append(str(value))

            if not args.parallel:
                print(f"{'#'*100}")
                print(f"Running analysis set:  {analysis_set} with {processes} processes")
                run_cmd_in_subprocess(cmd)
                print(f"{'#'*100}")
            else:
                future = executor.submit(run_cmd_in_subprocess, cmd)
                futures.append(future)
    if args.parallel:
        for future in futures:
            future.result()