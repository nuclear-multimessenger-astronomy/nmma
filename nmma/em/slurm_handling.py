import os
import numpy as np
from .em_parsing import slurm_lc_parser, parsing_and_logging, slurm_analysis_parser
from ..joint.base_parsing import nmma_base_parsing
from ..joint.injection_handling import NMMAInjectionCreator
from ..joint.utils import read_injection_file


def main():
    args = parsing_and_logging(slurm_lc_parser)
    injection_creator = NMMAInjectionCreator(args)
    dataframe = injection_creator.generate_prelim_dataframe()

    for index, row in dataframe.iterrows():
        with open(args.analysis_file, "r") as file:
            analysis = file.read()

        outdir = os.path.join(args.outdir, str(index))
        os.makedirs(outdir, exist_ok=True)

        injection_creator.priors.to_file(outdir, label="injection")
        priorfile = os.path.join(outdir, "injection.prior")
        injfile = os.path.join(outdir, "lc.csv")

        analysis = analysis.replace("PRIOR", priorfile)
        analysis = analysis.replace("OUTDIR", outdir)
        analysis = analysis.replace("INJOUT", injfile)
        analysis = analysis.replace("INJNUM", str(index))
        analysis_file = os.path.join(outdir, "inference.sh")

        fid = open(analysis_file, "w")
        fid.write(analysis)
        fid.close()


def lc_creation():
    args = parsing_and_logging(slurm_lc_parser)

    logdir = os.path.join(args.outdir, "logs")
    os.makedirs(logdir, exist_ok=True)

    injection_df = read_injection_file(args)
    number_jobs = int(
        np.ceil(len(injection_df) / args.lightcurves_per_job)
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

def slurm_analysis(args=None):
    parser = nmma_base_parsing(slurm_analysis_parser, return_parser=True)
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
    


    # Write slurm script based on inputs
    sbatch_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={args.job_name}.job",
        f"#SBATCH --output={args.logs_dir_name}/{args.job_name}_%A_%a.out",
        f"#SBATCH --error={args.logs_dir_name}/{args.job_name}_%A_%a.err",
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


if __name__ == "__main__":
    main()