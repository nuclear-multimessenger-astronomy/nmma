import yaml
import subprocess
from .analysis import get_parser
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import sys


def get_parser_here():
    parser = argparse.ArgumentParser(
        description="Multi config analysis script for NMMA."
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="To run multiple configs in parallel",
    )

    parser.add_argument(
        "-p",
        "--process",
        type=int,
        help="No of processess each configuration should have, if --parallel is set then process will be divided equally among all configs, else each config will run sequentially with given no of process. Strictly required if --process-per-config is not given",
    )
    parser.add_argument(
        "--process-per-config",
        type=int,
        help="If multiple configurations are given, how many MPI process should be assigned to each configuration. In the yaml file, indicate the number of process for each configuration with the key 'process-per-config'. If not given, all configurations will be run depending on the state and value of --parallel and --process. This takes precedence over --process"
    )

    return parser


def run_cmd_in_subprocess(cmd):
    subprocess.run(cmd)


def main(args=None):
    parser2 = get_parser_here()
    args2 = parser2.parse_known_args()

    parser = get_parser()
    args, _ = parser.parse_known_args(args2[1])

    for key, value in vars(args2[0]).items():
        setattr(args, key, value)

    yaml_dict = yaml.safe_load(Path(args.config).read_text())
    total_configs = len(list(yaml_dict.keys()))

    futures = []

    total_processes = 0

    for analysis_set in yaml_dict.keys():
        params = yaml_dict[analysis_set]

        if "process-per-config" in params or args.process is None:
            processes = params.get("process-per-config", 0)
            total_processes += processes
        elif args.parallel and args.process is not None:
            processes = args.process // total_configs
        else:
            processes = args.process
            total_processes += processes

    print(f"Total number of processes: {total_processes}")
    sys.stdout.flush()
    
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

                if key not in args:
                    print(f"{key} not a known argument... please remove")
                    exit()
                key = key.replace("_", "-")

                if isinstance(value, bool) and value:
                    cmd.append(f"--{key}")
                else:
                    cmd.append(f"--{key}")
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