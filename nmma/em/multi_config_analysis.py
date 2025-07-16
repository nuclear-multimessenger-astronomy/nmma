import yaml
import subprocess
from .em_parsing import multi_wavelength_analysis_parser, multi_config_parser
from ..joint.base_parsing import nmma_base_parsing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os

def run_cmd_in_subprocess(cmd):
    subprocess.run(cmd)


def main(args=None):
    parser = nmma_base_parsing(multi_config_parser, return_parser=True)
    args, _ = parser.parse_known_args(namespace=args)

    main_args = nmma_base_parsing(multi_wavelength_analysis_parser, cli_args=[])

    yaml_dict = yaml.safe_load(os.path.expandvars(Path(args.config).read_text()))

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
