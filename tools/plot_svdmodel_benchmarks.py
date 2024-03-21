import json
import argparse
import matplotlib.pyplot as plt
import pathlib
import numpy as np


BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()

plt.rcParams["font.size"] = 14


def plot_benchmarks(
    outdir: str = "benchmark_output",
    interpolation_type: str = "tensorflow",
):
    """
    make barplots of 25th, 50th and 75th percentiles of reduced chi2 distributions for trained models

    :param outdir: Path to the output directory (str)
    :param interpolation_type: Type of interpolation (str)

    """
    benchmark_path = BASE_DIR / outdir
    interpolation_suffix = "_tf" if interpolation_type == "tensorflow" else ""
    json_files = benchmark_path.glob(f"*{interpolation_suffix}/*.json")

    benchmark_dict = {}
    for file in json_files:
        str_file = str(file)
        model = str_file.split("/")[-2]

        with open(file) as f:
            percentiles = json.load(f)

        benchmark_dict.update(percentiles)

    for model in benchmark_dict.keys():
        model_benchmarks = benchmark_dict[model]
        filter_names = [x for x in model_benchmarks.keys()]
        filter_benchmarks = [x for x in model_benchmarks.values()]

        pctls_25 = [x[1] for x in filter_benchmarks]
        pctls_50 = [x[2] for x in filter_benchmarks]
        pctls_75 = [x[3] for x in filter_benchmarks]
        pctls_100 = [x[4] for x in filter_benchmarks]

        fig, ax = plt.subplots(figsize=(12, 8))

        plt.tight_layout()
        ax.bar(filter_names, pctls_75, label="<= 75th", color="red", hatch="\\")
        ax.bar(filter_names, pctls_50, label="<= 50th", color="slategray")
        ax.bar(filter_names, pctls_25, label="<= 25th", color="blue", hatch="/")
        ax.set_ylim(0, 1.1 * np.max(pctls_75))
        ax.tick_params(axis="x", labelrotation=75)
        ax.set_xlabel("Filter")
        ax.set_ylabel(r"Reduced $\chi^{2}$")
        ax.legend(
            title=f"Percentile\n (max $\chi^{2}$ = {np.round(np.max(pctls_100),1)})",  # noqa
            loc=2,
        )
        ax.set_title(f"{model} benchmark percentiles")
        plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.95)

        fig.savefig(
            benchmark_path / f"benchmark_percentiles_{model}.pdf", bbox_inches="tight"
        )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="benchmark_output",
        help="path to the output directory",
    )
    parser.add_argument(
        "--interpolation-type",
        type=str,
        default="tensorflow",
        help="type of interpolation",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    plot_benchmarks(**vars(args))
