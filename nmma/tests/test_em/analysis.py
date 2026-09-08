from pathlib import Path
import os
import pytest
import shutil
import copy
from argparse import Namespace

from nmma.em import analysis, em_parsing, cluster_handling

test_dir = Path(__file__).resolve().parent.parent
DATA_DIR = test_dir / "data"


@pytest.fixture(autouse=True)
def cleanup_outdir(args):
    yield
    if Path(args.outdir).exists():
        shutil.rmtree(args.outdir, ignore_errors=True)


@pytest.fixture(scope="module")
def args():
    args = em_parsing.parsing_and_logging(
        em_parsing.multi_wavelength_analysis_parser, []
    )
    non_default_args = dict(
        em_transient_class="fiesta_kn",
        label="injection",
        prior_file=f"{DATA_DIR}/Bu2026_simplified.prior",
        bestfit=True,
        filters="ztfr",
        Ebv_max=0.0,
        nlive=64,
        sampler="pymultinest",
        # sampler_kwargs={"max_iter": 100},
        injection=True,
        plot=True,
    )
    for key, value in non_default_args.items():
        setattr(args, key, value)

    return args


def test_with_Hubble(args):
    test_args = copy.deepcopy(args)
    test_args.prior_file = f"{DATA_DIR}/Bu2026_simplified_Hubble.prior"
    test_args.Hubble = True
    analysis.main(test_args)


def test_analysis_systematics_with_time(args):
    args.systematics_file = f"{DATA_DIR}/systematics_with_time.yaml"
    analysis.main(args)


def test_analysis_systematics_with_time_and_filters(args):
    args.filters = ["ztfr", "sdssu", "2massks"]
    args.systematics_file = f"{DATA_DIR}/systematics_with_time_combined_filters.yaml"
    analysis.main(args)


def test_analysis_systematics_without_time(args):
    args.filters = "ztfr"
    args.systematics_file = f"{DATA_DIR}/systematics_without_time.yaml"
    analysis.main(args)


def test_analysis_slurm(args):
    args_slurm = dict(
        Ncore=8,
        job_name="lightcurve-analysis",
        base_dir=Path(args.outdir),
        logs_dir_name="slurm_logs",
        cluster_name="Expanse",
        partition_type="shared",
        nodes=1,
        gpus=0,
        memory_GB=64,
        time="24:00:00",
        mail_type="NONE",
        mail_user="",
        python_env_name="nmma_env",
        script_name="slurm.sub",
    )
    args.__dict__.update(args_slurm)

    cluster_handling.slurm_analysis(args)
    shutil.rmtree(Path(args.base_dir) / args.logs_dir_name, ignore_errors=True)


def test_analysis_multi():
    config = DATA_DIR / "multi_config.yaml"
    os.environ["DATA_DIR"] = str(DATA_DIR)

    args = Namespace(config=str(config), process=2, parallel=False)
    cluster_handling.multi_config_analysis(args)
