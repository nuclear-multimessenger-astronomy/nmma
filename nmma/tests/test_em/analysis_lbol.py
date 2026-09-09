from pathlib import Path
import shutil
import toml
import pytest


from nmma.em import analysis, em_parsing

# Uses the analytic Arnett_modified model but fails in CI with the same
# numpy "zero-size array to reduction" cascade as the SVD tests. Skipping
# here so CI is green; needs a focused investigation separate from the
# fiesta migration.
# pytestmark = pytest.mark.skip(reason="lbol test failing in CI; tracked separately")

dataDir = Path(__file__).parent.parent / "data"
non_default_file = dataDir / "config.toml"


@pytest.fixture(autouse=True)
def cleanup_outdir(args):
    yield
    if Path(args.outdir).exists():
        shutil.rmtree(args.outdir, ignore_errors=True)
    non_default_file.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def args():

    non_default_args = dict(
        em_model="Arnett_modified",
        outdir="outdir",
        label="lbol_test",
        trigger_time=0,  # 60168.79041667,
        data="example_files/lbol/ztf23bqun/23bqun_bbdata.csv",
        prior_file="example_files/lbol/ztf23bqun/Arnett_modified.priors",
        bestfit=True,
        error_budget=0.0001,
        nlive=64,
        plot=True,
    )
    non_default_args = {k.replace("_", "-"): v for k, v in non_default_args.items()}
    with open(non_default_file, "w") as f:
        toml.dump(non_default_args, f)

    args = em_parsing.parsing_and_logging(
        em_parsing.bolometric_parser, [str(non_default_file)]
    )
    args.__dict__.update(non_default_args)
    return args


def test_analysis_lbol(args):

    analysis.lbol_main(args)
