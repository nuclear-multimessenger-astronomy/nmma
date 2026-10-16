import pickle
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from astropy.time import Time

from nmma.core import utils
from nmma.core.parsing import parsing_and_logging
from nmma.em import em_parsing
from nmma.joint import generation

DATA_DIR = Path(__file__).parent.parent / "data"
OUTDIR = "outdir_data_tmin"


def generated_data(tmp_path, data_file=None, **kwargs):
    """Run nmma-generation (EM only) on a mock injection, or on `data_file` if given.

    Returns (light_curve_data, trigger MJD).
    """
    em_args = parsing_and_logging((em_parsing.multi_wavelength_analysis_parser,), [])
    args = Namespace(**vars(em_args))
    for key, val in dict(
        label="injection",
        outdir=str(tmp_path / OUTDIR),
        prior_file=f"{DATA_DIR}/Bu2026_simplified.prior",
        em_tmin=0.1,
        em_tmax=10.0,
        injection_em_tmax=9.0,
        em_tstep=0.5,
        em_error_budget=0.1,
        filters="ztfr",
        em_transient_class="fiesta_kn",
        injection=data_file is None,
        sampler="dynesty",
        nlive=64,
        local_only=True,
        ejecta_conversion=False,
        trigger_time=60000.0,
        **kwargs,
    ).items():
        setattr(args, key, val)
    if data_file is None:
        args.injection_dict = utils.injection_from_prior(args)
    else:
        args.light_curve_data = str(data_file)
    generation.generate_runner(**vars(args))
    dump_file = next((tmp_path / OUTDIR).rglob("*_data_dump.pickle"))
    with open(dump_file, "rb") as f:
        data_dump = pickle.load(f)
    return data_dump["light_curve_data"], 60000.0


@pytest.fixture(autouse=True)
def cleanup():
    yield
    shutil.rmtree(OUTDIR, ignore_errors=True)


def test_data_tmin_tmax_applied_in_generation(tmp_path):
    full, t0 = generated_data(tmp_path)
    times = np.concatenate([d["time"] for d in full.values()]) - t0
    assert times.min() < 1.5 and times.max() > 6.0  # something to cut

    cut, t0 = generated_data(tmp_path, data_tmin=1.5, data_tmax=6.0)
    times = np.concatenate([d["time"] for d in cut.values()]) - t0
    assert len(times) > 0
    assert times.min() >= 1.5 and times.max() <= 6.0


def test_data_tmin_tmax_applied_to_observation_file(tmp_path):
    data_file = tmp_path / "lightcurve.dat"
    days = np.arange(0.25, 10.0, 0.5)
    mjd = 60000.0 + days
    with open(data_file, "w") as f:
        for t in mjd:
            iso = Time(t, format="mjd").isot
            f.write(f"{iso} ztfr {20.0 + 0.1 * t % 1:.3f} 0.1\n")

    cut, t0 = generated_data(
        tmp_path, data_file=data_file, data_tmin=1.5, data_tmax=6.0
    )
    times = cut["ztfr"]["time"] - t0
    assert len(times) > 0
    assert times.min() >= 1.5 and times.max() <= 6.0
    assert len(times) < len(days)
