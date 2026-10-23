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
OUTDIR = "outdir_injection_with_data"
T0 = 60000.0


def generated_light_curve(tmp_path, data_file=None):
    """nmma-generation (EM only) with an injection, and `--light-curve-data` if given."""
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
        injection=True,
        sampler="dynesty",
        nlive=64,
        local_only=True,
        ejecta_conversion=False,
        trigger_time=T0,
    ).items():
        setattr(args, key, val)
    args.injection_dict = utils.injection_from_prior(args)
    if data_file is not None:
        args.light_curve_data = str(data_file)
    generation.generate_runner(**vars(args))
    dump_file = next((tmp_path / OUTDIR).rglob("*_data_dump.pickle"))
    with open(dump_file, "rb") as f:
        return pickle.load(f)["light_curve_data"]


@pytest.fixture(autouse=True)
def cleanup():
    yield
    shutil.rmtree(OUTDIR, ignore_errors=True)


def test_injection_uses_the_light_curve_file_instead_of_simulating(tmp_path):
    days = np.array([0.5, 1.0, 2.0, 4.0])
    mags = np.array([20.0, 20.5, 21.0, 22.0])
    data_file = tmp_path / "lightcurve.dat"
    with open(data_file, "w") as f:
        for d, m in zip(days, mags):
            f.write(f"{Time(T0 + d, format='mjd').isot} ztfr {m:.3f} 0.1\n")

    simulated = generated_light_curve(tmp_path)
    assert len(simulated["ztfr"]["time"]) != len(days)  # the simulation has its own grid

    data = generated_light_curve(tmp_path, data_file)
    np.testing.assert_allclose(data["ztfr"]["time"] - T0, days, atol=1e-6)
    np.testing.assert_allclose(data["ztfr"]["mag"], mags)
