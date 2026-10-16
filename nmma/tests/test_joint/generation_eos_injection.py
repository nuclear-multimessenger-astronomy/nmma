import pickle
from argparse import Namespace
from pathlib import Path

import numpy as np

from nmma.core.parsing import parsing_and_logging
from nmma.em import em_parsing
from nmma.joint import generation

DATA_DIR = Path(__file__).parent.parent / "data"


def test_injection_with_tabulated_eos_and_no_constraints(tmp_path):
    """A tabulated EOS without EOS constraints must still convert the injection, so
    that the ejecta masses are derived from masses + EOS (here: BNS, Bu2026)."""
    em_args = parsing_and_logging((em_parsing.multi_wavelength_analysis_parser,), [])
    args = Namespace(**vars(em_args))
    for key, val in dict(
        label="injection",
        outdir=str(tmp_path / "outdir"),
        prior_file=f"{DATA_DIR}/Bu2026_simplified.prior",
        em_tmin=0.1,
        em_tmax=10.0,
        em_tstep=0.5,
        em_error_budget=0.1,
        filters="ztfr",
        em_transient_class="fiesta_kn",
        injection=True,
        sampler="dynesty",
        nlive=64,
        local_only=True,
        trigger_time=60000.0,
        eos_data=DATA_DIR / "eos_macro",
        eos_to_ram=True,
        Neos=10,
        injection_dict=dict(
            mass_1_source=1.4,
            mass_2_source=1.3,
            EOS=0,
            chi_1=0.0,
            chi_2=0.0,
            alpha=0.0,
            ratio_zeta=0.3,
            luminosity_distance=100.0,
            inclination_EM=0.0,
            timeshift=0.0,
            v_ej_dyn=0.2,
            Ye_dyn=0.25,
            v_ej_wind=0.06,
            Ye_wind=0.3,
        ),
    ).items():
        setattr(args, key, val)
    generation.generate_runner(**vars(args))

    dump_file = next((tmp_path / "outdir").rglob("*_data_dump.pickle"))
    with open(dump_file, "rb") as f:
        converted = pickle.load(f)["args"].injection_dict
    assert np.isfinite(converted["log10_mej_dyn"])
    assert np.isfinite(converted["log10_mej_wind"])
