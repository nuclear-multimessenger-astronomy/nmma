"""Test data and builders shared across the test modules.

Heavy objects (light curve surrogates, tabulated EOS sets) are built once per
session by the cached builders; tests that modify them must work on a copy.
"""

import json
from copy import copy
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from bilby.core.utils import BilbyJsonEncoder

from nmma.core.conversion import MassGrid

REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_DIR / "tests" / "data"
PRIOR_DIR = REPO_DIR / "priors"
EXAMPLE_DIR = REPO_DIR / "example_files"
EOS_MACRO_DIR = DATA_DIR / "eos_macro"
EOS_MICRO_DIR = DATA_DIR / "eos_micro"
N_EOS_MACRO = 10

KILONOVA_MODEL = "Bu2026_MLP"
KILONOVA_TRANSIENT_CLASS = "fiesta_kn"
KILONOVA_PRIOR = DATA_DIR / "Bu2026_simplified.prior"


def eos_files(n_eos: int = N_EOS_MACRO) -> list[Path]:
    "The tabulated test EOSs ``1.dat .. {n_eos}.dat``"
    return [EOS_MACRO_DIR / f"{j + 1}.dat" for j in range(n_eos)]


def read_eos_table(eos_file: Path, log_lambda: bool = True) -> np.ndarray:
    "(radii, masses, (ln-)lambdas) of a tabulated EOS file"
    table = np.loadtxt(eos_file, usecols=[0, 1, 2]).T
    if log_lambda:
        table[2] = np.log(table[2])
    return table


@cache
def eos_macro_tables(log_lambda: bool = True) -> tuple[np.ndarray, ...]:
    "The (radii, masses, (ln-)lambdas) tables of all test EOSs; read-only"
    tables = tuple(read_eos_table(f, log_lambda) for f in eos_files())
    for table in tables:
        table.flags.writeable = False
    return tables


@cache
def eos_macro_generator():
    """TabulatedEoSGenerator over all test EOSs on the default mass grid; it is not
    modified by conversions, so converters may share it."""
    from argparse import Namespace

    from nmma.eos.eos_processing import TabulatedEoSGenerator

    return TabulatedEoSGenerator(Namespace(eos_data=EOS_MACRO_DIR, Neos=N_EOS_MACRO))


class TableEmulator:
    """Stands in for a trained TOVEmulator: the first input column selects a test
    EOS, whose table is returned resampled onto the mass grid of the metadata, in
    the emulator output layout [radii | ln Lambdas | M_tov]."""

    def __init__(self, mass_grid: MassGrid, tables=None):
        self.metadata = {"mass_grid": mass_grid.config()}
        tables = eos_macro_tables() if tables is None else tables
        self.outputs = np.array(
            [
                np.concatenate(
                    [mass_grid.resample(mass, radius, ln_lam).ravel(), [mass[-1]]]
                )
                for radius, mass, ln_lam in tables
            ]
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.outputs[np.asarray(x)[:, 0].astype(int)]


@cache
def _loaded_kilonova_model(filters: tuple[str, ...]):
    from nmma.em.model import FiestaKilonovaModel

    return FiestaKilonovaModel(model=KILONOVA_MODEL, filters=list(filters))


def kilonova_model(filters: list[str]):
    """The Bu2026_MLP kilonova surrogate for ``filters``, loaded once per session;
    every call returns its own wrapper around the shared surrogate."""
    return copy(_loaded_kilonova_model(tuple(filters)))


def write_injection_file(path: Path, injections: pd.DataFrame) -> Path:
    """``injections`` in NMMA's json injection format, with a simulation_id."""
    if "simulation_id" not in injections:
        injections = injections.assign(simulation_id=range(len(injections)))
    content = {key: list(map(float, val)) for key, val in injections.items()}
    injection = {"injections": {"__dataframe__": True, "content": content}}
    path.write_text(json.dumps(injection, cls=BilbyJsonEncoder))
    return path
