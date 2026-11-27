"""Session-wide test setup and fixtures.

Points sncosmo and dustmaps at the vendored nmma-data submodule, to avoid hitting
the SVO / sncosmo CDN / Harvard dataverse when it is checked out
(``git submodule update --init``); without it, the upstream defaults apply.
Shared test data and builders live in ``helpers.py``.
"""

import os
from pathlib import Path

import helpers
import pytest

NMMA_DATA_DIR = Path(os.environ.get("NMMA_DATA", helpers.REPO_DIR / "nmma-data"))


def wire_sncosmo(sncosmo_dir: Path) -> None:
    import sncosmo

    sncosmo.conf.data_dir = str(sncosmo_dir)


def wire_dustmaps(dustmaps_dir: Path) -> None:
    try:
        from dustmaps.config import config
    except ImportError:
        return
    config["data_dir"] = str(dustmaps_dir)


if (NMMA_DATA_DIR / "sncosmo").is_dir():
    wire_sncosmo(NMMA_DATA_DIR / "sncosmo")
if (NMMA_DATA_DIR / "dustmaps").is_dir():
    wire_dustmaps(NMMA_DATA_DIR / "dustmaps")


@pytest.fixture(autouse=True, scope="class")
def class_outdir(request, tmp_path_factory):
    """An empty output directory per test class, as ``cls.outdir``; available in
    ``setup_class`` and removed by pytest, so no test writes into the repository."""
    if request.cls is not None:
        request.cls.outdir = tmp_path_factory.mktemp(request.cls.__name__)


@pytest.fixture(scope="session")
def eos_macro_generator():
    "TabulatedEoSGenerator over all test EOSs, see ``helpers.eos_macro_generator``"
    return helpers.eos_macro_generator()
