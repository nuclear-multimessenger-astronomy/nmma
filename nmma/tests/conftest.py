"""Point sncosmo and dustmaps at the vendored nmma-data submodule.

Avoids hitting the SVO / sncosmo CDN / Harvard dataverse during tests when
the submodule is checked out (``git submodule update --init``). No-op when
the directory is absent — tests fall back to the upstream defaults.
"""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NMMA_DATA = Path(os.environ.get("NMMA_DATA", _REPO_ROOT / "nmma-data"))


def _wire_sncosmo(sncosmo_dir: Path) -> None:
    import sncosmo

    sncosmo.conf.data_dir = str(sncosmo_dir)


def _wire_dustmaps(dustmaps_dir: Path) -> None:
    try:
        from dustmaps.config import config
    except ImportError:
        return
    config["data_dir"] = str(dustmaps_dir)


if _NMMA_DATA.is_dir():
    sncosmo_dir = _NMMA_DATA / "sncosmo"
    if sncosmo_dir.is_dir():
        _wire_sncosmo(sncosmo_dir)
    dustmaps_dir = _NMMA_DATA / "dustmaps"
    if dustmaps_dir.is_dir():
        _wire_dustmaps(dustmaps_dir)
