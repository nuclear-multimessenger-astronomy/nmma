"""
SkyPortal-AnalysisService → NMMA bridge.

Takes a SkyPortal analysis-service payload (photometry + redshift + free-form
``analysis_parameters``), assembles the NMMA argv, and invokes
:func:`nmma.em.analysis.main`. Returns a small dict pointing at the produced
posterior / plot / json-result files so callers (e.g. the
``osg-skyportal-plugin`` wrapper) can package them however they like.

This is a thin port of the analysis service in skyportal/skyportal#3199; the
NMMA-side logic lives here so plugins don't re-implement it.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from astropy.time import Time

# Knobs that mirror the legacy analysis service. Override per-call via
# ``analysis_parameters`` in the SkyPortal payload.
DEFAULTS = {
    "source": "Me2017",
    "nlive": 32,
    "tmin": 0.01,
    "tmax": 7.0,
    "dt": 0.1,
    "Ebv_max": 0.5724,
    "error_budget": 1.0,
    "interpolation_type": "sklearn_gp",
    "sampler": "dynesty",
}


def _params(payload: dict) -> dict:
    return {**DEFAULTS, **(payload.get("analysis_parameters") or {})}


def _resolve_redshift(payload: dict) -> float | None:
    src = payload.get("redshift")
    if src is None:
        return None
    # Lazy import: keeps test imports cheap.
    from astropy.table import Table

    table = Table.read(src, format="ascii.csv")
    if len(table) == 0 or "redshift" not in table.colnames:
        return None
    return float(table["redshift"][0])


def _prepare_prior(
    prior_dir: Path,
    source: str,
    fix_z: bool,
    redshift: float | None,
    outdir: Path,
) -> Path:
    """Locate the prior file and pin luminosity_distance when fix_z is set."""
    import bilby
    from astropy.cosmology import Planck18 as cosmo

    candidate = prior_dir / f"{source}.prior"
    if not candidate.exists():
        raise FileNotFoundError(
            f"prior file for model {source!r} not found at {candidate}"
        )
    priors = bilby.gw.prior.PriorDict(filename=str(candidate))
    if fix_z:
        if redshift is None:
            raise ValueError("fix_z=True requires a redshift in the SkyPortal payload")
        priors["luminosity_distance"] = float(cosmo.luminosity_distance(redshift).value)
    priors.to_file(outdir=str(outdir), label=source)
    return outdir / f"{source}.prior"


def _write_data_file(payload: dict, outdir: Path) -> tuple[Path, float]:
    """Convert SkyPortal photometry to NMMA's `time filter mag magerr` format."""
    from astropy.table import Table

    table = Table.read(payload["photometry"], format="ascii.csv")
    data_path = outdir / "data.dat"
    with data_path.open("w") as fh:
        for row in table:
            iso = Time(row["mjd"], format="mjd").isot
            filt = str(row["filter"])[-1]
            fh.write(f"{iso} {filt} {row['mag']} {row['magerr']}\n")
    t0 = float(np.min(table["mjd"]))
    return data_path, t0


def build_argv(
    payload: dict,
    *,
    label: str,
    outdir: Path,
    prior_path: Path,
    data_path: Path,
    svdmodel_dir: Path,
    trigger_time: float,
) -> list[str]:
    """Assemble the argv list NMMA's ``main(args=...)`` consumes."""
    p = _params(payload)
    return [
        "--model",
        str(p["source"]),
        "--svd-path",
        str(svdmodel_dir),
        "--outdir",
        str(outdir),
        "--label",
        label,
        "--trigger-time",
        str(trigger_time),
        "--data",
        str(data_path),
        "--prior",
        str(prior_path),
        "--tmin",
        str(p["tmin"]),
        "--tmax",
        str(p["tmax"]),
        "--dt",
        str(p["dt"]),
        "--error-budget",
        str(p["error_budget"]),
        "--nlive",
        str(p["nlive"]),
        "--Ebv-max",
        str(p["Ebv_max"]),
        "--interpolation-type",
        str(p["interpolation_type"]),
        "--sampler",
        str(p["sampler"]),
        "--plot",
    ]


def run_from_skyportal_inputs(
    payload: dict[str, Any],
    *,
    outdir: Path | None = None,
    prior_dir: Path | None = None,
    svdmodel_dir: Path | None = None,
    resource_id: str = "obj",
    invoke: Any | None = None,
) -> dict[str, Any]:
    """Run NMMA's EM analysis against a SkyPortal AnalysisService payload.

    Parameters
    ----------
    payload:
        The ``inputs`` dict the SkyPortal AnalysisService POSTs. Expected keys:
        ``photometry`` (CSV path/URL), ``redshift`` (CSV path/URL, optional),
        ``analysis_parameters`` (dict, optional; see ``DEFAULTS``).
    outdir, prior_dir, svdmodel_dir:
        Override default paths. ``outdir`` defaults to a fresh ``mkdtemp``;
        ``prior_dir`` defaults to NMMA's vendored priors directory;
        ``svdmodel_dir`` defaults to ``./svdmodels`` (NMMA downloads on demand).
    invoke:
        Hook for testing — defaults to :func:`nmma.em.analysis.main`. Tests
        pass a callable that records its ``args=`` kwarg.

    Returns
    -------
    dict
        Always contains ``status`` (``"success"`` or ``"failure"``) and
        ``message``. On success also: ``posterior_file``, ``json_result_file``,
        ``plot_file`` (may be ``None``), ``log_bayes_factor``, ``outdir``.
    """
    params = _params(payload)
    source = str(params["source"])
    fix_z = params.get("fix_z") in (True, "True", "true", "t", 1)

    if outdir is None:
        outdir = Path(tempfile.mkdtemp(prefix="nmma_osg_"))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if prior_dir is None:
        prior_dir = Path(__file__).resolve().parents[1] / "priors"
    if svdmodel_dir is None:
        svdmodel_dir = Path("svdmodels")

    redshift = _resolve_redshift(payload)
    prior_path = _prepare_prior(Path(prior_dir), source, fix_z, redshift, outdir)
    data_path, trigger_time = _write_data_file(payload, outdir)

    label = f"{resource_id}_{source}"
    argv = build_argv(
        payload,
        label=label,
        outdir=outdir,
        prior_path=prior_path,
        data_path=data_path,
        svdmodel_dir=Path(svdmodel_dir),
        trigger_time=trigger_time,
    )

    if invoke is None:
        from nmma.em.analysis import main as invoke
    invoke(args=argv)

    posterior_file = outdir / f"{label}_posterior_samples.dat"
    json_file = outdir / f"{label}_result.json"
    plot_file = outdir / "lightcurves.png"

    if not posterior_file.exists():
        return {
            "status": "failure",
            "message": f"NMMA fit did not produce {posterior_file.name}",
            "outdir": str(outdir),
        }

    log_bayes_factor: float | None = None
    if json_file.exists():
        with json_file.open() as fh:
            meta = json.load(fh)
        log_bayes_factor = meta.get("log_bayes_factor")

    return {
        "status": "success",
        "message": f"NMMA fit complete (log Bayes factor={log_bayes_factor})",
        "posterior_file": str(posterior_file),
        "json_result_file": str(json_file) if json_file.exists() else None,
        "plot_file": str(plot_file) if plot_file.exists() else None,
        "log_bayes_factor": log_bayes_factor,
        "outdir": str(outdir),
    }
