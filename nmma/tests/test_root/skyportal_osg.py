"""Tests for the SkyPortal/OSG bridge: argv assembly + happy/failure paths.

The actual NMMA invocation is patched out — we only verify the bridge
translates SkyPortal-shaped payloads into the right NMMA argv and packages
the produced files correctly.
"""

import json
from pathlib import Path

import pytest

from nmma import skyportal_osg


# ---- helpers ----------------------------------------------------------------------------------

CSV_PHOT = (
    "mjd,filter,mag,magerr\n"
    "59000.1,ztfg,18.4,0.05\n"
    "59001.2,ztfr,18.7,0.06\n"
    "59002.3,ztfg,19.0,0.07\n"
)
CSV_Z = "redshift\n0.123\n"


def _write_payload(tmp_path: Path, params=None) -> dict:
    phot = tmp_path / "phot.csv"
    z = tmp_path / "z.csv"
    phot.write_text(CSV_PHOT)
    z.write_text(CSV_Z)
    return {
        "photometry": str(phot),
        "redshift": str(z),
        "analysis_parameters": params or {},
    }


def _stub_prior_dir(tmp_path: Path, source: str) -> Path:
    """Make a tmp priors dir with a trivial bilby-loadable prior file."""
    prior_dir = tmp_path / "priors"
    prior_dir.mkdir()
    # Minimal bilby PriorDict needs at least one Uniform-like entry.
    (prior_dir / f"{source}.prior").write_text(
        "luminosity_distance = Uniform(minimum=10.0, maximum=1000.0, "
        "name='luminosity_distance', latex_label='$d_L$', unit='Mpc')\n"
    )
    return prior_dir


# ---- tests ------------------------------------------------------------------------------------


def test_defaults_merge_only_overrides():
    p = skyportal_osg._params(
        {"analysis_parameters": {"source": "Piro2021", "nlive": 64}}
    )
    assert p["source"] == "Piro2021"
    assert p["nlive"] == 64
    # Untouched defaults survive.
    assert p["sampler"] == skyportal_osg.DEFAULTS["sampler"]
    assert p["tmin"] == skyportal_osg.DEFAULTS["tmin"]


def test_resolve_redshift_handles_missing_table(tmp_path):
    assert skyportal_osg._resolve_redshift({}) is None


def test_resolve_redshift_reads_csv(tmp_path):
    p = tmp_path / "z.csv"
    p.write_text("redshift\n0.42\n")
    assert skyportal_osg._resolve_redshift({"redshift": str(p)}) == pytest.approx(0.42)


def test_write_data_file_format(tmp_path):
    payload = _write_payload(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    data_path, t0 = skyportal_osg._write_data_file(payload, out)
    rows = data_path.read_text().splitlines()
    assert len(rows) == 3
    # Each row: ISO_time band mag magerr
    first = rows[0].split()
    assert len(first) == 4
    assert first[1] in {"g", "r"}  # single-letter band
    assert t0 == pytest.approx(59000.1)


def test_build_argv_contains_all_required_flags(tmp_path):
    payload = _write_payload(tmp_path, params={"source": "Piro2021", "nlive": 128})
    argv = skyportal_osg.build_argv(
        payload,
        label="src_Piro2021",
        outdir=tmp_path,
        prior_path=tmp_path / "Piro2021.prior",
        data_path=tmp_path / "data.dat",
        svdmodel_dir=Path("svdmodels"),
        trigger_time=59000.1,
    )
    # argv comes as flag/value pairs (plus a lone --plot at the end).
    pairs = dict(zip(argv[::2], argv[1::2], strict=False))
    assert pairs["--model"] == "Piro2021"
    assert pairs["--label"] == "src_Piro2021"
    assert pairs["--nlive"] == "128"
    assert pairs["--sampler"] == skyportal_osg.DEFAULTS["sampler"]
    assert "--plot" in argv


def test_run_from_skyportal_inputs_happy_path(tmp_path, monkeypatch):
    # Stub out the bits that talk to the real world.
    prior_dir = _stub_prior_dir(tmp_path, "Me2017")
    outdir = tmp_path / "out"
    payload = _write_payload(tmp_path, params={"fix_z": False})

    captured: dict = {}

    def fake_invoke(args=None):
        captured["argv"] = list(args)
        # Pretend NMMA produced the expected outputs.
        label = "ZTF1_Me2017"
        (outdir / f"{label}_posterior_samples.dat").write_text("samples\n")
        (outdir / f"{label}_result.json").write_text(
            json.dumps({"log_bayes_factor": 1.5})
        )
        (outdir / "lightcurves.png").write_bytes(b"\x89PNG")

    result = skyportal_osg.run_from_skyportal_inputs(
        payload,
        outdir=outdir,
        prior_dir=prior_dir,
        svdmodel_dir=tmp_path / "svdmodels",
        resource_id="ZTF1",
        invoke=fake_invoke,
    )
    assert result["status"] == "success"
    assert result["log_bayes_factor"] == 1.5
    assert Path(result["posterior_file"]).exists()
    assert Path(result["plot_file"]).exists()
    # The argv really did reach the invoker.
    assert "--model" in captured["argv"]
    assert "Me2017" in captured["argv"]


def test_run_from_skyportal_inputs_failure_when_no_posterior(tmp_path):
    prior_dir = _stub_prior_dir(tmp_path, "Me2017")
    payload = _write_payload(tmp_path)
    result = skyportal_osg.run_from_skyportal_inputs(
        payload,
        outdir=tmp_path / "out",
        prior_dir=prior_dir,
        svdmodel_dir=tmp_path / "svdmodels",
        resource_id="ZTF1",
        invoke=lambda args=None: None,  # NMMA "ran" but produced nothing
    )
    assert result["status"] == "failure"
    assert "did not produce" in result["message"]


def test_run_fix_z_without_redshift_raises(tmp_path):
    prior_dir = _stub_prior_dir(tmp_path, "Me2017")
    payload = {
        "photometry": str(tmp_path / "phot.csv"),
        "analysis_parameters": {"fix_z": True},
    }
    (tmp_path / "phot.csv").write_text(CSV_PHOT)
    with pytest.raises(ValueError, match="fix_z=True requires"):
        skyportal_osg.run_from_skyportal_inputs(
            payload,
            outdir=tmp_path / "out",
            prior_dir=prior_dir,
            svdmodel_dir=tmp_path / "svdmodels",
            invoke=lambda args=None: None,
        )


def test_missing_prior_file_raises(tmp_path):
    # No Me2017.prior in this dir.
    prior_dir = tmp_path / "empty_priors"
    prior_dir.mkdir()
    payload = _write_payload(tmp_path)
    with pytest.raises(FileNotFoundError, match="prior file for model 'Me2017'"):
        skyportal_osg.run_from_skyportal_inputs(
            payload,
            outdir=tmp_path / "out",
            prior_dir=prior_dir,
            svdmodel_dir=tmp_path / "svdmodels",
            invoke=lambda args=None: None,
        )
