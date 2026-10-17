from argparse import Namespace

import numpy as np
import pytest

from nmma.core.conversion import KilonovaEjectaFitting
from nmma.eos.eos_processing import EoSConverter

# toy EOS: columns (radius [km], mass [Msun], lambda), MTOV = 2.2
MASSES = np.linspace(1.0, 2.2, 13)
RADII = np.full_like(MASSES, 12.0)
LAMBDAS = np.linspace(1500.0, 5.0, 13)


@pytest.fixture
def eos_file(tmp_path):
    path = tmp_path / "toy_eos.dat"
    np.savetxt(path, np.column_stack([RADII, MASSES, LAMBDAS]))
    return str(path)


def converted(eos_file, m1, m2, **kwargs):
    converter = EoSConverter(Namespace(eos_file=eos_file, **kwargs), "tabulated")
    params = dict(EOS=0, mass_1_source=m1, mass_2_source=m2)
    params = converter.compute_macro_parameters(params)
    return converter.system_props_from_eos(params)


def test_default_below_mtov_is_bns(eos_file):
    out = converted(eos_file, 1.4, 1.2)
    assert out["radius_1"] == pytest.approx(12.0)
    assert out["radius_2"] == pytest.approx(12.0)
    assert out["lambda_1"] == pytest.approx(np.exp(np.interp(1.4, MASSES, np.log(LAMBDAS))))
    assert out["lambda_1"] > 0


def test_default_above_mtov_gives_zero_lambda_and_radius(eos_file):
    out = converted(eos_file, 2.5, 1.2)
    assert out["lambda_1"] == 0.0
    assert out["radius_1"] == 0.0
    assert out["radius_2"] == pytest.approx(12.0)


@pytest.mark.parametrize("m1", [1.4, 2.5])
def test_force_nsbh_zeroes_primary_only(eos_file, m1):
    reference = converted(eos_file, m1, 1.2)
    out = converted(eos_file, m1, 1.2, force_nsbh=True)
    assert out["lambda_1"] == 0.0
    assert out["radius_1"] == 0.0
    assert out["lambda_2"] == pytest.approx(reference["lambda_2"])
    assert out["radius_2"] == pytest.approx(reference["radius_2"])
    assert out["lambda_2"] > 0


@pytest.mark.parametrize("force", [False, True])
def test_scalar_inputs_give_python_scalars(eos_file, force):
    """lalsimulation's SWIG setters reject 0-d arrays (TypeError ... REAL8): the
    single-messenger (GW only) path does not run val_to_scalar after the conversion"""
    out = converted(eos_file, 1.4, 1.2, force_nsbh=force)
    for key in ["lambda_1", "lambda_2", "radius_1", "radius_2"]:
        assert isinstance(out[key], float), (key, type(out[key]))
    import lalsimulation
    import lal

    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(
        lal.CreateDict(), out["lambda_1"]
    )


def test_force_nsbh_false_is_default(eos_file):
    default = converted(eos_file, 1.4, 1.2)
    explicit = converted(eos_file, 1.4, 1.2, force_nsbh=False)
    assert explicit["lambda_1"] == default["lambda_1"]
    assert explicit["radius_1"] == default["radius_1"]


def test_force_nsbh_multiple_eos(eos_file):
    """list-of-EOS branch: arrays of two samples with the same toy EOS"""
    converter = EoSConverter(Namespace(eos_file=eos_file, force_nsbh=True), "tabulated")
    converter.macro_parameters = {
        "radii": [RADII, RADII],
        "masses": [MASSES, MASSES],
        "lambdas": [LAMBDAS, LAMBDAS],
    }
    out = converter.system_props_from_eos(
        dict(mass_1_source=[1.4, 1.5], mass_2_source=[1.2, 1.1])
    )
    np.testing.assert_array_equal(out["lambda_1"], [0.0, 0.0])
    np.testing.assert_array_equal(out["radius_1"], [0.0, 0.0])
    assert np.all(np.asarray(out["radius_2"]) == pytest.approx(12.0))


def test_force_nsbh_selects_nsbh_ejecta(eos_file):
    """Below MTOV the default is the BNS fit; forced NSBH must give the NSBH fit."""
    fitter = KilonovaEjectaFitting()
    extra = dict(
        alpha=np.array([0.04]),
        ratio_zeta=np.array([0.5]),
        chi_1=np.array([0.5]),
        TOV_mass=np.array([2.2]),
        R_16=np.array([12.0]),
    )

    def system(**eos_kwargs):
        out = converted(eos_file, 1.6, 1.2, **eos_kwargs)
        return extra | {
            k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in out.items()
        }

    default, forced = system(), system(force_nsbh=True)
    np.testing.assert_allclose(
        fitter.ejecta_parameter_conversion(forced),
        fitter.nsbh_parameter_conversion(forced),
    )
    np.testing.assert_allclose(
        fitter.ejecta_parameter_conversion(default),
        fitter.bns_parameter_conversion(default),
    )
    assert not np.allclose(
        fitter.ejecta_parameter_conversion(default),
        fitter.ejecta_parameter_conversion(forced),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "m1, rejected", [(1.4, 1.0), (2.2, 1.0), (2.21, 0.0), (2.5, 0.0)]
)
def test_force_nsbh_nonoverlapping_rejection_threshold(eos_file, m1, rejected):
    """BH must be strictly above MTOV (2.2); lambda_1 = radius_1 = 0 as force_nsbh."""
    out = converted(eos_file, m1, 1.2, force_nsbh_nonoverlapping=True)
    assert out["force_nsbh_rejected"] == rejected
    assert out["lambda_1"] == 0.0
    assert out["radius_1"] == 0.0
    assert out["radius_2"] == pytest.approx(12.0)


def test_force_nsbh_nonoverlapping_default_off(eos_file):
    out = converted(eos_file, 1.4, 1.2, force_nsbh=True)
    assert "force_nsbh_rejected" not in out


def test_force_nsbh_nonoverlapping_multiple_eos(eos_file):
    converter = EoSConverter(
        Namespace(eos_file=eos_file, force_nsbh_nonoverlapping=True), "tabulated"
    )
    converter.macro_parameters = {
        "radii": [RADII, RADII],
        "masses": [MASSES, MASSES],
        "lambdas": [LAMBDAS, LAMBDAS],
    }
    out = converter.system_props_from_eos(
        dict(mass_1_source=[1.4, 2.5], mass_2_source=[1.2, 1.1])
    )
    np.testing.assert_array_equal(out["force_nsbh_rejected"], [1.0, 0.0])


def test_force_nsbh_nonoverlapping_exclusive_with_bns(eos_file):
    with pytest.raises(ValueError):
        EoSConverter(
            Namespace(
                eos_file=eos_file, force_nsbh_nonoverlapping=True, force_bns=True
            ),
            "tabulated",
        )
