from argparse import Namespace

import numpy as np
import pytest

from nmma.core.base import NMMALikelihoodMixin
from nmma.core.conversion import (
    KilonovaEjectaFitting,
    max_mass_ratio_from_spin,
    mtov_to_mmax_spin,
)
from nmma.eos.eos_processing import EoSConverter

# toy EOS: columns (radius [km], mass [Msun], lambda), MTOV = 2.2
MASSES = np.linspace(1.0, 2.2, 13)
RADII = np.full_like(MASSES, 12.0)
LAMBDAS = np.linspace(1500.0, 5.0, 13)
MTOV = 2.2

# Breu & Rezzolla (2016) Eq. 3 with chi_Kep = 0.7, evaluated by hand:
# 1 + 0.1316 (chi/0.7)^2 + 0.07111 (chi/0.7)^4
RATIO_AT_0P5 = 1.085654  # chi = 0.5
RATIO_AT_0P35 = 1.037344  # chi = 0.35
RATIO_AT_KEP = 1.20271  # chi = 0.7


@pytest.fixture
def eos_file(tmp_path):
    path = tmp_path / "toy_eos.dat"
    np.savetxt(path, np.column_stack([RADII, MASSES, LAMBDAS]))
    return str(path)


def converted(eos_file, m1, m2, spins=None, **kwargs):
    converter = EoSConverter(Namespace(eos_file=eos_file, **kwargs), "tabulated")
    params = dict(EOS=0, mass_1_source=m1, mass_2_source=m2)
    params.update(spins or {})
    params = converter.compute_macro_parameters(params)
    return converter.system_props_from_eos(params)


# ---------------------------------------------------------------- spin relation
def test_ratio_known_values():
    assert max_mass_ratio_from_spin(0.0) == 1.0
    assert max_mass_ratio_from_spin(0.35) == pytest.approx(RATIO_AT_0P35, abs=1e-6)
    assert max_mass_ratio_from_spin(0.5) == pytest.approx(RATIO_AT_0P5, abs=1e-6)
    assert max_mass_ratio_from_spin(0.7) == pytest.approx(RATIO_AT_KEP, abs=1e-5)


def test_ratio_monotone_and_vectorised():
    chi = np.linspace(0.0, 0.7, 30)
    ratio = max_mass_ratio_from_spin(chi)
    assert ratio.shape == chi.shape
    assert np.all(np.diff(ratio) > 0)


@pytest.mark.parametrize("chi", [0.71, 0.9, 1.0, -0.1, np.nan, np.inf, -np.inf])
def test_ratio_falls_back_to_mtov_when_invalid(chi):
    assert max_mass_ratio_from_spin(chi) == 1.0
    assert mtov_to_mmax_spin(MTOV, chi) == MTOV


def test_ratio_invalid_entries_in_array_only_affect_themselves():
    out = max_mass_ratio_from_spin(np.array([0.5, np.nan, 0.9, 0.7]))
    np.testing.assert_allclose(out, [RATIO_AT_0P5, 1.0, 1.0, RATIO_AT_KEP], atol=1e-5)


def test_mmax_scales_with_mtov():
    assert mtov_to_mmax_spin(2.2, 0.5) == pytest.approx(2.2 * RATIO_AT_0P5, abs=1e-5)
    assert mtov_to_mmax_spin(3.0, 0.5) == pytest.approx(3.0 * RATIO_AT_0P5, abs=1e-5)


# ------------------------------------------------------------------- converter
def test_force_bns_default_off(eos_file):
    out = converted(eos_file, 2.3, 1.2, spins=dict(chi_1=0.5))
    assert "force_bns_rejected" not in out
    assert out["lambda_1"] == 0.0 and out["radius_1"] == 0.0  # unchanged old behaviour


def test_force_bns_below_mtov_matches_default_and_not_rejected(eos_file):
    ref = converted(eos_file, 1.4, 1.2)
    out = converted(eos_file, 1.4, 1.2, force_bns=True)
    assert out["lambda_1"] == pytest.approx(ref["lambda_1"])
    assert out["radius_1"] == pytest.approx(ref["radius_1"])
    assert out["lambda_2"] == pytest.approx(ref["lambda_2"])
    assert out["force_bns_rejected"] == 0.0


def test_force_bns_above_mtov_accepted_when_spin_supports_it(eos_file):
    # 2.3 > MTOV = 2.2 but < 2.2 * 1.0857 = 2.388 at chi = 0.5
    out = converted(eos_file, 2.3, 1.2, spins=dict(chi_1=0.5), force_bns=True)
    assert out["force_bns_rejected"] == 0.0
    assert out["mmax_spin_1"] == pytest.approx(MTOV * RATIO_AT_0P5, abs=1e-4)
    assert out["mmax_spin_2"] == pytest.approx(MTOV)  # no spin given for the secondary
    # BNS routing needs radius_1 > 0; the table is evaluated at its top end (MTOV)
    assert out["radius_1"] == pytest.approx(12.0)
    assert out["lambda_1"] == pytest.approx(LAMBDAS[-1])


@pytest.mark.parametrize(
    "m1, spins, rejected",
    [
        (2.3, {}, 1.0),  # above MTOV, no spin
        (2.3, dict(chi_1=0.0), 1.0),
        (2.39, dict(chi_1=0.5), 1.0),  # just above 2.388
        (2.38, dict(chi_1=0.5), 0.0),  # just below
        (2.64, dict(chi_1=0.7), 0.0),  # 2.2 * 1.2027 = 2.6459
        (2.66, dict(chi_1=0.7), 1.0),
        (2.3, dict(chi_1=0.9), 1.0),  # chi beyond Kepler -> fallback MTOV -> rejected
        (2.3, dict(chi_1=np.nan), 1.0),
        (2.2, {}, 0.0),  # exactly MTOV is fine
    ],
)
def test_force_bns_rejection_threshold(eos_file, m1, spins, rejected):
    out = converted(eos_file, m1, 1.2, spins=spins, force_bns=True)
    assert out["force_bns_rejected"] == rejected


def test_force_bns_spin_key_priority_and_sign(eos_file):
    # a_1 (magnitude) wins over chi_1; negative aligned chi_1 uses |chi_1|
    by_a = converted(eos_file, 2.3, 1.2, spins=dict(a_1=0.5, chi_1=0.0), force_bns=True)
    assert by_a["mmax_spin_1"] == pytest.approx(MTOV * RATIO_AT_0P5, abs=1e-4)
    neg = converted(eos_file, 2.3, 1.2, spins=dict(chi_1=-0.5), force_bns=True)
    assert neg["mmax_spin_1"] == pytest.approx(MTOV * RATIO_AT_0P5, abs=1e-4)
    z = converted(eos_file, 2.3, 1.2, spins=dict(spin_1z=0.5), force_bns=True)
    assert z["mmax_spin_1"] == pytest.approx(MTOV * RATIO_AT_0P5, abs=1e-4)


def test_force_bns_checks_secondary_too(eos_file):
    # both heavy: m2 = 2.3 > MTOV with chi_2 = 0 -> rejected, with chi_2 = 0.5 -> fine
    spins = dict(chi_1=0.5, chi_2=0.0)
    assert (
        converted(eos_file, 2.35, 2.3, spins, force_bns=True)["force_bns_rejected"]
        == 1.0
    )
    spins = dict(chi_1=0.5, chi_2=0.5)
    assert (
        converted(eos_file, 2.35, 2.3, spins, force_bns=True)["force_bns_rejected"]
        == 0.0
    )


def test_force_bns_rejects_masses_below_the_eos_table(eos_file):
    # secondary below the tabulated minimum (1.0): radius_2 = 0 cannot be a NS here
    out = converted(eos_file, 1.4, 0.8, force_bns=True)
    assert out["force_bns_rejected"] == 1.0


def test_force_bns_multiple_eos(eos_file):
    converter = EoSConverter(Namespace(eos_file=eos_file, force_bns=True), "tabulated")
    converter.macro_parameters = {
        "radii": [RADII, RADII],
        "masses": [MASSES, MASSES],
        "lambdas": [LAMBDAS, LAMBDAS],
    }
    out = converter.system_props_from_eos(
        dict(
            mass_1_source=np.array([2.3, 2.3]),
            mass_2_source=np.array([1.2, 1.1]),
            chi_1=np.array([0.5, 0.0]),
        )
    )
    np.testing.assert_array_equal(out["force_bns_rejected"], [0.0, 1.0])
    np.testing.assert_allclose(
        out["mmax_spin_1"], [MTOV * RATIO_AT_0P5, MTOV], atol=1e-4
    )
    assert np.all(np.asarray(out["radius_1"]) > 0)


def test_force_bns_and_force_nsbh_are_exclusive(eos_file):
    with pytest.raises(ValueError):
        EoSConverter(
            Namespace(eos_file=eos_file, force_bns=True, force_nsbh=True), "tabulated"
        )


def test_force_bns_selects_bns_ejecta_above_mtov(eos_file):
    """Above MTOV the default routes to the NSBH fit; force_bns keeps the BNS fit."""
    fitter = KilonovaEjectaFitting()
    extra = dict(
        alpha=np.array([0.04]),
        ratio_zeta=np.array([0.5]),
        TOV_mass=np.array([MTOV]),
        R_16=np.array([12.0]),
    )

    def system(**eos_kwargs):
        out = converted(eos_file, 2.3, 1.2, spins=dict(chi_1=0.5), **eos_kwargs)
        return extra | {
            k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in out.items()
        }

    default, forced = system(), system(force_bns=True)
    np.testing.assert_allclose(
        fitter.ejecta_parameter_conversion(default),
        fitter.nsbh_parameter_conversion(default),
    )
    np.testing.assert_allclose(
        fitter.ejecta_parameter_conversion(forced),
        fitter.bns_parameter_conversion(forced),
        equal_nan=True,
    )
    assert np.all(np.isfinite(fitter.ejecta_parameter_conversion(forced)[:3]))


# ------------------------------------------------------------------ likelihood
class _ToyLikelihood(NMMALikelihoodMixin):
    def __init__(self, rejected):
        super().__init__()
        self.rejected = rejected
        self._constraints = {}

    def parameter_conversion(self, parameters):
        return dict(parameters, force_bns_rejected=self.rejected)

    def sub_log_likelihood(self, parameters):
        return -1.0


@pytest.mark.parametrize(
    "rejected, expected", [(0.0, -1.0), (1.0, np.nan_to_num(-np.inf))]
)
def test_rejected_samples_get_minus_infinity(rejected, expected):
    assert _ToyLikelihood(rejected).log_likelihood({}) == expected
