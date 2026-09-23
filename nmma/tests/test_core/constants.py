from argparse import Namespace

import pytest
from astropy import cosmology
from astropy import units as u
from bilby.gw import cosmology as bilby_cosmo

from nmma.core import constants


class TestFundamentalConstants:
    """Test the values of fundamental constants have not changed."""

    # snapshots from 15/09/2026
    constants = Namespace(
        mc2=u.Quantity(1.7870936687412792e47, "m2 kg / s2"),
        G_per_c2=u.Quantity(7.4261602691186655e-28, "m / kg"),
        seconds_a_day=86400,
        msun_cgs=1.988409870698051e33,
        c_cgs=29979245800.0,
        c_SI=299792458.0,
        c_kms=299792.458,
        G_in_ns_units=132712440000.0,
        h=6.62607015e-27,
        kb=1.380649e-16,
        Mpc=3.0856775814913676e24,
        D=3.0856775814913675e19,
        sigSB=5.6703744191844314e-05,
        arad=7.565733250280007e-15,
        eV_per_h_SI=241798924208491.8,
        particle_mass=u.Quantity(8.411856872862986e-58),
        geom_msun_km=1.476625038050125,
        msun_to_ergs=1.787093668741279e54,
        MeV_per_fm3_to_Msun_per_km3=8.965263892006713e-07,
        msun_s=4.9254909476412675e-06,
        msun_mus=4.925490947641268,
        einstein_factor=0.0002894896337539025,
    )
    cosmology_params = Namespace(
        H0=u.Quantity(67.66, "km / (Mpc s)"),
        Om0=0.30966,
        Tcmb0=u.Quantity(2.7255, "K"),
        Neff=3.046,
        m_nu=u.Quantity([0.0, 0.0, 0.06], "eV"),
        Ob0=0.04897,
    )
    default_cosmology = cosmology.Planck18

    def _assert_matches_snapshot(self, source, snapshot):
        """Compare every field of `snapshot` against the same-named attribute
        on `source`, raising once with all mismatches rather than failing on
        the first one."""
        mismatches = []
        for name, snap_val in vars(snapshot).items():
            source_val = getattr(source, name)

            try:
                if isinstance(snap_val, u.Quantity):
                    assert source_val.unit == snap_val.unit
                    assert source_val.value == pytest.approx(snap_val.value)
                else:
                    assert source_val == pytest.approx(snap_val)
            except AssertionError as e:
                mismatches.append((name, snap_val, source.name))
        if mismatches:
            details = "\n".join(
                f"{name}: expected {expected}, got {actual}"
                for name, (expected, actual) in mismatches.items()
            )
            raise AssertionError(f"Not matching the expected values: \n{details}")

    def test_constants_remain_constant(self):
        self._assert_matches_snapshot(constants, self.constants)

    def test_cosmology_remains_constant(self):
        self._assert_matches_snapshot(constants.get_cosmology(), self.cosmology_params)

    def test_cosmologies(self):
        default_cosmo = constants.default_cosmology
        test_cosmo = constants.set_cosmology(self.default_cosmology.name)
        assert test_cosmo == self.default_cosmology
        assert default_cosmo == self.default_cosmology

    def test_set_cosmology_resets_bilby(self):
        # NMMA and bilby must not disagree about the cosmology, since the
        # GW side of a joint run converts distances through bilby's global.
        constants.set_cosmology("Planck15")
        assert bilby_cosmo.get_cosmology().name == "Planck15"
