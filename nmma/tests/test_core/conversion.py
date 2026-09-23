import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.cosmology import Planck15, z_at_value
from bilby.core.prior import DeltaFunction, PriorDict, Uniform
from bilby.gw.prior import UniformComovingVolume

from nmma.core.constants import default_cosmology, set_cosmology
from nmma.core.conversion import (
    BNSEjectaFitting,
    CosmologyConverter,
    KilonovaEjectaFitting,
)


class TestCosmologyConverter:
    """Tests for CosmologyConverter."""

    alternative_cosmology = Planck15
    standard_z = 0.05
    standard_distance = 200.0
    standard_H0 = 70.0
    standard_Om0 = 0.3
    dlum_bounds = (10.0, 1000.0)  # Mpc
    z_bounds = (0.001, 0.5)
    H0_bounds = (60.0, 80.0)
    luminosity_distance_prior = Uniform(*dlum_bounds, name="luminosity_distance")
    alt_dlum_prior = UniformComovingVolume(
        *dlum_bounds, cosmology=alternative_cosmology, name="luminosity_distance"
    )
    peak_dlum_prior = DeltaFunction(peak=standard_distance, name="luminosity_distance")
    redshift_prior = Uniform(*z_bounds, name="redshift")
    peak_z_prior = DeltaFunction(peak=standard_z, name="redshift")
    Hubble_prior = Uniform(*H0_bounds, name="Hubble_constant")

    def setup_method(self):
        set_cosmology()
        self.converter = CosmologyConverter()

    def teardown_method(self):
        set_cosmology(default_cosmology)

    def true_redshift(self, distance, cosmology=default_cosmology):
        """Helper to compute the true redshift for a given luminosity distance
        using the default cosmology.
        """
        return z_at_value(cosmology.luminosity_distance, distance * u.Mpc).value

    def true_distance(self, redshift, cosmology=default_cosmology):
        """Helper to compute the true luminosity distance for a given redshift
        using the default cosmology.
        """
        return cosmology.luminosity_distance(redshift).value

    def test_setup(self):
        assert self.converter.cosmology == default_cosmology
        assert self.converter.conversion_function == self.converter.convert

        converter = CosmologyConverter(cosmology=self.alternative_cosmology)
        assert converter.cosmology.name == self.alternative_cosmology.name

    def test_default_conversion(self):
        parameters = {
            "luminosity_distance": self.standard_distance,
            "redshift": self.standard_z,
        }
        result_0 = self.converter(parameters)
        assert result_0 is parameters

        result_1 = self.converter({"redshift": self.standard_z})
        assert result_1["redshift"] == pytest.approx(self.standard_z)
        assert result_1["luminosity_distance"] == pytest.approx(
            self.true_distance(self.standard_z)
        )

        result_2 = self.converter({"luminosity_distance": self.standard_distance})
        assert result_2["luminosity_distance"] == pytest.approx(self.standard_distance)
        assert result_2["redshift"] == pytest.approx(
            self.true_redshift(self.standard_distance)
        )

        with pytest.raises(ValueError):
            self.converter.convert({"Hubble_constant": self.standard_H0})

    def test_inversion(self):
        dlum_1 = self.converter.luminosity_distance(self.standard_z)
        dlum_2 = self.converter.luminosity_distance(
            self.standard_z, cosmology=self.alternative_cosmology
        )
        assert dlum_1 != pytest.approx(dlum_2)

        inverted_z_1 = self.converter.redshift(dlum_1)
        inverted_z_2 = self.converter.redshift(
            dlum_2, cosmology=self.alternative_cosmology
        )
        assert inverted_z_1 == pytest.approx(self.standard_z)
        assert inverted_z_2 == pytest.approx(self.standard_z)

        redshift_1 = self.converter.redshift(self.standard_distance)
        redshift_2 = self.converter.redshift(
            self.standard_distance, cosmology=self.alternative_cosmology
        )
        assert redshift_1 != pytest.approx(redshift_2)

        inverted_dlum_1 = self.converter.luminosity_distance(redshift_1)
        inverted_dlum_2 = self.converter.luminosity_distance(
            redshift_2, cosmology=self.alternative_cosmology
        )
        assert inverted_dlum_1 == pytest.approx(self.standard_distance)
        assert inverted_dlum_2 == pytest.approx(self.standard_distance)

    def test_redshift_computation(self):
        distances = np.linspace(*self.dlum_bounds, 100)
        expected = self.true_redshift(distances)

        # interpolsation based
        redshifts = self.converter.redshift(distances)
        assert redshifts == pytest.approx(expected, rel=1e-3)

        # check pandas input
        distances = pd.Series(distances)
        redshifts = self.converter.redshift(distances)
        assert redshifts == pytest.approx(expected, rel=1e-3)

        # direct computation
        direct_redshifts = self.converter.redshift(distances[:40])
        assert direct_redshifts == pytest.approx(expected[:40])

    def test_distmod(self):
        assert self.converter.distmod(1e-5) == pytest.approx(0.0)  # 10 pc in Mpc
        distance_pc = self.standard_distance * 1e6
        assert self.converter.distmod(self.standard_distance) == pytest.approx(
            5.0 * np.log10(distance_pc / 10.0)
        )

    def test_cosmo_grid(self):
        dmin, dmax = self.dlum_bounds
        dist_grid, z_grid = self.converter.get_cosmo_grids(dmin, dmax)
        assert dist_grid[0] == pytest.approx(dmin)
        assert dist_grid[-1] == pytest.approx(dmax)
        assert z_grid[0] == pytest.approx(self.true_redshift(dmin))
        assert z_grid[-1] == pytest.approx(self.true_redshift(dmax))

    def test_sampled_cosmology_scalar(self):
        ref_cosmo = default_cosmology.clone(H0=self.standard_H0, Om0=self.standard_Om0)
        test_parameters = {
            "Hubble_constant": self.standard_H0,
            "Omega_matter": self.standard_Om0,
        }
        with pytest.raises(KeyError):
            self.converter.cosmology_to_distance(test_parameters)

        test_parameters_1 = test_parameters | {
            "luminosity_distance": self.standard_distance,
        }
        result_1 = self.converter.cosmology_to_distance(test_parameters_1)
        assert result_1["redshift"] == pytest.approx(
            self.converter.redshift(self.standard_distance, cosmology=ref_cosmo)
        )

        test_parameters_2 = test_parameters | {"redshift": self.standard_z}
        result_2 = self.converter.cosmology_to_distance(test_parameters_2)
        assert result_2["luminosity_distance"] == pytest.approx(
            self.converter.luminosity_distance(self.standard_z, cosmology=ref_cosmo)
        )

    def test_sampled_cosmology_array(self):
        hubble_constants = np.linspace(*self.H0_bounds, 3)
        distances = np.array([self.standard_distance for _ in hubble_constants])
        redshifts = np.array([self.standard_z for _ in hubble_constants])

        result_1 = self.converter.cosmology_to_distance(
            {"Hubble_constant": hubble_constants, "luminosity_distance": distances}
        )
        expect_1 = [
            self.true_redshift(d, default_cosmology.clone(H0=h0))
            for h0, d in zip(hubble_constants, distances)
        ]
        assert result_1["redshift"] == pytest.approx(expect_1)

        result_2 = self.converter.cosmology_to_distance(
            {"Hubble_constant": hubble_constants, "redshift": redshifts}
        )

        expect_2 = [
            self.true_distance(z, self.converter.cosmology.clone(H0=h0))
            for h0, z in zip(hubble_constants, redshifts)
        ]
        assert result_2["luminosity_distance"] == pytest.approx(expect_2)

    def test_from_priors(self):
        # default to absolute mags
        converter = CosmologyConverter.from_priors(PriorDict())
        result_0 = converter({})
        assert result_0["luminosity_distance"] == pytest.approx(1e-5)
        assert result_0["redshift"] == 0.0

        # Hubble_constant alone is rejected
        with pytest.raises(ValueError):
            CosmologyConverter.from_priors({"Hubble_constant": self.Hubble_prior})

        # fixed values
        converter = CosmologyConverter.from_priors({"redshift": self.peak_z_prior})
        result_1 = converter({})
        assert converter.conversion_function == converter._constant_cosmo
        assert result_1["redshift"] == pytest.approx(self.standard_z)
        assert result_1["luminosity_distance"] == pytest.approx(
            self.true_distance(self.standard_z)
        )
        converter = CosmologyConverter.from_priors(
            {"luminosity_distance": self.peak_dlum_prior}
        )
        result_2 = converter({})
        assert converter.conversion_function == converter._constant_cosmo
        assert result_2["luminosity_distance"] == pytest.approx(self.standard_distance)
        assert result_2["redshift"] == pytest.approx(
            self.true_redshift(self.standard_distance)
        )

        # sampled values
        converter = CosmologyConverter.from_priors({"redshift": self.redshift_prior})
        assert converter.conversion_function == converter.convert
        result_3 = converter({"redshift": self.standard_z})
        assert result_3["luminosity_distance"] == pytest.approx(
            self.true_distance(self.standard_z)
        )
        assert result_3["redshift"] == pytest.approx(self.standard_z)

        converter = CosmologyConverter.from_priors(
            {"luminosity_distance": self.luminosity_distance_prior}
        )
        assert converter.conversion_function == converter._grid_cosmo

        result4 = converter({"luminosity_distance": self.standard_distance})

        assert result4["redshift"] == pytest.approx(
            self.true_redshift(self.standard_distance), rel=1e-4
        )
        assert result4["luminosity_distance"] == pytest.approx(self.standard_distance)

    def test_from_cosmological_priors(self):
        converter = CosmologyConverter.from_priors(
            {"luminosity_distance": self.alt_dlum_prior}
        )
        result_0 = converter({"luminosity_distance": self.standard_distance})

        assert converter.cosmology == self.alternative_cosmology
        assert result_0["redshift"] == pytest.approx(
            self.true_redshift(self.standard_distance, self.alternative_cosmology),
            abs=1e-5,
        )
        assert result_0["luminosity_distance"] == pytest.approx(self.standard_distance)

        converter = CosmologyConverter.from_priors(
            {
                "luminosity_distance": self.luminosity_distance_prior,
                "Hubble_constant": self.Hubble_prior,
            }
        )
        assert converter.conversion_function == converter.cosmology_to_distance
        result_1 = converter(
            {
                "luminosity_distance": self.standard_distance,
                "Hubble_constant": self.standard_H0,
            }
        )
        assert result_1["redshift"] == pytest.approx(
            self.true_redshift(
                self.standard_distance,
                self.alternative_cosmology.clone(H0=self.standard_H0),
            )
        )

        converter = CosmologyConverter.from_priors(
            {
                "redshift": self.redshift_prior,
                "Hubble_constant": self.Hubble_prior,
            }
        )
        result_2 = converter(
            {
                "redshift": self.standard_z,
                "Hubble_constant": self.standard_H0,
            }
        )
        assert result_2["luminosity_distance"] == pytest.approx(
            self.true_distance(
                self.standard_z,
                self.alternative_cosmology.clone(H0=self.standard_H0),
            )
        )
        assert result_2["luminosity_distance"] != pytest.approx(
            self.true_distance(self.standard_z)
        )


class TestBNSEjectaFitting:
    def test_bns_ejecta_conversion_rejects_non_ns_component(self):
        """FIXME Weizmann: regression test for a bug in BNSEjectaFitting.bns_ejecta_conversion.

        radius_1/radius_2 are 0 (not a real Schwarzschild radius) for a mass
        outside the EOS's tabulated range, so compactness = mass*geom_msun_km/radius
        is inf for that component. dynamic_mass_fitting_KrFo/log10_disk_mass_fitting
        clip negative fit values via np.maximum(0, .), which silently turned that
        inf into a finite mdyn_fit=0.0 before np.isfinite() downstream had any
        chance to notice, log10(0 + alpha) then came out as an ordinary finite
        number for a system that isn't actually a BNS under this EOS. Verified
        directly against the exact formula: with mass_1_source=2.5 (above ALF2's
        ~2.09 Msun TOV mass) and radius_1=0, this used to return a finite
        log10_mej_dyn instead of -inf.
        """
        fitter = BNSEjectaFitting()
        params = dict(
            mass_1_source=np.array([2.5]),
            mass_2_source=np.array([1.3]),
            radius_1=np.array([0.0]),  # outside ALF2's mass range -> not a NS
            radius_2=np.array([13.1]),
            alpha=np.array([0.04]),
            ratio_zeta=np.array([0.5]),
            TOV_mass=np.array([2.0854]),
            R_16=np.array([12.0]),
        )
        (
            log10_mej_dyn,
            log10_mej_wind,
            log10_mej_total,
            _,
        ) = fitter.bns_ejecta_conversion(params)
        assert not np.isfinite(log10_mej_dyn[0]), (
            "log10_mej_dyn should be -inf when mass_1 isn't a real NS under this EOS"
        )
        assert not np.isfinite(log10_mej_wind[0]), (
            "log10_mej_wind should be -inf when mass_1 isn't a real NS under this EOS"
        )
        assert not np.isfinite(log10_mej_total[0])


class TestKilonovaEjectaFitting:
    def test_kn_ejecta_fitting_requires_both_components_to_be_ns(self):
        """FIXME Weizmann: regression test for KilonovaEjectaFitting's routing.

        It used to route to bns_parameter_conversion whenever radius_1>0 alone,
        without also checking radius_2>0. mass_1 >= mass_2 by convention, so
        radius_1>0 usually implies radius_2>0 too (a lighter mass is inside the
        EOS's mass range whenever a heavier one is), but not always: mass_2
        can fall below the EOS table's tabulated minimum while mass_1 is a
        valid NS mass, and that row was still (wrongly) treated as BNS,
        silently publishing a finite ejecta mass instead of -inf.
        """
        # two rows, to force numpy's vectorized np.where routing path (the
        # if/elif scalar path only ever runs for single-injection calls, which
        # the real pipeline, always operating on a whole dataframe, never
        # does; a length-1 array can be truth-tested directly by numpy without
        # raising, so it wouldn't actually exercise the routing bug here)
        fitter = KilonovaEjectaFitting()
        params = dict(
            mass_1_source=np.array([1.8, 1.4]),
            mass_2_source=np.array([1.3, 1.3]),
            radius_1=np.array(
                [13.0, 13.2]
            ),  # row 0: >0, would wrongly route to BNS alone
            radius_2=np.array(
                [0.0, 13.1]
            ),  # row 0: <=0, mass_2 isn't a real NS; row 1: genuine BNS
            alpha=np.array([0.04, 0.04]),
            ratio_zeta=np.array([0.5, 0.5]),
            TOV_mass=np.array([2.0854, 2.0854]),
            R_16=np.array([12.0, 12.0]),
            # np.where evaluates both the bns_ and nsbh_parameter_conversion
            # branches eagerly for every row (only the result is masked
            # afterwards), so nsbh_parameter_conversion's inputs must be valid
            # for all rows too, even the ones that will end up routed to BNS.
            chi_1=np.array([0.0, 0.0]),
        )
        (
            log10_mej_dyn,
            log10_mej_wind,
            log10_mej_total,
            _,
        ) = fitter.ejecta_parameter_conversion(params)
        assert not np.isfinite(log10_mej_dyn[0]), "invalid secondary should give -inf"
        assert not np.isfinite(log10_mej_wind[0]), "invalid secondary should give -inf"
        assert not np.isfinite(log10_mej_total[0]), "invalid secondary should give -inf"
        assert np.isfinite(log10_mej_dyn[1]), (
            "a genuine BNS row should not be affected by the fix"
        )
