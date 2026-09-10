import numpy as np
from nmma.core.conversion import BNSEjectaFitting, KilonovaEjectaFitting


def test_bns_ejecta_conversion_rejects_non_ns_component():
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
    log10_mej_dyn, log10_mej_wind, log10_mej_total, _ = fitter.bns_ejecta_conversion(
        params
    )
    assert not np.isfinite(
        log10_mej_dyn[0]
    ), "log10_mej_dyn should be -inf when mass_1 isn't a real NS under this EOS"
    assert not np.isfinite(
        log10_mej_wind[0]
    ), "log10_mej_wind should be -inf when mass_1 isn't a real NS under this EOS"
    assert not np.isfinite(log10_mej_total[0])


def test_kn_ejecta_fitting_requires_both_components_to_be_ns():
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
        radius_1=np.array([13.0, 13.2]),  # row 0: >0, would wrongly route to BNS alone
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
    assert np.isfinite(
        log10_mej_dyn[1]
    ), "a genuine BNS row should not be affected by the fix"
