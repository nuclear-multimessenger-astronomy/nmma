import pytest
import numpy as np
import astropy.units as u
from scipy.constants import c
import re

from nmma.em.utils import lightcurve_HoNa


def test_lightcurve_HoNa_basic():
    t = np.logspace(-2, 1, 50)
    mass = 0.01
    velocities = np.array([0.1, 0.2, 0.3])
    opacities = np.array([0.1, 0.5])
    n = 3.5
    L, T, r = lightcurve_HoNa(t, mass, velocities, opacities, n)

    assert L.shape == T.shape == r.shape == t.shape, (
        "Output shapes must match input time array"
    )
    assert np.all(L > 0), "Luminosity must be positive"
    assert np.all(T > 0), "Temperature must be positive"
    assert np.all(r > 0), "Radius must be positive"


def test_lightcurve_HoNa_invalid_time():
    t = np.array([1e-4, 0.1, 1])
    mass = 0.01
    velocities = np.array([0.1, 0.2, 0.3])
    opacities = np.array([0.1, 0.5])
    n = 3.5

    with pytest.raises(ValueError, match="Times must be >"):
        lightcurve_HoNa(t, mass, velocities, opacities, n)


def test_lightcurve_HoNa_invalid_velocity():
    t = np.logspace(-2, 1, 50)
    mass = 0.01
    velocities = np.array([0.1, 0.2])
    opacities = np.array([0.1, 0.5])
    n = 3.5

    with pytest.raises(
        ValueError, match=re.escape("len(velocities) must be len(opacities) + 1")
    ):
        lightcurve_HoNa(t, mass, velocities, opacities, n)


def test_lightcurve_HoNa_units():
    t = np.logspace(-2, 1, 50) * u.day
    mass = 0.01 * u.Msun
    velocities = np.array([0.1, 0.2, 0.3]) * c * u.m / u.s
    opacities = np.array([0.1, 0.5]) * u.cm**2 / u.g
    n = 3.5
    with pytest.raises(u.core.UnitConversionError, match="dimensionless"):
        # input args should be dimensionless
        lightcurve_HoNa(t, mass, velocities, opacities, n)