
import numpy as np

from nmma.em.lightcurve_generation import setup_HoNa_params, temp_photosphere_HoNa


def test_lightcurve_HoNa_basic():
    t = np.logspace(-1, 1, 50)
    param_dict = {
        "log10_mej": -2.,  # 0.01 Msun
    "vej_min": 0.1,
    "vej_max": 0.3,
    "vej_frac": 0.5,
    "log10_kappa_low_vej": -1.,  # 0.1 cm^2/g
    "log10_kappa_high_vej": np.log10(0.5),  # 0.5 cm^2/g
    "n": 3.5
    }
    
    inv_temp, r = temp_photosphere_HoNa(*setup_HoNa_params(t, param_dict), param_dict["n"])

    assert inv_temp.shape == r.shape == t.shape, (
        "Output shapes must match input time array"
    )
    assert np.all(inv_temp > 0), "Temperature must be positive"
    assert np.all(r > 0), "Radius must be positive"
