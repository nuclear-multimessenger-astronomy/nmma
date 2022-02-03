import numpy as np

from nmma.em.model import SimpleKilonovaLightCurveModel
from nmma.em.injection import create_light_curve_data

def test_Me2017():

    tmin, tmax, dt = 0.5, 20.0, 0.5
    sample_times = np.arange(tmin, tmax + dt, dt)

    filters = ['u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'K']
    lc_model = SimpleKilonovaLightCurveModel(sample_times=sample_times)

    bestfit_params = {'luminosity_distance': 56.4894661695928, 'beta': 3.6941470839046575, 'log10_kappa_r': 1.6646575950692677, 'KNtimeshift': -0.0583516607107672, 'log10_vej': -1.431500234170688, 'log10_Mej': -2.2511096455393518, 'Ebv': 0.0, 'log_likelihood': -309.52597696948493, 'log_prior': -10.203592144986466}
    _, mag = lc_model.generate_lightcurve(sample_times, bestfit_params)

    assert all([filt in mag for filt in filters])
