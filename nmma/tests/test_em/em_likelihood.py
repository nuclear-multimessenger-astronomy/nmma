from types import SimpleNamespace

import numpy as np
from scipy.stats import norm

from nmma.em.em_likelihood import BasicEMTransient


def score_example_data(error_model):
    est_mag = np.array([20.00, 20.00, 20.00])
    data_mag = np.array([20.25, 20.50, 21.00])
    data_sigma = np.array([0.20, 0.25, np.inf])
    upperlim_sigma = np.array([1.00, 1.00, 1.00])
    return BasicEMTransient.chisquare_gaussianlog_from_lc_data(
        SimpleNamespace(error_model=error_model),
        est_mag,
        data_mag,
        data_sigma,
        upperlim_sigma,
    )


def test_chisquare_gaussianlog_from_lc_data_uses_selected_model():
    def mock_model(*args, **kwargs):
        return np.full_like(args[0], -1)

    minus_chisquare, gausslogsf = score_example_data(mock_model)

    assert minus_chisquare == -2
    assert gausslogsf == norm.logsf(21.00, 20.00, 1.00)


def test_parameter_rejection_by_error_model():
    def mock_model(*args, **kwargs):
        return np.full_like(args[0], np.nan)

    assert score_example_data(mock_model) == (False, -np.inf)
