import numpy as np
from scipy.stats import uniform, truncnorm


class NeutronStarPopulation:
    """
    Object to compute the likelihood of a binary to align with
    a given population model from Landry & Read.
    (https://doi.org/10.3847/2041-8213/ac2f3e)
    """
    def __init__(self, model_name, beta=0.0):
        self.beta = beta
        if model_name.lower() == 'flat':
            m_min, m_max = 1.1, 2.0
            self.distribution = uniform(loc=m_min, scale=m_max)
        elif model_name.lower() == 'peak':
            m_min, m_max = 1.1, 2.1
            loc = 1.5
            scale = 1.0
            trunc_low = (m_min - loc) / scale
            trunc_high = (m_max - loc) / scale
            self.distribution = truncnorm(trunc_low, trunc_high,
                                          loc=loc, scale=scale)

    def log_likelihood(self, parameters):
        return (self.distribution.logpdf(parameters['mass_1_source'])
                + self.distribution.logpdf(parameters['mass_2_source'])
                + np.log(parameters['mass_ratio']**self.beta))
