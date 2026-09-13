import numpy as np
from scipy.stats import uniform, truncnorm


class NeutronStarPopulation:
    """
    Uniform (flat) neutron-star mass distribution of Landry & Read, by
    default spanning 1.1 to 2.0 solar masses.
    (https://doi.org/10.3847/2041-8213/ac2f3e)
    """
    def __init__(self, m_min=1.1, m_max=2.0, beta=0.0):
        self.m_min = m_min
        self.m_max = m_max
        self.beta = beta
        self.distribution = uniform(loc=m_min, scale=m_max - m_min)

    def log_likelihood(self, parameters):
        return (self.distribution.logpdf(parameters['mass_1_source'])
                + self.distribution.logpdf(parameters['mass_2_source'])
                + np.log(parameters['mass_ratio']**self.beta))


class PeakNeutronStarPopulation(NeutronStarPopulation):
    """
    Peaked neutron-star mass distribution: a normal distribution centred on
    `loc`, truncated to [m_min, m_max]. By default centred at 1.5 solar
    masses and truncated to between 1.1 and 2.1.
    """
    def __init__(self, m_min=1.1, m_max=2.1, loc=1.5, scale=1.0, beta=0.0):
        self.m_min = m_min
        self.m_max = m_max
        self.beta = beta
        trunc_low = (m_min - loc) / scale
        trunc_high = (m_max - loc) / scale
        self.distribution = truncnorm(trunc_low, trunc_high,
                                       loc=loc, scale=scale)


POPULATION_MODELS = {
    'flat': NeutronStarPopulation,
    'peak': PeakNeutronStarPopulation,
}


def build_population_model(model_name, **kwargs):
    """Look up a population model class by its CLI name and construct it."""
    return POPULATION_MODELS[model_name.lower()](**kwargs)
