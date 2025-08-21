
import numpy as np
from scipy.stats import uniform, truncnorm

# class NeutronStarPopulationLikelihood(NMMABaseLikelihood):
#     """
#     Base class for neutron star population likelihoods.
#     This class is intended to be subclassed for specific population models.
#     """
    
#     def __init__(self, pop_model):
#         super().__init__(pop_model)

class NeutronStarPopulation:
    #based on https://doi.org/10.3847/2041-8213/ac2f3e
    """
    Object to compute the likelihood of a binary to align with 
    a given population model from Landry & Read."""
    def __init__(self, model_name, **kwargs):
        self.beta = 0.0
        if model_name.lower() == 'flat':
            m_min, m_max = 1.1, 2.0
            self.distribution = uniform(loc=m_min, scale=m_max)
        elif model_name.lower() == 'peak':
            m_min, m_max = 1.1, 2.1 
            loc   = 1.5
            scale = 1.0
            trunc_low, trunc_high = (m_min - loc) / scale, (m_max - loc) / scale
            self.distribution = truncnorm(trunc_low, trunc_high, loc=loc, scale=scale)


    def log_likelihood(self, parameters):
        # 
        return (  self.distribution.logpdf(self.parameters['mass_1_source']) 
                + self.distribution.logpdf(self.parameters['mass_2_source']) 
                + np.log(self.parameters['mass_ratio']**self.beta)
        )
