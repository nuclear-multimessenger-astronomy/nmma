import numpy as np
from scipy.stats import uniform, truncnorm


class NeutronStarPopulation:
    """
    Object to compute the likelihood of a binary to align with
    a given population model from Landry & Read.
    (https://doi.org/10.3847/2041-8213/ac2f3e)
    """
    def __init__(self, model_name, beta=0.0):
        """
        Parameters
        ----------
        model_name : str
            Name of the source-frame component-mass distribution to use.
            One of ``'flat'`` (uniform between 1.1 and 2.0 solar masses)
            or ``'peak'`` (normal distribution centered at 1.5 solar
            masses, truncated to [1.1, 2.1] solar masses).
        beta : float
            Power-law index on the mass ratio, used to (de)weight
            unequal-mass systems in the log-likelihood.
        """
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
        ### FIXME: Unrecognized model_name fails silently at construction, then crashes unhelpfully later: there's no else branch, so passing e.g. NeutronStarPopulation('gaussian') succeeds without error and produces an object with no self.distribution at all. 

    def log_likelihood(self, parameters):
        """
        Compute the log-likelihood of a binary's component masses and
        mass ratio under the configured population model.

        Parameters
        ----------
        parameters : dict
            Must contain ``'mass_1_source'`` and ``'mass_2_source'``
            (source-frame component masses, in solar masses) and
            ``'mass_ratio'``.

        Returns
        -------
        float
            Sum of the log-pdf of both component masses under
            ``self.distribution`` plus ``beta * log(mass_ratio)``.
        """
        return (self.distribution.logpdf(parameters['mass_1_source'])
                + self.distribution.logpdf(parameters['mass_2_source'])
                + np.log(parameters['mass_ratio']**self.beta))
