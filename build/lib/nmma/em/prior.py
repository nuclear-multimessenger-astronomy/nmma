from bilby.core.prior import Prior
from bilby.core.prior.conditional import ConditionalTruncatedGaussian


class ConditionalGaussianIotaGivenThetaCore(ConditionalTruncatedGaussian):
    """
    This prior characterizes the conditional prior on the viewing angle
    of a GRB afterglow given the opening angle of the jet such that
    the viewing angle is follwing a half-Gaussian distribution
    with the opening angle (thetaCore) as the sigma;
    therefore, sigma = thetaCore / N_sigma
    a higher value of N_sigma, the distribution would be tigher around 0, vice versa

    This prior should only be used if the event is a prompt emission!
    """

    def __init__(self, minimum, maximum, name, N_sigma=1, latex_label=None, unit=None, boundary=None):

        super(ConditionalTruncatedGaussian, self).__init__(
            mu=0, sigma=1,
            minimum=minimum, maximum=maximum, name=name, latex_label=latex_label,
            unit=unit, boundary=boundary, condition_func=self._condition_function)

        self._required_variables = ['thetaCore']
        self.N_sigma = N_sigma
        self.__class__.__name__ = 'ConditionalGaussianIotaGivenThetaCore'
        self.__class__.__qualname__ = 'ConditionalGaussianIotaGivenThetaCore'

    def _condition_function(self, reference_params, **kwargs):
        return dict(sigma=(1. / self.N_sigma) * kwargs[self._required_variables[0]])

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        for key, value in self.reference_params.items():
            if key in instantiation_dict:
                instantiation_dict[key] = value
        return instantiation_dict