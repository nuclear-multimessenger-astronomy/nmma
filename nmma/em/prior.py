import bilby
import bilby.core
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

    def __init__(
        self,
        minimum,
        maximum,
        name,
        N_sigma=1,
        latex_label=None,
        unit=None,
        boundary=None,
    ):

        super(ConditionalTruncatedGaussian, self).__init__(
            mu=0,
            sigma=1,
            minimum=minimum,
            maximum=maximum,
            name=name,
            latex_label=latex_label,
            unit=unit,
            boundary=boundary,
            condition_func=self._condition_function,
        )

        self._required_variables = ["thetaCore"]
        self.N_sigma = N_sigma
        self.__class__.__name__ = "ConditionalGaussianIotaGivenThetaCore"
        self.__class__.__qualname__ = "ConditionalGaussianIotaGivenThetaCore"

    def _condition_function(self, reference_params, **kwargs):
        return dict(sigma=(1.0 / self.N_sigma) * kwargs[self._required_variables[0]])

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        for key, value in self.reference_params.items():
            if key in instantiation_dict:
                instantiation_dict[key] = value
        return instantiation_dict


def create_prior_from_args(model_names, args):

    if "AnBa2022" in model_names:

        def convert_mtot_mni(parameters):

            for param in ["mni", "mtot", "mrp"]:
                if param not in parameters:
                    parameters[param] = 10**parameters[f"log10_{param}"]

            parameters["mni_c"] = parameters["mni"] / parameters["mtot"]
            parameters["mrp_c"] = (
                parameters["xmix"] * (parameters["mtot"] - parameters["mni"])
                - parameters["mrp"]
            )
            return parameters

        priors = bilby.core.prior.PriorDict(
            args.prior, conversion_function=convert_mtot_mni
        )
    else:
        priors = bilby.gw.prior.PriorDict(args.prior)

    # setup for Ebv
    if args.Ebv_max > 0.0:
        Ebv_c = 1.0 / (0.5 * args.Ebv_max)
        priors["Ebv"] = bilby.core.prior.Interped(
            name="Ebv",
            minimum=0.0,
            maximum=args.Ebv_max,
            latex_label="$E(B-V)$",
            xx=[0, args.Ebv_max],
            yy=[Ebv_c, 0],
        )
    else:
        priors["Ebv"] = bilby.core.prior.DeltaFunction(
            name="Ebv", peak=0.0, latex_label="$E(B-V)$"
        )

    # re-setup the prior if the conditional prior for inclination is used

    if args.conditional_gaussian_prior_thetaObs:
        priors_dict = dict(priors)
        original_iota_prior = priors_dict["inclination_EM"]
        setup = dict(
            minimum=original_iota_prior.minimum,
            maximum=original_iota_prior.maximum,
            name=original_iota_prior.name,
            latex_label=original_iota_prior.latex_label,
            unit=original_iota_prior.unit,
            boundary=original_iota_prior.boundary,
            N_sigma=args.conditional_gaussian_prior_N_sigma,
        )

        priors_dict["inclination_EM"] = ConditionalGaussianIotaGivenThetaCore(**setup)
        priors = bilby.gw.prior.ConditionalPriorDict(priors_dict)

    return priors
