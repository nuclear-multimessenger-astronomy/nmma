from __future__ import division

import numpy as np
from bilby.core.likelihood import JointLikelihood

from ..joint.base import NMMABaseLikelihood
from ..gw.gw_likelihood import GravitationalWaveTransientLikelihood, setup_gw_kwargs
from ..eos.eos_likelihood import EquationofStateLikelihood, setup_eos_kwargs
from ..em.em_likelihood import EMTransientLikelihood, setup_em_kwargs
from ..population.pop_likelihood import NeutronStarPopulation


class MultiMessengerLikelihood(JointLikelihood):
    """A multi-messenger likelihood object

    This likelihood evaluates various sources of physical information at once.

    Parameters
    ----------
    priors: dict
        The analysis priors
    messenger_likelihoods: list
        list of likelihoods to be used in the analysis
    param_conv_func: callable
        Conversion function executed in the different subroutines
    

    """
    def __init__(self, messenger_likelihoods, priors):
        self.priors = priors
        super().__init__(*messenger_likelihoods)

    def __repr__(self):
        if len(self.likelihoods) == 1:
            return f"{self.__class__.__name__} with {self.likelihoods[0].__repr__()}"
        else:
            reprs=[lhood.__repr__() for lhood in self.likelihoods]
            return f"{self.__class__.__name__} with {', '.join(map(str, reprs[:-1]))} and {reprs[-1]}"

    def log_likelihood(self, parameters, local_parameters = None):
        if not self.priors.evaluate_constraints(parameters):
            return np.nan_to_num(-np.inf)
        self.sub_log_likelihood(parameters, local_parameters)

    def sub_log_likelihood(self, parameters, local_parameters = None):
        logL=0
        for lhood in self.likelihoods:
            lhood.local_parameters = local_parameters
            logL+= lhood.sub_log_likelihood(parameters)
        if not np.isfinite(logL):
            return np.nan_to_num(-np.inf)
        else:
            return logL
    

def setup_nmma_likelihood(data_dump, priors, args,messengers, logger):
    """Takes the kwargs and sets up and returns
    MultiMessengerLikelihood.

    Parameters
    ----------
    data_dump: dict
        collection of objects to be read in from the generation
    priors: dict
        The priors, used for setting up marginalization
    args: Namespace
        The parser arguments
    messengers: list
        list of messengers to be used in analysis
    logger: bilby.core.utils.logger
        Used for coherent logging
        
    Returns
    -------
    likelihood: nmma.joint.likelihood.MultiMessengerLikelihood

    """

    messenger_lhoods= []
    likelihood_kwargs={}

    if "eos" in messengers:
        logger.info("Sampling over EOS generated on the fly")
        eos_kwargs = setup_eos_kwargs(data_dump, args, logger)
        if eos_kwargs['constraint_dict']:
            # only evaluate if specific constraints are given
            messenger_lhoods.append(EquationofStateLikelihood(priors,  **eos_kwargs))
            likelihood_kwargs.update(eos_kwargs)

    if "gw" in messengers:
        gw_kwargs = setup_gw_kwargs(data_dump, args, logger)
        messenger_lhoods.append(GravitationalWaveTransientLikelihood(priors,**gw_kwargs))
        likelihood_kwargs.update(gw_kwargs)

    if "em" in messengers:
        em_kwargs= setup_em_kwargs(priors, data_dump, args, logger)
        messenger_lhoods.append(EMTransientLikelihood(**em_kwargs))
        em_kwargs.pop("priors")
        likelihood_kwargs.update(em_kwargs)

    if "pop" in messengers:
        pop_model = NeutronStarPopulation(args.population_model)
        messenger_lhoods.append(NMMABaseLikelihood(pop_model))
        # likelihood_kwargs.update(pop_kwargs)
    
    # if "spec" in messengers:  # FUTURE
    #     spec_kwargs = setup_spectroscopy_kwargs(data_dump, args, ...)
    #     messenger_lhoods.append(SpectroscopicLikelihood(priors, **spec_kwargs))
    logger.info(
        f"Initialise {MultiMessengerLikelihood} with kwargs: \n{likelihood_kwargs}"
    )
    
    return MultiMessengerLikelihood(priors, messenger_lhoods,**likelihood_kwargs)
