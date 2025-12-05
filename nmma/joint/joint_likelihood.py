from __future__ import division

import numpy as np
from bilby.core.likelihood import JointLikelihood

from ..joint.base import NMMABaseLikelihood,NMMALikelihoodMixin
from ..joint.conversion import MultimessengerConversion, EoSConverter
from ..gw.gw_likelihood import GravitationalWaveTransientLikelihood, setup_gw_kwargs
from ..eos.eos_likelihood import EquationofStateLikelihood, setup_eos_kwargs
from ..em.em_likelihood import EMTransientLikelihood, setup_em_kwargs
from ..population.pop_likelihood import NeutronStarPopulation


class MultiMessengerLikelihood(NMMALikelihoodMixin,JointLikelihood):
    """A multi-messenger likelihood object

    This likelihood evaluates various sources of physical information at once.

    Parameters
    ----------
    messenger_likelihoods: list
        list of likelihoods to be used in the analysis
    priors: bilby.core.prior.PriorDict
        The priors used for the analysis
    args: Namespace
        parsed arguments
    connected_params: bool, optional
        Whether the different messenger likelihoods have connected parameters, using a MultimessengerConversion-object (default: True)
    

    """
    def __init__(self, messenger_likelihoods, priors, args=None, connected_params=True):
        super().__init__(*messenger_likelihoods)

        self.priors = priors
        if connected_params:
            self.setup_multi_conversion(args)
        else:
            self._conversion_function = self.basic_parameter_conversion

        self._noise_logl = super().noise_log_likelihood()

    def __repr__(self):
        if len(self.likelihoods) == 1:
            return f"{self.__class__.__name__} with {self.likelihoods[0].__repr__()}"
        else:
            reprs=[lhood.__repr__() for lhood in self.likelihoods]
            return f"{self.__class__.__name__} with {', '.join(map(str, reprs[:-1]))} and {reprs[-1]}"
        
    def basic_parameter_conversion(self, parameters):
        for lhood in self.likelihoods:
            parameters = lhood.parameter_conversion(parameters)
        return parameters
    
    def setup_multi_conversion(self, args):
        gw_conversion, eos_conversion,  em_conversion = False, False, False
        for lhood in self.likelihoods:
            if isinstance(lhood, EMTransientLikelihood):
                em_conversion = lhood.parameter_conversion
            elif isinstance(lhood, GravitationalWaveTransientLikelihood):
                # FIXME: this is currently redundant, but potentially useful in future
                gw_conversion = lhood.parameter_conversion
            elif isinstance(lhood, EquationofStateLikelihood):
                eos_conversion = lhood.parameter_conversion

        if eos_conversion is False:
            eos_conversion = EoSConverter(args)

        self.multi_conversion = MultimessengerConversion(eos_conversion, gw_conversion, 
            em_conversion, cosmology=getattr(args, "cosmology", None))
        self._conversion_function = self.multi_conversion.convert_to_multimessenger_parameters

    def setup_parameter_conversion(self):
        pass
    
    def sanity_checks(self):
        return np.prod([lhood.sanity_checks() for lhood in self.likelihoods])

    def sub_log_likelihood(self, parameters):
        logl = sum([lhood.sub_log_likelihood(parameters) for lhood in self.likelihoods])
        if np.isfinite(logl):
            return logl
        else:
            return np.nan_to_num(-np.inf)
        
    def final_diagnostics(self, bestfit_params, args, result=None):
        for lhood in self.likelihoods:
            lhood.final_diagnostics(bestfit_params, args, result)
            
    def parameter_conversion(self, samples):
        return self._conversion_function(samples)
    
    
    def posterior_conversion(self, posterior_samples):
        posterior = self.multi_conversion.core_conversion(posterior_samples)
        return posterior.select_dtypes([np.number])
 

def setup_nmma_likelihood(data_dump, priors, args, logger):
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
    logger: bilby.core.utils.logger
        Used for coherent logging
        
    Returns
    -------
    likelihood: nmma.joint.likelihood.MultiMessengerLikelihood

    """
    messengers= data_dump["messengers"]

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
        messenger_lhoods.append(GravitationalWaveTransientLikelihood(priors, **gw_kwargs))
        likelihood_kwargs.update(gw_kwargs)

    if "em" in messengers:
        em_kwargs = setup_em_kwargs(priors, data_dump, args, logger)
        messenger_lhoods.append(EMTransientLikelihood(**em_kwargs))
        likelihood_kwargs.update(em_kwargs)

    if "pop" in messengers:
        pop_model = NeutronStarPopulation(args.population_model)
        messenger_lhoods.append(NMMABaseLikelihood(pop_model))
        # likelihood_kwargs.update(pop_kwargs)
    
    # if "spec" in messengers:  # FUTURE
    #     spec_kwargs = setup_spectroscopy_kwargs(data_dump, args, ...)
    #     messenger_lhoods.append(SpectroscopicLikelihood(priors, **spec_kwargs))

    if len(messenger_lhoods) == 0:
        raise ValueError("No messenger likelihoods were set up.")
    elif len(messenger_lhoods) == 1:
        logger.info("Using the individual likelihood class instead.")
        return messenger_lhoods[0]
    
    logger.info(
        f"Initialise {MultiMessengerLikelihood} with kwargs: \n{likelihood_kwargs}"
    )
    return MultiMessengerLikelihood(messenger_lhoods, priors, args)