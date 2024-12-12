from __future__ import division

import numpy as np
from bilby.core.likelihood import Likelihood

from ..gw.gw_likelihood import GravitationalWaveTransientLikelihood, setup_gw_kwargs
from ..eos.eos_likelihood import setup_eos_kwargs, EquationofStateLikelihood
from ..em.em_likelihood import EMTransientLikelihood, setup_em_kwargs


class MultiMessengerLikelihood(Likelihood):
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
    def __init__(self, priors, messenger_likelihoods,  **kwargs):
        super(MultiMessengerLikelihood, self).__init__(parameters={})
        self.reprs=[lhood.__repr__() for lhood in messenger_likelihoods]
        self.sub_likelihoods=messenger_likelihoods
        self.priors = priors
        self.__sync_parameters()

    def __repr__(self):
        if len(self.reprs) == 1:
            return f"{self.__class__.__name__} with {self.reprs[0]}"
        else:
            return f"{self.__class__.__name__} with {', '.join(map(str, self.reprs[:-1]))} and {self.reprs[-1]}"

    def __sync_parameters(self):
        for lhood in self.sub_likelihoods:
            lhood.parameters= self.parameters


    def log_likelihood(self):
        if not self.priors.evaluate_constraints(self.parameters):
            return np.nan_to_num(-np.inf)
        self.sub_log_likelihood()

    def sub_log_likelihood(self):
        logL=0
        for lhood in self.sub_likelihoods:
            lhood.parameters = self.parameters
            logL+= lhood.sub_log_likelihood()
        if not np.isfinite(logL):
            return np.nan_to_num(-np.inf)
        else:
            return logL

    def noise_log_likelihood(self):
        noise_logL=0
        for lhood in self.sub_likelihoods:
            noise_logL+= lhood.noise_log_likelihood()
        return noise_logL
    

def setup_nmma_likelihood(data_dump, priors, args,messengers, logger, param_conv_func= None, **kwargs
    #messengers, interferometers, waveform_generator, light_curve_data, priors, args
    ):
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
    param_conv_func: callable
        Conversion function executed in the different subroutines
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
        eos_kwargs = setup_eos_kwargs(priors, data_dump, 
                                      args, logger, **kwargs)
        messenger_lhoods.append(EquationofStateLikelihood(priors,  **eos_kwargs))
        likelihood_kwargs.update(eos_kwargs)

    if "gw" in messengers:
        gw_kwargs = setup_gw_kwargs(data_dump, 
                                    args, logger, **kwargs)
        messenger_lhoods.append(GravitationalWaveTransientLikelihood(priors, param_conv_func, **gw_kwargs))
        likelihood_kwargs.update(gw_kwargs)

    if "em" in messengers:
        # if "grb" in messengers:
        #     ###modifies the em parameters, does not (as of 06/24) have its own likelihood
        #     em_kwargs= setup_grb_kwargs(param_conv_func, data_dump,
        #                                 args, logger, **kwargs)
        # else:
        em_kwargs= setup_em_kwargs(param_conv_func, data_dump,
                                        args, logger, **kwargs)

        
        messenger_lhoods.append(
            EMTransientLikelihood(priors, **em_kwargs))
        likelihood_kwargs.update(em_kwargs)
    
    # if "spec" in messengers:
    #     spec_kwargs = setup_spectroscopy_kwargs(data_dump, args, ...)
    #     messenger_lhoods.append(SpectroscopicLikelihood(priors, **spec_kwargs))
    logger.info(
        f"Initialise {Likelihood} with kwargs: \n{likelihood_kwargs}"
    )
    
    return MultiMessengerLikelihood(priors, messenger_lhoods,**likelihood_kwargs)


def reorder_loglikelihoods(unsorted_loglikelihoods, unsorted_samples, sorted_samples):
    """Reorders the stored log-likelihood after they have been reweighted

    This creates a sorting index by matching the reweights `result.samples`
    against the raw samples, then uses this index to sort the
    loglikelihoods

    Parameters
    ----------
    sorted_samples, unsorted_samples: array-like
        Sorted and unsorted values of the samples. These should be of the
        same shape and contain the same sample values, but in different
        orders
    unsorted_loglikelihoods: array-like
        The loglikelihoods corresponding to the unsorted_samples

    Returns
    -------
    sorted_loglikelihoods: array-like
        The loglikelihoods reordered to match that of the sorted_samples


    """

    idxs = []
    for ii in range(len(unsorted_loglikelihoods)):
        idx = np.where(np.all(sorted_samples[ii] == unsorted_samples, axis=1))[0]
        if len(idx) > 1:
            print(
                "Multiple likelihood matches found between sorted and "
                "unsorted samples. Taking the first match."
            )
        idxs.append(idx[0])
    return unsorted_loglikelihoods[idxs]
