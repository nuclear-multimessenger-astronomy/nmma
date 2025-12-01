from __future__ import division

import numpy as np
from bilby.core.likelihood import JointLikelihood
from bilby.core.prior import Constraint

from ..joint.base import NMMABaseLikelihood
from ..joint.conversion import MultimessengerConversion
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
    def __init__(self, messenger_likelihoods, constraints, args, messengers, modifiers):
        super().__init__(*messenger_likelihoods)

        self.constraints = constraints
        self.messengers = messengers
        internal_conversions = {}
        for lhood in self.likelihoods:
            if isinstance(lhood, EMTransientLikelihood):
                internal_conversions['em'] = lhood.parameter_conversion
            elif isinstance(lhood, GravitationalWaveTransientLikelihood):
                # FIXME: this is currently redundant, but potentially useful in future
                internal_conversions['gw'] = lhood.parameter_conversion
            elif isinstance(lhood, EquationofStateLikelihood):
                internal_conversions['eos'] = lhood.parameter_conversion

        self.multi_conversion = MultimessengerConversion(args, messengers, modifiers, internal_conversions)
        self._noise_logl = super().noise_log_likelihood()

    def __repr__(self):
        if len(self.likelihoods) == 1:
            return f"{self.__class__.__name__} with {self.likelihoods[0].__repr__()}"
        else:
            reprs=[lhood.__repr__() for lhood in self.likelihoods]
            return f"{self.__class__.__name__} with {', '.join(map(str, reprs[:-1]))} and {reprs[-1]}"
            
    def sanity_checks(self):
        return np.prod([lhood.sanity_checks() for lhood in self.likelihoods])

    def evaluate_constraints(self, out_sample):
        return np.prod([con.prob(out_sample[k] ) for k, con in self.constraints.items()])   
    
    def log_likelihood(self, parameters):
        parameters, _ = self.parameter_conversion(parameters)
        if self.evaluate_constraints(parameters) and self.sanity_checks():
            return self.sub_log_likelihood(parameters)
        else:
            return np.nan_to_num(-np.inf)

    def sub_log_likelihood(self, parameters):
        logl = sum([lhood.sub_log_likelihood(parameters) for lhood in self.likelihoods])
        if np.isfinite(logl):
            return logl
        else:
            return np.nan_to_num(-np.inf)
        
    def noise_log_likelihood(self):
        return self._noise_logl
        
    def final_diagnostics(self, bestfit_params, args, result=None):
        for lhood in self.likelihoods:
            lhood.final_diagnostics(bestfit_params, args, result)
            
    def parameter_conversion(self, samples):
        return self.multi_conversion.convert_to_multimessenger_parameters(samples)
    
    def posterior_conversion(self, posterior_samples):
        posterior = self.multi_conversion.core_conversion(posterior_samples)
        return posterior.select_dtypes([np.number])
    
    def check_parameter_equivalencies(self, parameter_names):
        """Check for equivalent parameters and terminate if found"""
        #FIXME: to be extended
        single_equivalency_groups = [
            ["inclination_EM", "KNtheta", "theta_jn", "cos_theta_jn", "thetaObs"],
        ]
        for group in single_equivalency_groups:
            intersection = set(parameter_names).intersection(set(group))
            if len(intersection)>1:
                raise ValueError(f"Multiple equivalent parameters found: {intersection}. Please only provide one of these.")
            
        double_equivalency_groups = [
            ['redshift', 'luminosity_distance', 'Hubble_constant'], # FIXME: this can work if Omega_matter is investigated
            ['mass_1', 'mass_1_source', 'chirp_mass', 'mass_ratio', 'eta','mass_2', 'mass_2_source'],
        ]
        for group in double_equivalency_groups:
            intersection = set(parameter_names).intersection(set(group))
            if len(intersection)>2:
                raise ValueError(f"Mutually dependent parameters found: {intersection}. Please only provide up to two of these.")
        
    

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
    modifiers = data_dump["analysis_modifiers"]

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

    logger.info(
        f"Initialise {MultiMessengerLikelihood} with kwargs: \n{likelihood_kwargs}"
    )
    constraints = {k: v for k, v in priors.items() if isinstance(v, Constraint)}
    
    return MultiMessengerLikelihood(messenger_lhoods, constraints, args, messengers, modifiers)