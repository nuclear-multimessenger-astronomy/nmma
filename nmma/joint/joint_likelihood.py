import numpy as np
from bilby.core.likelihood import JointLikelihood

from ..core.base import NMMABaseLikelihood,NMMALikelihoodMixin
from ..core.conversion import MultimessengerConversion
from ..gw.gw_likelihood import GravitationalWaveTransientLikelihood, setup_gw_kwargs
from ..eos.eos_likelihood import EquationofStateLikelihood, setup_eos_kwargs, EoSConverter
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
    conversion_instructions: dict
        Instructions for parameter conversion

    """
    def __init__(self, messenger_likelihoods, priors, conversion_instructions = {}):
        super().__init__(*messenger_likelihoods)

        self.priors = priors
        self.conversion_instructions = conversion_instructions
        self.setup_parameter_conversion()
        self._noise_logl = JointLikelihood.noise_log_likelihood(self)

    def __repr__(self):
        if len(self.likelihoods) == 1:
            return f"{self.__class__.__name__} with {self.likelihoods[0].__repr__()}"
        else:
            reprs=[lhood.__repr__() for lhood in self.likelihoods]
            return f"{self.__class__.__name__} with {', '.join(map(str, reprs[:-1]))} and {reprs[-1]}"
            
    def setup_parameter_conversion(self):
        """Sets up the multimessenger conversion object, based on a corresponding dict."""
        for lhood in self.likelihoods:
            if isinstance(lhood, EMTransientLikelihood):
                self.conversion_instructions['em'] = lhood.parameter_conversion
            elif isinstance(lhood, GravitationalWaveTransientLikelihood):
                # FIXME: this is currently redundant, but potentially useful in future
                self.conversion_instructions['gw'] = lhood.parameter_conversion
            elif isinstance(lhood, EquationofStateLikelihood):
                self.conversion_instructions['eos'] = lhood.parameter_conversion

        self.multi_conversion = MultimessengerConversion.from_dict(self.conversion_instructions)
    
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
        return self.multi_conversion.convert_to_multimessenger_parameters(samples)
    
    
    def posterior_conversion(self, posterior_samples):
        posterior = self.multi_conversion.core_conversion(posterior_samples)
        for lhood in self.likelihoods:
            posterior = lhood.posterior_conversion(posterior)   
        return posterior.select_dtypes([np.number])
 
    @classmethod
    def setup_from_args(cls, data_dump, priors, args, logger):
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
        analysis_modifiers = data_dump["analysis_modifiers"]

        messenger_lhoods= []
        conversion_instructions = {}
        if "Hubble" in analysis_modifiers:
            conversion_instructions['cosmo'] = getattr(args, "cosmology", None)

        if "eos" in messengers:
            logger.info("Sampling over EOS generated on the fly")
            eos_kwargs = setup_eos_kwargs(data_dump, args, logger)
            if eos_kwargs['constraint_dict']:
                # only evaluate if specific constraints are given
                messenger_lhoods.append(EquationofStateLikelihood(priors,  **eos_kwargs))
            else:
                conversion_instructions['eos'] = eos_kwargs['eos_converter']
        elif "tabulated_eos" in analysis_modifiers:
            logger.info("Using tabulated EoS")
            conversion_instructions['eos'] = EoSConverter(args, 'tabulated')
        elif "lambda_1" in priors and "lambda_2" in priors:
            logger.info("Using universal relations for tidal deformabilities")
            conversion_instructions['eos'] = EoSConverter(args, 'qur')

        if "gw" in messengers:
            logger.info("Setting up GW likelihood")
            gw_kwargs = setup_gw_kwargs(data_dump, args, logger)
            messenger_lhoods.append(GravitationalWaveTransientLikelihood(priors, **gw_kwargs))

        if "em" in messengers:
            logger.info("Setting up EM likelihood")
            em_kwargs = setup_em_kwargs(priors, data_dump, args, logger)
            messenger_lhoods.append(EMTransientLikelihood(**em_kwargs))
            conversion_instructions['em'] = 'likelihood'

        if "pop" in messengers:
            pop_model = NeutronStarPopulation(args.population_model)
            messenger_lhoods.append(NMMABaseLikelihood(pop_model))
            # conversion_instructions['pop'] = 'model'
        
        # FIXME: Find a better way to check this automatically:
        #NOTE: this is nasty because we might have corresponding constraints, but do not need to...
        if args.ejecta_conversion:
            conversion_instructions['ejecta'] = True

        # if "spec" in messengers:  # FUTURE
        #     spec_kwargs = setup_spectroscopy_kwargs(data_dump, args, ...)
        #     messenger_lhoods.append(SpectroscopicLikelihood(priors, **spec_kwargs))

        if len(messenger_lhoods) == 0:
            raise ValueError("No messenger likelihoods were set up.")
        elif len(messenger_lhoods) == 1:
            lhood = messenger_lhoods[0]
            lhood.setup_parameter_conversion()
            logger.info(f"Only {lhood} is used. Using this instead.")
        else:
            lhood = cls(messenger_lhoods, priors, conversion_instructions)
            logger.info(f"Created {lhood}")
        return lhood