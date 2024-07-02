from __future__ import division
import inspect
import numpy as np
import json
from constraints import JointEoSConstraint, setup_joint_eos_constraint
from bilby.core.likelihood import Likelihood


def setup_eos_kwargs(priors, data_dump, args, logger, **kwargs):
    signature = inspect.signature(EquationofStateLikelihood) 
    default_eos_kwargs= {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}
    
    eos_kwargs= default_eos_kwargs.update(
        constraint_dict=data_dump['eos_constraint_dict']
    )

    ##########FIXME######## : This should go to some settings for analysis_modifiers
    if args.tabulated_eos:

        logger.info("Sampling over precomputed EOSs")
        
        eos_kwargs = dict(
            eos_path=args.eos_data,
            Neos=args.Neos,
            eos_weight_path=args.eos_weight,
        )
        if args.eos_from_neps:
            raise ValueError("Can only sample over either precomputed eos or NEPs. Set tabulated-eos or eos-from-neps to False!")
    
    elif args.eos_from_neps:
        
        eos_kwargs= dict(
            crust_path=args.eos_crust_file
        )
        if args.tabulated_eos:
            raise ValueError("Can only sample over either precomputed eos or NEPs. Set tabulated-eos or eos-from-neps to False!")
        
    return eos_kwargs

    
### To do: Routine to evaluate EOS likelihoods
class EquationofStateLikelihood(Likelihood):
    def __init__(self, constraint_dict, **kwargs):
        self.JointEoSConstraint=setup_joint_eos_constraint(constraint_dict)
        super(EquationofStateLikelihood).__init__(parameters= {})



    def __repr__(self):
        return self.__class__.__name__ 


    def log_likelihood(self):
        logL_EoS = self.JointEoSConstraint.log_likelihood(self.parameters)
        if not np.isfinite(logL_EoS):
            return np.nan_to_num(-np.inf)
        return logL_EoS

    def noise_log_likelihood(self):
        ###FIXME???
        return self.log_likelihood()
    
    def log_micro(self):
        ###FIXME!!!
        '''
        Routine to evaluate microphysical constraints on the EoS
        '''
        return 1
    

    def log_macro(self):
        ###FIXME!!!
        '''
        Routine to evaluate microphysical constraints on the EoS
        '''
        return 1
    


