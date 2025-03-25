import inspect
import numpy as np
from bilby.core.likelihood import Likelihood

class NMMABaseLikelihood(Likelihood):
    """ The base likelihood object for modular multi-messenger analysis

    Parameters
    ----------
    sub_model: bilby.core.likelihood.Likelihood
        The submodel to be used in each messenger. Must have a log_likelihood method.
    priors: dict
        The analysis priors, required for marginalization in some submodels
    parameters: dict
        The parameters to be used in the analysis
    param_conversion_func: callable, None
        Conversion function executed in the different subroutines. This should not be provided if a joint likelihood is used, as that one will handle the conversion for all submodels at once.

    """

    def __init__(self,sub_model, priors = None, parameters = {}, param_conversion_func = None, **kwargs):


        super().__init__(parameters=parameters)
        self.sub_model = sub_model
        if param_conversion_func is None:
            self.sub_model.parameter_conversion = self.identity_conversion
        else:
            self.sub_model.parameter_conversion = param_conversion_func
            
        if priors is not None:
            self.priors = priors
        self.__sync_parameters()

    def __repr__(self):
        return self.__class__.__name__ + ' with ' + self.sub_model.__repr__()

    def __sync_parameters(self):
        self.sub_model.parameters = self.parameters

    def identity_conversion(self, parameters):
        return parameters, []

    def log_likelihood(self):
        try:
            if not self.priors.evaluate_constraints(self.parameters):
                return np.nan_to_num(-np.inf)
        except AttributeError:
            print('No priors found, could not evaluate constraints')
            pass
        
        return self.sub_log_likelihood()
    
    def sub_log_likelihood(self):
        self.__sync_parameters()
        logL_model = self.sub_model.log_likelihood()
        if not np.isfinite(logL_model):
            return np.nan_to_num(-np.inf)

        return logL_model

    def noise_log_likelihood(self):
        return 0.
        # return self.sub_model.noise_log_likelihood()


def initialisation_args_from_signature_and_namespace(_callable, namespace, prefixes = []):
    prefixes.append('')
    signature = inspect.signature(_callable) 
    #step 1: get all default kwargs from the signature
    default_kwargs= {key: val.default for key, val in signature.parameters.items() if val.default is not inspect.Parameter.empty}

    #step 2: get all available kwargs from the namespace
    for key in signature.parameters.keys():
        ## this checks if further parameters from args-Namespace are only used as shorthands in the class definition (e.g. tmin, tmax)
        for prefix in prefixes:
            if hasattr(namespace, prefix+key):
                default_kwargs[key] = getattr(namespace, prefix+key)
                break
    return default_kwargs