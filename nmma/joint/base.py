import inspect
import h5py
from ast import literal_eval
import numpy as np
import pandas as pd

from bilby import run_sampler
from bilby.core.likelihood import Likelihood, ZeroLikelihood
from bilby.core.prior import (Prior, Interped, ConditionalPriorDict, PriorDict,
                              MultivariateGaussianDist, MultivariateGaussian)
from bilby.core.result import FileMovedError
from .utils import input_obj_to_str, load_yaml, read_bestfit_from_posterior

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
                kwarg = getattr(namespace, prefix+key)
                if kwarg is not None:
                    default_kwargs[key] = kwarg
                break
    return default_kwargs

class NMMABaseLikelihood(Likelihood):
    """ The base likelihood object for modular multi-messenger analysis

    Parameters
    ----------
    sub_model: bilby.core.likelihood.Likelihood
        The submodel to be used in each messenger. Must have a log_likelihood method.
    priors: dict
        The analysis priors, required for marginalization in some submodels

    """

    def __init__(self,sub_model, priors = None, **kwargs):


        super().__init__()
        self.sub_model = sub_model
        try:
            self.noise_logl = self.sub_model.noise_log_likelihood()
        except AttributeError:
            self.noise_logl = 0.
        self.local_parameters = None
        if priors is not None:
            self.priors = priors

    def __repr__(self):
        return self.__class__.__name__ + ' with ' + self.sub_model.__repr__()

    def identity_conversion(self, parameters):
        return parameters, []
    
    def sanity_checks(self):
        return True
    
    def parameter_conversion(self, parameters):
        return self.identity_conversion(parameters)

    def log_likelihood(self, parameters):
        # try:
        #     out_sample = self.priors.conversion_function(parameters)
        #     for k, p in self.priors.items():
        #         if isinstance(p, Constraint) and not p.prob(out_sample[k]):
        #             return np.nan_to_num(-np.inf)
        # except AttributeError:
        #     print('No priors found, could not evaluate constraints')
        #     out_sample = parameters
        
        return self.sub_log_likelihood(parameters)
    
    def sub_log_likelihood(self, parameters):
        logL_model = self.sub_model.log_likelihood(parameters)
        if not np.isfinite(logL_model):
            return np.nan_to_num(-np.inf)
        return logL_model

    def noise_log_likelihood(self):
        return self.noise_logl
        # return self.sub_model.noise_log_likelihood()
    

    def post_process_bestfit(self, args, result):
        bestfit_params = read_bestfit_from_posterior(args)
        self.final_diagnostics(bestfit_params, args, result)
    
    def final_diagnostics(self, bestfit_params, args, result=None):
        """Plot the best-fit light curve against the data

        Parameters
        ----------
        bestfit_params: dict
            Dictionary of best-fit parameters

        Returns
        -------
        fig: matplotlib.figure.Figure
            The figure object containing the plot

        """
        pass

class NMMADummyPrior(Prior):
    """ A dummy prior that can be read from a prior-file into a prior dict, but is set to be replaced later """
    def __init__(self, setup_props):
        super().__init__(name='NMMADummyPrior')
        self.setup_props = setup_props
        
    @classmethod
    def from_repr(cls, repr_str):
        setup_props = literal_eval(repr_str)
        return cls(setup_props)

def adjust_priors_for_nmma(priors, logger=None):
    """ Adjust the priors dictionary for NMMA analysis

    Parameters
    ----------
    priors: dict
        The analysis priors
    args: argparse.Namespace
        The analysis arguments
    logger: logging.Logger, optional
        The logger to use for logging messages

    Returns
    -------
    dict
        The adjusted priors dictionary
    """
    if isinstance(priors, str):
        priors = PriorDict(priors)
    for key, prior in priors.copy().items():
        if not isinstance(prior, NMMADummyPrior):
            continue
        elif 'h5' in key:
            priors.pop(key)  # Remove the dummy prior
            if logger:
                logger.info(f"Replacing dummy prior for {key} with multivariate Gaussian prior from HDF5 file")
            priors = h5_to_multivar_prior(prior.setup_props, priors)
        elif 'hubble' in key.lower():
            priors.pop(key)  # Remove the dummy prior
            if logger:
                logger.info(f"Replacing dummy prior for {key} with Interped prior from Hubble weighting file")
            priors = adjust_hubble_prior(priors, prior.setup_props, logger)
        # to be extended
    return priors

def adjust_hubble_prior(priors, args, logger=None):
    hubble_weight = input_obj_to_str(args, 'Hubble_weight')
    if hubble_weight:
        if logger:
            logger.info("Sampling over Hubble constant with pre-calculated prior")
            logger.info("Overwriting any Hubble prior in the prior file")
        try:
            Hubble_prior_data = pd.read_csv(hubble_weight, delimiter=" ", header=0)
            xx = Hubble_prior_data.Hubble.to_numpy()
            yy = Hubble_prior_data.prior_weight.to_numpy()
        except:
            xx, yy =  np.loadtxt(hubble_weight).T

        Hmin = xx[0]
        Hmax = xx[-1]

        priors["Hubble_constant"] = Interped(
            xx, yy, minimum=Hmin, maximum=Hmax, name="Hubble_constant"
        )
    return priors

def h5_to_multivar_prior(h5_file_path, priors = {}):
    h5_file_path = input_obj_to_str(h5_file_path, 'h5 file path')
    with h5py.File(h5_file_path, "r") as f:
        # Load the data from the HDF5 file
        keys = list(f.keys())
        data_array = np.column_stack([f[key][:] for key in keys])
    mean = np.mean(data_array, axis=0)
    cov = np.cov(data_array, rowvar=False)

    eos_dist = MultivariateGaussianDist(keys, mus=[mean], covs=[cov])
    priors.update({key: MultivariateGaussian(eos_dist, key) for key in keys})

    # We need "at least" a conditional Prior Dict, but should not "downgrade" CBC-dicts
    if isinstance(priors, ConditionalPriorDict):
        return priors
    return ConditionalPriorDict(priors)



def bilby_sampling(likelihood, priors, args, injection_parameters=None):

    if args.bilby_zero_likelihood_mode:
        likelihood = ZeroLikelihood(likelihood)

    # fetch the additional sampler kwargs
    sampler_kwargs = literal_eval(args.sampler_kwargs)
    print("Running with the following additional sampler_kwargs:")
    print(sampler_kwargs)

    # check if it is running with reactive sampler
    nlive = None if args.reactive_sampling else args.nlive
    if nlive is None and args.sampler != "ultranest":
        raise ValueError("reactive sampling is only available for ultranest, "
                         "please set nlive or use ultranest sampler")


    if args.skip_sampling:
        print("Sampling for 1 iteration and plotting checkpointed results.")
        if args.sampler == "pymultinest":
            sampler_kwargs["max_iter"] = 1
        elif args.sampler == "ultranest":
            sampler_kwargs["niter"] = 1
        elif args.sampler == "dynesty":
            sampler_kwargs["maxiter"] = 1

    result = run_sampler(
        likelihood,
        priors,
        sampler=args.sampler,
        outdir=args.outdir,
        label=args.label,
        nlive=nlive,
        seed=args.sampling_seed,
        soft_init=args.soft_init,
        queue_size=args.cpus,
        check_point_delta_t=3600,
        save=False,
        **sampler_kwargs,
    )

    # check if it is running under mpi
    try:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            exit()
    except ImportError:
        pass

    try:
        result.save_to_file()
        result.save_posterior_samples()
    except FileMovedError:
        # We assume the result was only moved here, no need to save
        result.outdir = args.outdir
        result.label = args.label
        result.parameter_labels_with_unit = None
        result_prior = result.priors.copy()
        for k, v in result_prior.items():
            if k in priors:
                v.latex_label = priors[k].latex_label
                result_prior[k] = v
        result.priors = result_prior
        # result.save_posterior_samples()

    if injection_parameters: 
        var_columns = {col for col in result.posterior 
                       if len(result.posterior[col].unique()) > 1}
        injection_parameters = {k: v for k, v in injection_parameters.items()
                        if k in var_columns}
    try:
        result.plot_corner(injection_parameters, priors)
    except RuntimeError:
        result.parameter_labels_with_unit = None
        for k, v in priors.copy().items():
            v.latex_label = None
            priors[k] = v
        result.priors = priors 
        result.plot_corner(injection_parameters, priors)
        
    if args.bestfit or args.plot:
        likelihood.post_process_bestfit(args, result)


def multi_analysis_loop(args, analysis_setup):
    if getattr(args, 'config', None):
        yaml_dict = load_yaml(args.config)
        for params in yaml_dict.values():
            for key, value in params.items():
                key = key.replace("-", "_")
                if key not in args:
                    print(f"{key} not a known argument... please remove")
                    exit()
                setattr(args, key, value)
            priors, likelihood, injection_parameters = analysis_setup(args)
            bilby_sampling(likelihood, priors, args, injection_parameters)
    else:
        priors, likelihood, injection_parameters = analysis_setup(args)
        bilby_sampling(likelihood, priors, args, injection_parameters)