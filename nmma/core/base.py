import inspect
import os
import h5py
from ast import literal_eval
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product

from bilby import run_sampler
from bilby.core.likelihood import Likelihood
from bilby.core.prior import (Prior, Constraint, Interped, ConditionalPriorDict, PriorDict,
                              MultivariateGaussianDist, MultivariateGaussian)
from bilby.core.result import FileMovedError
from .utils import input_obj_to_str, read_bestfit_from_posterior
from .constants import  set_cosmology
from .conversion import cosmology_to_distance
from .parsing import single_messenger_analysis_parsing, nmma_base_parsing

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

class NMMALikelihoodMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    @property
    def priors(self):
        return self._priors
    
    @priors.setter
    def priors(self, value):
        self.constraints = value
        sampling_keys = [k for k in value.keys() if k not in self.constraints]
        self.check_parameter_equivalencies(sampling_keys)
        self._priors = value

    @property
    def constraints(self):
        return self._constraints
    
    @constraints.setter
    def constraints(self, value):
        if isinstance(value, PriorDict):
            constr = {k: v for k, v in value.items() if isinstance(v, Constraint)}
        elif isinstance(value, Constraint):
            constr = {value.name: value}
        elif isinstance(value, dict):
            constr = value
            assert all(isinstance(v, Constraint) for v in value.values()), \
                "All entries in constraints dict must be of type Constraint"
        self._constraints = constr

    def evaluate_constraints(self, out_sample):
        return np.prod([con.prob(out_sample[k] ) for k, con in self.constraints.items()])   
    
    def identity_conversion(self, parameters):
        return parameters
    
    
    def __call__(self, parameters):
        return np.exp(self.log_likelihood(parameters))
    
    def log_likelihood(self, parameters):
        parameters = self.parameter_conversion(parameters)
        if self.evaluate_constraints(parameters) and self.sanity_checks():
            return self.sub_log_likelihood(parameters)
        else:
            return np.nan_to_num(-np.inf)
        
    def sanity_checks(self):
        return True
    
    
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
        try:
            return self.sub_model.final_diagnostics(bestfit_params, args, result)
        except AttributeError:
            pass

    def post_process_bestfit(self, args, result=None):
        bestfit_params = read_bestfit_from_posterior(args)
        bestfit_params = self.parameter_conversion(bestfit_params)
        return self.final_diagnostics(bestfit_params, args, result)
           
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
            ['redshift', 'luminosity_distance', 'Hubble_constant'], # FIXME: this would be ok if Omega_matter is investigated
            ['mass_1', 'mass_1_source', 'chirp_mass', 'mass_ratio', 'eta','mass_2', 'mass_2_source'],
        ]
        for group in double_equivalency_groups:
            intersection = set(parameter_names).intersection(set(group))
            if len(intersection)>2:
                raise ValueError(f"Mutually dependent parameters found: {intersection}. Please only provide up to two of these.")

    
class NMMALikelihood(NMMALikelihoodMixin,Likelihood):
    """ The base likelihood object for modular multi-messenger analysis

    Parameters
    ----------
    sub_model: bilby.core.likelihood.Likelihood
        The submodel to be used in each messenger. Must have a log_likelihood method.
    priors: dict
        The analysis priors, required for marginalization in some submodels

    """

    def __init__(self,sub_model, priors, **kwargs):
        super().__init__()

        self.sub_model = sub_model
        try:
            self._noise_logl = self.sub_model.noise_log_likelihood()
        except AttributeError:
            self._noise_logl = 0.
        self.conv_functions = []
        self.priors = priors
        self.setup_submodel_conversion()

    def __repr__(self):
        return self.__class__.__name__ + ' with ' + self.sub_model.__repr__()
        

    def setup_parameter_conversion(self):
        # FUTURE: add more standard conversions here
        if "Hubble_constant" in self.priors:
            self.conv_functions.append(cosmology_to_distance)

    def setup_submodel_conversion(self):
        pass
        
    def parameter_conversion(self, parameters):
        #reverse because "main conversion" are added last
        for conv in reversed(self.conv_functions):
            parameters = conv(parameters)
        return parameters
    
    def posterior_conversion(self, parameters):
        return self.parameter_conversion(parameters)
    
    def sub_log_likelihood(self, parameters):
        logL_model = self.sub_model.log_likelihood(parameters)
        if not np.isfinite(logL_model):
            return np.nan_to_num(-np.inf)
        return logL_model
    
    def noise_log_likelihood(self):
        return self._noise_logl

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
    if getattr(args, 'Hubble', False) or "Hubble_constant" in priors:  
        set_cosmology(getattr(args, 'cosmology', None))

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

def check_priors_and_likelihood_for_nmma(priors, likelihood):
    # remove constraints from priors and add to likelihood (should have happened already, but just in case)
    constraints = {k: priors.pop(k) for k in priors.copy().keys()
                    if isinstance(priors[k], Constraint)}
    likelihood.constraints.update(constraints)

    test_draw = priors.sample(1)
    test_conversion = priors.conversion_function(test_draw)
    if len(set(test_conversion.keys()) ) != len(test_conversion.keys()):
        priors.conversion_function = priors.default_conversion_function
        likelihood.conv_functions.append(likelihood.priors.conversion_function)
        
    # add final conversions
    likelihood.setup_parameter_conversion()
    return priors, likelihood

def bilby_sampling(likelihood, priors, args, injection_parameters=None, rank=0):
    if isinstance(args, dict):
        def_args = nmma_base_parsing(single_messenger_analysis_parsing)
        def_args.__dict__.update(args)
        args = def_args
    # fetch the additional sampler kwargs
    sampler_kwargs = getattr(args, "sampler_kwargs", {})
    print("Running with the following additional sampler_kwargs:")
    print(sampler_kwargs)

    # check if it is running with reactive sampler
    nlive = None if getattr(args, 'reactive_sampling', False) else args.nlive
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

    if rank != 0:
        return

    try:
        result.posterior = likelihood.posterior_conversion(result.posterior)
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
        result.posterior = likelihood.posterior_conversion(result.posterior)
        likelihood.post_process_bestfit(args, result)
    return result

def multi_analysis_loop(args, analysis_setup):
    USE_MPI = False
    rank = 0
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        if MPI.COMM_WORLD.Get_size() > 1:
            USE_MPI = True
    except:
        pass
        
    if rank != 0 and not getattr(args, 'verbose', False):
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        
    if getattr(args, 'multi', None):
        sub_runs = []
        if len(args.multi) == 1:
            arg, vals = list(args.multi.items())[0]
            for i, val in enumerate(vals):
                run_args = deepcopy(args)
                setattr(run_args, arg, val)
                setattr(run_args, 'label', f"{args.label}_{i}")
                sub_runs.append(run_args)
        else:
            for run_name, changes in args.multi.items():
                run_args = deepcopy(args)
                setattr(run_args, 'label', f"{args.label}_{run_name}")
                for key, value in changes.items():
                    if key not in args:
                        raise KeyError(f"{key} not a known argument... please remove")
                    setattr(run_args, key, value)
                sub_runs.append(run_args)
    elif getattr(args, 'matrix', None):
        sub_runs = []
        keys = args.matrix.keys()
        vals = args.matrix.values()
        for arg_variation in product(*vals):
            run_args = deepcopy(args)
            run_name = args.label
            for i, var in enumerate(arg_variation):
                rep = f'_{var}'
                if len(rep)>20:
                    key = keys[i]
                    var_idx = vals[i].index(var)
                    rep = f"_{key}_{var_idx}"
                run_name += rep
            setattr(run_args, 'label', run_name)
            for key, val in zip(keys, arg_variation):
                if key not in args:
                    raise KeyError(f"{key} not a known argument... please remove")
                setattr(run_args, key, val)
            sub_runs.append(run_args)
            
    else:
        sub_runs = [args]
    for run_args in sub_runs:
        priors, likelihood, injection_parameters = analysis_setup(run_args)
        priors, likelihood = check_priors_and_likelihood_for_nmma(priors, likelihood)
        if USE_MPI and run_args.sampler =='dynesty':
            from .mpi_setup import pbilby_sampling
            run_function = pbilby_sampling
        else:
            run_function = bilby_sampling
        out = run_function(likelihood, priors, run_args, injection_parameters, rank)
    return out