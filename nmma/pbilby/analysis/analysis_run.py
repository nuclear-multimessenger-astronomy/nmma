import logging
import datetime
import os
from glob import glob
import pickle

import numpy as np
from numpy.random import Generator, PCG64, SeedSequence

from bilby.core.result import Result
from bilby.core.sampler import dynesty3_utils  as dy_utils
from bilby.core.utils import logger
from bilby.core.prior import PriorDict, Constraint

import dynesty

from ...joint.joint_likelihood import setup_nmma_likelihood
from ...joint.utils import reorder_loglikelihoods, rejection_sample, read_bestfit_from_posterior


class MainRun:
   
    def __init__(
        self,
        sampling_keys,
        pooled_log_likelihood_function,
        pooled_prior_transform_function,
        pooled_initial_point_function,
        args=None,
        outdir=None,
        label=None,
        periodic=None,
        reflective = None,
        sample="acceptance-walk",
        nlive=5,
        bound="live",
        walks=100,
        maxmcmc=5000,
        naccept=60,
        nact=2,
        facc=0.5,
        min_eff=10,
        enlarge=1.5,
        sampling_seed=0
    ):
        
        # Create a random generator, which is saved across restarts
        # This ensures that runs are fully deterministic, which is important
        # for reproducibility
        self.rstate = Generator(PCG64(sampling_seed))
        logger.debug(
            f"Setting random state = {self.rstate} (seed={sampling_seed})"
        )
        ## Set some basic attributes
        self.sampling_keys = sampling_keys
        logger.info(f"sampling_keys={sampling_keys}")
        if periodic:
            logger.info(
                f"Periodic keys: {[sampling_keys[ii] for ii in periodic]}"
            )
        if reflective:
            logger.info(
                f"Reflective keys: {[sampling_keys[ii] for ii in reflective]}"
            )
        self.nlive = nlive
        self.args = args
        self.pooled_log_likelihood_function = pooled_log_likelihood_function
        self.pooled_prior_transform_function= pooled_prior_transform_function
        self.pooled_initial_point_function =  pooled_initial_point_function



        # If the run dir has not been specified, get it from the args
        if outdir is None:
            outdir = self.args.outdir
        else:
            # Create the run dir
            os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir

        # If the label has not been specified, get it from the args
        if label is None:
            label = self.args.label
        self.label = label


        
        # dynesty3 init kwargs
        dyn_kwargs = dict(
        ndim=len(sampling_keys),
        nlive=nlive,
        bound=bound,
        sample=sample,
        periodic=periodic,
        reflective=reflective,
        first_update=dict(min_eff=min_eff, min_ncall=2 * nlive),
        enlarge=enlarge,
        walks=walks,
        facc=facc
        )

        self._init_sampler_kwargs(dyn_kwargs, nact, naccept, maxmcmc)

    def _init_sampler_kwargs(self, kwargs,  nact, naccept, maxmcmc):
        """
        Stolen from bilby.core.sampler.dynesty to set up the internal sampler kwargs
        """
        
        internal_kwargs = dict(
            ndim=kwargs['ndim'],
            nonbounded=None,
            periodic = kwargs["periodic"],
            reflective = kwargs["reflective"],
            maxmcmc = maxmcmc
        )
        if kwargs["sample"] == "act-walk":
            internal_kwargs["nact"] = nact
            kwargs["sample"] = dy_utils.ACTTrackingEnsembleWalk(**internal_kwargs)
            kwargs["bound"] = "none"

            logger.info(
                f"Using the bilby-implemented ensemble rwalk sampling tracking the "
                f"autocorrelation function and thinning by {kwargs['sample'].thin} with "
                f"maximum length {kwargs['sample'].thin * kwargs['sample'].maxmcmc}."
            )

        elif kwargs["sample"] == "acceptance-walk":
            internal_kwargs["naccept"] = naccept
            internal_kwargs["walks"] = kwargs["walks"]
            kwargs["sample"] = dy_utils.EnsembleWalkSampler(**internal_kwargs)
            kwargs["bound"] = "none"
            logger.info(
                f"Using the bilby-implemented ensemble rwalk sampling method with an "
                f"average of {kwargs['sample'].naccept} accepted steps up to chain "
                f"length {kwargs['sample'].maxmcmc}."
            )

        elif kwargs["sample"] == "rwalk":
            internal_kwargs["nact"] = nact
            kwargs["sample"] = dy_utils.AcceptanceTrackingRWalk(**internal_kwargs)
            kwargs["bound"] = "none"
            logger.info(
                f"Using the bilby-implemented ensemble rwalk sampling method with ACT "
                f"estimated chain length. An average of {2 * kwargs['sample'].nact} "
                f"steps will be accepted up to chain length {kwargs['sample'].maxmcmc}."
            )

        elif kwargs["bound"] == "live":
            logger.info(
                "Live-point based bound method requested with dynesty sample "
                f"'{kwargs['sample']}', overwriting to 'multi'"
            )
            kwargs["bound"] = "multi"

        self.init_sampler_kwargs = kwargs


    def get_nested_sampler(self, live_points, pool):
        """
        Returns the dynesty nested sampler, getting most arguments
        from the object's attributes

        Parameters
        ----------
        live_points: (numpy.ndarraym, numpy.ndarray, numpy.ndarray)
            The set of live points, in the same format as returned by
            get_initial_points_from_prior

        pool: schwimmbad.MPIPool
            Schwimmbad pool for MPI parallelisation
            (pbilby implements a modified version: MPIPoolFast)


        Returns
        -------
        dynesty.NestedSampler

        """
        return dynesty.NestedSampler(
            self.pooled_log_likelihood_function,
            self.pooled_prior_transform_function,
            pool=pool,
            live_points=live_points,
            rstate=self.rstate,
            **self.init_sampler_kwargs,
        )

    def get_initial_points_from_prior(self, pool):
        """
        Generates a set of initial points, drawn from the prior

        Parameters
        ----------
        pool: schwimmbad.MPIPool
            Schwimmbad pool for MPI parallelisation

        Returns
        -------
        (numpy.ndarraym, numpy.ndarray, numpy.ndarray)
            Returns a tuple (unit, theta, logl) where
            unit: point in the unit cube
            theta: scaled value to prior space
            logl: log(likelihood)

        """
        # Create a new rstate for each point, otherwise each task will generate
        # the same random number, and the rstate on master will not be incremented.
        # The argument to self.rstate.integers() is a very large integer.
        # These rstates aren't used after this map, but each time they are created,
        # a different (but deterministic) seed is used.
        seed_gen = SeedSequence(self.rstate.integers(9223372036854775807))
        rstates = [Generator(PCG64(n)) for n in seed_gen.spawn(self.nlive)]
        initial_points = pool.map(self.pooled_initial_point_function, rstates)

        u_list = [point[0] for point in initial_points]
        v_list = [point[1] for point in initial_points]
        l_list = [point[2] for point in initial_points]

        return np.array(u_list), np.array(v_list), np.array(l_list)

    def format_result(
        self,
        worker_run,
        data_dump,
        out,
        nested_samples,
        sampler_kwargs,
        sampling_time,
        rejection_sample_posterior=True,
    ):
        """
        Packs the variables from the run into a bilby result object

        Parameters
        ----------
        worker_run: WorkerRun
        data_dump: str
            Path to the *_data_dump.pickle file
        out: dynesty.results.Results
            Results from the dynesty sampler
        nested_samples: pandas.core.frame.DataFrame
            DataFrame of the weights and likelihoods
        sampler_kwargs: dict
            Dictionary of keyword arguments for the sampler
        sampling_time: float
            Time in seconds spent sampling
        rejection_sample_posterior: bool
            Whether to generate the posterior samples by rejection sampling the
            nested samples or resampling with replacement

        Returns
        -------
        result: bilby.core.result.Result
            result object with values written into its attributes
        """

        result = Result(self.label, self.outdir, search_parameter_keys=self.sampling_keys)
        result.priors = worker_run.priors
        result.nested_samples = nested_samples
        result.meta_data = worker_run.data_dump["meta_data"]
        result.meta_data["command_line_args"]["sampler"] = "parallel_bilby"
        result.meta_data["data_dump"] = data_dump
        result.meta_data["likelihood"] = worker_run.likelihood.meta_data
        result.meta_data["sampler_kwargs"] = self.init_sampler_kwargs
        result.meta_data["run_sampler_kwargs"] = sampler_kwargs
        result.meta_data["injection_parameters"] = worker_run.injection_parameters
        result.injection_parameters = worker_run.injection_parameters

        weights = np.exp(out["logwt"] - out["logz"][-1])
        if rejection_sample_posterior:
            result.samples, keep = rejection_sample(out.samples, weights, self.rstate)
            result.log_likelihood_evaluations = out.logl[keep]
            logger.info(
                f"Rejection sampling nested samples to obtain {sum(keep)} posterior samples"
            )
        else:
            result.samples = dynesty.utils.resample_equal(out.samples, weights)
            result.log_likelihood_evaluations = reorder_loglikelihoods(
                unsorted_loglikelihoods=out.logl,
                unsorted_samples=out.samples,
                sorted_samples=result.samples,
            )
            logger.info("Resampling nested samples to posterior samples in place.")

        result.log_evidence = out.logz[-1] + worker_run.likelihood.noise_log_likelihood()
        result.log_evidence_err = out.logzerr[-1]
        result.log_bayes_factor = result.log_evidence - result.log_noise_evidence
        result.sampling_time = sampling_time
        result.num_likelihood_evaluations = np.sum(out.ncall)

        result.samples_to_posterior(likelihood=worker_run.likelihood, priors=result.priors)
        return result

class WorkerRun:
    """
    An object with methods to be called in parallelised tasks.

    Parameters: 
    data_dump: a pickle-file containing all relevant data to create priors and likelihoods.
    """
    def __init__(
            self, 
            data_dump,
            bilby_zero_likelihood_mode=False
            ):
        
        ## Load the data dump
        if not data_dump.endswith("_dump.pickle"):
            test_out = os.path.join(os.getcwd(), data_dump)
            test_dump = glob(f"{test_out}/data/*_dump.pickle")
            data_dump = test_dump[0]

        # Read data dump from the pickle file
        with open(data_dump, "rb") as file:
            data_dump = pickle.load(file)

        ## Set properties from the data dump
        self.data_dump = data_dump
        self.args = data_dump["args"]
        self.injection_parameters = data_dump.get("injection_parameters", None)
        self.zero_likelihood_mode=bilby_zero_likelihood_mode

        ## Set up the priors
        self.compose_priors(data_dump["prior_file"])

        ## Set up the likelihood
        logger.setLevel(logging.WARNING)
        self.likelihood = setup_nmma_likelihood(data_dump,
            self.priors, self.args, logger)
        logger.setLevel(logging.INFO)

        check_keys = self.sampling_keys.copy()
        check_keys.extend(self.fixed_keys)
        self.likelihood.check_parameter_equivalencies(check_keys)


    def compose_priors(self, prior_file):
        """
        Routine to create a bilby-Prior object from a prior-file and to modify it for NMMA

        Parameters
        ----------
        prior_file: str
            The path to the prior-file

        Returns
        -------
        priors: bilby.gw.prior.PriorDict
            a bilby-Prior object

        """
        priors=PriorDict.from_json(prior_file)

        sampling_keys = []
        fixed_keys = []
        for k, prior in priors.items():
            if isinstance(prior, Constraint):
                continue
            elif prior.is_fixed:
                fixed_keys.append(k)
            else:
                sampling_keys.append(k)

        self.sampling_keys = sampling_keys
        self.ndim = len(sampling_keys)
        self.fixed_keys = fixed_keys
        self.fixed_prior = {key: priors[key].peak for key in fixed_keys}

        periodic = []
        reflective = []
        for ii, key in enumerate(sampling_keys):
            if priors[key].boundary == "periodic":
                logger.debug(f"Setting periodic boundary for {key}")
                periodic.append(ii)
            elif priors[key].boundary == "reflective":
                logger.debug(f"Setting reflective boundary for {key}")
                reflective.append(ii)

        if len(periodic) == 0:
            periodic = None
        if len(reflective) == 0:
            reflective = None
        
        self.periodic = periodic
        self.reflective = reflective
        self.priors = priors
    

    def prior_transform_function(self, u_array):
        """
        Calls the bilby rescaling function on an array of values

        Parameters
        ----------
        u_array: (float, array-like)
            The values to rescale

        Returns
        -------
        (float, array-like)
            The rescaled values

        """
        return self.priors.rescale(self.sampling_keys, u_array)


    def log_likelihood_function(self, v_array):
        """
        Calculates the log(likelihood)

        Parameters
        ----------
        v_array: (float, array-like)
            The sampling parameters' values

        Returns
        -------
        (float, array-like)
            The resulting likelihood.

        """
        if self.zero_likelihood_mode:
            return 0
        parameters = {key: v for key, v in zip(self.sampling_keys, v_array)}
        parameters.update(self.fixed_prior)
        # During sampling, working in likelihood ratio space is preferable
        # we add the noise log-likelihood later to retrieve the full log-likelihood
        return self.likelihood.log_likelihood_ratio(parameters)

    def log_prior_function(self, v_array):
        """
        Calculates the log of the prior

        Parameters
        ----------
        v_array: (float, array-like)
            The prior values

        Returns
        -------
        (float, array-like)
            The log probability of the values

        """
        params = {key: t for key, t in zip(self.sampling_keys, v_array)} 
        # params.update({key: self.priors[key].peak for key in self.fixed_keys})
        # print(params.keys())
        return self.priors.ln_prob(params)


    # @staticmethod
    def get_initial_point_from_prior(self, rstate):
        """
        Draw initial points from the prior subject to constraints applied both to
        the prior and the likelihood.

        We remove any points where the likelihood or prior is infinite or NaN.

        The `log_likelihood_function` often converts infinite values to large
        finite values so we catch those.
        """
        bad_values = [ np.inf, np.nan_to_num(np.inf),
                      -np.inf, np.nan_to_num(- np.inf), np.nan]
        
        while True:
            unit = rstate.random(self.ndim)
            theta = self.prior_transform_function(unit)

            if self.log_prior_function(theta) in bad_values:
                continue
            logl = self.log_likelihood_function(theta)
            if logl in bad_values:
                continue
            return unit, theta, logl
                
    def final_diagnostics(self, result):

        print(f"Sampling time = {datetime.timedelta(seconds=result.sampling_time)}s")
        print(f"Number of lnl calls = {result.num_likelihood_evaluations}")
        print(result)
        if self.args.plot:
            result.plot_corner()
            bestfit_params = read_bestfit_from_posterior(self.args)
            for lhood in self.likelihood.likelihoods:
                lhood.final_diagnostics(bestfit_params, self.args, result)
