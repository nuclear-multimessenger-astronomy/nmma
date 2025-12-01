import os
import sys
from io import BufferedWriter
from glob import glob
import pickle
from functools import wraps
from time import time
from datetime import timedelta

from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
from numpy.random import Generator, PCG64, SeedSequence

from bilby.core.utils import logger
from bilby.core.prior import PriorDict, Constraint
from bilby.core.result import Result
from bilby.core.sampler import dynesty3_utils  as dy_utils
from bilby.core.sampler.dynesty import dynesty_stats_plot
import dynesty
from dynesty.plotting import traceplot, runplot 


from ..joint.joint_likelihood import setup_nmma_likelihood
from ..joint.utils import reorder_loglikelihoods, rejection_sample, read_bestfit_from_posterior


def time_storage(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        duration = timedelta(seconds=time()-start)
        logger.info(f"{func.__name__} took {duration} ")
        return result

    return wrapper

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
        maxmcmc=5000,
        naccept=60,
        nact=2,
        sampling_seed=42,
        sampler_kwargs={},
        sampler_init_kwargs={}
    ):
        
        self.args = args
        # If the run dir has not been specified, get it from the args
        if outdir is None:
            outdir = self.args.outdir
        os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir

        # If the label has not been specified, get it from the args
        if label is None:
            label = self.args.label
        self.label = label

        self.resume_file = f"{self.outdir}/{self.label}_checkpoint_resume.pickle" 
        self.temp_ext='parquet'

        # Create a random generator, which is saved across restarts
        # This ensures that runs are fully deterministic, which is important
        # for reproducibility
        self.rstate = Generator(PCG64(sampling_seed))
        logger.debug(
            f"Setting random state = {self.rstate} (seed={sampling_seed})"
        )
        ## Set some basic attributes
        self.sampling_keys = sampling_keys
        logger.info(f"sampling keys:{sampling_keys}")
        for name in ['periodic', 'reflective']:
            if name in sampler_init_kwargs:
                logger.info(
                    f"{name} keys: {[sampling_keys[ii] for ii in sampler_init_kwargs[name]]}"
                )
        self.pooled_log_likelihood_function = pooled_log_likelihood_function
        self.pooled_prior_transform_function= pooled_prior_transform_function
        self.pooled_initial_point_function =  pooled_initial_point_function

        # dynesty3 sampler kwargs
        self.dlogz = sampler_kwargs['dlogz']
        self.sampler_kwargs = sampler_kwargs
        self._init_sampler_kwargs(sampler_init_kwargs, nact, naccept, maxmcmc)
        self.nlive = sampler_init_kwargs['nlive']

    def _init_sampler_kwargs(self, kwargs, nact, naccept, maxmcmc):
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

    def start_sampler(self, pool):
        """
        Start from a saved state or initialise.

        The required information to reconstruct the state of the run is read from a
        pickle file.

        Parameters
        ----------
        pool: a Schwimmbad-pool object

        Returns
        -------
        sampler: dynesty.NestedSampler
            If a resume file exists and was successfully read, the nested sampler 
            instance updated with the values stored to disk. If unavailable, create the initial state from scratch.
        sampling_time: float
            The current sampling time
        """

        if os.path.isfile(self.resume_file):
            logger.info(f"Reading resume file {self.resume_file}")
            with open(self.resume_file, "rb") as file:
                sampler = pickle.load(file)
                if sampler.added_live:
                    sampler._remove_live_points()

                #reset pool
                sampler.nqueue = -1
                sampler.pool = pool
                sampler.queue_size = pool.size
                sampler.mapper = pool.map
                try:
                    sampling_time = sampler.sampling_time
                except AttributeError:
                    sampling_time = sampler.resume_kwargs["sampling_time"]

            sampler.prior_transform = self.pooled_prior_transform_function
            sampler.loglikelihood = dynesty.utils.LogLikelihood(
                self.pooled_log_likelihood_function, sampler.ndim)
            return sampler, sampling_time
        
        else:
            logger.info(f"Resume file {self.resume_file} does not exist. "
            f"Initializing sampling points with pool size={pool.size}")
            live_points = self.get_initial_points_from_prior(pool)
            logger.info( f"Initialize NestedSampler with {self.init_sampler_kwargs}")
            return self.get_nested_sampler(live_points, pool), 0

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


    def stdout_sampling_log(self, **kwargs):
        """Logs will look like:
        #:282|eff(%):26.406|logl*:-inf<-160.2<inf|logz:-165.5+/-0.1|dlogz:1038.1>0.1

        Adapted from dynesty
        https://github.com/joshspeagle/dynesty/blob/bb1c5d5f9504c9c3bbeffeeba28ce28806b42273/py/dynesty/utils.py#L349
        """
        niter, _, mid_str, _ = dynesty.utils.get_print_fn_args(dlogz = self.dlogz,**kwargs)
        custom_str = [f"#: {niter:d}"] + mid_str
        custom_str = "|".join(custom_str).replace(" ", "")
        sys.stdout.write("\033[K" + custom_str + "\r")
        sys.stdout.flush()

    def checkpointing(self, sampler, sampling_time, checkpoint_plot=False):
        self.write_current_state(sampler, sampling_time )
        self.write_sample_dump(sampler.saved_run.D)
        if checkpoint_plot:
            self.plot_current_state(sampler)

    @time_storage
    def write_sample_dump(self, data):
        """Writes a checkpoint file """
        samples_file = f'{self.outdir}/{self.label}_samples.{self.temp_ext}'
        weights = np.exp(data["logwt"] - data["logz"][-1])
        samples, keep = rejection_sample(data["v"], weights, self.rstate)

        logger.info(f"Writing {np.sum(keep)} current samples to {samples_file}")
        df = DataFrame(samples, columns=self.sampling_keys)
        if samples_file.endswith(".dat"):
            df.to_csv(samples_file, index=False, header=True, sep=" ")
        else:
            df.to_parquet(samples_file, index=False)

    @time_storage
    def write_current_state(self, sampler, sampling_time):
        """Writes a checkpoint file

        Parameters
        ----------
        sampler: dynesty.NestedSampler
            The sampler object itself
        sampling_time: float
            The total sampling time in seconds
        """
        print("")
        try:
            time_elapsed = timedelta(seconds= time()-os.path.getmtime(self.resume_file))
            logger.info(f"Start checkpoint writing (last checkpoint {time_elapsed} ago)"
            )
        except FileNotFoundError:
            logger.info("Start checkpoint writing (no previous checkpoint)")

        # avoid expensive pickling of easily rebuilt objects
        pool = sampler.pool
        logl_func = sampler.loglikelihood
        prior_func = sampler.prior_transform
        try:
            # Temporarily remove to accelerate pickling
            sampler.pool = None
            sampler.loglikelihood = None
            sampler.prior_transform = None
            sampler.mapper = map

            sampler.sampling_time = sampling_time
            temp_filename = f"{self.resume_file}.temp"
            with open(temp_filename, "wb") as file:
                with BufferedWriter(file) as buffer:
                    pickle.dump(sampler, buffer, protocol=pickle.HIGHEST_PROTOCOL)
            os.rename(temp_filename, self.resume_file)
            logger.info(f"Written checkpoint file {self.resume_file}")

            # reset after succesful pickle
            sampler.pool = pool
            sampler.mapper = pool.map
            sampler.loglikelihood = logl_func
            sampler.prior_transform = prior_func
        except:
            logger.warning("Cannot write pickle resume file!")

    @time_storage
    def plot_current_state(self, sampler):
        # labels = [label.replace("_", " ") for label in search_parameter_keys]
        for name, func, obj in zip (
            ["trace", "run", "stats"],
            [traceplot, runplot, dynesty_stats_plot],
            [sampler.results, sampler.results, sampler]):

            try: 
                fig, _ = func(obj)
                fig.tight_layout()
                fig.savefig(f"{self.outdir}/{self.label}_checkpoint_{name}.png")

            except Exception as e:
                logger.warning(e)
                logger.warning(f"Failed to create dynesty {name} plot at checkpoint")
            finally:
                plt.close("all")


    def format_result(
        self,
        worker_run,
        sampler_result,
        result_format,
        rejection_sample_posterior=True,
    ):
        """
        Packs the variables from the run into a bilby result object

        Parameters
        ----------
        worker_run: WorkerRun
        out: dynesty.results.Results
            Results from the dynesty sampler
        result_format: str
            The format to save the result
        rejection_sample_posterior: bool
            Whether to generate the posterior samples by rejection sampling the
            nested samples or resampling with replacement
        """

        nested_samples = DataFrame(sampler_result.samples, columns=self.sampling_keys)
        nested_samples["log_likelihood"] = sampler_result.logl
        log_noise_evidence= worker_run.likelihood.noise_log_likelihood()
        log_bayes = sampler_result.logz[-1]
        
        meta_data = worker_run.data_dump
        meta_data["args"] = vars(self.args) # convert Namespace to dict for storing
        meta_data["likelihood"] = worker_run.likelihood.meta_data
        meta_data["sampler_kwargs"] = self.init_sampler_kwargs
        meta_data["run_sampler_kwargs"] = self.sampler_kwargs

        result = Result(self.label, self.outdir, search_parameter_keys=self.sampling_keys, 
                        priors =  worker_run.priors, nested_samples = nested_samples, 
                        injection_parameters= worker_run.injection_parameters, 
                        sampling_time=self.sampling_time, meta_data=meta_data,
                        num_likelihood_evaluations=np.sum(sampler_result.ncall),
                        log_noise_evidence=log_noise_evidence, log_bayes_factor=log_bayes,
                        log_evidence_err=sampler_result.logzerr[-1], 
                        log_evidence=log_noise_evidence + log_bayes
        )

        weights = np.exp(sampler_result["logwt"] - log_bayes)
        if rejection_sample_posterior:
            result.samples, keep = rejection_sample(sampler_result.samples, weights, self.rstate)
            result.log_likelihood_evaluations = sampler_result.logl[keep]
            logger.info(f"Rejection sampling nested samples to obtain {sum(keep)} posterior samples")

        else:
            result.samples = dynesty.utils.resample_equal(sampler_result.samples, weights)
            result.log_likelihood_evaluations = reorder_loglikelihoods(
                unsorted_loglikelihoods=sampler_result.logl,
                unsorted_samples=sampler_result.samples,
                sorted_samples=result.samples,
            )
            logger.info("Resampling nested samples to posterior samples in place.")


        result.samples_to_posterior(priors=result.priors)
        result.posterior = worker_run.likelihood.posterior_conversion(result.posterior)
        result.save_posterior_samples()

        logger.info(f"Saving result to {self.outdir}/{self.label}_result.{result_format}")
        try:
            result.save_to_file(extension=result_format)
        except Exception as e:
            logger.warning(f"Failed to save result to {result_format}: {e}"
                           "Trying to save as json instead.")
            result.save_to_file(extension="json")


        worker_run.final_diagnostics(result)

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
        self.likelihood = setup_nmma_likelihood(data_dump,
            self.priors, self.args, logger)

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

        print(f"Sampling time = {result.sampling_time} seconds")
        print(f"Number of lnl calls = {result.num_likelihood_evaluations}")
        print(result)
        if self.args.plot:
            result.plot_corner()
            bestfit_params = read_bestfit_from_posterior(self.args)
            self.likelihood.final_diagnostics(bestfit_params, self.args, result)
