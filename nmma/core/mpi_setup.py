import os
import sys
import traceback
from io import BufferedWriter
from copy import deepcopy
import pickle
import signal
from functools import wraps
from time import time
from datetime import timedelta
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
from numpy.random import Generator, PCG64, SeedSequence
from schwimmbad import MPIPool, MultiPool

from bilby.core.sampler import base_sampler as bs, dynesty3_utils  as dy_utils
from bilby.core.sampler.dynesty import dynesty_stats_plot
import dynesty
from dynesty.plotting import traceplot, runplot 

from .conversion import label_mapping
from .utils import  rejection_sample, read_bestfit_from_posterior, logger
from .parsing import process_sampler_kwargs


def time_storage(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        duration = timedelta(seconds=time()-start)
        logger.info(f"{func.__name__} took {duration} ")
        return result

    return wrapper

class Worker(bs.NestedSampler):
    """
    An object with methods to be called in parallelised tasks.

    Parameters: 
    data_dump: a pickle-file containing all relevant data to create priors and likelihoods.
    """
    def __init__(
        self, args, prior, likelihood,
        injection_parameters,  plot = False,  
        skip_import_verification = True,
        ):
        
        args.plot = plot
        self.args = args
        self.outdir = args.outdir
        self.label = args.label

        os.makedirs(args.outdir, exist_ok=True)


        super().__init__(
            likelihood, prior, self.outdir, self.label,
            injection_parameters = injection_parameters,
            skip_import_verification = skip_import_verification,
            plot= plot,
            soft_init=True,
            use_ratio = True,
            
        )

    
    def log_likelihood(self, theta):
        """

        Parameters
        ==========
        theta: list
            List of values for the likelihood parameters

        Returns
        =======
        float: Log-likelihood or log-likelihood-ratio given the current
            likelihood.parameter values

        """

        params = deepcopy(self.parameters)
        params.update({key: t for key, t in zip(self._search_parameter_keys, theta)})
        # During sampling, working in likelihood ratio space is preferable
        # we add the noise log-likelihood later to retrieve the full log-likelihood
        return self.likelihood.log_likelihood_ratio(params)


    def get_initial_point_from_prior(self, rstate):
        """
        Draw initial points from the prior subject to constraints applied both to
        the prior and the likelihood.

        We remove any points where the likelihood or prior is infinite or NaN.

        The `log_likelihood` often converts infinite values to large
        finite values so we catch those.
        """
        bad_values = [ np.inf, np.nan_to_num(np.inf),
                      -np.inf, np.nan_to_num(- np.inf), np.nan]
        while True:
            unit = rstate.random(self.ndim)
            theta = self.prior_transform(unit)

            if self.log_prior(theta) in bad_values:
                continue
            logl = self.log_likelihood(theta)
            if logl in bad_values:
                continue

            return unit, theta, logl
                
    def checkpointing(self, checkpoint_plot=False, message=None):
        """
        Checkpointing function to be called periodically during sampling.

        Parameters
        ==========
        checkpoint_plot: bool
            Whether to create checkpoint plots
        message: str
            Message to log after checkpointing
        """
        os.wait()
        pass  # only to be executed in main process

class Dynesty(Worker):
   
    def __init__(
        self,
        args, prior, likelihood,
        injection_parameters = None,
        maxmcmc=5000,
        naccept=60,
        nact=2,
        sampling_seed=42,
        sampler_kwargs={},
        sampler_init_kwargs={},
        plot= False,
        meta_data = {},
    ):  
        breakpoint()
        super().__init__(args, prior, likelihood, injection_parameters, 
                        plot, skip_import_verification = False)
        
        self.resume_file = f"{self.outdir}/{self.label}_checkpoint_resume.pickle" 
        self.samples_file= f'{self.outdir}/{self.label}_samples.parquet'

        # Create a random generator, which is saved across restarts
        # This ensures that runs are fully deterministic, which is important
        # for reproducibility
        self.rstate = Generator(PCG64(sampling_seed))
        logger.debug(f"Setting random state = {self.rstate} (seed={sampling_seed})")
        
        # dynesty3 sampler kwargs
        self.dlogz = sampler_kwargs['dlogz']
        self.sampler_kwargs = sampler_kwargs
        self._init_sampler_kwargs(sampler_init_kwargs, nact, naccept, maxmcmc)
        self.nlive = sampler_init_kwargs['nlive']
        self.meta_data = meta_data

    
    def _init_sampler_kwargs(self, kwargs, nact, naccept, maxmcmc):
        """
        Mostly stolen from bilby.core.sampler.dynesty to set up the internal sampler kwargs
        """
        

        periodic = []
        reflective = []
        for ii, key in enumerate(self._search_parameter_keys):
            if self.priors[key].boundary == "periodic":
                logger.debug(f"Setting periodic boundary for {key}")
                periodic.append(ii)
            elif self.priors[key].boundary == "reflective":
                logger.debug(f"Setting reflective boundary for {key}")
                reflective.append(ii)

        if len(periodic) == 0:
            periodic = None
        if len(reflective) == 0:
            reflective = None
        kwargs |= dict(   
            ndim=len(self._search_parameter_keys),
            periodic=periodic,
            reflective=reflective)
        
        internal_kwargs = dict(
            ndim=kwargs['ndim'],
            nonbounded=None,
            periodic = kwargs["periodic"],
            reflective = kwargs["reflective"],
            maxmcmc = maxmcmc
        )
        if kwargs["sample"] == "act-walk":
            internal_kwargs["nact"] = nact
            sample_meth = dy_utils.ACTTrackingEnsembleWalk(**internal_kwargs)
            kwargs["sample"] = sample_meth
            kwargs["bound"] = "none"

            logger.info(
                f"Using the bilby-implemented ensemble rwalk sampling tracking the "
                f"autocorrelation function and thinning by {sample_meth.thin} with "
                f"maximum length {sample_meth.thin * sample_meth.maxmcmc}."
            )

        elif kwargs["sample"] == "acceptance-walk":
            internal_kwargs["naccept"] = naccept
            internal_kwargs["walks"] = kwargs["walks"]
            sample_meth = dy_utils.EnsembleWalkSampler(**internal_kwargs)
            kwargs["sample"] = sample_meth
            kwargs["bound"] = "none"
            logger.info(
                f"Using the bilby-implemented ensemble rwalk sampling method with an "
                f"average of {sample_meth.naccept} accepted steps up to chain "
                f"length {sample_meth.maxmcmc}."
            )

        elif kwargs["sample"] == "rwalk":
            internal_kwargs["nact"] = nact
            sample_meth = dy_utils.AcceptanceTrackingRWalk(**internal_kwargs)
            kwargs["sample"] = sample_meth
            kwargs["bound"] = "none"
            logger.info(
                f"Using the bilby-implemented ensemble rwalk sampling method with ACT "
                f"estimated chain length. An average of {2 * sample_meth.nact} "
                f"steps will be accepted up to chain length {sample_meth.maxmcmc}."
            )

        elif kwargs["bound"] == "live":
            logger.info(
                "Live-point based bound method requested with dynesty sample "
                f"'{kwargs['sample']}', overwriting to 'multi'"
            )
            kwargs["bound"] = "multi"

        self.init_sampler_kwargs = kwargs
    
    @time_storage
    def start_sampler(self, pool, log_likelihood, prior_transform, find_live):
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

            sampler.prior_transform = prior_transform
            sampler.loglikelihood = dynesty.utils.LogLikelihood(
                log_likelihood, sampler.ndim)
            
            self.sampling_time = sampler.sampling_time
            self.sampler = sampler
        else:
            logger.info(f"Resume file {self.resume_file} does not exist. "
            f"Initializing sampling points with pool size={pool.size}")
            live_points = self.get_initial_points_from_prior(pool, find_live)
            logger.info( f"Initialize NestedSampler with {self.init_sampler_kwargs}")
            self.sampler = dynesty.NestedSampler(log_likelihood, prior_transform,
                                            pool=pool,
                                            live_points=live_points,
                                            rstate=self.rstate,
                                            **self.init_sampler_kwargs,
                                        )
            self.sampling_time = 0.0


        self.init_sampler_kwargs.pop('sample')
        logger.info(f"Run criteria: {self.sampler_kwargs}")

        logger.info(f"Starting sampling for job {self.label}, with pool size={pool.size}")

    
    @time_storage
    def get_initial_points_from_prior(self, pool, find_live):
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
        initial_points = pool.map(find_live, rstates)

        u_list = [point[0] for point in initial_points]
        v_list = [point[1] for point in initial_points]
        l_list = [point[2] for point in initial_points]

        return np.array(u_list), np.array(v_list), np.array(l_list)

    def get_step_info_str(self, **kwargs):
        """Generate a log string for the current sampling step.
        Logs will look like:
        #:282|eff(%):26.406|logl*:-inf<-160.2<inf|logz:-165.5+/-0.1|dlogz:1038.1>0.1

        Adapted from dynesty
        https://github.com/joshspeagle/dynesty/blob/bb1c5d5f9504c9c3bbeffeeba28ce28806b42273/py/dynesty/utils.py#L349

        Parameters
        ----------
        kwargs: dict
            keyword arguments passed to dynesty.utils.get_print_fn_args

        Returns
        -------
        str
            Formatted log string for the current sampling step.
        """
        niter, _, mid_str, _ = dynesty.utils.get_print_fn_args(
            dlogz = self.dlogz, ncall=self.sampler.ncall,**kwargs)
        custom_str = [f"#: {niter:d}"] + mid_str
        custom_str = "|".join(custom_str).replace(" ", "")
        return custom_str
    

    def run_sampler(self, check_point_delta_t=1800, n_check_point=1000, max_its=1e10, max_run_time=1e10, checkpoint_plot=False):
        logger.info(f"Beginning sampling with checkpoints every {check_point_delta_t} seconds or {n_check_point} iterations \n "
                    f" until max {max_its} iterations or max run time {timedelta(seconds=max_run_time)}.")
        run_time = 0.
        t_start = time()
        last_checkpoint_time= t_start
        last_checkpoint_it = 0
        
        for it, res in enumerate(self.sampler.sample(**self.sampler_kwargs)):

            self.stdout_sampling_log(results=res, niter=it)
            run_time = time() - t_start
            checkpoint_interval = time()- last_checkpoint_time

            if it >= max_its or run_time > max_run_time:
                self.sampling_time +=  checkpoint_interval  
                self.checkpointing(checkpoint_plot,
                    f"{it} of max {max_its} iterations completed after {timedelta(seconds=run_time)} " 
                    f" sampling time of max {timedelta(seconds=max_run_time)}. Stopping." )
                return 
            
            elif (
                # checkpoint criteria
                checkpoint_interval >= check_point_delta_t
                or (it - last_checkpoint_it >= n_check_point) 
            ):
                self.sampling_time +=  checkpoint_interval
                last_checkpoint_time = time() 
                last_checkpoint_it = it
                self.checkpointing(checkpoint_plot, 
                    self.get_step_info_str(results=res, niter=it))

        # Adding the final set of live points.
        for it_final, res in enumerate(self.sampler.add_live_points()):
            pass

        # Create a final checkpoint in case anything happens during the formatting
        self.sampling_time +=  time() - last_checkpoint_time
        self.write_current_state()
        self.plot_current_state()
        return self.sampler.results
    
    def stdout_sampling_log(self, **kwargs):
        sys.stdout.write(f"\033[K {self.get_step_info_str(**kwargs)}\r")
        sys.stdout.flush()

    def checkpointing(self, checkpoint_plot=False, message= None):
        self.write_current_state()
        self.write_sample_dump(self.sampler.saved_run.D)
        if checkpoint_plot:
            self.plot_current_state()
        if message:
            logger.info(message)

    @time_storage
    def write_sample_dump(self, data):
        """Writes a checkpoint file """
        weights = np.exp(data["logwt"] - data["logz"][-1])
        samples, keep = rejection_sample(data["v"], weights, self.rstate)

        logger.info(f"Writing {np.sum(keep)} current samples to {self.samples_file}")
        df = DataFrame(samples, columns=self._search_parameter_keys)
        df.to_parquet(self.samples_file, index=False)

    @time_storage
    def write_current_state(self):
        """Writes a checkpoint file

        Parameters
        ----------
        sampler: dynesty.NestedSampler
            The sampler object itself
        sampling_time: float
            The total sampling time in seconds
        """
        print("")
        cp_time = time()-os.path.getmtime(self.resume_file) if os.path.isfile(self.resume_file) else self.sampling_time
        logger.info(f"Write new checkpoint after {timedelta(seconds = cp_time)}")

        # avoid expensive pickling of easily rebuilt objects
        pool = self.sampler.pool
        logl_func = self.sampler.loglikelihood
        prior_func = self.sampler.prior_transform
        try:
            # Temporarily remove to accelerate pickling
            self.sampler.pool = None
            self.sampler.loglikelihood = None
            self.sampler.prior_transform = None
            self.sampler.mapper = map

            self.sampler.sampling_time = self.sampling_time
            temp_filename = f"{self.resume_file}.temp"
            with open(temp_filename, "wb") as file:
                with BufferedWriter(file) as buffer:
                    pickle.dump(self.sampler, buffer, protocol=pickle.HIGHEST_PROTOCOL)
            os.rename(temp_filename, self.resume_file)
            logger.info(f"Written checkpoint file {self.resume_file}")

            # reset after succesful pickle
            self.sampler.pool = pool
            self.sampler.mapper = pool.map
            self.sampler.loglikelihood = logl_func
            self.sampler.prior_transform = prior_func
        except:
            logger.warning("Cannot write checkpoint file!")

    @time_storage
    def plot_current_state(self):
        # labels = [label.replace("_", " ") for label in search_parameter_keys]
        for name, func, obj in zip (
            ["trace", "run", "stats"],
            [traceplot, runplot, dynesty_stats_plot],
            [self.sampler.results, self.sampler.results, self.sampler]):

            try: 
                fig, _ = func(obj)
                fig.tight_layout()
                fig.savefig(f"{self.outdir}/{self.label}_checkpoint_{name}.png")

            except Exception as e:
                logger.warning(e)
                logger.warning(f"Failed to create dynesty {name} plot at checkpoint")
            finally:
                plt.close("all")

    def storable_metadata(self):
        meta_data = self.meta_data
        meta_data["args"] = vars(self.args).copy() # convert Namespace to dict for storing
        meta_data["likelihood"] = self.likelihood.meta_data
        meta_data["sampler_kwargs"] = self.init_sampler_kwargs
        meta_data["run_sampler_kwargs"] = self.sampler_kwargs
        meta_data = self.floatify_dict(meta_data)
        return meta_data
    
    def floatify_dict(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = self.floatify_dict(v)
            elif isinstance(v, np.floating):
                d[k] = float(v)
        return d

    def format_result(
        self,
        sampler_result,
        result_format,
        rejection_sample_posterior=True,
    ):
        """
        Packs the variables from the run into a bilby result object

        Parameters
        ----------
        sampler_result: dynesty.results.Results
            Results from the dynesty sampler
        result_format: str
            The format to save the result
        rejection_sample_posterior: bool
            Whether to generate the posterior samples by rejection sampling the
            nested samples or resampling with replacement
        """

        nested_samples = DataFrame(sampler_result.samples, columns=self._search_parameter_keys)
        nested_samples["log_likelihood"] = sampler_result.logl
        log_noise_evidence= self.likelihood.noise_log_likelihood()
        log_bayes = sampler_result.logz[-1]
        
        result = self.result

        result.nested_samples = nested_samples 
        result.sampling_time=timedelta(seconds=self.sampling_time)
        logger.info(f"Sampling time = {result.sampling_time}")
        result.meta_data=self.storable_metadata()
        result.num_likelihood_evaluations=np.sum(sampler_result.ncall)
        logger.info(f"Number of lnl calls = {result.num_likelihood_evaluations}")
        result.log_noise_evidence=log_noise_evidence
        result.log_bayes_factor=log_bayes
        result.log_evidence_err=sampler_result.logzerr[-1]
        result.log_evidence=log_noise_evidence + log_bayes
    
        weights = np.exp(sampler_result["logwt"] - log_bayes)
        if rejection_sample_posterior:
            result.samples, keep = rejection_sample(sampler_result.samples, weights, self.rstate)
            result.log_likelihood_evaluations = sampler_result.logl[keep]
            logger.info(f"Rejection sampling nested samples to obtain {sum(keep)} posterior samples")

        else:
            result.samples = dynesty.utils.resample_equal(sampler_result.samples, weights, self.rstate)
            result.log_likelihood_evaluations = self.reorder_loglikelihoods(
                unsorted_loglikelihoods=sampler_result.logl,
                unsorted_samples=sampler_result.samples,
                sorted_samples=result.samples,
            )
            logger.info("Resampling nested samples to posterior samples in place.")


        result.samples_to_posterior(priors=result.priors)
        result.posterior = self.likelihood.posterior_conversion(result.posterior)
        result.save_posterior_samples()
        extra_keys = set(result.posterior.columns) - set(self._search_parameter_keys)
        posterior_keys = self._search_parameter_keys.copy()
        posterior_keys.extend(extra_keys)
        result.meta_data["posterior_keys"] = posterior_keys
        posterior_labels = result.parameter_labels_with_unit.copy()
        posterior_labels.extend([label_mapping.get(k, k) for k in extra_keys])
        result.meta_data["posterior_labels"] = posterior_labels
        if os.path.isfile(self.samples_file):
            os.remove(self.samples_file)  # remove temp file after succesful run

        logger.info(f"Saving result to {self.outdir}/{self.label}_result.{result_format}")
        result.save_to_file(extension=result_format)

        if self.plot:
            logger.info("Creating corner plot of posterior samples.")
            try:
                injection_parameters = {
                    k: v for k, v in self.injection_parameters.items()
                    if k in self._search_parameter_keys} if self.injection_parameters else None
                result.plot_corner(parameters = injection_parameters, priors=True, dpi = 200)
            except Exception as e:
                logger.warning(f"Failed to create corner plot: {e}")    
            try:
                logger.info("Creating diagnostic plots.")
                bestfit_params = read_bestfit_from_posterior(result.posterior, 'max_posterior')
                self.likelihood.final_diagnostics(bestfit_params, self.args, result)
            except Exception as e:
                logger.warning(f"Failed to create diagnostic plots: {e} \n{traceback.format_exc()}")
        logger.info("Finished formatting result.")
        return result



def pbilby_sampling(
    likelihood, prior, args, 
    injection_parameters, rank,
    pool_type = 'mpi',
    meta_data = {},
    **kwargs
):
    
    default_kwargs = dict(
        sampler_kwargs={},
        sampling_seed=42,
        plot=True,
        #
        maxmcmc=5000,
        naccept=60,
        nact=2,
        check_point_delta_t=1800,
        n_check_point=2000,
        max_its=1e10,
        max_run_time=1e10,
        checkpoint_plot=False,
        #
        rejection_sample_posterior=True,
        result_format="hdf5",
    )

    # priority: kwargs > args > defaults
    use_kwargs = {}
    for key in default_kwargs.keys():
        if key in kwargs:
            use_kwargs[key] = kwargs[key]
        elif hasattr(args, key):
            use_kwargs[key] = getattr(args, key)
        else:
            use_kwargs[key] = default_kwargs[key]

    use_kwargs |= kwargs # in case there are additional kwargs not in default_kwargs

    # Initialise a worker. this needs a global scope to allow 
    # persistence of states beyond the pool's scope.
    # Otherwise emulators retrace on each evaluation.
    global worker
    if rank == 0:
        sampler_init_kwargs, run_kwargs = process_sampler_kwargs(
            kwargs.pop('sampler_kwargs', {}), use_kwargs)

        worker = Dynesty(
            args, prior, likelihood,
            injection_parameters,
            maxmcmc=use_kwargs['maxmcmc'],
            nact=use_kwargs['nact'],
            naccept=use_kwargs['naccept'],
            sampling_seed=use_kwargs['sampling_seed'],
            sampler_kwargs = run_kwargs,
            sampler_init_kwargs=sampler_init_kwargs,
            plot=use_kwargs['plot'],
            meta_data=meta_data,
        )

    else:
        worker = Worker(args, prior, likelihood,
            injection_parameters,  plot = use_kwargs['plot'])

    ## graceful handling of preemptive shutdowns
    def handle_sigterm(signum, frame):
        try:
            worker.checkpointing(False,
                'Received termination signal. Checkpointing and exiting gracefully.')
            sys.exit()
        except Exception:
            pass

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT , handle_sigterm)
    signal.signal(signal.SIGUSR1, handle_sigterm) 

    POOL = MPIPool if pool_type == 'mpi' else MultiPool
    with POOL() as pool:
        result = None
        if pool.is_master():           
            worker.start_sampler(
                pool,
                pooled_log_likelihood, 
                pooled_prior_transform,
                pooled_initial_point_from_prior)

            results = worker.run_sampler(
                check_point_delta_t=use_kwargs['check_point_delta_t'],
                n_check_point=use_kwargs['n_check_point'],
                max_its=use_kwargs['max_its'],
                max_run_time=use_kwargs['max_run_time'],
                checkpoint_plot=use_kwargs['checkpoint_plot']
            )
            result = worker.format_result(
            results, use_kwargs['result_format'],
            use_kwargs['rejection_sample_posterior'])
    return result


# Worker functions. These are read in the global scope by each worker
def pooled_initial_point_from_prior(args):
    return worker.get_initial_point_from_prior(args)

def pooled_log_likelihood(v_array):
    return worker.log_likelihood(v_array)

def pooled_prior_transform(u_array):
    return worker.prior_transform(u_array)
