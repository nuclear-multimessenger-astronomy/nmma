import logging
import os
from glob import glob
import pickle

import dynesty
import numpy as np
import pandas as pd
from bilby.core.sampler.base_sampler import _SamplingContainer
from bilby.core.sampler.dynesty import DynestySetupError, _set_sampling_kwargs
from bilby.core.sampler.dynesty_utils import (
    AcceptanceTrackingRWalk,
    ACTTrackingRWalk,
    FixedRWalk,
    LivePointSampler,
    MultiEllipsoidLivePointSampler,
)
from bilby.core.utils import logger
from bilby_pipe.utils import convert_string_to_list
from bilby.core.prior import Interped, DeltaFunction, PriorDict, Constraint

from ...joint.joint_likelihood import setup_nmma_likelihood
from ...joint.conversion import MultimessengerConversion


class MainRun(object):
   
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
        dynesty_sample="acceptance-walk",
        nlive=5,
        dynesty_bound="live",
        walks=100,
        maxmcmc=5000,
        naccept=60,
        nact=2,
        facc=0.5,
        min_eff=10,
        enlarge=1.5,
        sampling_seed=0,
        proposals=None,
    ):
        ## Set some basic attributes
        self.sampling_keys = sampling_keys
        self.ndim = len(sampling_keys)
        self.maxmcmc = maxmcmc
        self.nact = nact
        self.naccept = naccept
        self.proposals = convert_string_to_list(proposals)
        self.nlive = nlive
        self.periodic = periodic
        self.reflective = reflective
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

        self.init_sampler_kwargs = dict(
            nlive=nlive,
            sample=dynesty_sample,
            bound=dynesty_bound,
            walks=walks,
            facc=facc,
            first_update=dict(min_eff=min_eff, min_ncall=2 * nlive),
            enlarge=enlarge,
        )

        self._set_sampling_method()

        # Create a random generator, which is saved across restarts
        # This ensures that runs are fully deterministic, which is important
        # for reproducibility
        self.sampling_seed = sampling_seed
        self.rstate = np.random.Generator(np.random.PCG64(self.sampling_seed))
        logger.debug(
            f"Setting random state = {self.rstate} (seed={self.sampling_seed})"
        )

    def _set_sampling_method(self):

        sample = self.init_sampler_kwargs["sample"]
        bound = self.init_sampler_kwargs["bound"]

        _set_sampling_kwargs((self.nact, self.maxmcmc, self.proposals, self.naccept))

        if sample not in ["rwalk", "act-walk", "acceptance-walk"] and bound in [
            "live",
            "live-multi",
        ]:
            logger.info(
                "Live-point based bound method requested with dynesty sample "
                f"'{sample}', overwriting to 'multi'"
            )
            self.init_sampler_kwargs["bound"] = "multi"
        elif bound == "live":
            dynesty.dynamicsampler._SAMPLERS["live"] = LivePointSampler
        elif bound == "live-multi":
            dynesty.dynamicsampler._SAMPLERS[
                "live-multi"
            ] = MultiEllipsoidLivePointSampler
        elif sample == "acceptance-walk":
            raise DynestySetupError(
                "bound must be set to live or live-multi for sample=acceptance-walk"
            )
        elif self.proposals is None:
            logger.warning(
                "No proposals specified using dynesty sampling, defaulting "
                "to 'volumetric'."
            )
            self.proposals = ["volumetric"]
            _SamplingContainer.proposals = self.proposals
        elif "diff" in self.proposals:
            raise DynestySetupError(
                "bound must be set to live or live-multi to use differential "
                "evolution proposals"
            )

        if sample == "rwalk":
            logger.info(
                "Using the bilby-implemented rwalk sample method with ACT estimated walks. "
                f"An average of {2 * self.nact} steps will be accepted up to chain length "
                f"{self.maxmcmc}."
            )
            if self.init_sampler_kwargs["walks"] > self.maxmcmc:
                raise DynestySetupError("You have maxmcmc < walks (minimum mcmc)")
            if self.nact < 1:
                raise DynestySetupError("Unable to run with nact < 1")
            dynesty.nestedsamplers._SAMPLING["rwalk"] = AcceptanceTrackingRWalk()
        elif sample == "acceptance-walk":
            logger.info(
                "Using the bilby-implemented rwalk sampling with an average of "
                f"{self.naccept} accepted steps per MCMC and maximum length {self.maxmcmc}"
            )
            dynesty.nestedsamplers._SAMPLING["acceptance-walk"] = FixedRWalk()
        elif sample == "act-walk":
            logger.info(
                "Using the bilby-implemented rwalk sampling tracking the "
                f"autocorrelation function and thinning by "
                f"{self.nact} with maximum length {self.nact * self.maxmcmc}"
            )
            dynesty.nestedsamplers._SAMPLING["act-walk"] = ACTTrackingRWalk()
        elif sample == "rwalk_dynesty":
            sample = sample.strip("_dynesty")
            self.init_sampler_kwargs["sample"] = sample
            logger.info(f"Using the dynesty-implemented {sample} sample method")

    def get_nested_sampler(self, live_points, pool, pool_size):
        """
        Returns the dynested nested sampler, getting most arguments
        from the object's attributes

        Parameters
        ----------
        live_points: (numpy.ndarraym, numpy.ndarray, numpy.ndarray)
            The set of live points, in the same format as returned by
            get_initial_points_from_prior

        pool: schwimmbad.MPIPool
            Schwimmbad pool for MPI parallelisation
            (pbilby implements a modified version: MPIPoolFast)

        pool_size: int
            Number of workers in the pool

        Returns
        -------
        dynesty.NestedSampler

        """
        sampler = dynesty.NestedSampler(
            self.pooled_log_likelihood_function,
            self.pooled_prior_transform_function,
            self.ndim,
            pool=pool,
            queue_size=pool_size,
            periodic=self.periodic,
            reflective=self.reflective,
            live_points=live_points,
            rstate=self.rstate,
            use_pool=dict(
                update_bound=True,
                propose_point=True,
                prior_transform=True,
                loglikelihood=True,
            ),
            **self.init_sampler_kwargs,
        )

        return sampler

    def get_initial_points_from_prior(self, pool, calculate_likelihood=True):
        """
        Generates a set of initial points, drawn from the prior

        Parameters
        ----------
        pool: schwimmbad.MPIPool
            Schwimmbad pool for MPI parallelisation
            (pbilby implements a modified version: MPIPoolFast)

        calculate_likelihood: bool
            Option to calculate the likelihood for the generated points
            (default: True)

        Returns
        -------
        (numpy.ndarraym, numpy.ndarray, numpy.ndarray, None)
            Returns a tuple (unit, theta, logl, blob)
            unit: point in the unit cube
            theta: scaled value
            logl: log(likelihood)
            blob: None

        """
        # Create a new rstate for each point, otherwise each task will generate
        # the same random number, and the rstate on master will not be incremented.
        # The argument to self.rstate.integers() is a very large integer.
        # These rstates aren't used after this map, but each time they are created,
        # a different (but deterministic) seed is used.
        sg = np.random.SeedSequence(self.rstate.integers(9223372036854775807))
        map_rstates = [
            np.random.Generator(np.random.PCG64(n)) for n in sg.spawn(self.nlive)
        ]

        args_list = [
            (
                calculate_likelihood,
                map_rstates[i],
            )
            for i in range(self.nlive)
        ]
        initial_points = pool.map(self.pooled_initial_point_function, args_list)
        u_list = [point[0] for point in initial_points]
        v_list = [point[1] for point in initial_points]
        l_list = [point[2] for point in initial_points]
        blobs = None

        return np.array(u_list), np.array(v_list), np.array(l_list), blobs


class WorkerRun(object):
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
        if data_dump is None:
            test_out = os.path.join(os.getcwd(), 'outdir')
            test_dump = glob(f"{test_out}/data/*_dump.pickle")
            data_dump = test_dump[0]

        # Read data dump from the pickle file
        with open(data_dump, "rb") as file:
            data_dump = pickle.load(file)

        ## Set properties from the data dump
        self.data_dump = data_dump
        self.args = data_dump["args"]
        self.messengers= data_dump["messengers"]
        self.analysis_modifiers = data_dump['analysis_modifiers']
        self.injection_parameters = data_dump.get("injection_parameters", None)
        self.zero_likelihood_mode=bilby_zero_likelihood_mode

        ## Set up the priors
        self.compose_priors(
                data_dump["prior_file"], self.args, 
                self.analysis_modifiers, logger
            )


        self.parameter_conversion=MultimessengerConversion(self.args, self.messengers, self.analysis_modifiers)
        # priors.conversion_function = param_conv.priors_conversion_function

        logger.setLevel(logging.WARNING)
        
        self.likelihood= setup_nmma_likelihood(data_dump,
            self.priors, self.args, self.messengers,  logger
            )
        
        logger.setLevel(logging.INFO)


    def compose_priors(self, prior_file, args, ana_modifiers, logger):
        """
        Routine to create a bilby-Prior object from a prior-file and to modify it for NMMA

        Parameters
        ----------
        prior_file: str
            The path to the prior-file
        args: Namespace
            The parser arguments

        Returns
        -------
        priors: bilby.gw.prior.PriorDict
            a bilby-Prior object

        """
        priors = PriorDict.from_json(prior_file)
        priors.convert_floats_to_delta_functions()

        ###adjust hubble prior if applicable
        if args.Hubble_weight:
            logger.info("Sampling over Hubble constant with pre-calculated prior")
            logger.info("Assuming the redshift prior is the Hubble flow")
            logger.info("Overwriting any Hubble prior in the prior file")
            try:
                Hubble_prior_data = pd.read_csv(args.Hubble_weight, delimiter=" ", header=0)
                xx = Hubble_prior_data.Hubble.to_numpy()
                yy = Hubble_prior_data.prior_weight.to_numpy()
            except:
                xx, yy =  np.loadtxt(args.Hubble_weight).T

            Hmin = xx[0]
            Hmax = xx[-1]

            priors["Hubble_constant"] = Interped(
                xx, yy, minimum=Hmin, maximum=Hmax, name="Hubble_constant"
            )

        # construct the eos prior
        if "tabulated_eos" in ana_modifiers:
            
            logger.info("Sampling over precomputed EOSs")
            xx = np.arange(0, args.Neos + 1)
            if args.eos_weight:
                eos_weight = np.loadtxt(args.eos_weight)
                yy = np.concatenate((eos_weight, [eos_weight[-1]]))
            else: 
                yy = np.ones_like(xx)/len(xx)
            eos_prior = Interped(xx, yy, minimum=0, maximum=args.Neos, name="EOS")
            priors["EOS"] = eos_prior


        # add the ratio_epsilon in case it is not present (for no-grb case)
        if "ratio_epsilon" not in priors:
            priors["ratio_epsilon"] = DeltaFunction(0.01, name="ratio_epsilon")
        
        self.priors=priors

        # check prior properties
        sampling_keys = []
        fixed_keys = []
        for p in priors:
            if isinstance(priors[p], Constraint):
                continue
            elif priors[p].is_fixed:
                fixed_keys.append(p)
            else:
                sampling_keys.append(p)

        self.sampling_keys = sampling_keys
        self.ndim = len(sampling_keys)
        self.fixed_keys = fixed_keys

        fixed_prior = {key: priors[key].peak for key in fixed_keys}
        # FIXME should not the RL-likelihood set this intrinsically?
        # if self.args.likelihood_type ==  'RelativeBinningGravitationalWaveTransient':
        #     fixed_prior.update(fiducial=0)
        self.fixed_prior = fixed_prior

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
    
    def evaluate_constraints(self, out_sample):
        prob = 1
        for key in self.priors:
            if isinstance(self.priors[key], Constraint) and key in out_sample:
                prob *= self.priors[key].prob(out_sample[key])
        return prob


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
        parameters, added_keys = self.parameter_conversion.convert_to_multimessenger_parameters(parameters)
        if self.evaluate_constraints(parameters) > 0:
            self.likelihood.parameters = parameters
            return (
                self.likelihood.sub_log_likelihood()
                - self.likelihood.noise_log_likelihood()
            )
        else:
            return np.nan_to_num(-np.inf)

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
    def get_initial_point_from_prior(self, args):
        """
        Draw initial points from the prior subject to constraints applied both to
        the prior and the likelihood.

        We remove any points where the likelihood or prior is infinite or NaN.

        The `log_likelihood_function` often converts infinite values to large
        finite values so we catch those.
        """

        (
            calculate_likelihood,
            rstate,
        ) = args
        bad_values = [np.inf, np.nan_to_num(np.inf), np.nan]
        while True:
            unit = rstate.random(self.ndim)
            theta = self.prior_transform_function(unit)

            if abs(self.log_prior_function(theta)) not in bad_values:
                if calculate_likelihood:
                    logl = self.log_likelihood_function(theta)
                    if abs(logl) not in bad_values:
                        return unit, theta, logl
                else:
                    return unit, theta, np.nan