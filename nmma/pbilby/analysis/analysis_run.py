import logging
import os
import pickle

import bilby
import dynesty
import numpy as np
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

from .likelihood import setup_nmma_likelihood, setup_nmma_gw_likelihood


class AnalysisRun(object):
    """
    An object with methods for driving the sampling run.

    Parameters: arguments to set the output path and control the dynesty sampler.
    """

    def __init__(
        self,
        data_dump,
        inference_favour,
        outdir=None,
        label=None,
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
        bilby_zero_likelihood_mode=False,
    ):
        self.maxmcmc = maxmcmc
        self.nact = nact
        self.naccept = naccept
        self.proposals = convert_string_to_list(proposals)

        # Read data dump from the pickle file
        with open(data_dump, "rb") as file:
            data_dump = pickle.load(file)

        ifo_list = data_dump["ifo_list"]
        waveform_generator = data_dump["waveform_generator"]
        waveform_generator.start_time = ifo_list[0].time_array[0]
        args = data_dump["args"]
        injection_parameters = data_dump.get("injection_parameters", None)

        args.weight_file = data_dump["meta_data"].get("weight_file", None)

        # If the run dir has not been specified, get it from the args
        if outdir is None:
            outdir = args.outdir
        else:
            # Create the run dir
            os.makedirs(outdir, exist_ok=True)

        # If the label has not been specified, get it from the args
        if label is None:
            label = args.label

        priors = bilby.gw.prior.PriorDict.from_json(data_dump["prior_file"])

        logger.setLevel(logging.WARNING)
        # split the likelihood for difference inference_favour
        assert inference_favour in ['nmma', 'nmma_gw'], "Invalid inference_favour"
        if inference_favour == 'nmma':
            light_curve_data = data_dump["light_curve_data"]
            likelihood, priors = setup_nmma_likelihood(
                interferometers=ifo_list,
                waveform_generator=waveform_generator,
                light_curve_data=light_curve_data,
                priors=priors,
                args=args,
            )

        elif inference_favour == 'nmma_gw':
            likelihood, priors = setup_nmma_gw_likelihood(
                interferometers=ifo_list,
                waveform_generator=waveform_generator,
                priors=priors,
                args=args,
            )
        priors.convert_floats_to_delta_functions()
        logger.setLevel(logging.INFO)

        sampling_keys = []
        for p in priors:
            if isinstance(priors[p], bilby.core.prior.Constraint):
                continue
            elif priors[p].is_fixed:
                likelihood.parameters[p] = priors[p].peak
            else:
                sampling_keys.append(p)

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

        self.outdir = outdir
        self.label = label
        self.data_dump = data_dump
        self.priors = priors
        self.sampling_keys = sampling_keys
        self.likelihood = likelihood
        self.zero_likelihood_mode = bilby_zero_likelihood_mode
        self.periodic = periodic
        self.reflective = reflective
        self.args = args
        self.injection_parameters = injection_parameters
        self.nlive = nlive
        self.inference_favour = inference_favour

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
        u_array: (float, array-like)
            The values to rescale

        Returns
        -------
        (float, array-like)
            The rescaled values

        """
        if self.zero_likelihood_mode:
            return 0
        parameters = {key: v for key, v in zip(self.sampling_keys, v_array)}
        if self.priors.evaluate_constraints(parameters) > 0:
            self.likelihood.parameters.update(parameters)
            return (
                self.likelihood.log_likelihood()
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
        return self.priors.ln_prob(params)

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
        ndim = len(self.sampling_keys)

        args_list = [
            (
                self.prior_transform_function,
                self.log_prior_function,
                self.log_likelihood_function,
                ndim,
                calculate_likelihood,
                map_rstates[i],
            )
            for i in range(self.nlive)
        ]
        initial_points = pool.map(self.get_initial_point_from_prior, args_list)
        u_list = [point[0] for point in initial_points]
        v_list = [point[1] for point in initial_points]
        l_list = [point[2] for point in initial_points]
        blobs = None

        return np.array(u_list), np.array(v_list), np.array(l_list), blobs

    @staticmethod
    def get_initial_point_from_prior(args):
        """
        Draw initial points from the prior subject to constraints applied both to
        the prior and the likelihood.

        We remove any points where the likelihood or prior is infinite or NaN.

        The `log_likelihood_function` often converts infinite values to large
        finite values so we catch those.
        """
        (
            prior_transform_function,
            log_prior_function,
            log_likelihood_function,
            ndim,
            calculate_likelihood,
            rstate,
        ) = args
        bad_values = [np.inf, np.nan_to_num(np.inf), np.nan]
        while True:
            unit = rstate.random(ndim)
            theta = prior_transform_function(unit)

            if abs(log_prior_function(theta)) not in bad_values:
                if calculate_likelihood:
                    logl = log_likelihood_function(theta)
                    if abs(logl) not in bad_values:
                        return unit, theta, logl
                else:
                    return unit, theta, np.nan

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
        ndim = len(self.sampling_keys)
        sampler = dynesty.NestedSampler(
            self.log_likelihood_function,
            self.prior_transform_function,
            ndim,
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
