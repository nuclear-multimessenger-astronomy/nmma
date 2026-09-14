import numpy as np
from ast import literal_eval
from bilby.gw.likelihood import GravitationalWaveTransient, ROQGravitationalWaveTransient, RelativeBinningGravitationalWaveTransient, MBGravitationalWaveTransient
from bilby.gw.source import binary_neutron_star_frequency_sequence
from ..core.base import NMMALikelihood, initialisation_args_from_signature_and_namespace
from ..core.conversion import (bbh_source_frame, bns_source_frame, tidal_deformabilities_and_mass_ratio_to_eff_tidal_deformabilities as tidal_conversion)

def setup_gw_kwargs(data_dump, args, logger, **kwargs):
    """Assemble the keyword arguments for `GravitationalWaveTransientLikelihood`.

    Starts from the defaults inferred from `GravitationalWaveTransientLikelihood`'s
    signature via `args` (prefixed `gw_`), then reads any extra args required by
    the chosen `args.likelihood_type`:

    - ROQGravitationalWaveTransient: drops `time_marginalization`/`jitter_time`
      (unsupported for ROQ) and adds the ROQ kwargs from `roq_likelihood_kwargs`.
    - RelativeBinningGravitationalWaveTransient:
        fiducial_parameters: dict
            Fiducial parameters for the relative-binning reference waveform.
        update_fiducial_parameters: bool
            If True, tries to optimize fiducial parameters with the maximum
            likelihood. Defaults to False.
        epsilon: float
            Sets the precision of the binning for relative binning.
    - MBGravitationalWaveTransient: drops `time_marginalization`/`jitter_time`,
      strips bilby_pipe's default `minimum_frequency`/`maximum_frequency` from
      the waveform generator, and adds:
        reference_chirp_mass: float, optional
            A reference chirp mass for determining the frequency banding. This
            is set to the prior minimum of chirp mass if not specified. Hence a
            CBCPriorDict object needs to be passed to priors when this
            parameter is not specified.

    Parameters
    ----------
    data_dump : dict
        The generation-stage data dump; used for `ifo_list`, `waveform_generator`,
        and (for ROQ) `meta_data["weight_file"]`.
    args : argparse.Namespace
        Parsed analysis config, including `likelihood_type` and the `gw_`-prefixed
        likelihood options.
    logger : logging.Logger
        Logger used when loading ROQ weights.
    **kwargs
        Overrides merged into the returned kwargs last, taking precedence over
        everything computed above.

    Returns
    -------
    dict
        Keyword arguments ready to pass to `GravitationalWaveTransientLikelihood`.
    """

    default_gw_kwargs = initialisation_args_from_signature_and_namespace(
        GravitationalWaveTransientLikelihood, args, prefixes=['gw_'])
    gw_kwargs = default_gw_kwargs | dict(
            interferometers=data_dump["ifo_list"],
            waveform_generator=data_dump["waveform_generator"],
        
        )
    if args.likelihood_type == 'ROQGravitationalWaveTransient':
        gw_kwargs.pop("time_marginalization", None)
        gw_kwargs.pop("jitter_time", None)
        args.weight_file = data_dump["meta_data"].get("weight_file", None)
        gw_kwargs.update(roq_likelihood_kwargs(args, logger))

    elif args.likelihood_type == 'RelativeBinningGravitationalWaveTransient':
        if isinstance(args.fiducial_parameters, str):
            fiducial_parameters = literal_eval(args.fiducial_parameters)
        else:
            fiducial_parameters = args.fiducial_parameters
        gw_kwargs.update(
            fiducial_parameters=fiducial_parameters, epsilon=args.epsilon,
            update_fiducial_parameters=args.update_fiducial_parameters
        )
    elif args.likelihood_type == 'MBGravitationalWaveTransient':
        gw_kwargs.pop("time_marginalization", None)
        gw_kwargs.pop("jitter_time", None)
        ## NOTE: This is a temporary fix to remove defaults set by bilby-pipe. 
        # Will likely be adressed in bilby-pipe in the future.
        gw_kwargs['waveform_generator'].waveform_arguments.pop('minimum_frequency', None)
        gw_kwargs['waveform_generator'].waveform_arguments.pop('maximum_frequency', None)
        gw_kwargs.update(reference_chirp_mass=args.reference_chirp_mass)

    gw_kwargs.update(**kwargs)
    return gw_kwargs

def roq_likelihood_kwargs(args, logger):
    """Return the kwargs required for the ROQ setup

    Parameters
    ----------
    args: Namespace
        The parser arguments

    Returns
    -------
    kwargs: dict
        A dictionary of the required kwargs

    """

    kwargs = dict(
        weights=None,
        roq_params=None,
        linear_matrix=None,
        quadratic_matrix=None,
        roq_scale_factor=args.roq_scale_factor,
    )
    if hasattr(args, "likelihood_roq_params") and hasattr(
        args, "likelihood_roq_weights"
    ):
        kwargs["roq_params"] = args.likelihood_roq_params
        kwargs["weights"] = args.likelihood_roq_weights
    elif hasattr(args, "roq_folder") and args.roq_folder is not None:
        logger.info(f"Loading ROQ weights from {args.roq_folder}, {args.weight_file}")
        kwargs["roq_params"] = np.genfromtxt(
            args.roq_folder + "/params.dat", names=True
        )
        kwargs["weights"] = args.weight_file
    elif hasattr(args, "roq_linear_matrix") and args.roq_linear_matrix is not None:
        logger.info(f"Loading linear_matrix from {args.roq_linear_matrix}")
        logger.info(f"Loading quadratic_matrix from {args.roq_quadratic_matrix}")
        kwargs["linear_matrix"] = args.roq_linear_matrix
        kwargs["quadratic_matrix"] = args.roq_quadratic_matrix
    return kwargs

class GravitationalWaveTransientLikelihood(NMMALikelihood):
    """ A GravitationalWaveTransient likelihood object

    This likelihood uses the usual gravitational-wave transient
    but include an EOS handling for parameter conversion.

    Parameters
    ----------
    priors: dict
        The joint prior dict for the run (may already include EOS-derived
        priors merged in upstream by `MultiMessengerLikelihood`); used
        directly for distance and phase marginalization.
    interferometers: list, bilby.gw.detector.InterferometerList
        A list of `bilby.detector.Interferometer` instances - contains the
        detector data and power spectral densities
    waveform_generator: `bilby.waveform_generator.WaveformGenerator`
        An object which computes the frequency-domain strain of the signal, 
        given some set of parameters
    gw_likelihood_type: str
        The gravitational-wave likelihood to be taken
    time_marginalization: bool, optional
        If true, marginalize over time in the likelihood.
        This uses a FFT to calculate the likelihood over a regularly spaced
        grid.
        In order to cover the whole space the prior is set to be uniform over
        the spacing of the array of times.
        If using time marginalisation and jitter_time is True a "jitter"
        parameter is added to the prior which modifies the position of the
        grid of times.
    distance_marginalization: bool, optional
        If true, marginalize over distance in the likelihood.
        This uses a look up table calculated at run time.
        The distance prior is set to be a delta function at the minimum
        distance allowed in the prior being marginalised over.
    phase_marginalization: bool, optional
        If true, marginalize over phase in the likelihood.
        This is done analytically using a Bessel function.
        The phase prior is set to be a delta function at phase=0.
    distance_marginalization_lookup_table: (dict, str), optional
        If a dict, dictionary containing the lookup_table, distance_array,
        (distance) prior_array, and reference_distance used to construct
        the table.
        If a string the name of a file containing these quantities.
        The lookup table is stored after construction in either the
        provided string or a default location:
        '.distance_marginalization_lookup_dmin{}_dmax{}_n{}.npz'
    jitter_time: bool, optional
        Whether to introduce a `time_jitter` parameter. This avoids either
        missing the likelihood peak, or introducing biases in the
        reconstructed time posterior due to an insufficient sampling frequency.
        Using this parameter is strongly encouraged.
    reference_frame: (str, bilby.gw.detector.InterferometerList, list), optional
        Definition of the reference frame for the sky location.
        - "sky": sample in RA/dec, this is the default
        - e.g., "H1L1", ["H1", "L1"], InterferometerList(["H1", "L1"]):
          sample in azimuth and zenith, `azimuth` and `zenith` defined in the
          frame where the z-axis is aligned the the vector connecting H1
          and L1.
    time_reference: str, optional
        Name of the reference for the sampled time parameter.
        - "geocent"/"geocenter": sample in the time at the Earth's center,
          this is the default
        - e.g., "H1": sample in the time of arrival at H1    
    kwargs:
        Additional keyword arguments passed to the likelihood class. These might be required by the chosen gw_likelihood_type!

    Notes
    -----
    When `gw_likelihood_type='ROQGravitationalWaveTransient'`, `kwargs` must
    additionally supply:
        roq_params: str, array_like
            Parameters describing the domain of validity of the ROQ basis.
        roq_params_check: bool
            If true, run tests using the roq_params to check the prior and
            data are valid for the ROQ.
        roq_scale_factor: float
            The ROQ scale factor used.
    (`setup_gw_kwargs`/`roq_likelihood_kwargs` populate these automatically
    when building this likelihood via the normal pipeline.)
    """

    def __init__(self,priors, interferometers,  
                 waveform_generator, gw_likelihood_type='GravitationalWaveTransient', time_marginalization=False, distance_marginalization=False, phase_marginalization=False, distance_marginalization_lookup_table=None, jitter_time=True, reference_frame="sky", time_reference="geocenter", **kwargs):

        waveform_generator.parameter_conversion = self.gw_identity_conversion
        waveform_generator.start_time = interferometers[0].time_array[0]

        # initialize the GW likelihood
        gw_likelihood_kwargs = dict(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            priors=priors,
            phase_marginalization=phase_marginalization,
            distance_marginalization=distance_marginalization,
            distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            time_marginalization=time_marginalization,
            reference_frame=reference_frame,
            time_reference=time_reference,
            jitter_time=jitter_time,
            **kwargs
        )

        if gw_likelihood_type == 'GravitationalWaveTransient':
            gw_transient = GravitationalWaveTransient(**gw_likelihood_kwargs)

        elif gw_likelihood_type == 'ROQGravitationalWaveTransient':
            gw_transient = ROQGravitationalWaveTransient(**gw_likelihood_kwargs)

        elif gw_likelihood_type == 'RelativeBinningGravitationalWaveTransient':
            gw_transient = RelativeBinningGravitationalWaveTransient(**gw_likelihood_kwargs)

        elif gw_likelihood_type == 'MBGravitationalWaveTransient':
            gw_transient = MBGravitationalWaveTransient(**gw_likelihood_kwargs)
        else:
            ### FIXME: Not an f-string and no .format() call, so {} is never substituted.
            raise ValueError("Unknown GW Likelihood class {}")

        super().__init__(gw_transient, priors)

        if "neutron_star" in self.sub_model.waveform_generator.frequency_domain_source_model.__name__:
            self.parameter_conversion = bns_source_frame
        else:
            self.parameter_conversion = bbh_source_frame
    
    def posterior_conversion(self, posterior_samples):
        """Add derived spin/tidal summary parameters to posterior samples, in place.

        If not already present, computes `chi_eff` from `mass_ratio` and
        `chi_1`/`chi_2` (or `spin_1z`/`spin_2z`), and `lambda_tilde`/
        `delta_lambda_t` from `lambda_1`, `lambda_2`, `mass_ratio`. Silently
        skips a derived quantity if its required inputs are missing.

        Parameters
        ----------
        posterior_samples : dict or pandas.DataFrame
            Posterior samples, keyed by parameter name.

        Returns
        -------
        dict or pandas.DataFrame
            The same object, with derived keys added where possible.
        """

        if "chi_eff" not in posterior_samples:
            try:
                ### FIXME: Python evaluates dict.get's default argument eagerly, before checking whether the key exists. So posterior_samples['spin_1z'] is evaluated unconditionally — if spin_1z/spin_2z aren't in the samples (the normal case when a run uses chi_1/chi_2 instead), this raises KeyError, which gets caught by the surrounding except KeyError: pass and chi_eff is silently never added — even though chi_1/chi_2 were present. I confirmed this directly: a sample dict with chi_1/chi_2 (no spin_1z/spin_2z) produced no chi_eff key at all, while a dict with only spin_1z/spin_2z worked correctly. In practice, only the spin_1z/spin_2z path in the docstring actually works.
                q = posterior_samples['mass_ratio']
                chi_1 = posterior_samples.get('chi_1', posterior_samples['spin_1z'])
                chi_2 = posterior_samples.get('chi_2', posterior_samples['spin_2z'])
                posterior_samples['chi_eff'] = (chi_1 + q*chi_2)/(1+q)
            except KeyError:
                pass
        if "lambda_tilde" not in posterior_samples:
            try:
                lambda1 = posterior_samples['lambda_1']
                lambda2 = posterior_samples['lambda_2']
                q = posterior_samples['mass_ratio']
                # Calculate the effective tidal deformability
                lambdaT, delta_lambda_t  = tidal_conversion(lambda1, lambda2, q)
                posterior_samples['lambda_tilde'] = lambdaT
                posterior_samples['delta_lambda_t'] = delta_lambda_t
            except KeyError:
                pass
        
        return posterior_samples

    ### CHECKME: Check all functions below if they are needed or if they can be removed.

    def sanity_checks(self):
        """Validate the sub-likelihood configuration.

        Returns
        -------
        bool
            Always True currently; relative-binning-specific checks are not
            yet implemented (see inline TODO).
        """
        
        #TODO: add additional checks RelativeBinning!
        return True
    
    def final_diagnostics(self, bestfit_params, args, result=None):
        # TODO add some nice plotting for final waveform
        pass
    
    def noise_log_likelihood(self):
        return self.sub_model.noise_log_likelihood()

    def gw_identity_conversion(self, parameters):
        return parameters, []