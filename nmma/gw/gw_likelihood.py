import numpy as np
from ast import literal_eval
from bilby.gw.likelihood import GravitationalWaveTransient, ROQGravitationalWaveTransient, RelativeBinningGravitationalWaveTransient, MBGravitationalWaveTransient
from ..core.base import NMMABaseLikelihood, initialisation_args_from_signature_and_namespace
from ..core.conversion import (bbh_source_frame, tidal_deformabilities_and_mass_ratio_to_eff_tidal_deformabilities as tidal_conversion)

def setup_gw_kwargs(data_dump, args, logger, **kwargs):
    """
    Set up the gravitational-wave likelihood.
    We read some required args for the chosen gw-likelihood in this process.
    For multibanding, this includes:
        reference_chirp_mass: float, optional          
            A reference chirp mass for determining the frequency banding. This is set to prior minimum of chirp mass if not specified. Hence a CBCPriorDict object needs to be passed to priors when this parameter is not specified.
    For relative-binning, this includes:
        fiducial_parameters: dictionary
            fiducial parameters for relative binning reference waveform
        update_fiducial_parameters: bool
            if True, tries to optimize fiducial parameters with the maximum likelihood. Defaults to False
        epsilon: float
            sets the precision of the binning for relative binning
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
        fiducial_parameters = literal_eval(args.fiducial_parameters)
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

class GravitationalWaveTransientLikelihood(NMMABaseLikelihood):
    """ A GravitationalWaveTransient likelihood object

    This likelihood uses the usual gravitational-wave transient
    but include an EOS handling for parameter conversion.

    Parameters
    ----------
    priors: dict
        To be used in the distance and phase marginalization and
        having the corresponding eos prior appened to it
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
            """Additional params:
            roq_params: str, array_like
                Parameters describing the domain of validity of the ROQ basis.
            roq_params_check: bool
                If true, run tests using the roq_params to check the prior and data are
                valid for the ROQ
            roq_scale_factor: float
                The ROQ scale factor used."""
            gw_transient = ROQGravitationalWaveTransient(**gw_likelihood_kwargs)

        elif gw_likelihood_type == 'RelativeBinningGravitationalWaveTransient':
            gw_transient = RelativeBinningGravitationalWaveTransient(**gw_likelihood_kwargs)

        elif gw_likelihood_type == 'MBGravitationalWaveTransient':
            gw_transient = MBGravitationalWaveTransient(**gw_likelihood_kwargs)
        else:
            raise ValueError("Unknown GW Likelihood class {}")

        super().__init__(gw_transient, priors)

    def parameter_conversion(self, parameters):
        return bbh_source_frame(parameters)
    
    def posterior_conversion(self, posterior_samples):
        if "chi_eff" not in posterior_samples:
            try:
                q = posterior_samples['mass_ratio']
                chi_1 = posterior_samples.get('chi_1', posterior_samples.get('spin_1z'))
                chi_2 = posterior_samples.get('chi_2', posterior_samples.get('spin_2z'))
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

    def sanity_checks(self):
        #TODO: add additional checks RelativeBinning!
        return True
    
    def final_diagnostics(self, bestfit_params, args, result=None):
        # TODO add some nice plotting for final waveform
        pass
    def noise_log_likelihood(self):
        return self.sub_model.noise_log_likelihood()

    def gw_identity_conversion(self, parameters):
        return parameters, []