from __future__ import division

import inspect
import numpy as np

from ..em.model import (
    SVDLightCurveModel,
    GRBLightCurveModel,
    GenericCombineLightCurveModel,
)
from ..em.em_likelihood import OpticalLightCurve
from .conversion import MultimessengerConversion

from bilby.core.likelihood import Likelihood


from ..gw.gw_likelihood import GravitationalWaveTransientLikelihood, roq_likelihood_kwargs


def reorder_loglikelihoods(unsorted_loglikelihoods, unsorted_samples, sorted_samples):
    """Reorders the stored log-likelihood after they have been reweighted

    This creates a sorting index by matching the reweights `result.samples`
    against the raw samples, then uses this index to sort the
    loglikelihoods

    Parameters
    ----------
    sorted_samples, unsorted_samples: array-like
        Sorted and unsorted values of the samples. These should be of the
        same shape and contain the same sample values, but in different
        orders
    unsorted_loglikelihoods: array-like
        The loglikelihoods corresponding to the unsorted_samples

    Returns
    -------
    sorted_loglikelihoods: array-like
        The loglikelihoods reordered to match that of the sorted_samples


    """

    idxs = []
    for ii in range(len(unsorted_loglikelihoods)):
        idx = np.where(np.all(sorted_samples[ii] == unsorted_samples, axis=1))[0]
        if len(idx) > 1:
            print(
                "Multiple likelihood matches found between sorted and "
                "unsorted samples. Taking the first match."
            )
        idxs.append(idx[0])
    return unsorted_loglikelihoods[idxs]


def setup_nmma_likelihood(args, logger, priors, messengers, **kwargs
    #messengers, interferometers, waveform_generator, light_curve_data, priors, args
):
    """Takes the kwargs and sets up and returns
    MultiMessengerLikelihood with either an ROQ GW or GW likelihood.

    Parameters
    ----------
    messengers: list
        list of messengers to be used in analysis
    interferometers: bilby.gw.detectors.InterferometerList
        The pre-loaded bilby IFO
    waveform_generator: bilby.gw.waveform_generator.LALCBCWaveformGenerator
        The waveform generation
    light_curve_data: dict
        The light curve data with filter as key
    priors: dict
        The priors, used for setting up marginalization
    args: Namespace
        The parser arguments


    Returns
    -------
    likelihood: nmma.joint.likelihood.MultiMessengerLikelihood
    priors: dict
        The priors including eos setup in case --eos is used

    """
    likelihood_kwargs=dict(
        messengers=messengers
    )
    gw_kwargs=None
    em_kwargs=None
    eos_kwargs=None
    if "gw" in messengers:
        gw_kwargs= dict(
            binary_type=args.binary_type,
            gw_likelihood_type=args.likelihood_type,
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            priors=priors,
            phase_marginalization=args.phase_marginalization,
            distance_marginalization=args.distance_marginalization,
            distance_marginalization_lookup_table=args.distance_marginalization_lookup_table,
            time_marginalization=args.time_marginalization,
            reference_frame=args.reference_frame,
            time_reference=args.time_reference
        )
    if "em" in messengers:
        em_kwargs = dict(
            light_curve_data=light_curve_data,
            light_curve_model_name=args.kilonova_model,
            light_curve_interpolation_type=args.kilonova_interpolation_type,
            light_curve_SVD_path=args.kilonova_model_svd,
            em_trigger_time=args.kilonova_trigger_time,
            filters=args.filters,
            mag_ncoeff=args.svd_mag_ncoeff,
            lbol_ncoeff=args.svd_lbol_ncoeff,
            tmin=args.kilonova_tmin,
            tmax=args.kilonova_tmax,
            error_budget=args.kilonova_error,
            local_only=args.local_model_only,
            grb_resolution=args.grb_resolution,
            with_grb=args.with_grb
        )

    if "eos" in messengers:
        if args.tabulated_eos:

            logger.info("Sampling over precomputed EOSs")
            
            eos_kwargs = dict(
                eos_path=args.eos_data,
                Neos=args.Neos,
                eos_weight_path=args.eos_weight,
            )
            if args.eos_from_neps:
                raise ValueError("Can only sample over either precomputed eos or NEPs. Set tabulated-eos or eos-from-neps to False!")
        
        elif args.eos_from_neps:
            logger.info("Sampling over EOS generated on the fly")
            
            eos_kwargs= dict(
                crust_path=args.eos_crust_file
            )
            if args.tabulated_eos:
                raise ValueError("Can only sample over either precomputed eos or NEPs. Set tabulated-eos or eos-from-neps to False!")
        

    Likelihood = MultiMessengerLikelihood
    if args.gw_likelihood_type == "GravitationalWaveTransient":
        likelihood_kwargs.update(jitter_time=args.jitter_time)

    elif args.gw_likelihood_type == "ROQGravitationalWaveTransient":
        if args.time_marginalization:
            logger.warning(
                "Time marginalization not implemented for "
                "ROQGravitationalWaveTransient: option ignored"
            )
        likelihood_kwargs.pop("time_marginalization", None)
        likelihood_kwargs.pop("jitter_time", None)
        likelihood_kwargs.update(roq_likelihood_kwargs(args, logger))
    else:
        raise ValueError("Unknown GW Likelihood class {}")

    likelihood_kwargs = {
        key: likelihood_kwargs[key]
        for key in likelihood_kwargs
        if key in inspect.getfullargspec(Likelihood.__init__).args
    }

    logger.info(
        f"Initialise likelihood {Likelihood} with kwargs: \n{likelihood_kwargs}"
    )

    likelihood = Likelihood(**likelihood_kwargs)
    return likelihood

class MultiMessengerLikelihood(Likelihood):
    """A multi-messenger likelihood object

    This likelihood combines the usual gravitational-wave transient
    likelihood and the kilonova afterglow light curve likelihood.

    Parameters
    ----------
    
    messengers: list
        list of messengers to be used in analysisinterferometers: list, bilby.gw.detector.InterferometerList
        A list of `bilby.detector.Interferometer` instances - contains the
        detector data and power spectral densities
    waveform_generator: `bilby.waveform_generator.WaveformGenerator`
        An object which computes the frequency-domain strain of the signal,
        given some set of parameters
    light_curve_data: dict
        The light curve data processed with nmma.em.utils.loadEvent
    light_curve_model_name: str
        Name of the kilonova model to be used
    light_curve_SVD_path: str
        Path to the SVD files for the light curve model
    em_trigger_time: float
        GPS time of the kilonova trigger in days
    mag_ncoeff: int
        mag_ncoeff highest eigenvalues to be taken for mag's SVD evaluation
    lbol_ncoeff: int
        lbol_ncoeff highest eigenvalues to be taken for lbol's SVD evaluation
    eos_path: str
        Path to the sorted EOS directory
    Neos: int
        Number of EOS to be considered
    eos_weight_path: str
        Path to the eos weighting file
    binary_type: str
        The binary to be analysed is "BNS" or "NSBH"
    gw_likelihood_type: str
        The gravitational-wave likelihood to be taken is
        "GravitationalWaveTransient" or "ROQGravitationalWaveTransient"
    priors: dict
        To be used in the distance and phase marginalization and
        having the corresponding eos prior appened to it
    filters: list, str
        List of filters to be analysed. By default, using all th filters
        in the light curve data given.
    error_budget: float (default:1)
        Additionally introduced statistical error on the light curve data,
        so as to keep the systematic error in control
    tmin: float (default:0)
        Days from trigger_time to be started analysing
    tmax: float (default:14)
        Days from trigger_time to be ended analysing
    roq_params: str, array_like
        Parameters describing the domain of validity of the ROQ basis.
    roq_params_check: bool
        If true, run tests using the roq_params to check the prior and data are
        valid for the ROQ
    roq_scale_factor: float
        The ROQ scale factor used.
    distance_marginalization: bool, optional
        If true, marginalize over distance in the likelihood.
        This uses a look up table calculated at run time.
        The distance prior is set to be a delta function at the minimum
        distance allowed in the prior being marginalised over.
    time_marginalization: bool, optional
        If true, marginalize over time in the likelihood.
        This uses a FFT to calculate the likelihood over a regularly spaced
        grid.
        In order to cover the whole space the prior is set to be uniform over
        the spacing of the array of times.
        If using time marginalisation and jitter_time is True a "jitter"
        parameter is added to the prior which modifies the position of the
        grid of times.
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
        Default is False, however using this parameter is strongly encouraged.
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

    """

    def __init__(
        self,
        messengers,
        interferometers,
        waveform_generator,
        light_curve_data,
        light_curve_model_name,
        light_curve_SVD_path,
        em_trigger_time,
        mag_ncoeff,
        lbol_ncoeff,
        binary_type,
        gw_likelihood_type,
        priors,
        with_grb,
        grb_resolution,
        light_curve_interpolation_type,
        with_eos=True,
        filters=None,
        error_budget=1.0,
        tmin=0.0,
        tmax=14.0,
        roq_weights=None,
        roq_params=None,
        roq_scale_factor=None,
        time_marginalization=False,
        distance_marginalization=False,
        phase_marginalization=False,
        distance_marginalization_lookup_table=None,
        jitter_time=True,
        reference_frame="sky",
        time_reference="geocenter",
        local_only=False,
    ):

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
        )
        if gw_likelihood_type == "GravitationalWaveTransient":
            GWLikelihood = GravitationalWaveTransient(**gw_likelihood_kwargs)

        elif gw_likelihood_type == "ROQGravitationalWaveTransient":
            gw_likelihood_kwargs.pop("time_marginalization", None)
            gw_likelihood_kwargs.pop("jitter_time", None)
            gw_likelihood_kwargs.update(
                dict(
                    weights=roq_weights,
                    roq_params=roq_params,
                    roq_scale_factor=roq_scale_factor,
                )
            )
            GWLikelihood = ROQGravitationalWaveTransient(**gw_likelihood_kwargs)

        # initialize the EM likelihood
        if not filters:
            filters = list(light_curve_data.keys())
        sample_times = np.arange(tmin, tmax, 0.1)
        light_curve_model_kwargs = dict(
            model=light_curve_model_name,
            sample_times=sample_times,
            svd_path=light_curve_SVD_path,
            parameter_conversion=parameter_conversion,
            mag_ncoeff=mag_ncoeff,
            lbol_ncoeff=lbol_ncoeff,
            interpolation_type=light_curve_interpolation_type,
            filters=filters,
            local_only=local_only,
        )

        if with_grb:
            models = []
            models.append(SVDLightCurveModel(**light_curve_model_kwargs))
            models.append(
                GRBLightCurveModel(
                    sample_times=sample_times,
                    resolution=grb_resolution,
                    filters=filters,
                    parameter_conversion=parameter_conversion,
                )
            )
            light_curve_model = GenericCombineLightCurveModel(
                models=models, sample_times=sample_times
            )
        else:
            light_curve_model = SVDLightCurveModel(**light_curve_model_kwargs)

        em_likelihood_kwargs = dict(
            light_curve_model=light_curve_model,
            filters=filters,
            light_curve_data=light_curve_data,
            trigger_time=em_trigger_time,
            error_budget=error_budget,
            tmin=tmin,
            tmax=tmax,
        )
        EMLikelihood = OpticalLightCurve(**em_likelihood_kwargs)
        # EOSLikelihood= 

        super(MultiMessengerLikelihood, self).__init__(parameters={})
        self.parameter_conversion = parameter_conversion
        self.GWLikelihood = GWLikelihood
        self.EMLikelihood = EMLikelihood
        self.time_marginalization = time_marginalization
        self.phase_marginalization = phase_marginalization
        self.distance_marginalization = distance_marginalization
        self.__sync_parameters()

    def __repr__(self):
        return (
            self.__class__.__name__
            + " with "
            + self.GWLikelihood.__repr__()
            + " and "
            + self.EMLikelihood.__repr__()
        )

    def __sync_parameters(self):
        self.GWLikelihood.parameters = self.parameters
        self.EMLikelihood.parameters = self.parameters

    def log_likelihood(self):

        logL_EM = self.EMLikelihood.log_likelihood()
        if not np.isfinite(logL_EM):
            return np.nan_to_num(-np.inf)
        
        #logL_EOS

        logL_GW = self.GWLikelihood.log_likelihood()
        if not np.isfinite(logL_GW):
            return np.nan_to_num(-np.inf)

        return logL_EM + logL_GW

    def noise_log_likelihood(self):
        return (
            self.GWLikelihood.noise_log_likelihood()
            + self.EMLikelihood.noise_log_likelihood()
        )
