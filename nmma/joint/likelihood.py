from __future__ import division

import numpy as np

from ..em.model import SVDLightCurveModel, KilonovaGRBLightCurveModel
from ..em.likelihood import OpticalLightCurve
from .conversion import MultimessengerConversion, MultimessengerConversionWithLambdas

from bilby.gw.likelihood import GravitationalWaveTransient, ROQGravitationalWaveTransient
from bilby.core.likelihood import Likelihood
from bilby.core.prior import Interped


class MultiMessengerLikelihood(Likelihood):
    """ A multi-messenger likelihood object

    This likelihood combines the usual gravitational-wave transient
    likelihood and the kilonva afterglow light curve likelihood.

    Parameters
    ----------
    interferometers: list, bilby.gw.detector.InterferometerList
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
        Path to the SVD pickles for the light curve model
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

    def __init__(self, interferometers, waveform_generator,
                 light_curve_data, light_curve_model_name,
                 light_curve_SVD_path, em_trigger_time, mag_ncoeff, lbol_ncoeff,
                 eos_path, Neos, eos_weight_path, binary_type, gw_likelihood_type,
                 priors, with_grb, grb_resolution, light_curve_interpolation_type, with_eos=True,
                 filters=None, error_budget=1., tmin=0., tmax=14.,
                 roq_weights=None, roq_params=None, roq_scale_factor=None,
                 time_marginalization=False, distance_marginalization=False,
                 phase_marginalization=False, distance_marginalization_lookup_table=None,
                 jitter_time=True, reference_frame="sky", time_reference="geocenter"):

        # construct the eos prior
        if with_eos:
            xx = np.arange(0, Neos + 1)
            eos_weight = np.loadtxt(eos_weight_path)
            yy = np.concatenate((eos_weight, [eos_weight[-1]]))
            eos_prior = Interped(xx, yy, minimum=0, maximum=Neos, name='EOS')
            priors['EOS'] = eos_prior

            # construct the eos conversion
            parameter_conversion_class = MultimessengerConversion(eos_data_path=eos_path, Neos=Neos,
                                                                  binary_type=binary_type)
        else:
            parameter_conversion_class = MultimessengerConversionWithLambdas(binary_type=binary_type)

        priors.conversion_function = parameter_conversion_class.priors_conversion_function
        parameter_conversion = parameter_conversion_class.convert_to_multimessenger_parameters
        waveform_generator.parameter_conversion = parameter_conversion

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
        if gw_likelihood_type == 'GravitationalWaveTransient':
            GWLikelihood = GravitationalWaveTransient(**gw_likelihood_kwargs)

        elif gw_likelihood_type == 'ROQGravitationalWaveTransient':
            gw_likelihood_kwargs.pop("time_marginalization", None)
            gw_likelihood_kwargs.pop("jitter_time", None)
            gw_likelihood_kwargs.update(dict(weights=roq_weights, roq_params=roq_params,
                                             roq_scale_factor=roq_scale_factor))
            GWLikelihood = ROQGravitationalWaveTransient(**gw_likelihood_kwargs)

        # initialize the EM likelihood
        sample_times = np.arange(tmin, tmax, 0.1)
        light_curve_model_kwargs = dict(model=light_curve_model_name, sample_times=sample_times,
                                        svd_path=light_curve_SVD_path,
                                        parameter_conversion=parameter_conversion,
                                        mag_ncoeff=mag_ncoeff, lbol_ncoeff=lbol_ncoeff,
                                        interpolation_type=light_curve_interpolation_type)
        if with_grb:
            light_curve_model = KilonovaGRBLightCurveModel(sample_times=sample_times,
                                                           kilonova_kwargs=light_curve_model_kwargs,
                                                           GRB_resolution=grb_resolution)
        else:
            light_curve_model = SVDLightCurveModel(**light_curve_model_kwargs)
        if not filters:
            filters = list(light_curve_data.keys())
        em_likelihood_kwargs = dict(light_curve_model=light_curve_model, filters=filters,
                                    light_curve_data=light_curve_data, trigger_time=em_trigger_time,
                                    error_budget=error_budget, tmin=tmin, tmax=tmax)
        EMLikelihood = OpticalLightCurve(**em_likelihood_kwargs)

        super(MultiMessengerLikelihood, self).__init__(parameters={})
        self.parameter_conversion = parameter_conversion
        self.GWLikelihood = GWLikelihood
        self.EMLikelihood = EMLikelihood
        self.priors = priors
        self.time_marginalization = time_marginalization
        self.phase_marginalization = phase_marginalization
        self.distance_marginalization = distance_marginalization
        self.__sync_parameters()

    def __repr__(self):
        return self.__class__.__name__ + ' with ' + self.GWLikelihood.__repr__() +\
            ' and ' + self.EMLikelihood.__repr__()

    def __sync_parameters(self):
        self.GWLikelihood.parameters = self.parameters
        self.EMLikelihood.parameters = self.parameters

    def log_likelihood(self):

        if not self.priors.evaluate_constraints(self.parameters):
            return np.nan_to_num(-np.inf)

        logL_EM = self.EMLikelihood.log_likelihood()
        if not np.isfinite(logL_EM):
            return np.nan_to_num(-np.inf)

        logL_GW = self.GWLikelihood.log_likelihood()
        if not np.isfinite(logL_GW):
            return np.nan_to_num(-np.inf)

        return logL_EM + logL_GW

    def noise_log_likelihood(self):
        return self.GWLikelihood.noise_log_likelihood() + self.EMLikelihood.noise_log_likelihood()
