from __future__ import division

import numpy as np

from ..joint.conversion import (
    MultimessengerConversion,
    MultimessengerConversionWithLambdas
    )

from bilby.gw.likelihood import GravitationalWaveTransient, ROQGravitationalWaveTransient
from bilby.core.likelihood import Likelihood
from bilby.core.prior import Interped


class GravitationalWaveTransientLikelihoodwithEOS(Likelihood):
    """ A GravitationalWaveTransient likelihood object

    This likelihood uses the usual gravitational-wave transient
    but include an EOS handling for parameter conversion.

    Parameters
    ----------
    interferometers: list, bilby.gw.detector.InterferometerList
        A list of `bilby.detector.Interferometer` instances - contains the
        detector data and power spectral densities
    waveform_generator: `bilby.waveform_generator.WaveformGenerator`
        An object which computes the frequency-domain strain of the signal,
        given some set of parameters
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
                 eos_path, Neos, eos_weight_path, binary_type, gw_likelihood_type,
                 priors, with_eos=True,
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
            parameter_conversion_class = MultimessengerConversion(eos_data_path=eos_path,
                                                                  Neos=Neos,
                                                                  binary_type=binary_type,
                                                                  with_ejecta=False)
        else:
            parameter_conversion_class = MultimessengerConversionWithLambdas(binary_type=binary_type,
                                                                             with_ejecta=False)

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

        super(GravitationalWaveTransientLikelihoodwithEOS, self).__init__(parameters={})
        self.parameter_conversion = parameter_conversion
        self.GWLikelihood = GWLikelihood
        self.priors = priors
        self.time_marginalization = time_marginalization
        self.phase_marginalization = phase_marginalization
        self.distance_marginalization = distance_marginalization
        self.__sync_parameters()

    def __repr__(self):
        return self.__class__.__name__ + ' with ' + self.GWLikelihood.__repr__()

    def __sync_parameters(self):
        self.GWLikelihood.parameters = self.parameters

    def log_likelihood(self):

        if not self.priors.evaluate_constraints(self.parameters):
            return np.nan_to_num(-np.inf)

        logL_GW = self.GWLikelihood.log_likelihood()
        if not np.isfinite(logL_GW):
            return np.nan_to_num(-np.inf)

        return logL_GW

    def noise_log_likelihood(self):
        return self.GWLikelihood.noise_log_likelihood()
