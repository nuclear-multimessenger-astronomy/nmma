import inspect
from importlib import import_module

import numpy as np
import pandas as pd

import bilby
import bilby_pipe
from bilby.core.utils import logger
from bilby.core.prior import Interped

from ...joint.likelihood import MultiMessengerLikelihood
from ...gw.likelihood import GravitationalWaveTransientLikelihoodwithEOS

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


def roq_likelihood_kwargs(args):
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


def setup_nmma_likelihood(
        interferometers,
        waveform_generator,
        light_curve_data,
        priors,
        args
        ):
    """Takes the kwargs and sets up and returns 
    MultiMessengerLikelihood with either an ROQ GW or GW likelihood.

    Parameters
    ----------
    interferometers: bilby.gw.detectors.InterferometerList
        The pre-loaded bilby IFO
    waveform_generator: bilby.gw.waveform_generator.LALCBCWaveformGenerator
        The waveform generation
    light_curve_dat: dict
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

    if args.with_eos:

        logger.info("Sampling over precomputed EOSs")

        likelihood_kwargs = dict(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            priors=priors,
            phase_marginalization=args.phase_marginalization,
            distance_marginalization=args.distance_marginalization,
            distance_marginalization_lookup_table=args.distance_marginalization_lookup_table,
            time_marginalization=args.time_marginalization,
            reference_frame=args.reference_frame,
            time_reference=args.time_reference,
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
            with_grb=args.with_grb,
            grb_resolution=args.grb_resolution,
            eos_path=args.eos_data,
            Neos=args.Neos,
            eos_weight_path=args.eos_weight,
            binary_type=args.binary_type,
            gw_likelihood_type=args.likelihood_type,
        )

    else:

        logger.info("Sampling over Lambdas, i.e. fully using quasi-universal relations")

        likelihood_kwargs = dict(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            priors=priors,
            phase_marginalization=args.phase_marginalization,
            distance_marginalization=args.distance_marginalization,
            distance_marginalization_lookup_table=args.distance_marginalization_lookup_table,
            time_marginalization=args.time_marginalization,
            reference_frame=args.reference_frame,
            time_reference=args.time_reference,
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
            with_grb=args.with_grb,
            grb_resolution=args.grb_resolution,
            eos_path=None,
            Neos=None,
            eos_weight_path=None,
            with_eos=False,
            binary_type=args.binary_type,
            gw_likelihood_type=args.likelihood_type,
        )

    if args.with_Hubble and args.Hubble_weight:
        logger.info("Sampling over Hubble constant with pre-calculated prior")
        logger.info("Assuming the redshift prior is the Hubble flow")
        logger.info("Overwriting any Hubble prior in the prior file")

        Hubble_prior_data = pd.read_csv(args.Hubble_weight,
                                        delimiter=' ', header=0)
        xx = Hubble_prior_data.Hubble.to_numpy()
        yy = Hubble_prior_data.prior_weight.to_numpy()
        Hmin = xx[0]
        Hmax = xx[-1]

        priors['Hubble_constant'] = Interped(xx, yy, minimum=Hmin, maximum=Hmax,
                                              name='Hubble_constant')

    Likelihood = MultiMessengerLikelihood
    if args.likelihood_type == "GravitationalWaveTransient":
        likelihood_kwargs.update(jitter_time=args.jitter_time)

    elif args.likelihood_type == "ROQGravitationalWaveTransient":
        if args.time_marginalization:
            logger.warning(
                "Time marginalization not implemented for "
                "ROQGravitationalWaveTransient: option ignored"
            )
        likelihood_kwargs.pop("time_marginalization", None)
        likelihood_kwargs.pop("jitter_time", None)
        likelihood_kwargs.update(roq_likelihood_kwargs(args))
    else:
        raise ValueError("Unknown Likelihood class {}")

    likelihood_kwargs = {
        key: likelihood_kwargs[key]
        for key in likelihood_kwargs
        if key in inspect.getfullargspec(Likelihood.__init__).args
    }

    logger.info(
        f"Initialise likelihood {Likelihood} with kwargs: \n{likelihood_kwargs}"
    )

    likelihood = Likelihood(**likelihood_kwargs)
    return likelihood, priors


def setup_nmma_gw_likelihood(
        interferometers,
        waveform_generator,
        priors,
        args
        ):
    """Takes the kwargs and sets up and returns 
    GravitationalWaveTransientLikelihoodwithEOS
    with either an ROQ GW or GW likelihood.

    Parameters
    ----------
    interferometers: bilby.gw.detectors.InterferometerList
        The pre-loaded bilby IFO
    waveform_generator: bilby.gw.waveform_generator.LALCBCWaveformGenerator
        The waveform generation
    priors: dict
        The priors, used for setting up marginalization
    args: Namespace
        The parser arguments


    Returns
    -------
    likelihood: nmma.gw.likelihood.GravitationalWaveTransientLikelihoodwithEOS
    priors: dict
        The priors including eos setup in case --eos is used

    """

    if args.with_eos:

        logger.info("Sampling over precomputed EOSs")

        likelihood_kwargs = dict(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            priors=priors,
            phase_marginalization=args.phase_marginalization,
            distance_marginalization=args.distance_marginalization,
            distance_marginalization_lookup_table=args.distance_marginalization_lookup_table,
            time_marginalization=args.time_marginalization,
            reference_frame=args.reference_frame,
            time_reference=args.time_reference,
            eos_path=args.eos_data,
            Neos=args.Neos,
            eos_weight_path=args.eos_weight,
            binary_type=args.binary_type,
            gw_likelihood_type=args.likelihood_type,
        )

    else:

        logger.info("Sampling over Lambdas, i.e. fully using quasi-universal relations")

        likelihood_kwargs = dict(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            priors=priors,
            phase_marginalization=args.phase_marginalization,
            distance_marginalization=args.distance_marginalization,
            distance_marginalization_lookup_table=args.distance_marginalization_lookup_table,
            time_marginalization=args.time_marginalization,
            reference_frame=args.reference_frame,
            time_reference=args.time_reference,
            eos_path=None,
            Neos=None,
            eos_weight_path=None,
            with_eos=False,
            binary_type=args.binary_type,
            gw_likelihood_type=args.likelihood_type,
        )

    if args.with_Hubble and args.Hubble_weight:
        logger.info("Sampling over Hubble constant with pre-calculated prior")
        logger.info("Assuming the redshift prior is the Hubble flow")
        logger.info("Overwriting any Hubble prior in the prior file")

        Hubble_prior_data = pd.read_csv(args.Hubble_weight,
                                        delimiter=' ', header=0)
        xx = Hubble_prior_data.Hubble.to_numpy()
        yy = Hubble_prior_data.prior_weight.to_numpy()
        Hmin = xx[0]
        Hmax = xx[-1]

        priors['Hubble_constant'] = Interped(xx, yy, minimum=Hmin, maximum=Hmax,
                                              name='Hubble_constant')

    Likelihood = GravitationalWaveTransientLikelihoodwithEOS
    if args.likelihood_type == "GravitationalWaveTransient":
        likelihood_kwargs.update(jitter_time=args.jitter_time)

    elif args.likelihood_type == "ROQGravitationalWaveTransient":
        if args.time_marginalization:
            logger.warning(
                "Time marginalization not implemented for "
                "ROQGravitationalWaveTransient: option ignored"
            )
        likelihood_kwargs.pop("time_marginalization", None)
        likelihood_kwargs.pop("jitter_time", None)
        likelihood_kwargs.update(roq_likelihood_kwargs(args))
    else:
        raise ValueError("Unknown Likelihood class {}")

    likelihood_kwargs = {
        key: likelihood_kwargs[key]
        for key in likelihood_kwargs
        if key in inspect.getfullargspec(Likelihood.__init__).args
    }

    logger.info(
        f"Initialise likelihood {Likelihood} with kwargs: \n{likelihood_kwargs}"
    )

    likelihood = Likelihood(**likelihood_kwargs)
    return likelihood, priors
