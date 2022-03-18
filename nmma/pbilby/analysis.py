#!/usr/bin/env python
"""
Module to run parallel bilby using MPI
"""
import matplotlib
matplotlib.use("agg")
import datetime
import inspect
import json
import logging
import os
import pickle
import sys
import time

import bilby
import dill
import dynesty
import dynesty.plotting as dyplot
import matplotlib.pyplot as plt
import mpi4py
import nestcheck.data_processing
import numpy as np
from bilby.gw import conversion
from dynesty import NestedSampler
from pandas import DataFrame

from .parser import create_nmma_analysis_parser
from parallel_bilby.schwimmbad_fast import MPIPoolFast as MPIPool
from parallel_bilby.utils import (
    get_cli_args,
    get_initial_points_from_prior,
    safe_file_dump,
    stopwatch,
)

from ..joint.likelihood import MultiMessengerLikelihood
from .._version import __version__

__prog__ = "nmma_analysis"

mpi4py.rc.threads = False
mpi4py.rc.recv_mprobe = False


logger = bilby.core.utils.logger


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
    if hasattr(args, "likelihood_roq_params"):
        params = args.likelihood_roq_params
    else:
        params = np.genfromtxt(args.roq_folder + "/params.dat", names=True)

    if hasattr(args, "likelihood_roq_weights"):
        weights = args.likelihood_roq_weights
    else:
        weights = args.weight_file
        logger.info("Loading ROQ weights from {}".format(weights))

    return dict(
        roq_weights=weights, roq_params=params, roq_scale_factor=args.roq_scale_factor
    )


def setup_likelihood_priors(
    interferometers, waveform_generator, light_curve_data, priors, args
):
    """Takes the kwargs and sets up and returns an EM-GW likelihood.

    Parameters
    ----------
    interferometers: bilby.gw.detectors.InterferometerList
        The pre-loaded bilby IFO
    waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
        The waveform generator
    light_curve_dat: dict
        The light curve data with filter as key
    priors: dict
        The priors, used for setting up marginalization
    args: Namespace
        The parser arguments


    Returns
    -------
    likelihood: nmma.joint.likelihood.MultiMessengerLikelihood

    """

    if args.with_eos:

        logger.info("Sampling over EOSs")

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

        logger.info("Sampling over Lambdas")

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

    Likelihood = MultiMessengerLikelihood
    if args.likelihood_type == "GravitationalWaveTransient":
        likelihood_kwargs.update(jitter_time=args.jitter_time)
    elif args.likelihood_type == "ROQGravitationalWaveTransient":
        likelihood_kwargs.pop("jitter_time", None)
        likelihood_kwargs.update(roq_likelihood_kwargs(args))

    likelihood_kwargs = {
        key: likelihood_kwargs[key]
        for key in likelihood_kwargs
        if key in inspect.getfullargspec(Likelihood.__init__).args
    }

    logger.info(
        "Initialise likelihood {} with kwargs: \n{}".format(
            Likelihood, likelihood_kwargs
        )
    )

    likelihood = Likelihood(**likelihood_kwargs)
    return likelihood, priors


def main():
    """ Do nothing function to play nicely with MPI """
    pass


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


@stopwatch
def write_current_state(sampler, resume_file, sampling_time):
    """Writes a checkpoint file

    Parameters
    ----------
    sampler: dynesty.NestedSampler
        The sampler object itself
    resume_file: str
        The name of the resume/checkpoint file to use
    sampling_time: float
        The total sampling time in seconds
    """
    print("")
    logger.info("Start checkpoint writing")
    sampler.kwargs["sampling_time"] = sampling_time
    if dill.pickles(sampler):
        safe_file_dump(sampler, resume_file, dill)
        logger.info("Written checkpoint file {}".format(resume_file))
    else:
        logger.warning("Cannot write pickle resume file!")


def write_sample_dump(sampler, samples_file, search_parameter_keys):
    """Writes a checkpoint file

    Parameters
    ----------
    sampler: dynesty.NestedSampler
        The sampler object itself
    """

    ln_weights = sampler.saved_logwt - sampler.saved_logz[-1]
    weights = np.exp(ln_weights)
    samples = bilby.core.result.rejection_sample(np.array(sampler.saved_v), weights)
    nsamples = len(samples)

    # If we don't have enough samples, don't dump them
    if nsamples < 100:
        return

    logger.info("Writing {} current samples to {}".format(nsamples, samples_file))
    df = DataFrame(samples, columns=search_parameter_keys)
    df.to_csv(samples_file, index=False, header=True, sep=" ")


@stopwatch
def plot_current_state(sampler, search_parameter_keys, outdir, label):
    labels = [label.replace("_", " ") for label in search_parameter_keys]
    try:
        filename = "{}/{}_checkpoint_trace.png".format(outdir, label)
        fig = dyplot.traceplot(sampler.results, labels=labels)[0]
        fig.tight_layout()
        fig.savefig(filename)
    except (
        AssertionError,
        RuntimeError,
        np.linalg.linalg.LinAlgError,
        ValueError,
    ) as e:
        logger.warning(e)
        logger.warning("Failed to create dynesty state plot at checkpoint")
    finally:
        plt.close("all")
    try:
        filename = "{}/{}_checkpoint_run.png".format(outdir, label)
        fig, axs = dyplot.runplot(sampler.results)
        fig.tight_layout()
        plt.savefig(filename)
    except (RuntimeError, np.linalg.linalg.LinAlgError, ValueError) as e:
        logger.warning(e)
        logger.warning("Failed to create dynesty run plot at checkpoint")
    finally:
        plt.close("all")
    try:
        filename = "{}/{}_checkpoint_stats.png".format(outdir, label)
        fig, axs = plt.subplots(nrows=3, sharex=True)
        for ax, name in zip(axs, ["boundidx", "nc", "scale"]):
            ax.plot(getattr(sampler, f"saved_{name}"), color="C0")
            ax.set_ylabel(name)
        axs[-1].set_xlabel("iteration")
        fig.tight_layout()
        plt.savefig(filename)
    except (RuntimeError, ValueError) as e:
        logger.warning(e)
        logger.warning("Failed to create dynesty stats plot at checkpoint")
    finally:
        plt.close("all")


@stopwatch
def read_saved_state(resume_file, continuing=True):
    """
    Read a saved state of the sampler to disk.

    The required information to reconstruct the state of the run is read from a
    pickle file.

    Parameters
    ----------
    resume_file: str
        The path to the resume file to read

    Returns
    -------
    sampler: dynesty.NestedSampler
        If a resume file exists and was successfully read, the nested sampler
        instance updated with the values stored to disk. If unavailable,
        returns False
    sampling_time: int
        The sampling time from previous runs
    """

    if os.path.isfile(resume_file):
        logger.info("Reading resume file {}".format(resume_file))
        with open(resume_file, "rb") as file:
            sampler = dill.load(file)
            if sampler.added_live and continuing:
                sampler._remove_live_points()
            sampler.nqueue = -1
            sampler.rstate = np.random
            sampling_time = sampler.kwargs.pop("sampling_time")
        return sampler, sampling_time
    else:
        logger.info("Resume file {} does not exist.".format(resume_file))
        return False, 0


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "0"
os.environ["MPI_PER_NODE"] = "16"

analysis_parser = create_nmma_analysis_parser(__prog__, __version__)
cli_args = get_cli_args()
input_args = analysis_parser.parse_args(args=cli_args)

with open(input_args.data_dump, "rb") as file:
    data_dump = pickle.load(file)

ifo_list = data_dump["ifo_list"]
waveform_generator = data_dump["waveform_generator"]
waveform_generator.start_time = ifo_list[0].time_array[0]
args = data_dump["args"]
injection_parameters = data_dump.get("injection_parameters", None)
light_curve_data = data_dump["light_curve_data"]

args.weight_file = data_dump["meta_data"].get("weight_file", None)

outdir = args.outdir
if input_args.outdir is not None:
    outdir = input_args.outdir
label = args.label
if input_args.label is not None:
    label = input_args.label

priors = bilby.gw.prior.PriorDict.from_json(data_dump["prior_file"])

logger.setLevel(logging.WARNING)
likelihood, priors = setup_likelihood_priors(
    interferometers=ifo_list,
    waveform_generator=waveform_generator,
    light_curve_data=light_curve_data,
    priors=priors,
    args=args,
)
logger.setLevel(logging.INFO)


def prior_transform_function(u_array):
    return priors.rescale(sampling_keys, u_array)


def log_likelihood_function(v_array):
    if input_args.bilby_zero_likelihood_mode:
        return 0
    parameters = {key: v for key, v in zip(sampling_keys, v_array)}
    if priors.evaluate_constraints(parameters) > 0:
        likelihood.parameters.update(parameters)
        return likelihood.log_likelihood() - likelihood.noise_log_likelihood()
    else:
        return np.nan_to_num(-np.inf)


def log_prior_function(v_array):
    params = {key: t for key, t in zip(sampling_keys, v_array)}
    return priors.ln_prob(params)


sampling_keys = []
for p in priors:
    if isinstance(priors[p], bilby.core.prior.Constraint):
        continue
    if isinstance(priors[p], (int, float)):
        likelihood.parameters[p] = priors[p]
    elif priors[p].is_fixed:
        likelihood.parameters[p] = priors[p].peak
    else:
        sampling_keys.append(p)

periodic = []
reflective = []
for ii, key in enumerate(sampling_keys):
    if priors[key].boundary == "periodic":
        logger.debug("Setting periodic boundary for {}".format(key))
        periodic.append(ii)
    elif priors[key].boundary == "reflective":
        logger.debug("Setting reflective boundary for {}".format(key))
        reflective.append(ii)

# Setting marginalized parameters to their reference values
if likelihood.phase_marginalization:
    likelihood.parameters["phase"] = priors["phase"]
if likelihood.time_marginalization:
    likelihood.parameters["geocent_time"] = priors["geocent_time"]
if likelihood.distance_marginalization:
    likelihood.parameters["luminosity_distance"] = priors["luminosity_distance"]

if input_args.dynesty_sample == "rwalk":
    logger.debug("Using the bilby-implemented rwalk sample method")
    dynesty.dynesty._SAMPLING["rwalk"] = bilby.core.sampler.dynesty.sample_rwalk_bilby
    dynesty.nestedsamplers._SAMPLING[
        "rwalk"
    ] = bilby.core.sampler.dynesty.sample_rwalk_bilby
elif input_args.dynesty_sample == "rwalk_dynesty":
    logger.debug("Using the dynesty-implemented rwalk sample method")
    input_args.dynesty_sample = "rwalk"
else:
    logger.debug(
        "Using the dynesty-implemented {} sample method".format(
            input_args.dynesty_sample
        )
    )

t0 = datetime.datetime.now()
sampling_time = 0
with MPIPool(
    parallel_comms=input_args.fast_mpi,
    time_mpi=input_args.mpi_timing,
    timing_interval=input_args.mpi_timing_interval,
    use_dill=True,
) as pool:
    if pool.is_master():
        POOL_SIZE = pool.size

        logger.info("Setting sampling seed = {}".format(input_args.sampling_seed))
        np.random.seed(input_args.sampling_seed)

        logger.info(f"sampling_keys={sampling_keys}")
        logger.info("Periodic keys: {}".format([sampling_keys[ii] for ii in periodic]))
        logger.info(
            "Reflective keys: {}".format([sampling_keys[ii] for ii in reflective])
        )
        logger.info("Using priors:")
        for key in priors:
            logger.info(f"{key}: {priors[key]}")

        filename_trace = "{}/{}_checkpoint_trace.png".format(outdir, label)
        resume_file = "{}/{}_checkpoint_resume.pickle".format(outdir, label)
        samples_file = "{}/{}_samples.dat".format(outdir, label)

        dynesty_sample = input_args.dynesty_sample
        dynesty_bound = input_args.dynesty_bound
        nlive = input_args.nlive
        walks = input_args.walks
        maxmcmc = input_args.maxmcmc
        nact = input_args.nact
        facc = input_args.facc
        min_eff = input_args.min_eff
        vol_dec = input_args.vol_dec
        vol_check = input_args.vol_check
        enlarge = input_args.enlarge
        nestcheck_flag = input_args.nestcheck

        init_sampler_kwargs = dict(
            nlive=nlive,
            sample=dynesty_sample,
            bound=dynesty_bound,
            walks=walks,
            maxmcmc=maxmcmc,
            nact=nact,
            facc=facc,
            first_update=dict(min_eff=min_eff, min_ncall=2 * nlive),
            vol_dec=vol_dec,
            vol_check=vol_check,
            enlarge=enlarge,
            save_bounds=False,
        )

        ndim = len(sampling_keys)
        sampler, sampling_time = read_saved_state(resume_file)

        if sampler is False:
            logger.info(f"Initializing sampling points with pool size={POOL_SIZE}")
            live_points = get_initial_points_from_prior(
                ndim,
                nlive,
                prior_transform_function,
                log_prior_function,
                log_likelihood_function,
                pool,
            )

            logger.info(
                "Initialize NestedSampler with {}".format(
                    json.dumps(init_sampler_kwargs, indent=1, sort_keys=True)
                )
            )
            sampler = NestedSampler(
                log_likelihood_function,
                prior_transform_function,
                ndim,
                pool=pool,
                queue_size=POOL_SIZE,
                print_func=dynesty.results.print_fn_fallback,
                periodic=periodic,
                reflective=reflective,
                live_points=live_points,
                use_pool=dict(
                    update_bound=True,
                    propose_point=True,
                    prior_transform=True,
                    loglikelihood=True,
                ),
                **init_sampler_kwargs,
            )
        else:
            # Reinstate the pool and map (not saved in the pickle)
            logger.info(
                "Read in resume file with sampling_time = {}".format(sampling_time)
            )
            sampler.pool = pool
            sampler.M = pool.map

        logger.info(
            f"Starting sampling for job {label}, with pool size={POOL_SIZE} "
            f"and check_point_deltaT={input_args.check_point_deltaT}"
        )

        sampler_kwargs = dict(
            n_effective=input_args.n_effective,
            dlogz=input_args.dlogz,
            save_bounds=not input_args.do_not_save_bounds_in_resume,
        )
        logger.info("Run criteria: {}".format(json.dumps(sampler_kwargs)))

        run_time = 0

        for it, res in enumerate(sampler.sample(**sampler_kwargs)):

            (
                worst,
                ustar,
                vstar,
                loglstar,
                logvol,
                logwt,
                logz,
                logzvar,
                h,
                nc,
                worst_it,
                boundidx,
                bounditer,
                eff,
                delta_logz,
            ) = res

            i = it - 1
            dynesty.results.print_fn_fallback(
                res, i, sampler.ncall, dlogz=input_args.dlogz
            )

            if (
                it == 0 or it % input_args.n_check_point != 0
            ) and it != input_args.max_its:
                continue

            iteration_time = (datetime.datetime.now() - t0).total_seconds()
            t0 = datetime.datetime.now()

            sampling_time += iteration_time
            run_time += iteration_time

            if os.path.isfile(resume_file):
                last_checkpoint_s = time.time() - os.path.getmtime(resume_file)
            else:
                last_checkpoint_s = np.inf

            if (
                last_checkpoint_s > input_args.check_point_deltaT
                or it == input_args.max_its
                or run_time > input_args.max_run_time
            ):
                # dill does not work well with the MPI
                # have these attributes removed before checkpointing
                sampler.pool = None
                sampler.M = None
                write_current_state(sampler, resume_file, sampling_time)
                write_sample_dump(sampler, samples_file, sampling_keys)
                # add the MPI attributes back to the sampler
                sampler.pool = pool
                sampler.M = pool.map
                if input_args.no_plot is False:
                    plot_current_state(sampler, sampling_keys, outdir, label)

                if it == input_args.max_its:
                    logger.info(f"Max iterations ({it}) reached; stopping sampling.")
                    sys.exit(0)

                if run_time > input_args.max_run_time:
                    logger.info(
                        f"Max run time ({input_args.max_run_time}) reached; stopping sampling."
                    )
                    sys.exit(0)

        # Adding the final set of live points.
        for it_final, res in enumerate(sampler.add_live_points()):
            pass

        # Create a final checkpoint and set of plots
        write_current_state(sampler, resume_file, sampling_time)
        write_sample_dump(sampler, samples_file, sampling_keys)
        if input_args.no_plot is False:
            plot_current_state(sampler, sampling_keys, outdir, label)

        sampling_time += (datetime.datetime.now() - t0).total_seconds()

        out = sampler.results

        if nestcheck_flag is True:
            logger.info("Creating nestcheck files")
            ns_run = nestcheck.data_processing.process_dynesty_run(out)
            nestcheck_path = os.path.join(outdir, "Nestcheck")
            bilby.core.utils.check_directory_exists_and_if_not_mkdir(nestcheck_path)
            nestcheck_result = "{}/{}_nestcheck.pickle".format(nestcheck_path, label)

            with open(nestcheck_result, "wb") as file_nest:
                pickle.dump(ns_run, file_nest)

        weights = np.exp(out["logwt"] - out["logz"][-1])
        nested_samples = DataFrame(out.samples, columns=sampling_keys)
        nested_samples["weights"] = weights
        nested_samples["log_likelihood"] = out.logl

        result = bilby.core.result.Result(
            label=label, outdir=outdir, search_parameter_keys=sampling_keys
        )
        result.priors = priors
        result.samples = dynesty.utils.resample_equal(out.samples, weights)
        result.nested_samples = nested_samples
        result.meta_data = data_dump["meta_data"]
        result.meta_data["command_line_args"] = vars(input_args)
        result.meta_data["command_line_args"]["sampler"] = "parallel_bilby"
        result.meta_data["config_file"] = vars(args)
        result.meta_data["data_dump"] = input_args.data_dump
        result.meta_data["likelihood"] = likelihood.meta_data
        result.meta_data["sampler_kwargs"] = init_sampler_kwargs
        result.meta_data["run_sampler_kwargs"] = sampler_kwargs
        result.meta_data["injection_parameters"] = injection_parameters
        result.injection_parameters = injection_parameters

        result.log_likelihood_evaluations = reorder_loglikelihoods(
            unsorted_loglikelihoods=out.logl,
            unsorted_samples=out.samples,
            sorted_samples=result.samples,
        )

        result.log_evidence = out.logz[-1] + likelihood.noise_log_likelihood()
        result.log_evidence_err = out.logzerr[-1]
        result.log_noise_evidence = likelihood.noise_log_likelihood()
        result.log_bayes_factor = result.log_evidence - result.log_noise_evidence
        result.sampling_time = sampling_time

        result.samples_to_posterior()

        posterior = result.posterior

        nsamples = len(posterior)
        logger.info("Using {} samples".format(nsamples))

        posterior = conversion.fill_from_fixed_priors(posterior, priors)

        # DEBUG required
        #        logger.info(
        #            "Generating posterior from marginalized parameters for"
        #            f" nsamples={len(posterior)}, POOL={pool.size}"
        #        )

        #        fill_args = [(ii, row, likelihood) for ii, row in posterior.iterrows()]
        #        samples = pool.map(fill_sample, fill_args)
        #        result.posterior = pd.DataFrame(samples)

        posterior, _ = likelihood.parameter_conversion(
            posterior,
        )

        result.posterior = conversion._generate_all_cbc_parameters(
            posterior,
            likelihood.GWLikelihood.waveform_generator.waveform_arguments,
            conversion.convert_to_lal_binary_neutron_star_parameters,
        )

        logger.debug("Updating prior to the actual prior")
        for par, name in zip(
            ["distance", "phase", "time"],
            ["luminosity_distance", "phase", "geocent_time"],
        ):
            if getattr(likelihood, "{}_marginalization".format(par), False):
                priors[name] = likelihood.priors[name]
        result.priors = priors

        if args.convert_to_flat_in_component_mass:
            try:
                result = bilby.gw.prior.convert_to_flat_in_component_mass_prior(result)
            except Exception as e:
                logger.warning(f"Unable to convert to the LALInference prior: {e}")

        logger.info(
            f"Saving posterior samples to {outdir}/{label}_posterior_samples.dat"
        )
        result.save_posterior_samples()
        logger.info(f"Saving result to {outdir}/{label}_result.json")
        result.save_to_file(extension="json")
        print(
            "Sampling time = {}s".format(
                datetime.timedelta(seconds=result.sampling_time)
            )
        )
        print(result)
