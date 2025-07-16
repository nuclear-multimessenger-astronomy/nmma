import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

from gwpy.table import Table
from ligo.skymap import bayestar, distance
from ligo.skymap.io import read_sky_map

from .lightcurve_generation import create_light_curve_data
from . import io, utils, em_parsing as emp
from .model import  create_light_curve_model_from_args, create_injection_model
from .plotting_utils import lc_plot_with_histogram

from ..joint.conversion import (MultimessengerConversion, 
                                   mass_ratio_to_eta,
                                   component_masses_to_mass_quantities,
                                   chirp_mass_and_eta_to_component_masses)
from ..joint.utils import read_injection_file, set_filename

from ..eos.eos_processing import load_tabulated_macro_eos_set_to_dict

def get_all_gw_quantities(data_out):
    try:
        data_out["mchirp"], data_out["eta"], data_out["q"] = component_masses_to_mass_quantities(
            data_out["m1"], data_out["m2"]
        )
    except KeyError:
        data_out["eta"] = mass_ratio_to_eta(data_out["q"])
        data_out["mchirp"] = data_out["mc"]
        data_out["m1"], data_out["m2"] = chirp_mass_and_eta_to_component_masses(data_out["mchirp"], data_out["eta"])
    
    
    data_out["weight"] = 1.0 / len(data_out["m1"])


    data_out["chi_eff"] = (
        data_out["m1"] * data_out["a1"] + data_out["m2"] * data_out["a2"]
    ) / (data_out["m1"] + data_out["m2"])

    for key in ["a1", "a2", "theta_jn", "tilt1", "tilt2"]:
        if key not in data_out.keys():
            data_out[key] = 0.
    try:
        data_out["a1"], data_out["a2"] = data_out["spin1z"], data_out["spin2z"]
    except KeyError:
        pass
    return data_out


def call_lc_validation (args=None):
    args = emp.parsing_and_logging(emp.lc_validation_parser, args)
    return validate_lightcurve(data_file=args.light_curve_data,filters=args.filters,min_obs=args.min_obs,cutoff_time=args.cutoff_time,verbose=args.verbose)

def validate_lightcurve(data_file,filters=None,min_obs=3,cutoff_time=0,verbose=False):
    """
    Evaluates whether the lightcurve has the requisite user-defined number of observations in the filters provided within the defined time window from the first observation. In the case where one wants to check that at least one filter has the requisite number of observations, the function should be called multiple times with different filter arguments.

    Parameters:
        data_file (str): Path to the data file in nmma-compliant format (.dat, .json, etc.)
        filters (str): Comma separated list of filters to validate against. If not provided, all filters in the data will be used.
        min_obs (int): Minimum number of observations required in each filter before cutoff time.
        cutoff_time (float): Cutoff time (relative to the first data point) that the minimum observations must be in. If not provided, the entire lightcurve will be evaluated
        verbose (bool): Verbose output

    Returns:
        bool: True if the lightcurve meets the minimum number of observations in the defined time window, False otherwise.
    """
    data = io.load_em_observations(data_file, format='observations')

    ## determine filters to consider
    if filters:
        filters_to_check = filters.replace(" ", "").split(",")
    else:
        filters_to_check = list(data.keys())

    ## determine time window to consider
    min_time = np.min([np.min(filt_dict['time']) for filt_dict in data.values()])
    max_time = min_time + cutoff_time if cutoff_time > 0 else np.max([np.max(filt_dict['time']) for filt_dict in data.values()])

    ## evaluate lightcurve for each filter
    for filter in filters_to_check:
        if filter not in utils.DEFAULT_FILTERS:
            raise ValueError(f"Filter {filter} not in supported filter list")
        elif filter not in data.keys():
            print(f"{filter} not present in data file, cannot validate") if verbose else None
            return False
        filter_idcs = (data[filter]['time']<= max_time)
        num_detections = np.sum(np.isfinite(data[filter]['mag_error'][filter_idcs]))
        if num_detections < min_obs:
            print(f"{filter} filter has {num_detections} detections, less than the required {min_obs}") if verbose else None
            return False
        else:
            continue
    print(f"Lightcurve has at least {min_obs} detections in the filters within the first {max_time-min_time} days") if verbose else None

    return True


def marginalised_lightcurve_expectation_from_gw_samples(args=None):
    """Routine to generate a marginalized set of light curves from a set of GW samples. These need to be parsed as template-files, h5-file or coincidence files."""
    args = emp.parsing_and_logging(emp.lc_marginalisation_parser, args)

    rng = np.random.default_rng(args.generation_seed)
    args.mag_error_scale = 0
    light_curve_model = create_light_curve_model_from_args(args.em_model, args)
    

    ## read eos and gw data
    EOS_data, weights, Neos = load_tabulated_macro_eos_set_to_dict(args.eos_data, args.eos_weights)
    args.Neos = Neos

    if args.template_file is not None:
        try:
            names = ["SNRdiff", "erf", "weight", "m1", "m2", "a1", "a2", "dist"]
            data_out = Table.read(args.template_file, names=names, format="ascii")
        except Exception:
            names = ["SNRdiff", "erf", "weight", "m1", "m2", "dist"]
            data_out = Table.read(args.template_file, names=names, format="ascii")

    elif args.hdf5_file is not None:
        f = h5py.File(args.hdf5_file, "r")
        posterior = f["lalinference"]["lalinference_mcmc"]["posterior_samples"][()]
        data_out = Table(posterior)
        args.gps = np.median(data_out["t0"])


    elif args.coinc_file is not None:
        data_out = Table.read(args.coinc_file, format="ligolw", tablename="sngl_inspiral")
        data_out["m1"], data_out["m2"] = data_out["mass1"], data_out["mass2"]
        skymap = read_sky_map(args.skymap, moc=True, distances=True)
        skymap = bayestar.rasterize(skymap, order=9)
        dist_mean, dist_std = distance.parameters_to_marginal_moments(
            skymap["PROB"], skymap["DISTMU"], skymap["DISTSIGMA"]
        )
        data_out["dist"] = dist_mean + rng.standard_normal(len(data_out["m1"])) * dist_std

    else:
        print("Needs template_file, hdf5_file, or coinc_file")
        exit(1)
    
    data_out = get_all_gw_quantities(data_out)


    idxs = rng.choice(np.arange(len(weights)), args.Nmarg, p=weights)
    idys = rng.choice(np.arange(len(data_out["m1"])), args.Nmarg,
        p=data_out["weight"] / np.sum(data_out["weight"]) )

    mag_ds, matter = [], []
    for ii in range(args.Nmarg):

        outdir = os.path.join(args.outdir, "%d" % ii)
        os.makedirs(outdir, exist_ok=True)

        lightcurve_outfile = os.path.join(outdir, "lc.dat")
        matter_outfile = os.path.join(outdir, "matter.dat")
        if os.path.isfile(lightcurve_outfile) and os.path.isfile(matter_outfile):
            mag_ds.append(io.read_lc_from_csv(lightcurve_outfile, args, format='model'))
            matter.append(np.loadtxt(matter_outfile))
            continue

        idx, idy = int(idxs[ii]), int(idys[ii])
        m1, m2 = data_out["m1"][idy], data_out["m2"][idy]

        mMax = np.max(EOS_data[idx]["M"]) 

        params = {
            "luminosity_distance": data_out["dist"][idy],
            "chirp_mass": data_out["mchirp"][idy],
            "ratio_epsilon": 1e-20,
            "theta_jn": data_out["theta_jn"][idy],
            "a_1": data_out["a1"][idy],
            "a_2": data_out["m2"][idy],
            "mass_1": m1,
            "mass_2": m2,
            "EOS": idx,
            "cos_tilt_1": np.cos(data_out["tilt1"][idy]),
            "cos_tilt_2": np.cos(data_out["tilt2"][idy]),
            "KNphi": 30,
        }

        log10zeta_min, log10zeta_max = -3, 0
        zeta = 10 ** rng.uniform(log10zeta_min, log10zeta_max)
        
        if (m1 < mMax) and (m2 < mMax):
            alpha_min, alpha_max = 1e-2, 2e-2
            alpha = rng.uniform(alpha_min, alpha_max)
            log10_alpha = np.log10(alpha)
        elif (m1 > mMax) and (m2 < mMax):
            log10_alpha_min, log10_alpha_max = -3, -1
            log10_alpha = rng.uniform(log10_alpha_min, log10_alpha_max)
            alpha = 10 ** log10_alpha
        params.update({ "alpha"         : alpha,
                        "log10_alpha"   : log10_alpha, 
                        "ratio_zeta"    : zeta})
        
        conversion = MultimessengerConversion(args, 
                messengers=['gw', 'em'], ana_modifiers=['tabulated_eos'])
        complete_parameters, _ = conversion.convert_to_multimessenger_parameters(params)

        log10_mej_dyn = complete_parameters["log10_mej_dyn"].item()
        log10_mej_wind = complete_parameters["log10_mej_wind"].item()
        with open(matter_outfile, "w") as fid:
            if np.isfinite(log10_mej_dyn):
                fid.write(f"1 {log10_mej_dyn:.5f} {log10_mej_wind:.5f}\n")
            else:
                fid.write("0 0 0\n")
        # initialize light curve model
        complete_parameters['trigger_time'] = utils.read_trigger_time(complete_parameters, args)
        sample_times = utils.setup_sample_times(args)
        data = create_light_curve_data(
            complete_parameters, args, light_curve_model, sample_times, rng=rng,
        )

        filters = args.filters.split(",")

        io.write_lc_to_csv(lightcurve_outfile, data, 'model')

        mag_ds.append(io.read_lc_from_csv(lightcurve_outfile, args, format='model'))
        matter.append( np.loadtxt(matter_outfile))

    if args.plot:
        NS, dyn, wind = [], [], []
        for matter_data in  matter:
            NS.append(matter_data[0])
            dyn.append(matter_data[1])
            wind.append(matter_data[2])

        print("Fraction of samples with NS: %.5f" % (np.sum(NS) / len(NS)))

        bins = np.linspace(-3, 0, 25)
        dyn_hist, bin_edges = np.histogram(dyn, bins=bins, density=True)
        wind_hist, bin_edges = np.histogram(wind, bins=bins, density=True)
        bins = (bin_edges[1:] + bin_edges[:-1]) / 2.0

        plotName = os.path.join(args.outdir, "matter.pdf")
        fig = plt.figure(figsize=(10, 6))
        plt.step(bins, dyn_hist, "k--", label="Dynamical")
        plt.step(bins, wind_hist, "b-", label="Wind")
        plt.xlabel(r"log10(Ejecta Mass / $M_\odot$)")
        plt.ylabel("Probability Density Function")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plotName, bbox_inches="tight")
        plt.close()

        plotpath= os.path.join(args.outdir, "lc.pdf")
        plot_dict = {filt: np.vstack(
                        [lc_data[filt]['mag'] for lc_data in mag_ds])
                        for filt in filters}
        times = next(iter(mag_ds[0].values()))['time']
        if args.absolute:
            ylim = getattr(args, 'ylim', [-12, -18])
        else:
            ylim = getattr(args, 'ylim', [24, 15])
        lc_plot_with_histogram(filters, plot_dict, times, plotpath, ylim=ylim, fontsize=30)


def lcs_from_injection_parameters(args=None):
    args = emp.parsing_and_logging(emp.lightcurve_parser, args)
    
    # initialize light curve model
    light_curve_model = create_light_curve_model_from_args(args.em_model, args)
    # lightcurve model will set them to usable list
    filters = light_curve_model.filters

    # read injection file
    injection_df = read_injection_file(args)
    mag_ds = create_multiple_injections(injection_df, args, light_curve_model)

    if args.plot:
        plotpath= os.path.join(args.outdir, f"injection_{args.em_model}_{args.label}_lc.pdf")
        plot_dict = {filt: np.vstack([lc_data[filt]['mag'] 
                        for lc_data in mag_ds.values()]) for filt in filters}
        first_lc_dict = next(iter(mag_ds.values()))
        times = next(iter(first_lc_dict.values()))['time']

        lc_plot_with_histogram(
            filters, plot_dict, times,  plotpath, 
            xlim=args.xlim, ylim=args.ylim, colorbar= True, 
            ylabel_kwargs = dict(fontsize=30, rotation=90, labelpad=8))

def create_multiple_injections(injection_df, args, light_curve_model=None, format = 'model'):
    mag_ds = {}

    rng = np.random.default_rng(args.generation_seed)
    for index, row in injection_df.iterrows():
        mag_ds[index] = make_injection_lightcurve_from_parameters(
            row.to_dict(), args, light_curve_model, rng, format)
    return mag_ds


def make_injection_lightcurve_from_parameters(
    injection_parameters, args, light_curve_model=None, rng=None, format='model'
):
    injection_outfile = set_filename(args.label, args, 
                                     f"_lc_{int(injection_parameters['simulation_id'])}")

    if os.path.isfile(injection_outfile):
        try:
            return io.load_em_observations(injection_outfile, format=format)

        except ValueError:
            raise ValueError(f"The previous run generated light curves with unreadable content for {injection_outfile}. Consider removing all output files in this format, format then retry.")

    data, _ = make_injection(injection_parameters, args, injection_model = light_curve_model, rng=rng, keep_infinite_data=True)

    if injection_outfile:
        #store and retrieve to double check
        io.write_em_observations(injection_outfile, data, format=format)
        return io.load_em_observations(injection_outfile, format=format)
    else:
        return data


def make_injection(injection_parameters, args, filters = None, injection_model = None,
                   rng=None, keep_infinite_data=False):
    

    injection_parameters["trigger_time"] = utils.read_trigger_time(injection_parameters, args)
    
    if args.ignore_timeshift:
        injection_parameters.pop('timeshift', None)

    injection_parameters["trigger_time"] += injection_parameters.get("timeshift",0)

    if args.prompt_collapse:
        injection_parameters["log10_mej_wind"] = -3.0

    # sanity check for eject masses
    for key in ["log10_mej_dyn", "log10_mej_wind"]:
        if key in injection_parameters and not np.isfinite(injection_parameters[key]):
            injection_parameters[key] = -3.0

    if injection_model is None:
        print("Creating injection light curve model")
        injection_model = create_injection_model(args, filters)

    data = create_light_curve_data(
        injection_parameters,
        args,
        light_curve_model=injection_model,
        keep_infinite_data=keep_infinite_data,
        rng = rng
    )
    print("Injection generated")

    return data, injection_parameters 