import os
import json
import numpy as np
from astropy import time
import h5py
import healpy as hp
import matplotlib.pyplot as plt

import bilby
import bilby.core
from gwpy.table import Table
from ligo.skymap import bayestar, distance
from ligo.skymap.io import read_sky_map

from nmma.em.lightcurve_generation import create_light_curve_data
from nmma.em import em_parsing as emp
from nmma.em.model import (
    GRBLightCurveModel,
    KilonovaGRBLightCurveModel,
    SupernovaGRBLightCurveModel,
    SVDLightCurveModel,
    create_light_curve_model_from_args
)
from nmma.em import io 
from nmma.em.plotting_utils import lc_plot_with_histogram
from .utils import setup_sample_times, DEFAULT_FILTERS

from nmma.joint.conversion import (MultimessengerConversion, 
                                   mass_ratio_to_eta,
                                   component_masses_to_mass_quantities,
                                   chirp_mass_and_eta_to_component_masses)

from nmma.eos.eos_processing import load_tabulated_macro_eos_set_to_dict

np.random.seed(0)

def read_trigger_time(parameters, args=None):
    try:
        tc_gps = time.Time(parameters["geocent_time_x"], format="gps")
    except KeyError:
        try:
            tc_gps = time.Time(parameters["geocent_time"], format="gps")
        except KeyError:
            try:
                tc_gps = time.Time(args.gps, format="gps")
            except AttributeError:
                raise AttributeError("Need either geocent_time or geocent_time_x")
    parameters["kilonova_trigger_time"] = tc_gps.mjd
    return parameters

def write_matter_file(matter_outfile, complete_parameters):
    fid = open(matter_outfile, "w")
    if np.isfinite(complete_parameters["log10_mej_dyn"]):
        fid.write(
            "1 %.5f %.5f\n"
            % (
                complete_parameters["log10_mej_dyn"],
                complete_parameters["log10_mej_wind"],
            )
        )
    else:
        fid.write(
            "0 0 0\n"
            % (
                complete_parameters["log10_mej_dyn"],
                complete_parameters["log10_mej_wind"],
            )
        )
    fid.close()

def write_lightcurve_file(lc_outfile, data, filters, sample_times):
    fid = open(lc_outfile, "w")
    fid.write("# t[days] ")
    fid.write(str(" ".join(filters)))
    fid.write("\n")
    for ii, time_step in enumerate(sample_times):
        fid.write("%.5f " % time_step)
        for filt in data.keys():
            if filters is not None and filt not in filters:
                continue
            fid.write("%.3f " % data[filt][ii, 1])
        fid.write("\n")
    fid.close()

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
    return validate_lightcurve(data=args.data,filters=args.filters,min_obs=args.min_obs,cutoff_time=args.cutoff_time,silent=args.silent)

def validate_lightcurve(data,filters=None,min_obs=3,cutoff_time=0,silent=False):
    """
    Evaluates whether the lightcurve has the requisite user-defined number of observations in the filters provided within the defined time window from the first observation. In the case where one wants to check that at least one filter has the requisite number of observations, the function should be called multiple times with different filter arguments.

    Parameters:
        data (str): Path to the data file in [nmma-compliant format (.dat, .json, etc.)
        filters (str): Comma separated list of filters to validate against. If not provided, all filters in the data will be used.
        min_obs (int): Minimum number of observations required in each filter before cutoff time.
        cutoff_time (float): Cutoff time (relative to the first data point) that the minimum observations must be in. If not provided, the entire lightcurve will be evaluated
        silent (bool): Suppress output

    Returns:
        bool: True if the lightcurve meets the minimum number of observations in the defined time window, False otherwise.
    """
    data = io.loadEvent(data)

    ## determine filters to consider
    if filters:
        filters_to_check = filters.replace(" ", "").split(",")
    else:
        filters_to_check = list(data.keys())

    ## determine time window to consider
    min_time = np.min([np.min(array[:,0]) for array in data.values()])
    max_time = min_time + cutoff_time if cutoff_time > 0 else np.max([np.max(array[:,0]) for array in data.values()])

    ## evaluate lightcurve for each filter
    for filter in filters_to_check:
        if filter not in DEFAULT_FILTERS:
            raise ValueError(f"Filter {filter} not in supported filter list")
        elif filter not in data.keys():
            print(f"{filter} not present in data file, cannot validate") if not silent else None
            return False
        filter_data_indices = np.where(data[filter][:,0] <= max_time)[0]
        filter_data = [data[filter][i] for i in filter_data_indices]
        
        ## evaluate the number of detections
        num_observations = sum(1 for value in filter_data if np.isfinite(value[1]))
        num_detections = sum(1 for value in filter_data if np.isfinite(value[2]) and value[2] != 99)
        if num_detections < min_obs:
            print(f"{filter} filter has {num_detections} detections, less than the required {min_obs}") if not silent else None
            return False
        else:
            continue
    print(f"Lightcurve has at least {min_obs} detections in the filters within the first {max_time-min_time} days") if not silent else None

    return True


def marginalised_lightcurve_expectation_from_gw_samples(args=None):
    """Routine to generate a marginalized set of light curves from a set of GW samples. These need to be parsed as template-files, h5-file or coincidence files."""
    args = emp.parsing_and_logging(emp.lc_marginalisation_parser, args)

    if args.joint_light_curve:

        assert args.model != "TrPi2018", "TrPi2018 is not a kilonova / supernova model"

        if args.model != "nugent-hyper" or args.model != "salt2":

            kilonova_kwargs = dict(
                model=args.model,
                svd_path=args.svd_path,
                svd_mag_ncoeff =args.svd_mag_ncoeff,
                svd_lbol_ncoeff=args.svd_lbol_ncoeff,
            )

            light_curve_model = KilonovaGRBLightCurveModel(
                kilonova_kwargs=kilonova_kwargs,
                grb_resolution=args.grb_resolution,
                jet_type=args.jet_type,
            )

        else:
            supernova_kwargs = dict(
                model = args.model
            )
            light_curve_model = SupernovaGRBLightCurveModel(
                supernova_kwargs = supernova_kwargs,
                grb_resolution=args.grb_resolution,
                jet_type=args.jet_type,
            )

    else:
        if args.model == "TrPi2018":
            light_curve_model = GRBLightCurveModel(
                resolution=args.grb_resolution,
                jet_type=args.jet_type,
            )

        else:
            light_curve_kwargs = dict(
                model=args.model,
                svd_path=args.svd_path,
                svd_mag_ncoeff=args.svd_mag_ncoeff,
                svd_lbol_ncoeff=args.svd_lbol_ncoeff,
            )
            light_curve_model = SVDLightCurveModel(  # noqa: F841
                **light_curve_kwargs, interpolation_type=args.interpolation_type
            )

    args.em_transient_error = 0

    ## read eos and gw data
    EOS_data, weights, Neos = load_tabulated_macro_eos_set_to_dict(args.eos_dir, args.eos_weights)

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
        order = hp.nside2order(512)
        skymap = bayestar.rasterize(skymap, order)
        dist_mean, dist_std = distance.parameters_to_marginal_moments(
            skymap["PROB"], skymap["DISTMU"], skymap["DISTSIGMA"]
        )
        data_out["dist"] = dist_mean + np.random.randn(len(data_out["m1"])) * dist_std

    else:
        print("Needs template_file, hdf5_file, or coinc_file")
        exit(1)
    
    data_out = get_all_gw_quantities(data_out)



    idxs = np.random.choice(np.arange(len(weights)), args.Nmarg, p=weights)
    idys = np.random.choice(
        np.arange(len(data_out["m1"])),
        args.Nmarg,
        p=data_out["weight"] / np.sum(data_out["weight"]),
    )

    mag_ds, matter = {}, {}
    for ii in range(args.Nmarg):

        outdir = os.path.join(args.outdir, "%d" % ii)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        injection_outfile = os.path.join(outdir, "lc.dat")
        matter_outfile = os.path.join(outdir, "matter.dat")
        if os.path.isfile(injection_outfile) and os.path.isfile(matter_outfile):
            mag_ds[ii] = np.loadtxt(injection_outfile)
            matter[ii] = np.loadtxt(matter_outfile)
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
        zeta = 10 ** np.random.uniform(log10zeta_min, log10zeta_max)
        
        if (m1 < mMax) and (m2 < mMax):
            alpha_min, alpha_max = 1e-2, 2e-2
            alpha = np.random.uniform(alpha_min, alpha_max)
            log10_alpha = np.log10(alpha)
        elif (m1 > mMax) and (m2 < mMax):
            log10_alpha_min, log10_alpha_max = -3, -1
            log10_alpha = np.random.uniform(log10_alpha_min, log10_alpha_max)
            alpha = 10 ** log10_alpha
        params.update({ "alpha"         : alpha,
                        "log10_alpha"   : log10_alpha, 
                        "ratio_zeta"    : zeta})
        args.with_eos = True
        args.eos_data = args.eos_dir
        conversion = MultimessengerConversion(args, messengers=['gw', 'em'])
        complete_parameters, _ = conversion.convert_to_multimessenger_parameters(params)

        write_matter_file(matter_outfile, complete_parameters)


        # initialize light curve model
        complete_parameters = read_trigger_time(complete_parameters, args)
        sample_times = setup_sample_times(args)
        data = create_light_curve_data(
            complete_parameters, args, light_curve_model,  sample_times, doAbsolute=args.absolute
        )

        filters = args.filters.split(",")

        write_lightcurve_file(injection_outfile, data, filters, sample_times)

        mag_ds[ii] = np.loadtxt(injection_outfile)
        matter[ii] = np.loadtxt(matter_outfile)

    if args.plot:
        NS, dyn, wind = [], [], []
        for jj, key in enumerate(matter.keys()):
            NS.append(matter[key][0])
            dyn.append(matter[key][1])
            wind.append(matter[key][2])

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
        plot_dict = {filt: np.vstack([lc_data[:, i+1] for lc_data in mag_ds.values()]) for i, filt in enumerate(filters)}
        if args.absolute:
            ylim = getattr(args, 'ylim', [-12, -18])
        else:
            ylim = getattr(args, 'ylim', [24, 15])
        lc_plot_with_histogram(filters, plot_dict, sample_times, save_path= plotpath, ylim = ylim, fontsize=30)


def lcs_from_injection_parameters(args=None):
    args = emp.parsing_and_logging(emp.lightcurve_parser, args)

    # initialize light curve model
    if args.filters:
        filters = args.filters.split(",")
    else:
        filters = None
    light_curve_model = create_light_curve_model_from_args(args.model, args, filters)

    # read injection file
    with open(args.injection, "r") as f:
        injection_dict = json.load(f, object_hook=bilby.core.utils.decode_bilby_json)

    ## io setup
    if args.outfile_type == "json":
        ext = "json"
        reading_function = io.return_from_json
        saving_function  = io.write_to_json
    elif args.outfile_type == "csv":
        ext = "dat"
        reading_function =  io.read_lightcurve_file
        saving_function  = write_lightcurve_file
    injection_outfile = getattr(args, 'injection_outfile', args.label)

    
    sample_times = setup_sample_times(args)
    injection_df = injection_dict["injections"]
    # save simulation_id from observing scenarios data
    # we save lighcurve for each event with its initial simulation ID
    # from observing scenarios
    simulation_id = injection_df["simulation_id"]
    mag_ds = {}

    for index, row in injection_df.iterrows():
        ## setup file
        injection_outfile = os.path.join(
            args.outdir, f"{injection_outfile}_{simulation_id[index]}.{ext}"
        )
        if os.path.isfile(injection_outfile):
            try:
                mag_ds[index] = reading_function(injection_outfile)
                continue

            except ValueError:
                raise ValueError(f"The previous run generated light curves with unreadable content. Please remove all output files in .{ext} format then retry.")

        ## do injection!
        injection_parameters = row.to_dict()
        injection_parameters = read_trigger_time(injection_parameters, args)

        if args.increment_seeds:
            args.generation_seed = args.generation_seed + 1
        # args.injection_detection_limit = np.inf

        data = create_light_curve_data(
            injection_parameters,
            args,
            light_curve_model=light_curve_model,
            sample_times=sample_times,
            doAbsolute=args.absolute,
            keep_infinite_data=True,
        )
        print("Injection generated")

        #store and retrieve to double check
        saving_function(injection_outfile, data, filters, sample_times)
        mag_ds[index] = reading_function(injection_outfile)

    if args.plot:
        plotpath= os.path.join(args.outdir, f"injection_{args.model}_{args.label}_lc.pdf")
        plot_data_dict = {}
        plot_data_dict
        for filt in filters:
            plot_data_filt = []
            for lc_data in mag_ds.values():
                data_vec = np.array(lc_data[filt])
                if data_vec.ndim == 2:
                    plot_data_filt.append(data_vec[:, 1].tolist())
                else:
                    plot_data_filt.append(data_vec.tolist())
            plot_data_dict[filt] = np.vstack(plot_data_filt)

        lc_plot_with_histogram(
            filters, plot_data_dict, sample_times=sample_times, save_path=plotpath, xlim=args.xlim, ylim=args.ylim, colorbar= True, ylabel_kwargs = dict(fontsize=30, rotation=90, labelpad=8))
