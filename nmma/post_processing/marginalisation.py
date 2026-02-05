import os
import numpy as np
import h5py
import matplotlib.pyplot as plt


from ..eos.eos_processing import load_tabulated_macro_eos_set_to_dict, EoSConverter
from gwpy.table import Table

from ..em.lightcurve_generation import create_light_curve_data
from ..em import io, model, utils, em_parsing as emp
from ..em.plotting_utils import lc_plot_with_histogram

from ..core import conversion as conv 
from ..core.utils import read_trigger_time

def marginalised_lightcurve_expectation_from_gw_samples(args=None):
    """Routine to generate a marginalized set of light curves from a set of GW samples. These need to be parsed as template-files, h5-file or coincidence files."""
    args = emp.parsing_and_logging(emp.lc_marginalisation_parser, args)

    rng = np.random.default_rng(args.generation_seed)
    args.mag_error_scale = 0
    filters = utils.set_filters(args)
    if filters is None:
        filters = 'u,g,r,i,z,y,J,H,K'
    light_curve_model = model.create_light_curve_model_from_args(args.em_model, args, filters)
    conversion = conv.MultimessengerConversion.basic_bns(EoSConverter(args, method = 'tabulated'),
        light_curve_model.parameter_conversion)

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
        from ligo.skymap import bayestar, distance, io as lio
        data_out = Table.read(args.coinc_file, format="ligolw", tablename="sngl_inspiral")
        data_out["m1"], data_out["m2"] = data_out["mass1"], data_out["mass2"]
        skymap = lio.fits.read_sky_map(args.skymap, moc=True, distances=True)
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
                        "ratio_zeta"    : zeta})
        
        complete_parameters = conversion.convert_to_multimessenger_parameters(params)

        log10_mej_dyn = complete_parameters["log10_mej_dyn"].item()
        log10_mej_wind = complete_parameters["log10_mej_wind"].item()
        with open(matter_outfile, "w") as fid:
            if np.isfinite(log10_mej_dyn):
                fid.write(f"1 {log10_mej_dyn:.5f} {log10_mej_wind:.5f}\n")
            else:
                fid.write("0 0 0\n")
        # initialize light curve model
        complete_parameters['trigger_time'] = read_trigger_time(complete_parameters, args)
        sample_times = utils.setup_sample_times(args)
        data = create_light_curve_data(
            complete_parameters, args, light_curve_model, sample_times, rng=rng,
        )

        filters = list(data.keys())

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

def get_all_gw_quantities(data_out):
    try:
        data_out["mchirp"], data_out["eta"], data_out["q"] = conv.component_masses_to_mass_quantities(
            data_out["m1"], data_out["m2"]
        )
    except KeyError:
        data_out["eta"] = conv.mass_ratio_to_eta(data_out["q"])
        data_out["mchirp"] = data_out["mc"]
        data_out["m1"], data_out["m2"] = conv.chirp_mass_and_eta_to_component_masses(data_out["mchirp"], data_out["eta"])
    
    
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
