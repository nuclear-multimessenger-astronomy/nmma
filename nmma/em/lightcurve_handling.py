import os
import shutil
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

import sncosmo
from astropy import units as u
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

from .lightcurve_generation import create_light_curve_data
from . import io, model, utils, em_parsing as emp
from .plotting_utils import lc_plot_with_histogram, basic_em_analysis_plot

from ..core.constants import get_cosmology, D, c_cgs
from ..core import conversion as conv 
from ..core.utils import read_injection_file, set_filename, read_trigger_time

def post_process_bestfit(transient, bestfit_params, args, result=None):
    
    lc_model = transient.light_curve_model
    observable_times, best_mags = lc_model.gen_detector_lc(bestfit_params)
    model_error_data = transient.systematics_handler(bestfit_params)
    model_error = {filt: utils.autocomplete_data(observable_times,
    transient.light_curve_times[filt], data) 
        for filt, data in model_error_data.items()}
    # model may not necessarily work on observed filters:
    for filt in set(transient.observed_filters) - set(best_mags.keys()):
        best_mags[filt] =  utils.get_filtered_mag(best_mags, filt)
        model_error[filt]= utils.get_filtered_mag(model_error, filt)

    
    # transient.parameters = bestfit_params
    chi2_dict, mismatches = compute_chisquare_dict(transient, best_mags, 
            observable_times, model_error, verbose=args.verbose)

    chi2_dict_raw, _ = compute_chisquare_dict(transient, best_mags, 
                                              observable_times, {filt: 0. for filt in model_error.keys()}, 
                                              verbose=args.verbose)

    if getattr(args, "bestfit", False):
        bestfit_to_write = bestfit_params.copy()
        if result is not None:
            bestfit_to_write["log_bayes_factor"] = result.log_bayes_factor
            bestfit_to_write["log_bayes_factor_err"] = result.log_evidence_err
        bestfit_to_write["Magnitudes"] = {filt: best_mags[filt].tolist() 
                                          for filt in transient.observed_filters}
        bestfit_to_write["model_error"] = {filt: model_error[filt].tolist() for filt in transient.observed_filters}
        bestfit_to_write["obs_times"] = observable_times.tolist()
        bestfit_to_write["chi2_per_dof"] = chi2_dict["total"]
        bestfit_to_write["chi2_dict"] = chi2_dict
        bestfit_to_write["chi2_per_dof_raw"] = chi2_dict_raw["total"]
        bestfit_to_write["chi2_dict_raw"] = chi2_dict_raw

        bestfit_file = os.path.join(args.outdir, f"{args.label}_bestfit_params.json")
        with open(bestfit_file, "w") as file:
            json.dump(bestfit_to_write, file, indent=4)

        print(f"Saved bestfit parameters and magnitudes to {bestfit_file}")

    if args.plot:
        filters_to_plot = [
            filt for filt in transient.observed_filters
            if not np.isnan(transient.light_curves[filt]).all()
        ]
        plot_error = {filt: model_error[filt] for filt in filters_to_plot}
        mags_to_plot = {filt: best_mags[filt] for filt in filters_to_plot}
        mags_to_plot["time"] = observable_times

        if isinstance(lc_model, model.CombinedLightCurveModelContainer):
            sub_models = lc_model.lc_models
            model_colors = plt.cm.Spectral(np.linspace(0, 1, len(sub_models)))[::-1]
            obs_times , mag_all = lc_model.gen_detector_lc(
                bestfit_params, return_all=True
            )
            sub_model_plot_props = {}
            for i, sub_model in enumerate(sub_models):
                sub_model_plot_props[sub_model.model] ={
                    'color': model_colors[i], 
                    'plot_times': obs_times[i]
                }
                plot_errors = []
                plot_mags = []
                for filt in filters_to_plot:
                    try:
                        plot_mags.append(utils.get_filtered_mag(mag_all[i], filt))
                    except KeyError:
                        plot_mags.append(np.full_like(obs_times[i], np.nan))
                    plot_errors.append(utils.autocomplete_data(obs_times[i],
                        transient.light_curve_times[filt], model_error_data[filt]))

                sub_model_plot_props[sub_model.model]['plot_mags'] = plot_mags
                sub_model_plot_props[sub_model.model]['plot_errors'] = plot_errors
        else: sub_model_plot_props = None

        
        basic_em_analysis_plot(
            transient, mags_to_plot, plot_error, chi2_dict, mismatches,
            sub_model_plot_props, xlim = args.xlim, ylim = args.ylim, 
            save_path = os.path.join(args.outdir, f"{args.label}_lightcurves.png")
        )

       
def compute_chisquare_dict(transient, model_data, model_time, model_error, verbose=False):
    chi2 = 0.0
    dof = 0.0
    chi2_dict = {}
    mismatches = {}
    for filt  in model_data.keys():
        mag = transient.light_curves[filt]
        t = transient.light_curve_times[filt]
        sigma_y = transient.light_curve_uncertainties[filt]
        # only the detection data are needed
        finite_idx = np.isfinite(sigma_y)
        n_finite = finite_idx.sum()
        if n_finite > 0:
            t_det, y_det, sigma_y_det = (
                t[finite_idx],
                mag[finite_idx],
                sigma_y[finite_idx],
            )
            
            offset = (y_det - np.interp(t_det,model_time, model_data[filt])) ** 2
            try:
                errors = np.interp(t_det,model_time, model_error[filt])
            except ValueError:
                errors = model_error[filt] 
            total_unc = sigma_y_det**2 + errors**2
            chi2_per_filt = np.sum(offset / total_unc)
            # store the data
            chi2 += chi2_per_filt
            dof += n_finite
            mismatches[filt] = (offset, total_unc)
            chi2_dict[filt] = float(chi2_per_filt / n_finite)

            if verbose:
                print(f"the {filt} data being analyzed is: ", t, mag, sigma_y)
                print(f"for {filt} the length of the detections array is: ", n_finite, "increasing the dof to", dof)

    chi2_dict["total"] = chi2 / dof if dof > 0 else np.inf
    chi2_dict["dof"] = dof

    return chi2_dict, mismatches

def lcs_from_injection_parameters(args=None):
    args = emp.parsing_and_logging(emp.lightcurve_parser, args)
    
    # initialize light curve model
    light_curve_model = model.create_light_curve_model_from_args(args.em_model, args)
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
                                     f"_{int(injection_parameters['simulation_id'])}_lc")

    if os.path.isfile(injection_outfile):
        try:
            return io.load_em_observations(injection_outfile, format=format)

        except ValueError:
            raise ValueError(f"The previous run generated light curves with unreadable content for {injection_outfile}. Consider removing all output files in this format, format then retry.")

    data, injection_parameters = make_injection(injection_parameters, args, injection_model = light_curve_model, rng=rng, keep_infinite_data=True)

    #store and retrieve to double check
    io.write_em_observations(injection_outfile, data, format=format)
    return io.load_em_observations(injection_outfile, format=format)


def make_injection(injection_params, args, injection_model, rng=None, keep_infinite_data=False):
    
    injection_params = adjust_injection_parameters(injection_params, args, injection_model)
    data = create_light_curve_data(
        injection_params,
        args,
        light_curve_model=injection_model,
        keep_infinite_data=keep_infinite_data,
        rng = rng
    )
    print("Injection generated")

    return data, injection_params 

def adjust_injection_parameters(injection_parameters, args, injection_model):

    injection_parameters["trigger_time"] = read_trigger_time(injection_parameters, args)
    if args.ignore_timeshift:
        injection_parameters.pop('timeshift', None)
    injection_parameters["trigger_time"] += injection_parameters.get("timeshift",0)

    # sanity check for eject masses
    if args.prompt_collapse:
        injection_parameters["log10_mej_wind"] = -3.0
    for key in ["log10_mej_dyn", "log10_mej_wind"]:
        if key in injection_parameters and not np.isfinite(injection_parameters[key]):
            injection_parameters[key] = -3.0
    
    return injection_model.parameter_conversion(injection_parameters)

def make_lcs(args = None):
    args = emp.parsing_and_logging(emp.multi_lc_parser, args)
    if args.file_type is None:
        lc_handler = LightCurveHandler(args)
    elif "lanl" in args.file_type.lower():
        lc_handler = LANLLightCurveHandler(args)
    elif "kasen" in args.file_type.lower():
        lc_handler = KasenLightCurveHandler(args)
    elif ('h5' in args.file_type.lower()) or ('hdf5' in args.file_type.lower()):
        lc_handler = H5LightCurveHandler(args)
    else:
        raise ValueError('Unknown file type for lc creation')
    
    lc_handler.generate_nmma_lcs_from_files()

class LightCurveHandler:
    def __init__(self, args):

        self.filters = utils.set_filters(args)
        cosmology = get_cosmology()
        # Use redshift or dMpc if z is not provided
        if args.redshift is None:
            self.dMpc = args.dMpc
            self.redshift = conv.luminosity_distance_to_redshift(self.dMpc, cosmology)
            dist_filler = f"dMpc{int(self.dMpc)}"
        else:
            self.redshift = args.redshift
            self.dMpc = cosmology.luminosity_distance(self.redshift).to("Mpc").value
            dist_filler = f"z{self.redshift}"
        
        if args.doAB:
            self.format = "model"
            self.compose_data = self.compose_filter_data
            self.dist_filler = dist_filler
        elif args.doLbol:
            self.format = "lbol"
            self.compose_data = self.compose_lbol_data
            self.dist_filler = dist_filler + '_Lbol'
        else:
            raise SystemExit( "ERROR: Neither --doAB nor --doLbol are enabled. " \
        "Please enable at least one.")

        self.modeldir = args.modeldir
        lcdir = args.lcdir
        os.makedirs(lcdir, exist_ok=True)
        self.extensions = [".dat", ".csv", ".txt"] # to be overwritten in subclass


    def generate_nmma_lcs_from_files(self, directory=None, target_extensions=None):
        if directory is None:
            directory = self.modeldir
        if target_extensions is None:
            target_extensions = self.extensions

        for f in tqdm(os.listdir(directory)):
            base, ext = os.path.splitext(f)
            if ext not in target_extensions:
                continue

            in_file = os.path.join(directory, f)

            iterator, data = self.open_source(in_file)

            for ind in iterator:
                out_file = self.set_filename(base, ind)
                if os.path.isfile(out_file):
                    continue
                
                processed_data = self.process_source(ind, data)
                lc_data = self.compose_data(processed_data)
                io.write_lc_to_csv(out_file, lc_data, format=self.format)

    def open_source(self, in_file):
        # Read header values from the first three lines
        with open(in_file) as f:
            Nobs = int(f.readline().strip())
            self.Nwave = int(f.readline().strip())
            Ntime, ti, tf = map(float, f.readline().strip().split())

        time, dt = np.linspace(ti, tf, int(Ntime), retstep=True)
        self.time = time + 0.5*dt

        cos = np.linspace(0, 1, Nobs)
        self.thetas = np.arccos(cos) * 180 / np.pi
        # Load the rest of the data
        data = np.loadtxt(in_file, skiprows=3)
        return range(Nobs), data
 
    def set_filename(self, base, index):
        return os.path.join(self.lcdir, 
            f"{base}_theta{self.thetas[index]:.2f}_{self.dist_filler}.dat")

    def process_source(self, i, data):
        wave = data[self.Nwave * i : self.Nwave * (i + 1), 0] * (1 + self.redshift)
        Istokes = data[self.Nwave * i : self.Nwave * (i + 1), 1 :len(self.time)+1]
        fl = Istokes.T * (1e-5 / self.dMpc) ** 2 / (1 + self.redshift)

        return (wave, fl)

    def compose_data(self, processed_data):
        return processed_data # Dummy, to be overwritten
    
    def compose_filter_data(self, processed_data):
        wave, fl = processed_data
        source = sncosmo.TimeSeriesSource(self.time, wave, fl)
        data = {}
        for filt in self.filters:
            bandpass = sncosmo.get_bandpass(filt, 5.0) if filt == "ultrasat" else sncosmo.get_bandpass(filt)
            m = source.bandmag(bandpass, "ab", self.time)
            data[filt] = {"time": self.time, "mag": m, "mag_error": np.full_like(m, np.nan)}
        return data

    def compose_lbol_data(self, processed_data):
        wave, fl = processed_data
        Lbol = np.trapezoid(fl * (4 * np.pi * D**2), x=wave)
        return { "time": self.time, "lbol": Lbol }

class LANLLightCurveHandler(LightCurveHandler):
    def __init__(self, args):
        # Initiate a LANL filereader object
        from cocteau import filereaders
        self.filereader = filereaders.LANLFileReader()
        super().__init__(args)

    def open_source(self, in_file):
        spectra = self.filereader.read_spectra(in_file, angles=np.arange(54), remove_zero=False)
        Nfiles = len(spectra)
        cos = np.linspace(-1, 1, Nfiles)
        self.thetas = np.arccos(cos) * 180 / np.pi
        return (range(Nfiles), spectra)
    
    def process_source(self, i, data):
        spectrum = data[i]
        wave = spectrum.spectra[0].wavelength_arr.to(u.angstrom).value
        self.time = spectrum.timesteps.value

        fl = [len(data) * spec.flux_density_arr.to(u.erg / u.s / u.cm**2 / u.angstrom).value
               for spec in spectrum.spectra] 
        return (wave, fl)

class H5LightCurveHandler(LightCurveHandler):
    def __init__(self, args):
        super().__init__(args)
        self.extensions = [".h5", ".hdf5"]

        if args.doLbol:
            self.process_source = self.process_lbol_source
        else:
            self.process_source = self.process_filter_source

    def open_source(self, in_file):
        with h5py.File(in_file) as f:
            data = f["observables"]
            stokes = np.array(data["stokes"])
            self.time = np.array(data["time"]) / (60 * 60 * 24)  # get time in days
            self.wave = np.array(data["wave"]) * (1 + self.redshift)  # wavelength spec w/ redshift
            self.Lbol = np.array(data["lbol"])

        Istokes = stokes[:, :, :, 0]  # get I stokes parameter
        Nobs = stokes.shape[0]  # num observing angles

        cos = np.linspace(0, 1, stokes.shape[0])  # num observing angles
        self.thetas = np.arccos(cos) * 180 / np.pi
        return range(Nobs), Istokes

    def process_filter_source(self, i, data):
        fl = data[i] * (1.0 / self.dMpc) ** 2 / (1 + self.redshift)
        return (self.wave, fl)
    
    def process_lbol_source(self, i, data):
        return self.Lbol[i]
    
    def compose_lbol_data(self, processed_data):
        return {'time': self.time, 'lbol': processed_data }

class KasenLightCurveHandler(LightCurveHandler):
    def __init__(self, args):
        super().__init__(args)
        self.extensions = [".h5", ".hdf5"]
        self.smoothing = args.doSmoothing

    def open_source(self, in_file):
        with h5py.File(in_file, "r") as f:
            nu = np.array(f["nu"], dtype="d")
            time = np.array(f["time"])
            Lnu = np.array(f["Lnu"], dtype="d")

        # smooth over missing data
        Lnu[Lnu == 0.0] = 1e20
        self.Lnu = 10 ** gaussian_filter(np.log10(Lnu), 3.0)

        nuS = np.tile(nu, (len(time), 1))

        Llam = self.Lnu * nuS**2.0 / c_cgs / 1e8  # ergs/s/Angstrom
        Llam = Llam / (4 * np.pi * D**2)  # ergs / s / cm^2 / Angstrom

        self.time = np.array(time) / (60 * 60 * 24)  # get time in days
        wave = c_cgs / nu * 1e8  # AA

        # flip axes to make wavelength increasing
        wave = np.flip(wave)
        Llam = np.flip(Llam, axis=1)
        self.nu = np.flipud(nu)
        iterator = range(1) # dummy, only one angle in Kasen data
        return iterator, (wave, Llam)

    def set_filename(self, base, index):
        return os.path.join(self.lcdir, f"{base}_{self.dist_filler}.dat")
    
    def process_source(self, i, data): # dummy for conformity
        return data

    def compose_filter_data(self, processed_data):
        data = super().compose_filter_data(processed_data)
        if self.smoothing:
            for filt in self.filters:
                prel_mag = utils.autocomplete_data(self.time, self.time, data[filt]['mag'])
                data[filt]['mag'] = savgol_filter(prel_mag, window_length=17, polyorder=3)
        return data

    def compose_lbol_data(self, processed_data):
        wave, _ = processed_data
        lbol = np.trapezoid( self.Lnu * self.nu ** 2.0 / c_cgs / 1e8 * (4 * np.pi * D**2),
            x=wave )
        if self.smoothing:
            lbol = 10**utils.autocomplete_data(self.time, self.time, np.log10(lbol))
            lbol = savgol_filter(lbol, window_length=17, polyorder=3)
        return {'time': self.time, 'lbol': lbol}

def resample_lightcurve_grid(args=None):
    args = emp.parsing_and_logging(emp.lc_grid_parser, args)
    
    if not args.gridpath.endswith(".h5"):
        raise ValueError("Resampling currently only supports grid files with a .h5 extension.")

    grid = Grid(args.gridpath, args.base_dirname, 
                args.base_filename, args.random_seed)

    if args.downsample:
        grid.downsample(factor=args.factor, shuffle=args.shuffle)

    if args.fragment:
        grid.fragment(factor=args.factor, shuffle=args.shuffle)
    
    if args.remove:
        grid.remove()

class Grid:
    def __init__(self, gridpath, base_dirname="lcs_grid", base_filename="lcs", random_seed=21):
        if not gridpath.startswith("/"):
            gridpath = os.path.join(os.getcwd(), gridpath)
        self.file = h5py.File(gridpath)
        self.keys = list(self.file.keys())
        self.base_dirname = base_dirname
        self.base_filename = base_filename
        self.rng = np.random.default_rng(random_seed)

    def downsample(self, factor=10, shuffle=False):
        save_dir, keys, tag = self._setup(f"downsampled_{factor}x", shuffle)
        keys = keys[::factor]
        save_file = os.path.join(save_dir, f"{self.base_filename}_{tag}.h5")
        self._save(keys, save_file)
        print("Downsampling done.")

    def fragment(self, factor=10, shuffle=False):
        save_dir, keys, tag = self._setup(f"fragmented", shuffle)
        chunks = np.array_split(keys, factor)
        for i, chunk in enumerate(chunks):
            save_file = os.path.join(save_dir, f"{self.base_filename}_{tag}_{i+1}_of_{factor}.h5")
            self._save(chunk, save_file)
            print(f"Fragment {i+1}/{factor} done.")
        print("Fragmenting done.")

    def _setup(self, tag, shuffle):
        keys = self.keys.copy()
        if shuffle:
            self.rng.shuffle(keys)
            tag = f"shuffled_{tag}"
        
        dirname = os.path.join(self.base_dirname, tag)
        os.makedirs(dirname, exist_ok=True)
        return dirname, keys, tag

    def _save(self, keys, filename):
        with h5py.File(filename, "w") as new_file:
            for key in keys:
                new_file.copy(self.file[key], key)

    def remove(self):
        # Clean up directories starting with base_dirname
        for item in os.listdir():
            if item.startswith(self.base_dirname) and os.path.isdir(item):
                shutil.rmtree(item)
                # shutil.rmtree(item_path)


def call_lc_validation (args=None):
    args = emp.parsing_and_logging(emp.lc_validation_parser, args)
    filters = utils.set_filters(args)
    return validate_lightcurve(data_file=args.light_curve_data,filters=filters,min_obs=args.min_obs,cutoff_time=args.cutoff_time,verbose=args.verbose)

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

    # Determine filters to consider
    filters = filters or list(data.keys())

    # Time window
    min_time = np.min([np.min(filt_dict['time']) for filt_dict in data.values()])
    max_time = min_time + cutoff_time if cutoff_time > 0 else np.max([np.max(filt_dict['time']) for filt_dict in data.values()])

    # Validate each filter
    for filt in filters:
        #FIXME this seems an unpractical restriction
        if filt not in utils.DEFAULT_FILTERS:
            raise ValueError(f"Unsupported filter: {filt}")
        if filt not in data:
            if verbose:
                print(f"{filt} not in data file")
            return False
        mask = data[filt]['time'] <= max_time
        detections = np.sum(np.isfinite(data[filt]['mag_error'][mask]))
        if detections < min_obs:
            if verbose:
                print(f"{filt}: only {detections} detections, required: {min_obs}")
            return False
    if verbose:
        print(f"Lightcurve meets minimum {min_obs} detections in all filters within {max_time-min_time:.2f} days")

    return True
