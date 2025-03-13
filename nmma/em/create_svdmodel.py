import copy
import glob
import inspect
import os

import numpy as np

from . import model_parameters
from .model import SVDLightCurveModel
from .training import SVDTrainingModel, KerasTrainingModel
from .utils import interpolate_nans, setup_sample_times
from .io import read_photometry_files, read_spectroscopy_files
from .em_parsing import parsing_and_logging, svd_training_parser
from .plotting_utils import lc_comparison_plot, basic_spec_plot, spec_subplot


def axial_symmetry(training_data):

    modelkeys = list(training_data.keys())
    if any(["KNtheta" not in training_data[key] for key in modelkeys]):
        raise ValueError("unknown symmetry parameter")

    for key in modelkeys:
        training = training_data[key]
        key_new = key + "_flipped"
        training_data[key_new] = copy.deepcopy(training)
        training_data[key_new]["KNtheta"] = -training_data[key_new]["KNtheta"]
        key_new = key + "_flipped_180"
        training_data[key_new] = copy.deepcopy(training)
        training_data[key_new]["KNtheta"] = 180 - training_data[key_new]["KNtheta"]

    return training_data


def main():

    args = parsing_and_logging(svd_training_parser)

    # initialize light curve model
    sample_times = setup_sample_times(args)

    MODEL_FUNCTIONS = {
        k: v for k, v in model_parameters.__dict__.items() if inspect.isfunction(v)
    }
    if args.model not in list(MODEL_FUNCTIONS.keys()):
        raise ValueError(
            f"{args.model} unknown. Please add to nmma.em.model_parameters"
        )
    model_function = MODEL_FUNCTIONS[args.model]

    file_extensions = ["dat", "csv", "dat.gz", "h5"]
    filenames = []
    for file_extension in file_extensions:
        if not args.ignore_bolometric:
            filenames = filenames + glob.glob(f"{args.data_path}/*.{file_extension}")
        else:
            filenames = filenames + glob.glob(
                f"{args.data_path}/*[!_Lbol].{file_extension}"
            )
    if len(filenames) == 0:
        raise ValueError("Need at least one file to interpolate.")

    if args.data_type == "photometry":
        try:
            data = read_photometry_files(filenames, datatype=args.data_file_type)
        except IndexError:
            raise IndexError(
                "If there are bolometric light curves in your --data-path, try setting --ignore-bolometric"
            )
    elif args.data_type == "spectroscopy":
        data = read_spectroscopy_files(
            filenames, wavelength_min=args.lmin, wavelength_max=args.lmax, smooth=True
        )
        keys = list(data.keys())
        filts = data[keys[0]]["lambda"]
    else:
        raise ValueError("data-type should be photometry or spectroscopy")

    data = interpolate_nans(data)

    training_data, parameters = model_function(data)
    if args.axial_symmetry:
        training_data = axial_symmetry(training_data)

    if args.filters is not None:
        filts = args.filters.split(",")
    else:
        keys = list(data.keys())
        filts = sorted(list(set(data[keys[0]].keys()) - {"t"}))

    training_args = [
        args.model,
        training_data,
        parameters,
        sample_times,
        filts,
        ]
    training_kwargs = dict(
            n_coeff=args.svd_mag_ncoeff,
            n_epochs=args.nepochs,
            svd_path=args.svd_path,
            data_type=args.data_type,
            data_time_unit=args.data_time_unit,
            plot=args.plot,
            plotdir=args.outdir,
            ncpus=args.ncpus,
            univariate_spline=args.use_UnivariateSpline,
            univariate_spline_s=args.UnivariateSpline_s,
            random_seed=args.random_seed,
            continue_training=args.continue_training,
    )
    try:
        training_model = KerasTrainingModel(*training_args, **training_kwargs)
    except:
        print("Your settings are not compatible with a keras training model.\n \
              Please consider adjusting your setup.\n \
            We will now try to train a legacy SVD model.")
        training_kwargs['interpolation_type'] = args.interpolation_type
        training_model = SVDTrainingModel(*training_args, **training_kwargs)

    light_curve_model = SVDLightCurveModel(
        args.model,
        svd_path=args.svd_path,
        svd_mag_ncoeff=args.svd_mag_ncoeff,
        interpolation_type=args.interpolation_type,
        model_parameters=training_model.model_parameters,
        local_only=True,
    )
    if args.plot:
        # we can plot an example where we compare the model performance
        # to the grid points

        modelkeys = list(training_data.keys())
        training = training_data[modelkeys[0]]
        parameters = training_model.model_parameters
        data = {param: training[param] for param in parameters}
        data["redshift"] = 0
        plotName = os.path.join(
            args.outdir, "injection_" + args.model + "_lightcurves.png"
        )
        if args.data_type == "photometry":
            lbol, mag = light_curve_model.generate_lightcurve(sample_times, data)
            lc_comparison_plot(mag_dict=mag, training_data=training["data"], filters= filts, sample_times=sample_times, save_path=plotName, ylabel_kwargs=dict( fontsize=30, rotation=0, labelpad=14))

        elif args.data_type == "spectroscopy":
            spec = light_curve_model.generate_spectra(
                sample_times, training_model.filters, data)
            
            training_data =np.log10(training["data"])
            interpolated_data = np.log10(np.array([spec[key] for key in filts]))
            residual = interpolated_data - training_data
            norm_residual = np.log10(np.abs(residual/training_data))
            plot_entries = {"Original": training_data.T, 
                             "Interpolated": interpolated_data,
                            "(Original - Interpolated) / Interpolated": norm_residual.T}
            def spec_plot_func(fig, ax, XX, YY, plot_data, label):
                if label == "(Original - Interpolated) / Interpolated":
                    vmin = -3
                    vmax = 0
                    cbar_label = "Relative Difference"
                else:
                    vmin = -10
                    vmax = -2
                    cbar_label = "log10(Flux)"
                fig, ax = spec_subplot(fig, ax, XX, YY, plot_data, label, vmin=vmin, vmax=vmax, cbar_label=cbar_label)
                return fig, ax
            basic_spec_plot(
                mesh_X=sample_times,
                mesh_Y=filts,
                spec_func = spec_plot_func,
                plot_entries=plot_entries,
                save_path=plotName,
                figsize=(32, 14))

