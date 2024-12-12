import os
import numpy as np
import json

from astropy import time

import bilby
import bilby.core

from .em_parsing import lightcurve_parser, parsing_and_logging
from .model import create_light_curve_model_from_args
from .injection import create_light_curve_data
from .io import read_lightcurve_file
from .utils import NumpyEncoder, setup_sample_times

def main(args=None):
    args = parsing_and_logging(lightcurve_parser, args)
    
    seed = args.generation_seed
    np.random.seed(seed)

    # initialize light curve model
    sample_times = setup_sample_times(args)

    if args.filters:
        filters = args.filters.split(",")
    else:
        filters = None

    _, _, light_curve_model = create_light_curve_model_from_args(
        args.model,
        args,
        sample_times,
        filters=filters,
    )

    # read injection file
    with open(args.injection, "r") as f:
        injection_dict = json.load(f, object_hook=bilby.core.utils.decode_bilby_json)

    args.kilonova_tmin = args.tmin
    args.kilonova_tmax = args.tmax
    args.kilonova_tstep = args.dt

    args.kilonova_injection_model = args.model
    args.kilonova_injection_svd = args.svd_path
    args.injection_svd_mag_ncoeff = args.svd_mag_ncoeff
    args.injection_svd_lbol_ncoeff = args.svd_lbol_ncoeff

    # args.injection_detection_limit = np.inf
    args.kilonova_error = args.photometric_error_budget

    injection_df = injection_dict["injections"]

    # save simulation_id from observing scenarios data
    # we save lighcurve for each event with its initial simulation ID
    # from observing scenarios
    simulation_id = injection_df["simulation_id"]

    mag_ds = {}
    for index, row in injection_df.iterrows():

        if args.outfile_type == "json":
            ext = "json"
        elif args.outfile_type == "csv":
            ext = "dat"

        injection_outfile = getattr(args, 'injection_outfile', args.label)
        if len(injection_df) == 1:
            injection_outfile = os.path.join(args.outdir, f"{injection_outfile}.{ext}")
        else:
            injection_outfile = os.path.join(
                args.outdir, f"{injection_outfile}_{simulation_id[index]}.{ext}"
            )
        if os.path.isfile(injection_outfile):
            try:
                if args.outfile_type == "json":
                    with open(injection_outfile) as f:
                        mag_ds[index] = json.load(f)
                elif args.outfile_type == "csv":
                    mag_ds[index] = read_lightcurve_file(injection_outfile)
                continue

            except ValueError:
                print(
                    "\n==================================================================="
                )
                print(
                    "The previous run generated light curves with unreadable content.\n"
                )
                print(f"Please remove all output files in .{ext} format then retry.")
                print(
                    "===================================================================\n"
                )
                exit()

        injection_parameters = row.to_dict()

        try:
            tc_gps = time.Time(injection_parameters["geocent_time_x"], format="gps")
        except KeyError:
            tc_gps = time.Time(injection_parameters["geocent_time"], format="gps")
        trigger_time = tc_gps.mjd

        injection_parameters["kilonova_trigger_time"] = trigger_time

        if args.increment_seeds:
            args.generation_seed = args.generation_seed + 1

        data = create_light_curve_data(
            injection_parameters,
            args,
            doAbsolute=args.absolute,
            light_curve_model=light_curve_model,
            keep_infinite_data=True,
        )
        print("Injection generated")

        if args.outfile_type == "json":
            with open(injection_outfile, "w") as f:
                json.dump(data, f, cls=NumpyEncoder, indent=4)
            with open(injection_outfile) as f:
                mag_ds[index] = json.load(f)
        elif args.outfile_type == "csv":
            try:
                fid = open(injection_outfile, "w")
                # fid.write('# t[days] u g r i z y J H K\n')
                # fid.write(str(" ".join(('# t[days]'," ".join(args.filters.split(',')),"\n"))))
                fid.write("# t[days] ")
                fid.write(str(" ".join(args.filters.split(","))))
                fid.write("\n")

                for ii, tt in enumerate(sample_times):
                    fid.write("%.5f " % sample_times[ii])
                    for filt in data.keys():
                        if args.filters:
                            if filt not in args.filters.split(","):
                                continue
                        fid.write("%.3f " % data[filt][ii, 1])
                    fid.write("\n")
                fid.close()

            except IndexError:
                print(
                    "\n==================================================================="
                )
                print(
                    "Sorry we could not use ZTF or Rubin flags to generate those statistics\n"
                )
                print(
                    "Please remove all flags rely on with ZTF or Rubin then retry again"
                )
                print(
                    "===================================================================\n"
                )
                exit()

            mag_ds[index] = read_lightcurve_file(injection_outfile)

    if args.plot:
        from .plotting_utils import lc_plot
        plotpath= os.path.join(args.outdir, f"injection_{args.model}_{args.label}_lc.pdf")
        plot_data_dict = {}
        for filt in filters:
            plot_data_filt = []
            for lc_data in mag_ds.values():
                data_vec = np.array(lc_data[filt])
                if data_vec.ndim == 2:
                    plot_data_filt.append(data_vec[:, 1].tolist())
                else:
                    plot_data_filt.append(data_vec.tolist())
            plot_data_dict[filt] = np.vstack(plot_data_filt)

        lc_plot(filters, plot_data_dict, sample_times=sample_times, plotpath=plotpath, xlim=args.xlim, ylim=args.ylim,
                colorbar= True, ylabel_kwargs = dict(fontsize=30, rotation=90, labelpad=8))
