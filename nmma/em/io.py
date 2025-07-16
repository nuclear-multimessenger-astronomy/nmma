import json
from argparse import Namespace
import astropy
from astropy.time import Time
import h5py
import numpy as np
import pandas as pd
from bilby.core.utils import decode_bilby_json
import scipy.signal
# from sncosmo.bandpasses import _BANDPASSES


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_em_observations(filename, args=None, format='observations'):
    """
    Reads in lightcurve data from a file and returns data in nmma standard format.

    Available formats are
    
    Args:
    - filename (str): 
        Path to lightcurve file
    - args (Namespace, optional):
        Namespace containing additional arguments, such as time format.
        If not provided, the function will use the default time format 'mjd'.
    - format (str): 
        The format of the input file. This can be 'standard', 'observations' or 'model'.
        'standard' uses the internal format of nmma lightcurve data as a dict in the form {filter_name: {'time': observation_times (list), 'mag': observed_magnitudes (list), 'mag_error': observed_errors (list)}}.
        'observations' uses the format of observations as a text file with columns: time, filter, mag, mag_error and transforms them to a dict in the same format as 'standard'.
        'model' uses the format of model lightcurves as a text file with columns: time, filter1, filter2, ..., filterN, filter1_error, ..., filterN_error and transforms them to a dict in the same format as 'standard'.

    Returns:
    - data (dict): Dictionary containing the lightcurve data from the file. The keys are generally 't' and each of the filters in the file as well as their accompanying error values.
    """
    if isinstance(filename, Namespace):
        args = filename
        filename = args.light_curve_data
    
    if filename.endswith(".json"):
        data =  read_lc_from_json(filename)
    
    else:
        data =  read_lc_from_csv(filename, args, format=format)
    return {filt: 
            {k: np.array(vals) for k, vals in filt_dict.items()} 
            for filt, filt_dict in data.items()}


def read_lc_from_json(filename):
    # we assume the json file is in the standard format
    with open(filename, "r") as f:
        data = json.load(f, object_hook=decode_bilby_json)

    # but we check - if given in 'model' format, we are ready to convert it
    if "time" in data: # indicates model_format
        new_data = {}
        for key, value in data.items():
            if key != "time" and not key.endswith("_error"):
                new_data[key] = {
                    "time": data["time"],
                    "mag": value,
                    "mag_error": data.get(f"{key}_error", np.zeros_like(data["time"]))
                }
        data = new_data

    return data

def read_lc_from_csv(filename, args, format):
    if "obs" in format:
        with open(filename, "r") as f:
            lines = [line.rstrip("\n") for line in f]
            lines = filter(None, lines) #get non-empty lines

            data = {}
            for line in lines:
                lineSplit = line.split(" ")
                lineSplit = list(filter(None, lineSplit))
                mjd = Time(lineSplit[0], format=getattr(args, "time_format", "mjd")).mjd
                filt = lineSplit[1]
                mag = float(lineSplit[2])
                dmag = float(lineSplit[3])

                try:
                    data[filt]['time'].append(mjd)
                    data[filt]['mag'].append(mag)
                    data[filt]['mag_error'].append(dmag)
                except KeyError:
                    data[filt] = {'time': [mjd], 'mag': [mag], 'mag_error': [dmag]}
        return data


    elif 'model' in format:
        #FIXME 
        # For model lightcurves, the format is a simple text file with columns:
        # time, filter1, filter2, ..., filterN. filter1_error, ..., filterN_error are optional
        try:
            data = pd.read_csv(filename, delim_whitespace=True)
        except:
            data = pd.read_json(filename, orient = 'columns')

        data = data.to_dict(orient="list")
        time = data.pop("time")
        data = {filt: {'time':time, 'mag': mag, 'mag_error': data.get(filt + "_error", np.zeros_like(time))} for filt, mag in data.items() if not filt.endswith("_error")}

        return data
    elif format == "standard":
        raise ValueError("Standard format is not supported for reading from csv files. Please use json files instead.")

def write_em_observations(filename, data, format='observations'):
    # write json file in standard format or csv file, either in observations or model format
    if filename.endswith(".json"):
        write_lc_to_json(filename, data)
    elif filename.endswith(".txt") or filename.endswith(".dat"):
        write_lc_to_csv(filename, data, format=format)

def write_lc_to_json(injection_outfile, data):
    with open(injection_outfile, "w") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)

def write_lc_to_csv(outfile, data, format= "observations"):
    if format == "observations":
        all_times, all_filters, all_mags, all_errs = [], [], [], []
        for filt, sub_dict in data.items():
            all_times.extend(sub_dict['time'])
            all_mags.extend(sub_dict['mag'])
            all_errs.extend(sub_dict['mag_error'])
            all_filters.extend([filt] * len(sub_dict['time']))
        sort_indices = np.argsort(all_times)
        out_data = [
            [Time(all_times[i], format="mjd").mjd, all_filters[i], all_mags[i], all_errs[i]] 
            for i in sort_indices]
        np.savetxt(outfile, out_data, fmt="%s %s %.3f %.3f", delimiter=" ", header="time filter mag mag_error", comments="#")
        
    elif format == "model":
        # Lightcurve as issued by model
        mags, errs = [], []
        for filt, sub_dict in data.items():
            mags.append(sub_dict['mag'])
            errs.append(sub_dict['mag_error'])
        time = sub_dict['time']
        out_data = np.column_stack((time, *mags, *errs))
        header = "time " + " ".join([filt for filt in data.keys()] + " ".join([filt + "_error" for filt in data.keys()]) )
        np.savetxt(outfile, out_data, fmt="%.5f " + " ".join(["%.3f"] * len(mags) * 2), delimiter=" ", header=header, comments="#")




def read_training_data(filenames, format, data_type = "photometry", args=None):

    # read the grid data
    if data_type == "photometry":
        try:
            return read_photometry_files(filenames, format=format)
        except IndexError:
            raise IndexError(
                "If there are bolometric light curves in your --data-path, try setting --ignore-bolometric"
            )

    elif data_type == "spectroscopy":
        return read_spectroscopy_files(
            filenames, wavelength_min=args.lmin, wavelength_max=args.lmax, smooth=True
        )
    else:
        raise ValueError("data-type should be photometry or spectroscopy")

def read_spectroscopy_files(
    files, wavelength_min=3000.0, wavelength_max=10000.0, smooth=False
):

    data = {}
    for filename in files:
        name = (
            filename.replace("_spec", "")
            .replace(".spec", "")
            .replace(".txt", "")
            .replace(".dat", "")
            .split("/")[-1]
        )
        df = pd.read_csv(filename, names=["wavelength", "time", "fnu"])
        df_group = df.groupby("time")

        t_d = []
        lambda_d = []
        spec_d = []
        for ii, (tt, group) in enumerate(df_group):
            t_d.append(tt)
            if ii == 0:
                lambda_d = group["wavelength"].to_numpy()
                jj = np.where(
                    (lambda_d >= wavelength_min) & ((lambda_d <= wavelength_max))
                )[0]
                lambda_d = lambda_d[jj]
            spec = group["fnu"].to_numpy()[jj]
            if smooth:
                spec = scipy.signal.medfilt(spec, kernel_size=9)
            spec_d.append(spec)

        data[name] = {}
        data[name]["t"] = np.array(t_d)
        data[name]["lambda"] = np.array(lambda_d)
        data[name]["fnu"] = np.array(spec_d)

    return data


def read_photometry_files(files: list, filters: list = None, tt: np.array = np.linspace(0, 14, 100), format:str ="bulla") -> dict:
    """
    Read in a list of photometry files with given filenames and process them in a dictionary
    
    Args:
        files (list): List of filenames with the photometry files.
        filters (list): List of photometry filters to be extracted.
        tt (np.array): Array containing the time grid at which photometry values are given.
        format (str): Which model we are considering. Currently supports
        
    Returns:
        data: Dictionary with keys being the given filenames and values being dictionaries themselves, with keys 
        being t (time) and specified filters and values being the time grid, and values the time grid and lightcurves.
    """
    
    # First, check whether given format is supported in this function
    supported_formats = ["ztf", "bulla", "standard", "hdf5"]
    if format not in supported_formats:
        space = " "
        raise ValueError(f"format {format} unknown. Currently supported formats are: {space.join(supported_formats)}")

    # Return value 
    data = {}
    
    # Iterate over all the given files and extract the lightcurve data from it
    for filename in files:
        name = (
            filename.replace(".csv", "")
            .replace(".txt", "")
            .replace(".dat", "")
            .replace(".hdf5", "")
            .replace(".h5", "")
            .split("/")[-1]
        )

        # ZTF rest style file
        if format == "ztf":
            df = pd.read_csv(filename)

            if "mag" in df:
                mag_key = "mag"
            elif "magpsf" in df:
                mag_key = "magpsf"
            else:
                raise ValueError("Unknown magnitude key")

            if "mag_unc" in df:
                magerror_key = "mag_unc"
            elif "sigmapsf" in df:
                magerror_key = "sigmapsf"
            else:
                raise ValueError("Unknown uncertainty key")

            idx = np.where(df[magerror_key] != 99.0)[0]
            if len(idx) < 2:
                print(f"{name} does not have enough detections to interpolate.")
                continue

            jd_min = df["jd"].iloc[idx[0]]

            data[name] = {}
            data[name]["t"] = tt
            for filt in ["u", "g", "r", "i", "z", "y", "J", "H", "K"]:
                data[name][filt] = np.nan * tt

            for filt, group in df.groupby("filter"):
                idx = np.where(group[magerror_key] != 99.0)[0]
                if len(idx) < 2:
                    continue
                data[name][filt] = np.interp(tt,
                    group["jd"].iloc[idx] - jd_min,
                    group[mag_key].iloc[idx],left=np.nan, right=np.nan )
        
        # Bulla format
        elif format == "bulla":
            with open(filename, "r") as f:
                header = list(filter(None, f.readline().rstrip().strip("#").split(" ")))
            df = pd.read_csv(
                filename,
                delimiter=" ",
                comment="#",
                header=None,
                names=header,
                index_col=False,
            )
            df.rename(columns={"t[days]": "t"}, inplace=True)
            data[name] = df.to_dict(orient="series")

        # Standard format
        elif format == "standard":
            mag_d = np.loadtxt(filename)
            mag_d_shape = mag_d.shape

            data[name] = {}
            data[name]["t"] = mag_d[:, 0]
            data[name]["u"] = mag_d[:, 1]
            data[name]["g"] = mag_d[:, 2]
            data[name]["r"] = mag_d[:, 3]
            data[name]["i"] = mag_d[:, 4]
            data[name]["z"] = mag_d[:, 5]
            data[name]["y"] = mag_d[:, 6]
            data[name]["J"] = mag_d[:, 7]
            data[name]["H"] = mag_d[:, 8]
            data[name]["K"] = mag_d[:, 9]

            if mag_d_shape[1] == 15:
                data[name]["U"] = mag_d[:, 10]
                data[name]["B"] = mag_d[:, 11]
                data[name]["V"] = mag_d[:, 12]
                data[name]["R"] = mag_d[:, 13]
                data[name]["I"] = mag_d[:, 14]

        # HDF5 format
        elif format == "hdf5":
            f = h5py.File(filename, "r")
            keys = list(f.keys())
            for key in keys:
                df = astropy.table.Table(f[key]).to_pandas()
                df.rename(
                    columns={
                        "2MASS_J": "2massj",
                        "2MASS_H": "2massh",
                        "2MASS_Ks": "2massks",
                        "SDSS_u": "sdssu",
                        "ZTF_g": "ztfg",
                        "ZTF_i": "ztfi",
                        "ZTF_r": "ztfr",
                        "atlas_c": "atlasc",
                        "atlas_o": "atlaso",
                        "ps_g": "ps1::g",
                        "ps_r": "ps1::r",
                        "ps_i": "ps1::i",
                        "ps_z": "ps1::z",
                        "ps_y": "ps1::y",
                        "sU": "bessellux",
                        "sB": "bessellb",
                        "sV": "bessellv",
                        "sR": "bessellr",
                        "sI": "besselli",
                        "uvot_b": "uvot::b",
                        "uvot_u": "uvot::u",
                        "uvot_v": "uvot::v",
                        "uvot_uvm2": "uvot::uvm2",
                        "uvot_uvw1": "uvot::uvw1",
                        "uvot_uvw2": "uvot::uvw2",
                        "uvot_white": "uvot::white",
                        "time": "t",
                    },
                    inplace=True,
                )
                data[key] = df.to_dict(orient="series")
                # data[key] = {
                #     k.replace(":", "_"): v.to_numpy() for k, v in data[key].items()
                # }

        # Finally, extract the desired filters from all filters present in the data
        if filters is not None:
            filters_to_remove = set(list(data[name].keys())) - set(filters + ["t"])
            for filt in filters_to_remove:
                del data[name][filt]

    return data


#FIXME Legacy??? seems unused
def loadEventSpec(filename):

    data_out = np.loadtxt(filename)
    spec = {}

    spec["lambda"] = data_out[:, 0]  # Angstroms
    spec["data"] = np.abs(data_out[:, 1])  # ergs/s/cm2./Angs
    spec["error"] = np.zeros(spec["data"].shape)  # ergs/s/cm2./Angs
    spec["error"][:-1] = np.abs(np.diff(spec["data"]))
    spec["error"][-1] = spec["error"][-2]
    idx = np.where(spec["error"] <= 0.5 * spec["data"])[0]
    spec["error"][idx] = 0.5 * spec["data"][idx]

    return spec