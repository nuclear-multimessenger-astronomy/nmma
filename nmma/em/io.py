import astropy
from astropy.time import Time
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interpolate as interp
import scipy.signal
from sncosmo.bandpasses import _BANDPASSES


def loadEvent(filename):
    lines = [line.rstrip("\n") for line in open(filename)]
    lines = filter(None, lines)

    sncosmo_filts = [val["name"] for val in _BANDPASSES.get_loaders_metadata()]
    sncosmo_maps = {name: name.replace(":", "_") for name in sncosmo_filts}

    data = {}
    for line in lines:
        lineSplit = line.split(" ")
        lineSplit = list(filter(None, lineSplit))
        mjd = Time(lineSplit[0], format="isot").mjd
        filt = lineSplit[1]

        if filt in sncosmo_maps:
            filt = sncosmo_maps[filt]

        mag = float(lineSplit[2])
        dmag = float(lineSplit[3])

        if filt not in data:
            data[filt] = np.empty((0, 3), float)
        data[filt] = np.append(data[filt], np.array([[mjd, mag, dmag]]), axis=0)

    return data


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


def read_photometry_files(
    files, filters=None, tt=np.linspace(0, 14, 100), datatype="bulla"
):

    data = {}
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
        if datatype == "ztf":
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
                lc = interp.interp1d(
                    group["jd"].iloc[idx] - jd_min,
                    group[mag_key].iloc[idx],
                    fill_value=np.nan,
                    bounds_error=False,
                    assume_sorted=True,
                )
                data[name][filt] = lc(tt)
        elif datatype == "bulla":
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
            data[name] = {
                k.replace(":", "_"): v.to_numpy() for k, v in data[name].items()
            }

        elif datatype == "standard":
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

        elif datatype == "hdf5":
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
                data[key] = {
                    k.replace(":", "_"): v.to_numpy() for k, v in data[key].items()
                }

        else:
            raise ValueError(f"datatype {datatype} unknown")

        if filters is not None:
            filters_to_remove = set(list(data[name].keys())) - set(filters + ["t"])
            for filt in filters_to_remove:
                del data[name][filt]

    return data


def read_lightcurve_file(filename):

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

    return df.to_dict(orient="series")
