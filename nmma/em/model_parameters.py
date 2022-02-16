import re
import numpy as np


def Bu2019lm_sparse(data):

    data_out = {}

    parameters = ["log10_mej_dyn", "log10_mej_wind"]
    parameters_idx = [1, 2]
    magkeys = data.keys()
    for jj, key in enumerate(magkeys):
        rr = [
            float(x)
            for x in re.findall(
                r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", key
            )
        ]

        # Best to interpolate mass in log10 space
        rr[1] = np.log10(rr[1])
        rr[2] = np.log10(rr[2])

        data_out[key] = {
            param: rr[idx] for param, idx in zip(parameters, parameters_idx)
        }
        data_out[key] = {**data_out[key], **data[key]}

    return data_out


def Bu2019lm(data):

    data_out = {}

    parameters = ["log10_mej_dyn", "log10_mej_wind", "KNphi", "KNtheta"]
    parameters_idx = [1, 2, 3, 4]
    magkeys = data.keys()
    for jj, key in enumerate(magkeys):
        rr = [
            float(x)
            for x in re.findall(
                r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", key
            )
        ]

        # Best to interpolate mass in log10 space
        rr[1] = np.log10(rr[1])
        rr[2] = np.log10(rr[2])

        data_out[key] = {
            param: rr[idx] for param, idx in zip(parameters, parameters_idx)
        }
        data_out[key] = {**data_out[key], **data[key]}

    return data_out


def Bu2022mv(data):

    data_out = {}

    parameters = ["log10_mej_dyn", "vej_dyn", "log10_mej_wind", "vej_wind", "KNtheta"]
    parameters_idx = [0, 1, 3, 4, 6]
    magkeys = data.keys()
    for jj, key in enumerate(magkeys):
        rr = [
            np.abs(float(x))
            for x in re.findall(
                r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", key
            )
        ]

        # Best to interpolate mass in log10 space
        rr[0] = np.log10(rr[0])
        rr[3] = np.log10(rr[3])

        data_out[key] = {
            param: rr[idx] for param, idx in zip(parameters, parameters_idx)
        }
        data_out[key] = {**data_out[key], **data[key]}

    return data_out
