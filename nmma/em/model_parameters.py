import re
import numpy as np

from .utils import get_knprops_from_LANLfilename


def AnBa2022(data):

    data_out = {}

    parameters = ["log10_mtot", "vej", "log10_mni", "log10_mrp", "xmix"]
    parameters_idx = [0, 1, 2, 3, 4]
    magkeys = data.keys()
    for jj, key in enumerate(magkeys):
        rr = [
            np.abs(float(x))
            for x in re.findall(
                r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",
                key.replace("m56", "mni"),
            )
        ]

        data_out[key] = {
            param: rr[idx] for param, idx in zip(parameters, parameters_idx)
        }
        data_out[key] = {**data_out[key], **data[key]}

    return data_out, parameters


def AnBa2022_sparse(data):

    data_out = {}

    parameters = ["mrp", "xmix"]
    parameters_idx = [3, 4]
    magkeys = data.keys()
    for jj, key in enumerate(magkeys):
        rr = [
            np.abs(float(x))
            for x in re.findall(
                r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", key
            )
        ]

        data_out[key] = {
            param: rr[idx] for param, idx in zip(parameters, parameters_idx)
        }
        data_out[key] = {**data_out[key], **data[key]}

    return data_out, parameters


def CV(data):

    data_out = {}

    parameters = ["example_num"]
    parameters_idx = [0]
    magkeys = data.keys()
    for jj, key in enumerate(magkeys):
        data_out[key] = {param: jj for param, idx in zip(parameters, parameters_idx)}
        data_out[key] = {**data_out[key], **data[key]}

    return data_out, parameters


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

    return data_out, parameters


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

    return data_out, parameters


def Bu2019nsbh(data):

    data_out = {}

    parameters = ["log10_mej_dyn", "log10_mej_wind", "KNtheta"]
    parameters_idx = [1, 2, 4]
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

    return data_out, parameters


def Bu2022Ye(data):

    data_out = {}

    parameters = [
        "log10_mej_dyn",
        "vej_dyn",
        "Yedyn",
        "log10_mej_wind",
        "vej_wind",
        "KNtheta",
    ]
    parameters_idx = [1, 2, 3, 4, 5, 6]
    magkeys = data.keys()
    for jj, key in enumerate(magkeys):
        rr = [
            np.abs(float(x))
            for x in re.findall(
                r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", key
            )
        ]

        # Best to interpolate mass in log10 space
        rr[1] = np.log10(rr[1])
        rr[4] = np.log10(rr[4])

        data_out[key] = {
            param: rr[idx] for param, idx in zip(parameters, parameters_idx)
        }
        data_out[key] = {**data_out[key], **data[key]}

    return data_out, parameters


def Ka2017(data):

    parameters = [
        "log10_mej",
        "log10_vej",
        "log10_Xlan",
    ]

    data_out = {}

    parameters_idx = [2, 3, 5]
    magkeys = data.keys()
    for jj, key in enumerate(magkeys):
        rr = [
            np.abs(float(x))
            for x in re.findall(
                r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", key
            )
        ]

        # Best to interpolate mass in log10 space
        rr[2] = np.log10(rr[2])
        rr[3] = np.log10(rr[3])
        rr[5] = np.log10(rr[5])

        data_out[key] = {
            param: rr[idx] for param, idx in zip(parameters, parameters_idx)
        }
        data_out[key] = {**data_out[key], **data[key]}

    return data_out, parameters


def LANL2022(data):

    parameters = [
        # "Ye_wind",
        "log10_mej_dyn",
        "vej_dyn",
        "log10_mej_wind",
        "vej_wind",
        "KNtheta",
    ]

    data_out = {}

    magkeys = data.keys()
    for jj, key in enumerate(magkeys):
        knprops = get_knprops_from_LANLfilename(key)

        # best to interpolate masses in log10
        knprops["log10_mej_dyn"] = np.log10(knprops["mej_dyn"])
        knprops["log10_mej_wind"] = np.log10(knprops["mej_wind"])
        del knprops["mej_dyn"]
        del knprops["mej_wind"]
        # del knprops["morphology"]

        data_out[key] = knprops
        data_out[key] = {**data_out[key], **data[key]}

    return data_out, parameters
