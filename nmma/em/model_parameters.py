"""Script with various functions to extract the parameters of models from their naming convention for filenames."""

import re

import numpy as np

from .utils import get_knprops_from_LANLfilename


def AnBa2022_linear(data):
    """Read the Anand 2022 parameters off the file names, in linear space.

    Parameters
    ----------
    data: dict
        Training grid, keyed by the file name each light curve came from.

    Returns
    -------
    data_out: dict
        The same grid, each entry carrying its extracted parameters.
    parameters: list of str
        Names of the parameters, in the order the surrogate expects them.
    """

    data_out = {}

    parameters = ["mtot", "mni", "vej", "mrp", "xmix"]
    parameters_idx = [0, 2, 1, 3, 4]
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


def AnBa2022_log(data):
    """Read the Anand 2022 parameters off the file names, in log space.

    The masses are interpolated in log space, where the grid is far more
    evenly sampled than in linear space.

    Parameters
    ----------
    data: dict
        Training grid, keyed by the file name each light curve came from.

    Returns
    -------
    data_out: dict
        The same grid, each entry carrying its extracted parameters.
    parameters: list of str
        Names of the parameters, in the order the surrogate expects them.
    """

    data_out = {}

    parameters = ["log10_mtot", "log10_mni", "vej", "log10_mrp", "xmix"]
    parameters_idx = [0, 2, 1, 3, 4]
    magkeys = data.keys()
    for jj, key in enumerate(magkeys):
        rr = [
            np.abs(float(x))
            for x in re.findall(
                r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",
                key.replace("m56", "mni"),
            )
        ]

        # Best to interpolate mass in log10 space
        rr[0] = np.log10(rr[0])
        rr[2] = np.log10(rr[2])
        rr[3] = np.log10(rr[3])

        data_out[key] = {
            param: rr[idx] for param, idx in zip(parameters, parameters_idx)
        }
        data_out[key] = {**data_out[key], **data[key]}

    return data_out, parameters


def AnBa2022_sparse(data):
    """Read only the two Anand 2022 parameters a sparse grid varies.

    Parameters
    ----------
    data: dict
        Training grid, keyed by the file name each light curve came from.

    Returns
    -------
    data_out: dict
        The same grid, each entry carrying its extracted parameters.
    parameters: list of str
        Names of the parameters, in the order the surrogate expects them.
    """

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
    """Index cataclysmic variable light curves by their example number.

    These grids carry no physical parameter in their file names, only a
    running index.

    Parameters
    ----------
    data: dict
        Training grid, keyed by the file name each light curve came from.

    Returns
    -------
    data_out: dict
        The same grid, each entry carrying its extracted parameters.
    parameters: list of str
        Names of the parameters, in the order the surrogate expects them.
    """

    data_out = {}

    parameters = ["example_num"]
    parameters_idx = [0]
    magkeys = data.keys()
    for jj, key in enumerate(magkeys):
        data_out[key] = {param: jj for param, idx in zip(parameters, parameters_idx)}
        data_out[key] = {**data_out[key], **data[key]}

    return data_out, parameters


def Bu2019lm_sparse(data):
    """Read only the two ejecta masses of a sparse Bulla 2019 grid.

    Parameters
    ----------
    data: dict
        Training grid, keyed by the file name each light curve came from.

    Returns
    -------
    data_out: dict
        The same grid, each entry carrying its extracted parameters.
    parameters: list of str
        Names of the parameters, in the order the surrogate expects them.
    """

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
    """Read the Bulla 2019 kilonova parameters off the file names.

    Covers the two ejecta components, the half-opening angle of the
    lanthanide-rich region, and the viewing angle.

    Parameters
    ----------
    data: dict
        Training grid, keyed by the file name each light curve came from.

    Returns
    -------
    data_out: dict
        The same grid, each entry carrying its extracted parameters.
    parameters: list of str
        Names of the parameters, in the order the surrogate expects them.
    """

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
    """Read the Bulla 2019 parameters of a neutron star black hole grid.

    Unlike the binary neutron star case, no opening angle is varied: the
    disc geometry is set by the tidal disruption.

    Parameters
    ----------
    data: dict
        Training grid, keyed by the file name each light curve came from.

    Returns
    -------
    data_out: dict
        The same grid, each entry carrying its extracted parameters.
    parameters: list of str
        Names of the parameters, in the order the surrogate expects them.
    """

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
    """Read the Bulla 2022 parameters, electron fraction included.

    Adds the velocity of each ejecta component and the electron fraction of
    the dynamical ejecta, which drives how many lanthanides form and hence
    how red the kilonova is.

    Parameters
    ----------
    data: dict
        Training grid, keyed by the file name each light curve came from.

    Returns
    -------
    data_out: dict
        The same grid, each entry carrying its extracted parameters.
    parameters: list of str
        Names of the parameters, in the order the surrogate expects them.
    """

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


def Bu2023Ye(data):
    """Read the Bulla 2023 parameters, with both electron fractions.

    Extends the 2022 grid by varying the electron fraction of the wind as
    well as that of the dynamical ejecta.

    Parameters
    ----------
    data: dict
        Training grid, keyed by the file name each light curve came from.

    Returns
    -------
    data_out: dict
        The same grid, each entry carrying its extracted parameters.
    parameters: list of str
        Names of the parameters, in the order the surrogate expects them.
    """

    data_out = {}

    parameters = [
        "log10_mej_dyn",
        "vej_dyn",
        "Yedyn",
        "log10_mej_wind",
        "vej_wind",
        "Yewind",
        "KNtheta",
    ]
    parameters_idx = [0, 1, 2, 3, 4, 5, 6]
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

    return data_out, parameters


def Ka2017(data):
    """Read the Kasen 2017 parameters off the file names.

    A one-component model: an ejecta mass, a velocity, and the lanthanide
    mass fraction that sets the opacity.

    Parameters
    ----------
    data: dict
        Training grid, keyed by the file name each light curve came from.

    Returns
    -------
    data_out: dict
        The same grid, each entry carrying its extracted parameters.
    parameters: list of str
        Names of the parameters, in the order the surrogate expects them.
    """

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


def LANLTP1(data):
    """Read the parameters of a LANL TP1 grid.

    TP stands for the toroidal-peanut ejecta geometry, 1 for the first
    wind configuration.

    Parameters
    ----------
    data: dict
        Training grid, keyed by the file name each light curve came from.

    Returns
    -------
    data_out: dict
        The same grid, each entry carrying its extracted parameters.
    parameters: list of str
        Names of the parameters, in the order the surrogate expects them.
    """

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


def LANLTS1(data):
    """Read the parameters of a LANL TS1 grid.

    TS stands for the toroidal-spherical ejecta geometry, 1 for the first
    wind configuration.

    Parameters
    ----------
    data: dict
        Training grid, keyed by the file name each light curve came from.

    Returns
    -------
    data_out: dict
        The same grid, each entry carrying its extracted parameters.
    parameters: list of str
        Names of the parameters, in the order the surrogate expects them.
    """

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


def LANLTP2(data):
    """Read the parameters of a LANL TP2 grid.

    Toroidal-peanut ejecta geometry, second wind configuration.

    Parameters
    ----------
    data: dict
        Training grid, keyed by the file name each light curve came from.

    Returns
    -------
    data_out: dict
        The same grid, each entry carrying its extracted parameters.
    parameters: list of str
        Names of the parameters, in the order the surrogate expects them.
    """

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


def LANLTS2(data):
    """Read the parameters of a LANL TS2 grid.

    Toroidal-spherical ejecta geometry, second wind configuration.

    Parameters
    ----------
    data: dict
        Training grid, keyed by the file name each light curve came from.

    Returns
    -------
    data_out: dict
        The same grid, each entry carrying its extracted parameters.
    parameters: list of str
        Names of the parameters, in the order the surrogate expects them.
    """

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
