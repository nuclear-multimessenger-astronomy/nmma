import numpy as np
from scipy.integrate import quad
import astropy.constants


def arnett_lc_get_int_A_non_vec(x, y):
    r = quad(lambda z: 2 * z * np.exp(-2 * z * y + z**2), 0, x)
    int_A = r[0]
    return int_A


arnett_lc_get_int_A = np.vectorize(arnett_lc_get_int_A_non_vec, excluded=["y"])


def arnett_lc_get_int_B_non_vec(x, y, s):
    r = quad(lambda z: 2 * z * np.exp(-2 * z * y + 2 * z * s + z**2), 0, x)
    int_B = r[0]
    return int_B


arnett_lc_get_int_B = np.vectorize(arnett_lc_get_int_B_non_vec, excluded=["y", "s"])


def arnett_lc(t_day, param_dict):
    day = 86400.0  # in seconds
    ts = t_day * day
    Mni = 10 ** param_dict["log10_mni"]
    Mni *= astropy.constants.M_sun.cgs.value  # in g
    tau_m = param_dict["tau_m"] * day

    epsilon_ni = 3.9e10  # erg / s / g
    epsilon_co = 6.78e9  # erg / s / g
    tau_ni = 8.8 * day  # s
    tau_co = 111.3 * day  # s

    x = ts / tau_m
    y = tau_m / (2 * tau_ni)
    s = tau_m * (tau_co - tau_ni) / (2 * tau_co * tau_ni)

    int_A = arnett_lc_get_int_A(x, y)
    int_B = arnett_lc_get_int_B(x, y, s)

    Ls = (
        Mni
        * np.exp(-(x**2))
        * ((epsilon_ni - epsilon_co) * int_A + epsilon_co * int_B)
    )

    return Ls


def arnett_modified_lc(t_day, param_dict):
    day = 86400.0  # in seconds
    ts = t_day * day  # in seconds
    Mni = 10 ** param_dict["log10_mni"]
    Mni *= astropy.constants.M_sun.cgs.value  # in g
    tau_m = param_dict["tau_m"] * day
    t0 = param_dict["t_0"] * day

    epsilon_ni = 3.9e10  # erg / s / g
    epsilon_co = 6.78e9  # erg / s / g
    tau_ni = 8.8 * day  # second
    tau_co = 111.3 * day  # second

    x = ts / tau_m
    y = tau_m / (2 * tau_ni)
    s = tau_m * (tau_co - tau_ni) / (2 * tau_co * tau_ni)

    int_A = arnett_lc_get_int_A(x, y)
    int_B = arnett_lc_get_int_B(x, y, s)

    Ls = (
        Mni
        * np.exp(-(x**2))
        * ((epsilon_ni - epsilon_co) * int_A + epsilon_co * int_B)
    )
    Ls_modified = Ls * (1.0 - np.exp(-1.0 * (t0 / ts) ** 2))

    return Ls_modified
