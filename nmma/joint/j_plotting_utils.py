
from bilby.core.prior import PriorDict, DeltaFunction
import numpy as np
import matplotlib
matplotlib.use("Agg")

def fig_setup():
    fig_width_pt = 750.0  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = 0.9 * fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    params = {
        "backend": "pdf",
        "axes.labelsize": 18,
        "legend.fontsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "text.usetex": True,
        "font.family": "Times New Roman",
        "figure.figsize": fig_size,
        "mathtext.fontset": "stix",
    }

    ## Alt choice
    # params = {
    #     # latex
    #     "text.usetex": True,
    #     # fonts
    #     "mathtext.fontset": "stix",
    #     "font.serif": "Computer Modern",
    #     "font.family":["serif", "STIXGeneral"],
    #     # figure and axes
    #     "axes.grid": False,
    #     "axes.titlesize": 10,
    #     # tick markers
    #     "xtick.direction": "in",
    #     "ytick.direction": "in",
    #     "xtick.labelsize": 13,
    #     "ytick.labelsize": 13,
    #     "xtick.major.size": 10.0,
    #     "ytick.major.size": 10.0,
    #     # legend
    #     "legend.fontsize": 20,
    # }
    matplotlib.rcParams.update(params)

    return  [
        "#22ADFC", # blue
        "#F42969", # red
        "#F4A429", # orange
        "#4635CE", # purple
        "#4CAF50", # green
        "#FFD700", # gold
        "#008080", # teal
        "#00008B", # deep blue
        "#00FFFF", # cyan
        "#8B4513", # brown
        "#FF6347", # tomato
    ]


def plotting_parameters_from_priors(priors, keys=None):
    """
    Extracts plotting parameters from the priors dictionary.

    Parameters
    ----------
    priors : dict
        Dictionary containing prior information.
    keys : list, optional
        List of keys to extract from the priors. If None, all keys are used.

    Returns
    -------
    dict
        Dictionary with plotting parameters.
    """
    if isinstance(priors, str):
        priors = PriorDict(filename=priors)
        
    priors.convert_floats_to_delta_functions()
    if keys is None:
        keys = priors.keys()

    return {k: v.latex_label for k, v in priors.items() if k in keys and not isinstance(v, DeltaFunction)}