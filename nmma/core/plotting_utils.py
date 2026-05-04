
from bilby.core.prior import PriorDict, DeltaFunction
import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import os
# matplotlib.use("Agg")
from matplotlib import pyplot as plt
import itertools

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
        "font.family": "serif",
        "figure.figsize": fig_size,
        "mathtext.fontset": "stix",
    }

    ## Alt choice
    # params = {
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
    matplotlib.rcParams['text.usetex'] = (os.environ.get("CI") != 'true')

    color_array = [
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
    return itertools.cycle(color_array)

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

def setup_multi_axes(num_axes, sharex=False, sharey=False, ncols=None):
    "Set up a multi-panel figure with the specified number of axes, essentially stolen from corner.py"
    
    if ncols is None:
        ncols = np.min([5, np.ceil(np.sqrt(num_axes)).astype(int)])
    nrows = np.ceil(num_axes / ncols).astype(int)
    

    factor = 2.0  # size of one side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.05  # w/hspace size
    whspace = trdim
    rowdim = lbdim + factor * ncols + factor * (ncols - 1.0) * whspace + trdim
    coldim = lbdim + factor * nrows + factor * (nrows - 1.0) * whspace + trdim



    # Create a new figure if one wasn't provided.
    fig, axes = plt.subplots(nrows, ncols, figsize=(rowdim, coldim), 
            sharex=sharex, sharey=sharey, constrained_layout=True)

    return fig, axes.flatten()


def fading_cmap(color):
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["white",color], gamma = 2)

    cdict = cmap._segmentdata.copy()
    vals = cdict['alpha'][:, 0]
    alpha = np.linspace(0, 1, len(vals))
    cdict['alpha'] = np.column_stack([vals, alpha, alpha])

    return LinearSegmentedColormap(cmap.name, cdict, cmap.N, cmap._gamma)