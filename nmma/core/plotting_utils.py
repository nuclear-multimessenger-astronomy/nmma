
from bilby.core.prior import PriorDict, DeltaFunction
import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import os
# matplotlib.use("Agg")
from matplotlib import pyplot as plt
import itertools
matplotlib.rcParams['text.usetex'] = (os.environ.get("CI") != 'true')

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

    color_array = [
        "#22ADFC", # blue
        "#F42969", # red
        "#F4A429", # orange
        "#4635CE", # purple
        "#008080", # teal
        "#FFD700", # gold
        "#00FFFF", # cyan
        "#8B4513", # brown
        "#4CAF50", # green
        "#FF6347", # tomato
        "#00008B", # deep blue
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

def setup_multi_axes(num_axes, sharex=False, sharey=False, ncols=None, dpi=250, **fig_kwargs):
    "Set up a multi-panel figure with the specified number of axes, essentially stolen from corner.py"
    fig_setup()
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

    if 'figsize' not in fig_kwargs:
        fig_kwargs['figsize'] = (rowdim, coldim)
    # Create a new figure if one wasn't provided.
    fig, axes = plt.subplots(nrows, ncols,
            sharex=sharex, sharey=sharey, constrained_layout=True, dpi=dpi, **fig_kwargs)
    try:
        out_axes = axes.flatten()
    except AttributeError:
        out_axes = axes

    return fig, out_axes


def fading_cmap(color):
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["white",color], gamma = 2)

    cdict = cmap._segmentdata.copy()
    vals = cdict['alpha'][:, 0]
    alpha = np.linspace(0, 1, len(vals))
    cdict['alpha'] = np.column_stack([vals, alpha, alpha])

    return LinearSegmentedColormap(cmap.name, cdict, cmap.N, cmap._gamma)


def format_title_offset(ax, title):
    formatter = ax.xaxis.get_major_formatter()
    offset_string = formatter.get_offset()
    if not offset_string:
        return title
    

    use_math = formatter._useMathText
    use_tex = formatter._usetex
    formatter._useMathText=False
    formatter._usetex=False
    offset_string = formatter.get_offset()
    formatter._useMathText = use_math
    formatter._usetex = use_tex

    mult, add = get_offset(offset_string)
    title_parts = title.split('$')
    sig_range = title_parts[-2].replace('{', '').replace('}', '')
    mean, unc = sig_range.split('_')
    mean = (float(mean) - add) / mult
    low, high = unc.split('^')
    low = float(low) / mult
    high = float(high) / mult

    title_parts[-2] = f"{mean:#.3g}_{{{low:#.2g}}}^{{+{high:#.2g}}}"
    return '$'.join(title_parts)


def get_offset(offset_string):
    offset = offset_string.replace('−', '-')
    #single addition
    if offset.startswith('+') or offset.startswith('-'):
        return (1., float(offset))
    
    parts = offset.split('e')
    # single multiplication
    if len(parts) == 2:
        return (float(offset), 0.)
    #addition and multiplication
    for i, char in enumerate(parts[1]):
        if char in ['+', '-'] and i != 0:
            break
    mult = 'e'.join([parts[0], parts[1][:i]])
    add = 'e'.join([parts[1][i:], parts[2]])
    return (float(mult), float(add))

def arange_titles(ax, move = False):
    lower_y = 1.07
    ax.title.set_visible(False) 
    if len(ax.texts) == 2:
        ax.texts[0].set_position((0., lower_y))
        ax.texts[0].set_ha('left')
        ax.texts[1].set_position((1., lower_y))
        ax.texts[1].set_ha('right')
        if move:
            move_up(ax.texts[1])
    elif len(ax.texts) == 3:

        ax.texts[1].set_position((0.5, lower_y))
        ax.texts[1].set_ha('center')
        move_up(ax.texts[1])
        ax.texts[2].set_position((1., lower_y))
        ax.texts[2].set_ha('right')
        if move:
            move_up(ax.texts[0], amount=0.44)
    elif len(ax.texts) ==4:
        move_up(ax.texts[0])
        ax.texts[1].set_position((0., lower_y))
        ax.texts[1].set_ha('left')
        ax.texts[2].set_position((1., lower_y))
        ax.texts[2].set_ha('right')
        move_up(ax.texts[2])
        ax.texts[3].set_position((1., lower_y))
        ax.texts[3].set_ha('right')
    else:
        print("Warning: Titles could not be nicely aligned.")
        title_texts = [t.get_text() for t in ax.texts]
        print(f"All intended title information was {ax.get_ylabel()}: {title_texts}")
        for t in ax.texts:
            t.set_visible(False)
    return ax

def texts_overlap(left_text, right_text):
    # FIXME: this needs to be reworked more thoroughly
    left_bounds = left_text.get_tightbbox().get_points()
    right_bounds = right_text.get_tightbbox().get_points()
    return True if (
        left_bounds[1, 0] > right_bounds[0, 0]
    ) and (
        left_bounds[1, 1] > right_bounds[0, 1]
    ) and(
        left_bounds[1, 1] <= right_bounds[1, 1]
    ) else False

def move_up(text, amount = 0.22):
        x, y = text.get_position()
        text.set_position((x, y + amount))