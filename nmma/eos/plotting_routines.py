import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from . import eos_likelihood
from scipy.stats import norm
import numpy as np

def plot_eos_vs_constraints(eos_data, constraints=[],  save_path = None, **kwargs):
    """
    Plot the EOS data against the provided constraints.

    Parameters:
    - eos_data: A dictionary containing 'radius', 'mass', and 'lambda' keys.
    - constraints: A dictionary containing 'mass' and 'lambda' keys for constraints.
    - save_path: Optional path to save the plot.
    """
    if isinstance(constraints, eos_likelihood.JointEoSConstraint):
        constraints = constraints.constraints
    radii, masses, lambdas = eos_data
    x_lim = (np.min(radii)-0.3, np.max(radii)+0.3)
    y_lim = (masses[0], masses[-1]+0.1)
    fig, ax = plt.subplots(figsize=(10, 6), xlim=x_lim, ylim=y_lim)
    for constraint in constraints:
        ax = constraint.plot(ax=ax, **kwargs)
        
    ax.plot(radii, masses, label='Best fit EOS')

    ax.set_xlabel(r'Radius [km]')
    ax.set_ylabel(r'Mass [M$_\odot$]')
    ax.legend()

    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()