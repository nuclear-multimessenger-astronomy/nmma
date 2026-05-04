from glob import glob
from argparse import Namespace
import os
import shutil
import json
from ast import literal_eval
import matplotlib
from tqdm.contrib.concurrent import process_map 
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
# from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
from bilby.core.prior import WeightedCategorical, PriorDict
from .eos_processing import EoSConverter
from ..core.base import NMMALikelihood
from ..core.utils import nan_level
from ..core.plotting_utils import fading_cmap
    
def setup_tabulated_eos_priors(args, priors, logger=None):
    if logger:    
        logger.info("Sampling over precomputed EOSs")
    weights = np.loadtxt(args.eos_weight) if getattr(args, 'eos_weight', None) else None
    if getattr(args, 'Neos', False):
        n_eos = args.Neos
    elif weights is not None:
        n_eos = len(weights)
    else:
        n_eos = len(os.listdir(args.eos_data))
    priors["EOS"] = WeightedCategorical(n_eos, weights, name="EOS")
    return priors

def setup_eos_kwargs(data_dump, args, logger):
    # default_eos_kwargs = initialisation_args_from_signature_and_namespace(EquationofStateLikelihood, args)
    # eos_kwargs = default_eos_kwargs | dict(
    eos_kwargs = dict(
        constraint_dict=data_dump['eos_constraint_dict'],
        eos_converter=EoSConverter(args, 'emulated'),
        # crust_path=args.eos_crust_file
    )
    return eos_kwargs 

def tabulated_eos_setup(args):
    priors = PriorDict()
    priors = setup_tabulated_eos_priors(args, priors)
    args.Neos = priors['EOS'].ncategories
    eos_converter = EoSConverter(args, 'tabulated')
    eos_converter.parameter_conversion = eos_converter.compute_macro_parameters
    eos_likelihood_kwargs = {
        "constraint_dict": compose_eos_constraints(args),
        "eos_converter": eos_converter,
    }
    eos_likelihood = EquationofStateLikelihood(priors, **eos_likelihood_kwargs)
    return priors, eos_likelihood, None
   
class EquationofStateLikelihood(NMMALikelihood):
    def __init__(self, priors, constraint_dict, eos_converter, **kwargs):
        constraint =JointEoSConstraint(constraint_dict, eos_converter=eos_converter)
        # TODO: to be extended for more complex likelihood expressions
        super().__init__(constraint, priors, **kwargs)

    def setup_submodel_conversion(self):
        self.conv_functions.append(self.sub_model.parameter_conversion)
        
     
    def final_diagnostics(self, bestfit_params, args, result=None, fig = None):
        matplotlib.rcParams.update({'font.size': 16, 'font.family': 'serif'})

        bestfit_params =self.parameter_conversion(bestfit_params)
        radii, masses, lambdas = self.sub_model.eos_converter.macro_parameters.values()
        
        if fig is None:
            x_lim = (min(np.min(radii)-0.3, 9), max(np.max(radii)+0.3, 15))
            y_lim = (masses[0], masses[-1]+0.1)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_xlabel(r'Radius [km]')
            ax.set_ylabel(r'Mass [M$_\odot$]') 

            for constraint in self.sub_model.constraints:
                color = ax._get_lines.get_next_color()
                ax = constraint.plot(ax=ax, color=color)
        else:
            ax = fig.axes[0]
            xlow, xhigh = ax.get_xlim()
            ylow, yhigh = ax.get_ylim()
            ax.set_xlim(min(np.min(radii)-0.3, xlow), max(np.max(radii)+0.3, xhigh))
            ax.set_ylim(min(masses[0], ylow), max(masses[-1]+0.1, yhigh))

            labels = [line.get_label() for line in fig.legends[0].legend_handles]
            for constraint in self.sub_model.constraints:
                if constraint.name in labels:
                    continue
                color = ax._get_lines.get_next_color()
                ax = constraint.plot(ax=ax, color=color)
            fig.legends.clear() ## remove old legend to avoid duplicates

        
        line =ax.plot(radii, masses, label=f'{args.label}',linewidth=3, zorder=10)
        
        if result is not None:
            cmap = fading_cmap(line[0].get_color())
            posterior = self.parameter_conversion(result.posterior)
            # posterior['log_post'] = posterior.log_likelihood + posterior.log_prior
            # post_weights = np.exp(posterior.log_post)
            post_weights = None
            eos_data = self.sub_model.eos_converter.macro_conversion(posterior)
            max_masses = np.max([eos[1][-1] for eos in eos_data])
            mass_range = np.linspace(1.0, max_masses, 151)
            show_radii = np.empty((len(mass_range), len(eos_data)))
            
            for i, eos in enumerate(eos_data):
                show_radii[:,i] = np.interp(mass_range, eos[1], eos[0], right = np.nan)
                
            for level in [0.5, 0.9]:
                bounds = np.array([nan_level(radii, level, post_weights) for radii in show_radii])
                ax.fill_betweenx(mass_range, bounds[:, 0], bounds[:, 1], color=cmap(1-0.5*level), zorder=1)
                
            if result.injection_parameters is not None:
                inj_eos = self.sub_model.eos_converter.macro_conversion(result.injection_parameters)
                ax.plot(inj_eos[0], inj_eos[1], label='Injection', color='black', linestyle='dashed', linewidth=3, zorder=10)

        fig.legend(ncols=2, loc='upper center',
                   bbox_to_anchor=(0.5, 0.00), handlelength=2)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"{args.label}_mr_curve.png"),bbox_inches='tight')
        return fig


def compose_eos_constraints(args, constraint_kinds=['lower_mtov', 'upper_mtov', 'mass_radius']):
    try:
        with open(args.eos_constraint_json, 'r') as f:
            constraint_dict = json.load(f) 
    except:
        constraint_dict = {}

    for constraint_kind in constraint_kinds:
        sub_dict= read_constraint_from_args(args,constraint_kind)
        if sub_dict is None:
            continue
        try:
            constraint_dict[constraint_kind].update(sub_dict)
        except KeyError:
            constraint_dict[constraint_kind] = sub_dict
        except AttributeError:
            constraint_dict[constraint_kind] = sub_dict

    try:
        with open(args.eos_constraint_json, "w") as f:
            json.dump(constraint_dict, f, indent=4)
    except:
        pass
    return constraint_dict

def read_constraint_from_args(args, constraint_kind):
    "Routine to read prepare constraint in expected dict-format from argparse namespace during generation process"
    ##preferred: Have the dict with the subconstraints already set up
    prep_dict= getattr(args, constraint_kind, None) 
    if prep_dict:
        if isinstance(prep_dict, str):
            prep_dict = literal_eval(prep_dict)
        return prep_dict
    
    ###otherwise try to construct it:
        ### read in provided attributes like constraint_kind-name, -mass, -error, -arxiv
    constraint_props = {
        key.removeprefix(constraint_kind+'_'): ##cut identifier
                getattr(args, key) 
                for key in dir(args)            ## search args for attrs 
                if key.startswith(constraint_kind+'_') ## related with kind
    } 

    new_constraints = constraint_props.pop('name', None)
    if new_constraints: ## there needs to be a unique label
        for k in list(constraint_props.keys()):
            v = constraint_props[k]
            if v is None:
                constraint_props.pop(k)
            elif len(v) != len(new_constraints): 
                raise ValueError(f'For {constraint_kind}, the number of entries for {k} ({len(v)}) does not match the number of names ({len(new_constraints)})!')
            
        ext_dict={} 
        ### iterate through constrs.
        for i, name in enumerate(new_constraints): 
            ext_dict[name] ={k:v[i] for k,v in constraint_props.items()} 
        return ext_dict
    else:
        return None
    
class JointEoSConstraint:
    def __init__(self, *constraints, eos_converter = None):
        self.constraints = self.initialise_constraints(constraints)

        if eos_converter is None:
            eos_converter = Namespace(macro_parameters={})
        self.eos_converter = eos_converter

    def __repr__(self):
        if len(self.constraints) == 1:
            return f"{self.constraints[0].__repr__()}"
        elif len(self.constraints) == 2:
            return f"{self.constraints[0].__repr__()} and {self.constraints[1].__repr__()}"
        else:
            return f"{self.__class__.__name__} of {', '.join([cons.__repr__() for cons in self.constraints[:-1]])} and {self.constraints[-1].__repr__()}"

    def initialise_constraints(self, constraint_tuple):
        constraint_list=[]
        for constraint in constraint_tuple:
            if isinstance(constraint, EoSConstraint):
                constraint_list.append(constraint)
            elif isinstance(constraint, JointEoSConstraint):
                constraint_list.extend(constraint.constraints)
            elif isinstance(constraint, dict):
                constraint_list.extend(self.initialise_from_dict(constraint))
        return constraint_list

    def initialise_from_dict(self, constraint_dict):
        constraint_list=[]
        for constraint_kind, sub_constraints in constraint_dict.items():
            if constraint_kind == 'lower_mtov':
                for label, constraint in sub_constraints.items():
                    constraint_list.append(LowerMTOVConstraint(
                        measured_mass=constraint['mass'],
                        measure_error=constraint.get('error',0.),
                        name=label,
                        arxiv_ref=constraint.get('arxiv', None)
                    ))
            elif constraint_kind == 'upper_mtov':
                for label, constraint in sub_constraints.items():
                    constraint_list.append(UpperMTOVConstraint(
                        measured_mass=constraint['mass'],
                        measure_error=constraint.get('error',0.),
                        name=label,
                        arxiv_ref=constraint.get('arxiv', None)
                    ))
            elif constraint_kind == 'mass_radius':
                for label, constraint in sub_constraints.items():
                    constraint_list.append(MassRadiusConstraint(
                        file_path=constraint.get('posterior', constraint.get('file_path', None)),
                        name=label,
                        arxiv_ref=constraint.get('arxiv', None)
                    ))
            else:
                raise ValueError('Unknown type of EoS Constraint. Must be "lower_mtov", \
                                "upper_mtov", "mass-radius" or "micro\
                                ')
        return constraint_list
        
    def parameter_conversion(self, parameters):
        return self.eos_converter.parameter_conversion(parameters)
    
    def log_likelihood(self, parameters):
        return sum([constraint.log_likelihood(parameters, self.eos_converter.macro_parameters)
                     for constraint in self.constraints])
    
    def tabulate_weighted_eos(self, parameters, outdir, weight_path=None, normalise=True):  
        """Given a directory of macroscopic EOSs and nmma.joint.Constraint,
        returns sorted EOSs and the corresponding prior weights
        
        Parameters
        ----------
        parameters: dict | str | int | float
            parameters for which the eos should be tabulated. If str, int or float, it is assumed to reweight all EOSs in the data directory
        outdir: str
            path to the directory where sorted EoSs should be stored
        weight_path: str | None
            If given, filename to save computed EOS weights. Default is None.
        normalise: Bool
            Whether to return normalised weights. Default is True"""

        file_path =os.path.join(outdir, 'sorted')
        if os.path.isdir(file_path) and os.path.isfile(os.path.join(outdir, "eos_weights.dat")):
            return os.path.join(outdir, "eos_weights.dat"), file_path, len(os.listdir(file_path))

        os.makedirs(file_path, exist_ok=True)
        if parameters is None:
            parameters = { 'EOS': np.arange(self.eos_converter.Neos)}
        if isinstance(parameters, (str, int, float)):
            parameters = { 'EOS': np.arange(int(parameters))}
        eos_data = self.eos_converter.macro_conversion(parameters)

        weight_data = process_map(self.eval_eos_data, eos_data, chunksize =5)

        log_weights = []
        good_data = []

        for i, weight in enumerate(weight_data):
            if weight is not None:
                log_weights.append(weight)
                good_data.append(eos_data[i])
        
        if isinstance(weight_path, str):
            try:
                previous_weights = np.loadtxt(weight_path)
                weight_path = os.path.join(outdir, "eos_weights.dat")
            except FileNotFoundError:
                previous_weights = np.ones_like(log_weights)
        else:
            previous_weights = np.ones_like(log_weights)
            weight_path = os.path.join(outdir, "eos_weights.dat")

        save_weights= np.array(log_weights)
        save_weights += np.log(previous_weights)
        if normalise:
            save_weights-= logsumexp(save_weights)
        save_weights=np.exp(save_weights)

        sort_idcs = np.argsort(save_weights)
        for i, idx in enumerate(sort_idcs):
            np.savetxt(os.path.join(file_path, f"{i+1}.dat"), np.column_stack(good_data[idx]))
        np.savetxt(weight_path, np.array(save_weights)[sort_idcs])
        return weight_path, file_path, len(good_data)

    def eval_eos_data(self, eos_data):
        try:
            R, M, _ = eos_data
            self.eos_converter.macro_parameters = {"radii": R, "masses": M}
            return self.log_likelihood({"TOV_mass": M[-1]})
        except ValueError:
            return None

class EoSConstraint:
    def __init__(self, name = None, arxiv_ref=None):
        self.repr_add = ''
        self.type = 'macro'
        if name is None:
            self.name = self.__class__.__name__
            self.base_repr = name 
        else:
            self.name = name
            self.base_repr = f'{self.__class__.__name__} based on {name}'
        self.arxiv_ref = arxiv_ref

    def __repr__(self):
        out = f'{self.base_repr} {self.repr_add}'
        if self.arxiv_ref:
            out = f'{out} (see arxiv:{self.arxiv_ref})'
        return out

class MassConstraint(EoSConstraint):
    def __init__(self, measured_mass, measure_error, name=None, arxiv_ref=None, lognorm_method=None):
        super().__init__(name, arxiv_ref)
        self.mass = measured_mass
        self.error = measure_error
        self.repr_add = f'of {measured_mass}+-{measure_error} M_sun'
        self.lognorm_method = lognorm_method
        self.linestyle = '--'

    
    def __repr__(self):
        out = f'{self.__class__.__name__} of {self.mass}+-{self.error} M_sun'
        if self.name != "Mass Constraint":
            out = f'{out} based on {self.name}'
        return  out
    
    def log_likelihood(self, parameters, local_parameters=None):
        tov_mass = parameters.get('TOV_mass', None)
        if tov_mass is None:
            if isinstance(local_parameters['masses'], list):
                tov_mass = [masses[-1] for masses in local_parameters['masses']]
            else:   
                tov_mass = local_parameters['masses'][-1]
        return self.lognorm_method(tov_mass, loc=self.mass, scale=self.error)
    
    def plot(self, ax, **kwargs):
        """Plot the mass constraint on the given figure."""
        x_lim = ax.get_xlim()
        dummy_line = ax.plot([], [], label=self.name, linestyle=self.linestyle, linewidth=2.5, **kwargs)
        line = ax.hlines(self.mass, *x_lim, linestyle=self.linestyle, linewidth=2.5, zorder=3, **kwargs)
        cmap = fading_cmap(dummy_line[0].get_color())
        levels = [0.95, 0.68]
        for i, level in enumerate(levels):
            ax.fill_between(x_lim, self.mass - (i+1)*self.error, self.mass + (i+1)*self.error, color=cmap(0.8*level), zorder=2-i)
            # ax.hlines(self.mass + (i+1)*self.error, *x_lim, color=cmap(0.9*level),  linewidth=1.5, zorder=1)
            # ax.hlines(self.mass - (i+1)*self.error, *x_lim, color=cmap(0.9*level), linewidth=1.5, zorder=1)
        return ax


class LowerMTOVConstraint(MassConstraint):
    '''Constraint that an EOS supports at least a certain TOV mass(within Gaussian uncertainty)'''
    def __init__(self, measured_mass, measure_error, name=None, arxiv_ref=None):
        """
        Parameters
        ----------
        measured_mass: float
            Observed mass (in solar masses)
        measure_error: float
            1-sigma uncertainty of the measurement
        name: str
            identifier of the measurement
        arxiv_ref: str
            Identifier of a relevant source
        """
        super().__init__(measured_mass, measure_error, name, arxiv_ref, lognorm_method=norm.logcdf)
    

class UpperMTOVConstraint(MassConstraint):
    '''Constraint that an EOS supports at most a certain TOV mass (within Gaussian uncertainty)'''
    def __init__(self, measured_mass, measure_error, name=None, arxiv_ref=None):
        """
        Parameters
        ----------
        measured_mass: float
            Observed mass (in solar masses)
        measure_error: float
            1-sigma uncertainty of the measurement
        name: str, optional
            identifier of the measurement
        arxiv_ref: str, optional
            Identifier of a relevant source
        """
        super().__init__(measured_mass, measure_error, name, arxiv_ref, lognorm_method=norm.logsf)
        self.linestyle = ':'
    

class MassRadiusConstraint(EoSConstraint):
    '''Constraint that an EOS adheres to  certain mass-radius region'''
    def __init__(self, mass_array=None, radius_array=None, weights = None, file_path=None, name=None, arxiv_ref=None):
        """
        Parameters
        ----------
        mass_array: np.array, optional
            Array (or similar) with mass posterior of M-R measurement, must be specified along an equal-length radius_array
        radius_array: np.array, optional
            Array (or similar) with radius posterior of M-R measurement, must be specified along an equal-length mass_array
        file_path: str, optional
            path to data_file that contains radius and mass posteriors. If provided, mass_array and radius_array are ignored
        name: str
            identifier of the measurement
        arxiv_ref: str
            Identifier of a relevant source
        """

        super().__init__(name, arxiv_ref)
        if file_path:
            mass_array, radius_array, weights = self.read_data(file_path)
        elif mass_array is None or radius_array is None:
            raise ValueError('Must provide data for masses and radii as arrays or file from which to load')
        self.set_grid(mass_array, radius_array, weights)
        self.test_masses = np.linspace(1.2, 2.5, 151) 
   
    def read_data(self, file_path):
        """Read mass-radius data from a file."""
        data = np.loadtxt(file_path, unpack=True)
        if data.shape[0] not in [2, 3]:
            data = data.T
        if data.shape[0] not in [2, 3]:
            raise ValueError("Data file must have two or three columns for mass, radius (, weights)")
        try:
            # if three columns it includes weights
            data_1, data_2, weights = data
        except ValueError:
            data_1, data_2 = data
            weights = None

        if (data_1 <=3.).any():
            # we assume radii in km and mass in solar masses. 3 km is an arbitrary-ish limit for neutron stars
            masses = data_1
            radius = data_2

        else:
            radius = data_1
            masses = data_2

        if not (masses > 0).all() and (masses < 5).all() and (radius > 3).all():
            min_mass = np.min(masses)
            max_mass = np.max(masses)
            median_mass = np.median(masses)
            min_radius = np.min(radius)
            max_radius = np.max(radius)
            median_radius = np.median(radius)
            raise ValueError("Failed to properly identify mass and radius. Masses should be in solar masses and radii in km, " \
            "but the identified values seem to fall outside reasonable ranges. " \
            f"Identified mass range: {min_mass:.2f} - {max_mass:.2f} (median: {median_mass:.2f}), " \
            f"Identified radius range: {min_radius:.2f} - {max_radius:.2f} (median: {median_radius:.2f}). " \
            "Please check the input data file format.")
        
        return masses, radius, weights

    def set_grid(self, masses, radii, weights, mass_step = 0.01, radius_step = 0.03):
        """Set up a grid upon which to build a histogram of mass-radius data to approximate the pdf.
        Note that when using multiple mass-radius measurements, all measurements should use the same stepsizes!
        Parameters
        ----------
        masses: np.array
            Array with mass posterior of M-R measurement
        radii: np.array
            Array with radius posterior of M-R measurement, must be specified along an equal-length mass_array
        weights: np.array, optional
            Array with weights of the M-R samples, must be specified along an equal-length mass_array
            mass_step: float
            step size for mass grid in solar masses, default is 0.01 Msun
        radius_step: float  
            step size for radius grid in km, default is 0.02 km (20 m)
        
        """
        mass_bins = self.set_bins(masses, mass_step)
        rad_bins = self.set_bins(radii, radius_step)
        if 3*len(mass_bins)*len(rad_bins) > len(masses):
            print("Warning: The histogram might be to sparsely populated to get meaningful results.")

        histogram, self.rad_edges, self.mass_edges = np.histogram2d(radii, masses, bins=[rad_bins, mass_bins], weights=weights, density=True)
        drad = self.rad_edges[1] - self.rad_edges[0]
        dmass = self.mass_edges[1] - self.mass_edges[0]

        self.histogram = gaussian_filter(histogram*dmass*drad, sigma=3)

    def set_bins(self, array, step_size, sensitivity=0.001):
        low, high = np.quantile(array, [sensitivity, 1.- sensitivity])
        bins = np.arange(0.95*low, 1.05*high, step_size, dtype=np.float64) 
        return bins
    
    def log_likelihood(self, parameters, local_parameters):
        try:
            tov_mass = parameters.get('TOV_mass', local_parameters['masses'][-1])
            return self.single_logl(tov_mass, local_parameters['masses'], local_parameters['radii'])
        except (ValueError, IndexError):
            self.single_logl(tov_mass, local_parameters['masses'], local_parameters['radii'])
            return [
                self.single_logl(masses[-1], masses, local_parameters['radii'][i])
                for i, masses in enumerate(local_parameters['masses'])
            ]

    def single_logl(self, tov_mass, masses, radii):
        ## interpolate radii along equally spaced mass grid up to MTov
        test_mass_range=self.test_masses[self.test_masses<tov_mass]
        test_radii=np.interp(test_mass_range, masses, radii)
        yi = np.searchsorted(self.mass_edges[1:], test_mass_range) -1
        xi = np.searchsorted(self.rad_edges[1:], test_radii) -1
        log_l = np.log(self.histogram[xi, yi].sum())
        
        return log_l
    def plot(self, ax, **kwargs):
        """Plot the mass-radius constraint on the given figure."""

        # for the legend
        dummy_line = ax.plot([], [], label=self.name, linewidth=2, **kwargs) 
        color = dummy_line[0].get_color()      
        Xc = 0.5 * (self.rad_edges[:-1] + self.rad_edges[1:])
        Yc = 0.5 * (self.mass_edges[:-1] + self.mass_edges[1:])

        flat = self.histogram.flatten()
        order = np.argsort(flat)[::-1]
        cumsum = np.cumsum(flat[order])
        levels = [0.95, 0.68]
        cmap = fading_cmap(color)
        colors = cmap(levels[::-1])
        plt.contour(
            Xc, Yc, self.histogram.T,
            levels=[flat[order][np.searchsorted(cumsum, p)] for p in levels], 
            colors = colors, linewidths=2
        )
        return ax
    
class PulsarConstraint(LowerMTOVConstraint):
    '''legacy synonym for general LowerMTOVConstraint'''
class MTOVUpperConstraint(UpperMTOVConstraint):
    '''legacy synonym for general UpperMTOVConstraint'''
class JointConstraint(JointEoSConstraint):
    '''legacy synonym for JointEoSConstraint'''


##### legacy routines to organise a collection of tabulated EoSs
def weights_for_tabulated_eos_from_constraints(macro_constraints=None, micro_constraints=None, macro_eos_path=None, micro_eos_path=None, eos_identifier='', save_path=None, normalise=True):
    """routine to obtain prior weights on pre-computed EOS

    Parameters
    ----------
    macro_constraints: iterable | None
        contains the Constraint objects for EoS constraints based on macroscopic NS observations, default is None
    micro_constraints: iterable | None
        contains the Constraint objects for EoS constraints based on macroscopic NS observations, default is None
    macro_eos_path: str | None
        path to directory containing the macroscopic eos data, default is None. Required if applying macro_constraints!
    micro_eos_path: str | None
        path to directory containing the microscopic eos data, default is None. Required if applying micro_constraints!
    eos_identifier: str
        string to identify eos-files in eos directory. Will be parsed with enclosing wildcards (f'*{eos_identifier}*'), default is an empty string.
    save_path: str, None
        If given, filename to save computed EOS weights. Default is None.
    normalise: Bool
        Whether to return normalised weights. Default is True


    Returns
    -------
    save_weights: ndarray
        A 1d-array with the normalised likelihoods of the tabulated EoS, to be used as prior weights for sampling
    eos_files: list
        Contains the eos files by which order the weights are returned

    """
    if micro_eos_path:
        log_weights=[]
        micro_eos_files=sorted(glob(f'{micro_eos_path}/*{eos_identifier}*'))
        for eos_file in micro_eos_files:
            log_weights.append(constraint_weight_from_micro_eos_file(eos_file, micro_constraints))
        log_weights=np.array(log_weights)

    if macro_eos_path:
        macro_log_weights=[]
        macro_eos_files=sorted(glob(f'{macro_eos_path}/*{eos_identifier}*'))
        for eos_file in macro_eos_files:
            macro_log_weights.append(constraint_weight_from_macro_eos_file(eos_file, macro_constraints))
        
        if micro_eos_path:
            eos_files= micro_eos_files
            try: 
                log_weights+=np.array(macro_log_weights)
            except ValueError: 
                raise ValueError(f'Your EoS directories contained unequal numbers of microscopic ({len(micro_eos_files)}) and macroscopic ({len(macro_eos_files)}) EoSs!')
        else:
            eos_files= macro_eos_files
            log_weights =np.array(macro_log_weights)

    save_weights= np.array(log_weights)
    if normalise:
        save_weights-= logsumexp(log_weights)
    save_weights=np.exp(save_weights)
    if save_path:
        np.savetxt(save_path, np.c_[save_weights])
    return save_weights, eos_files

def constraint_weight_from_micro_eos_file(filepath, micro_constraints):
    n, p, eps = np.loadtxt(filepath, usecols=[0,1,2], unpack=True)
    micro_params=dict(number_density=n, pressur=p, energy_density=eps)
    return eos_weight_from_constraints(micro_params, micro_constraints)

def constraint_weight_from_macro_eos_file(filepath, macro_constraints):
    R, M = np.loadtxt(filepath, usecols=[0,1], unpack=True)
    macro_params=dict(radii=R, masses=M, TOV_mass=M[-1])
    return eos_weight_from_constraints(macro_params, macro_constraints)

def eos_weight_from_constraints(eos_params, *constraints):
    logL_eos=0
    for constraint in constraints:
        logL_eos+=constraint.log_likelihood(eos_params)
    return logL_eos

def EOSSorting(eos_files, out_dir, sort_quantity_array):
    """routine to resort EoS files for more efficient sampling.
    Each EoS file will be saved as {integer}.dat

    Parameters
    ----------
    eos_files: list 
        list of eos_files, such that the ith entry of sort_quantity_array relates to the ith EoS in eos_files
    outdir: str | None
        path to directory where sorted EoSs should be stored
    sort_quantity_array: iterable 
        Contains the quantity by which EoS files should be sorted, typically prior probability or an associated observable (e.g. Lambda_1.4)
    """
    sort_quantity_array=np.atleast_1d(sort_quantity_array).argsort()
    sortIdcs = sort_quantity_array.argsort()
    for i, eos_file in enumerate(eos_files):
        sortedIdx = sortIdcs[i] + 1
        shutil.copy(f"{eos_file}.dat", f"{out_dir}/{sortedIdx}.dat")

def EOSConstraints2Prior(macro_eos_path, out_path, Constraint):
    """
    Given a directory of macroscopic EOSs and nmma.joint.Constraint,
    returns sorted EOSs and the corresponding prior

    Parameters
    ----------
    eos_path: str
        Path to the original EOSs
    out_path: str
        Path to the sorted EOSs
    Neos: int
        Number of EOSs
    Constraint: nmma.joint.Constraint
        The EOS constraint to be consider

    Returns
    -------
    prior: bilby.core.Prior
        Prior on the EOSs
    logNorm: float
        log normalization constant for the EOS prior
    """

    
    logLs, eos_files =weights_for_tabulated_eos_from_constraints(
        macro_constraints=Constraint, 
        macro_eos_path=macro_eos_path, 
        normalise=False
        )
    logNorm = logsumexp(logLs)

    weights = np.exp(logLs - logNorm)
    sorted_weights = np.sort(weights)
    EOSSorting(eos_files, out_path, weights)
    prior = WeightedCategorical(len(eos_files), sorted_weights, name='EOS')

    return prior, logNorm

