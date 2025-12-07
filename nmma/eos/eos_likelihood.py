from __future__ import division
from glob import glob
from argparse import Namespace
import os
import shutil
import json
from ast import literal_eval
from tqdm.contrib.concurrent import process_map 
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, gaussian_kde
from matplotlib import pyplot as plt
from bilby.core.prior import WeightedCategorical, PriorDict
from ..joint.base import NMMABaseLikelihood
from ..joint.utils import fading_cmap
from ..joint.conversion import EoSConverter
    
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
   
class EquationofStateLikelihood(NMMABaseLikelihood):
    def __init__(self, priors, constraint_dict, eos_converter, **kwargs):
        constraint =setup_joint_eos_constraint(constraint_dict, eos_converter)
        # TODO: to be extended for more complex likelihood expressions
        super().__init__(constraint, priors, **kwargs)

    def setup_submodel_conversion(self):
        self.conv_functions.append(self.sub_model.parameter_conversion)
        
     
    def final_diagnostics(self, bestfit_params, args, result=None):
        bestfit_params =self.parameter_conversion(bestfit_params)
        
        color_densities = []
        radii, masses, lambdas = self.sub_model.eos_converter.macro_parameters.values()
        x_lim = (np.min(radii)-0.3, np.max(radii)+0.3)
        y_lim = (masses[0], masses[-1]+0.1)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        for constraint in self.sub_model.constraints:
            color = ax._get_lines.get_next_color()
            ax = constraint.plot(ax=ax, resolution=100, color=color)
        ax.plot(radii, masses, label='Best fit EOS', zorder=10)

        ax.set_xlabel(r'Radius [km]')
        ax.set_ylabel(r'Mass [M$_\odot$]')
        ax.legend()
        fig.savefig(os.path.join(args.outdir, f"{args.label}_mr_curve.png"))
        
        plt.show()


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
    
def setup_joint_eos_constraint(constraint_dict, eos_converter=None):
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
                    file_path=constraint.get('posterior', None),
                    name=label,
                    arxiv_ref=constraint.get('arxiv', None)
                ))
        else:
            raise ValueError('Unknown type of EoS Constraint. Must be "lower_mtov", \
                             "upper_mtov", "mass-radius" or "micro\
                             ')
        
        if eos_converter is None:
            eos_converter = Namespace(macro_parameters={})

    return JointEoSConstraint(*constraint_list, eos_converter=eos_converter)

class JointEoSConstraint:
    def __init__(self, *constraints, eos_converter = None):
        self.constraints = constraints
        self.eos_converter = eos_converter

    def __repr__(self):
        if len(self.constraints) == 1:
            return f"{self.constraints[0].__repr__()}"
        elif len(self.constraints) == 2:
            return f"{self.constraints[0].__repr__()} and {self.constraints[1].__repr__()}"
        else:
            return f"{self.__class__.__name__} of {', '.join([cons.__repr__() for cons in self.constraints[:-1]])} and {self.constraints[-1].__repr__()}"

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

class MassConstraint:
    def __init__(self, measured_mass, measure_error, name=None, arxiv_ref=None, lognorm_method=None):
        self.mass = measured_mass
        self.error = measure_error
        self.type = 'macro'
        self.name= name if name else "Mass Constraint"
        self.arxiv_ref = arxiv_ref if arxiv_ref else None
        self.lognorm_method = lognorm_method

    
    def __repr__(self):
        out = f'{self.__class__.__name__} of {self.mass}+-{self.error} M_sun'
        if self.name != "Mass Constraint":
            out = f'{out} based on {self.name}'
        if self.arxiv_ref:
            out = f'{out} (see arxiv:{self.arxiv_ref})'
        return  out
    
    def log_likelihood(self, parameters, local_parameters=None):
        tov_mass = parameters.get('TOV_mass', None)
        if tov_mass is None:
            if isinstance(local_parameters['masses'], list):
                tov_mass = [masses[-1] for masses in local_parameters['masses']]
            else:   
                tov_mass = local_parameters['masses'][-1]
        return self.lognorm_method(tov_mass, loc=self.mass, scale=self.error)
    
    def plot(self, ax, resolution = 100, **kwargs):
        """Plot the mass constraint on the given figure."""
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        M_grid = np.linspace(*y_lim, num=resolution)
        show_x, show_y = np.meshgrid(np.linspace(*x_lim, resolution), M_grid)
        line = ax.hlines(self.mass, *x_lim, linestyle='--', linewidth=1.5, zorder=3, label=self.name, **kwargs)
        cmap = fading_cmap(line.get_color())
        shade_profile = norm.pdf(M_grid, loc=self.mass, scale=self.error)
        shading_matrix = np.repeat(shade_profile[:, np.newaxis], resolution, axis=1)
        ax.contourf(show_x, show_y, shading_matrix, levels=50, cmap=cmap)
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
    

class MassRadiusConstraint:
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

        self.type = 'macro'

        if file_path:
            mass_array, radius_array, weights = self.read_data(file_path)
        elif mass_array is None or radius_array is None:
            raise ValueError('Must provide data for masses and radii as arrays or file from which to load')

        if len(radius_array) > 10000:
            ratio = len(radius_array) // 10000
        else:
            ratio = 1
            
        radius = radius_array[::ratio]
        mass = mass_array[::ratio]
        if weights is not None:
            weights = weights[::ratio]
        self.KDE = gaussian_kde((radius, mass), weights=weights)

        self.test_masses= np.linspace(start=1., stop=2.5, num=150 ) # 1 to 2.5 Msun

        self.name = name if name else "Mass-Radius Constraint"
        self.arxiv_ref = arxiv_ref if arxiv_ref else None

        self.rng = np.random.default_rng()

    
    def __repr__(self):
        out = self.__class__.__name__
        if self.name:
            out = f'{out} based on {self.name}'
        if self.arxiv_ref:
            out = f'{out} (see arxiv:{self.arxiv_ref})'
        return  out

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

    def log_likelihood(self, parameters, local_parameters):
        
        try:
            tov_mass = parameters.get('TOV_mass', local_parameters['masses'][-1])
            return self.single_logl(tov_mass, local_parameters['masses'], local_parameters['radii'])
        except (ValueError, IndexError):
            return [
                self.single_logl(masses[-1], masses, local_parameters['radii'][i])
                for i, masses in enumerate(local_parameters['masses'])
            ]

    def single_logl(self, tov_mass, masses, radii):
        ## interpolate radii along equally spaced mass grid up to MTov
        test_mass_range=self.test_masses[self.test_masses<tov_mass]
        test_radii=np.interp(test_mass_range, masses, radii)
        return logsumexp(self.KDE.pdf((test_radii, test_mass_range)))

    def plot(self, ax, resolution = 100, **kwargs):
        """Plot the mass-radius constraint on the given figure."""
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

        # for the legend
        dummy_line = ax.plot([], [], label=self.name, linewidth=2, **kwargs) 
        color = dummy_line[0].get_color()

        show_x, show_y = np.meshgrid(
            np.linspace(*x_lim, resolution), np.linspace(*y_lim, resolution))
        # test_data = np.column_stack((show_x.flatten(), show_y.flatten()))
        test_data = np.stack((show_x.flatten(), show_y.flatten()))
        test_scores = self.KDE.pdf(test_data)
        test_scores = np.exp(test_scores).reshape(resolution, resolution)

        # levels = np.percentile(test_scores, [68, 95, 99.7])   
        levels = 50  
        # ax.contour(show_x, show_y, test_scores, levels=[level_90], colors=[color], linewidths=2) 
        cmap = fading_cmap(color)
        ax.contourf(show_x, show_y, 10*test_scores, levels=levels, cmap = cmap)

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

