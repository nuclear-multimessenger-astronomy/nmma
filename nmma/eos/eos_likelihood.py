from __future__ import division
from glob import glob
from ..joint.base import NMMABaseLikelihood, initialisation_args_from_signature_and_namespace
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, gaussian_kde
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
from bilby.core.prior import Interped, Categorical
from bilby_pipe.utils import convert_string_to_dict, convert_string_to_list


def read_constraint_from_args(args, constraint_kind):
    "Routine to read prepare constraint in expected dict-format from argparse namespace during generation process"
    ##preferred: Have the dict with the subconstraints already set up
    prep_dict= getattr(args, constraint_kind, None) 
    if prep_dict is not None:
        return convert_string_to_dict(prep_dict)
    
    ###otherwise try to construct it:
        ### read in provided attributes like constraint_kind-name, -mass, -error, -arxiv
    prel_dict= {
        key.removeprefix(constraint_kind+'_'): ##cut identifier
                convert_string_to_list(getattr(args, key)) ## read in list or float
                for key in dir(args)            ## search args for attrs 
                if key.startswith(constraint_kind+'_') ## related with kind
    } 
    new_constraints = prel_dict.pop('name', None)
    if new_constraints is not None: ## there needs to be a unique label
        ext_dict={} 
        ### iterate through constrs.
        for i, name in enumerate(new_constraints): 
            ext_dict[name] ={k:v[i] for k,v in prel_dict.items()} 
        return ext_dict

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


def setup_eos_kwargs(data_dump, args, logger):
    # default_eos_kwargs = initialisation_args_from_signature_and_namespace(EquationofStateLikelihood, args)
    # eos_kwargs = default_eos_kwargs | dict(
    eos_kwargs = dict(
        constraint_dict=data_dump['eos_constraint_dict'],
        # crust_path=args.eos_crust_file
    )
    return eos_kwargs

    

def setup_joint_eos_constraint(constraint_dict):
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
                constraint_list.append(MassRadiusConstraint(mass_array=constraint.get('masses', None),
                    radius_array= constraint.get('radii', None),
                    file_path=constraint.get('file_path', None),
                    name=label,
                    arxiv_ref=constraint.get('arxiv', None)
                ))
        else:
            raise ValueError('Unknown type of EoS Constraint. Must be "lower_mtov", \
                             "upper_mtov", "mass-radius" or "micro\
                             ')
    return JointEoSConstraint(*constraint_list)

        
class EquationofStateLikelihood(NMMABaseLikelihood):
    def __init__(self, priors, constraint_dict, **kwargs):
        sub_model =setup_joint_eos_constraint(constraint_dict)
        # to be extended for more complex likelihood expressions
        super().__init__(sub_model=sub_model, priors=priors, **kwargs)
      

class JointEoSConstraint(object):
    def __init__(self, *constraints):
        self.constraints = constraints
        self.local_parameters = {}

    def __repr__(self):
        if len(self.constraints) == 1:
            return f"{self.constraints[0].__repr__()}"
        elif len(self.constraints) == 2:
            return f"{self.constraints[0].__repr__()} and {self.constraints[1].__repr__()}"
        else:
            return f"{self.__class__.__name__} of {', '.join([cons.__repr__() for cons in self.constraints[:-1]])} and {self.constraints[-1].__repr__()}"

    def log_likelihood(self):
        logl = 0.

        for constraint in self.constraints:
            logl += constraint.log_likelihood(self.parameters, self.local_parameters)

        return np.squeeze(logl)
    
    def log_micro(self):
        ###FIXME!!!
        '''
        Routine to evaluate microphysical constraints on the EoS
        '''
        return 1
    

    def log_macro(self):
        ###FIXME!!!
        '''
        Routine to evaluate microphysical constraints on the EoS
        '''
        return 1
    

class MassConstraint(object):
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
        return self.lognorm_method(parameters['TOV_mass'], loc=self.mass, scale=self.error)
    
    def plot(self, ax, resolution = 100, **kwargs):
        """Plot the mass constraint on the given figure."""
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        line = ax.hlines(self.mass, x_min, x_max, linestyle='--', linewidth=1.5, label=self.name)
        cmap = LinearSegmentedColormap.from_list("custom_cmap", ["white", line.get_color()])
        M_grid = np.linspace(y_min, y_max,num=resolution)
        shade_profile = norm.pdf(M_grid, loc=self.mass, scale=self.error)
        shading_matrix = np.repeat(shade_profile[:, np.newaxis], resolution, axis=1)
        ax.imshow(shading_matrix, aspect='auto', cmap=cmap, alpha=0.5,
                  extent=[x_min, x_max, y_min, y_max], origin='lower')
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
    

class MassRadiusConstraint(object):
    '''Constraint that an EOS adheres to  certain mass-radius region'''
    def __init__(self, mass_array=None, radius_array=None, file_path=None, name=None, arxiv_ref=None):
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
            radius, mass = np.loadtxt(file_path, usecols=[0, 1], unpack=True)
            if len(radius) > 10000:
                ratio = len(radius) // 10000
            else:
                ratio = 1
            self.radius_estimate = radius[::ratio]
            self.mass_estimate = mass[::ratio]
        elif mass_array is not None and radius_array is not None:
            self.mass_estimate = mass_array
            self.radius_estimate = radius_array
        else:
            raise ValueError('Must provide data for masses and radii as arrays or file from which to load')
        
        # self.KDE = gaussian_kde((self.radius_estimate, self.mass_estimate))
        self.KDE = KernelDensity(kernel='gaussian', bandwidth="scott").fit((self.radius_estimate, self.mass_estimate))
        self.test_masses= np.linspace(start=1., stop=2.5, num=150 ) # 1 to 2.5 Msun

        self.name = name if name else "Mass-Radius Constraint"
        if arxiv_ref:
            self.arxiv_ref= arxiv_ref

        self.rng = np.random.default_rng()

    
    def __repr__(self):
        out = self.__class__.__name__
        if self.name:
            out = f'{out} based on {self.name}'
        if self.arxiv_ref:
            out = f'{out} (see arxiv:{self.arxiv_ref})'
        return  out

    def log_likelihood(self, parameters, local_parameters):
        ## interpolate radii along equally spaced mass grid up to MTov
        test_mass_range=self.test_masses[self.test_masses<parameters['TOV_mass']]
        test_radii=np.interp(test_mass_range, local_parameters['masses'], local_parameters['radii'])
        # return logsumexp(self.KDE((test_radii, test_mass_range)))
        return logsumexp(self.KDE.score_samples((test_radii, test_mass_range)))
    
    def plot(self, ax, resolution = 100, **kwargs):
        """Plot the mass-radius constraint on the given figure."""
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        show_x, show_y = np.meshgrid(
            np.linspace(*x_lim, resolution), np.linspace(*y_lim, resolution))
        test_data = np.column_stack((show_x.flatten(), show_y.flatten()))
        test_scores = self.KDE.score_samples(test_data)
        test_scores = np.exp(test_scores).reshape(resolution, resolution)
        level_90 = np.percentile(test_scores, 90)
        
        # just for legend and color cycle
        dummy_line = ax.plot([], [], label=self.name, linewidth=2) 
        colour = dummy_line[0].get_color()
        cmap = LinearSegmentedColormap.from_list("custom_cmap", ["white",colour])

        ax.contour(show_x, show_y, test_scores, levels=[level_90], colors=[colour], linewidths=2, label = "Contour")  
        ax.imshow(test_scores, aspect= 'auto', cmap = cmap, extent = [*x_lim, *y_lim], origin = 'lower')

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
    import os
    sort_quantity_array=np.atleast_1d(sort_quantity_array).argsort()
    sortIdcs = sort_quantity_array.argsort()
    for i, eos_file in enumerate(eos_files):
        sortedIdx = sortIdcs[i] + 1
        os.system(f"cp {eos_file}.dat {out_dir}/{sortedIdx}.dat")

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
    Neos= len(eos_files) 
    xx = np.arange(0, Neos + 1)
    yy = np.concatenate((sorted_weights, sorted_weights[-1]))
    prior = Interped(xx, yy, minimum=0, maximum=Neos, name='EOS')

    return prior, logNorm

def setup_tabulated_eos_priors(args, priors, logger=None):
    if logger:    
        logger.info("Sampling over precomputed EOSs")
    if args.eos_weight:
        xx = np.arange(0, args.Neos + 1)
        eos_weight = np.loadtxt(args.eos_weight)
        yy = np.concatenate((eos_weight, [eos_weight[-1]]))
        priors["EOS"] = Interped(xx, yy, minimum=0, maximum=args.Neos, name="EOS")
    else: 
        priors["EOS"] = Categorical(args.Neos, name="EOS")
    return priors

