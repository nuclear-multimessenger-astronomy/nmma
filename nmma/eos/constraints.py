from __future__ import division
from glob import glob
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, gaussian_kde

class JointEoSConstraint(object):
    def __init__(self, *constraints):
        self.constraints = constraints

    def __repr__(self):
        if len(self.constraints) == 1:
            return f"{self.constraints[0].__repr__()}"
        else:
            return f"{self.__class__.__name__} of {', '.join(map(str, self.constraints[:-1].__repr__()))} and {self.constraints[-1].__repr__()}"

    def log_likelihood(self, parameters):
        logl = 0.

        for constraint in self.constraints:
            logl += constraint.log_likelihood(parameters)

        return logl

class MassConstraint(object):
    def __init__(self, measured_mass, measure_error, name=None, arxiv_ref=None):
        self.mass = measured_mass
        self.error = measure_error
        self.type = 'macro'
        if name:
            self.name=name
        if arxiv_ref:
            self.arxiv_ref= arxiv_ref

    
    def __repr__(self):
        out = f'{self.__class__.__name__} of {self.mass}+-{self.error} M_sun'
        if self.name:
            out = f'{out} based on {self.name}'
        if self.arxiv_ref:
            out = f'{out} (see arxiv:{self.arxiv_ref})'
        return  out
    
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
        super().__init__(measured_mass, measure_error, name, arxiv_ref)
    

    def log_likelihood(self, parameters):
        return norm.logcdf(parameters['TOV_mass'], loc=self.mass, scale=self.error)

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
        super().__init__(measured_mass, measure_error, name, arxiv_ref)
    

    def log_likelihood(self, parameters):
        return norm.logsf(parameters['TOV_mass'], loc=self.mass, scale=self.error)

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
        
        self.KDE = gaussian_kde((self.radius_estimate, self.mass_estimate))
        self.test_masses= np.linspace(start=1., stop=2.5, num=150 )
        if name:
            self.name=name
        if arxiv_ref:
            self.arxiv_ref= arxiv_ref

    
    def __repr__(self):
        out = self.__class__.__name__
        if self.name:
            out = f'{out} based on {self.name}'
        if self.arxiv_ref:
            out = f'{out} (see arxiv:{self.arxiv_ref})'
        return  out

    def log_likelihood(self, parameters):
        ## interpolate radii along equally spaced mass grid up to MTov
        test_mass_range=self.test_masses[self.test_masses<parameters['TOV_mass']]
        test_radii=np.interp(test_mass_range, parameters['masses'], parameters['radii'])
        return logsumexp(self.KDE(test_radii, test_mass_range))

class PulsarConstraint(LowerMTOVConstraint):
    '''legacy synonym for general LowerMTOVConstraint'''
class MTOVUpperConstraint(UpperMTOVConstraint):
    '''legacy synonym for general UpperMTOVConstraint'''
class JointConstraint(JointEoSConstraint):
    '''legacy synonym for JointEoSConstraint'''


def setup_joint_eos_constraint(constraint_dict):
    constraint_list=[]
    for constraint_kind, sub_constraints in constraint_dict.items():
        if constraint_kind == 'lower_mtov':
            for label, constraint in sub_constraints:
                constraint_list.append(LowerMTOVConstraint(
                    measured_mass=constraint['mass'],
                    measure_error=constraint.get('error',0.),
                    name=label,
                    arxiv_ref=constraint.get('arxiv', None)
                ))
        elif constraint_kind == 'upper_mtov':
            for label, constraint in sub_constraints:
                constraint_list.append(UpperMTOVConstraint(
                    measured_mass=constraint['mass'],
                    measure_error=constraint.get('error',0.),
                    name=label,
                    arxiv_ref=constraint.get('arxiv', None)
                ))
        elif constraint_kind == 'mass_radius':
            for label, constraint in sub_constraints:
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

    from bilby.core.prior import Interped
    
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