import os
import numpy as np
import scipy.special
import scipy.interpolate
from bilby.core.prior import Interped


def EOSConstraintsLoglikelihood(eos_path, Neos, Constraint):

    parameters = {}
    logLs = []

    for eos in range(1, Neos + 1):
        radius, mass, Lambda = np.loadtxt('{0}/{1}.dat'.format(eos_path, eos),
                                          unpack=True, usecols=[0, 1, 2])
        interp_mass_lambda = scipy.interpolate.interp1d(mass, Lambda)
        interp_mass_radius = scipy.interpolate.interp1d(mass, radius)

        parameters['TOV_mass'] = mass[-1]
        parameters['TOV_radius'] = radius[-1]
        parameters['interp_mass_radius'] = interp_mass_radius
        parameters['interp_mass_lambda'] = interp_mass_lambda

        logLs.append(Constraint.log_likelihood(parameters))

    logLs = np.array(logLs)

    return logLs


def EOSSorting(eos_path, out_path, Neos, weights):

    sortIdx = weights.argsort()

    for sortedIdx in range(0, Neos):
        originalIdx = sortIdx[sortedIdx] + 1
        os.system("cp {0}/{1}.dat {2}/{3}.dat".format(eos_path, originalIdx, out_path, sortedIdx + 1))

    return


def EOSConstraints2Prior(eos_path, out_path, Neos, Constraint):
    """
    Given a directory of EOSs and nmma.joint.Constraint,
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

    logLs = EOSConstraintsLoglikelihood(eos_path, Neos, Constraint)
    logNorm = scipy.special.logsumexp(logLs)

    weights = np.exp(logLs - logNorm)
    sorted_weights = np.sort(weights)

    EOSSorting(eos_path, out_path, Neos, weights)

    xx = np.arange(0, Neos + 1)
    yy = np.concatenate((sorted_weights, sorted_weights[-1]))
    prior = Interped(xx, yy, minimum=0, maximum=Neos, name='EOS')

    return prior, logNorm
