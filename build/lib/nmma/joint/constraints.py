from __future__ import division
import os

import numpy as np
import scipy.stats
import scipy.special


class PulsarConstraint(object):

    def __init__(self, pulsar_masses, pulsar_masses_error):
        self.pulsar_masses = pulsar_masses
        self.pulsar_masses_error = pulsar_masses_error
        if len(self.pulsar_masses) != len(self.pulsar_masses_error):
            raise ValueError('Number of masses and mass errors are inconsistent')

    def __repr__(self):
        return self.__class__.__name__ + ' with {} pulsars of masses {} and errors of {}'\
            .format(len(self.pulsar_masses), self.pulsar_masses, self.pulsar_masses_error)

    def log_likelihood(self, parameters):
        MTOV = parameters['TOV_mass']
        logl = 0
        for mass, error in zip(self.pulsar_masses, self.pulsar_masses_error):
            logl += scipy.stats.norm.logcdf(MTOV, loc=mass, scale=error)
        return logl


class MTOVUpperConstraint(object):

    def __init__(self, maxMTOV, maxMTOV_error):
        self.maxMTOV = maxMTOV
        self.maxMTOV_error = maxMTOV_error

    def __repr__(self):
        return self.__class__.__name__ + ' maxTOV of {} and error of {}'\
            .format(self.maxMTOV, self.maxMTOV_error)

    def log_likelihood(self, parameters):
        MTOV = parameters['TOV_mass']
        logl = scipy.stats.norm.logsf(MTOV, loc=self.maxMTOV, scale=self.maxMTOV_error)
        return logl


class NICERConstraint(object):

    def __init__(self, NICER_path=None):

        if NICER_path is None:
            self.NICER_path = os.path.join(os.path.dirname(__file__), 'NICER/J0030_3spot_RM.txt')
        else:
            self.NICER_path = NICER_path

        radius, mass = np.loadtxt(self.NICER_path, usecols=[0, 1], unpack=True)
        if len(radius) > 10000:
            ratio = len(radius) // 10000
        else:
            ratio = 1
        self.radius = radius[::ratio]
        self.mass = mass[::ratio]
        self.KDE = scipy.stats.gaussian_kde((self.radius, self.mass))

    def __repr__(self):
        return self.__class__.__name__ + ' with the observation of PSR J0030+0451 ' +\
            'with three potentially overlapping ovals. Data taken from {}'\
            .format(self.NICER_path)

    def log_likelihood(self, parameters):

        interp = parameters['interp_mass_radius']
        masses = np.linspace(1., 2., num=1000)
        masses = masses[masses < interp.x[-1]]

        logl = scipy.special.logsumexp(self.KDE.logpdf((interp(masses), masses)))
        logl -= np.log(len(masses))

        return logl


class JointConstraint(object):

    def __init__(self, *constraints):
        self.constraints = constraints

    def log_likelihood(self, parameters):
        logl = 0.

        for constraint in self.constraints:
            logl += constraint.log_likelihood(parameters)

        return logl
