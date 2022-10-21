import os
import numpy as np
import scipy.special
import scipy.interpolate
import scipy.stats
from bilby.core.prior import Interped
from tqdm import tqdm
from operator import add


class PulsarConstraint(object):

    def __init__(self, pulsar_masses, pulsar_masses_error, EOSpath, Neos):
        self.pulsar_masses = pulsar_masses              # list of pulsar masses
        self.pulsar_masses_error = pulsar_masses_error  # list of errors of pulsar masses
        self.EOSpath = EOSpath                          # path of EOS data files
        self.Neos = Neos                                # number of EOS files
        if len(self.pulsar_masses) != len(self.pulsar_masses_error):
            raise ValueError('Number of masses and mass errors are inconsistent')

    def __repr__(self):
        return self.__class__.__name__ + ' with {} pulsars of masses {} and errors of {}'\
            .format(len(self.pulsar_masses), self.pulsar_masses, self.pulsar_masses_error)

    def pulsar_logweight(self, pulsar_masses, pulsar_masses_error, EOSpath, Neos):
        
        logweight = []
        for EOSIdx in range(1,Neos+1):
            EOSdata = '{0}/{1}.dat'.format(EOSpath, EOSIdx)
            R, M = np.loadtxt(EOSdata, usecols=[0,1], unpack=True)
            mMax = M[-1]

            logl = 0.
            for i in range(len(pulsar_masses)):
                mass = pulsar_masses[i]
                error = pulsar_masses_error[i]
                logl += scipy.stats.norm.logcdf(mMax, loc=mass, scale=error)
            logweight.append(logl)

        logweight -= scipy.special.logsumexp(logweight)
        pulsar_weight = np.exp(logweight)
        np.savetxt('pulsar_weight.dat', np.c_[pulsar_weight])
        return pulsar_weight

class MTOVUpperConstraint(object):

    def __init__(self, maxMTOV, maxMTOV_error, EOSpath, Neos):
        self.maxMTOV = maxMTOV                  # list of maximum TOV masses
        self.maxMTOV_error = maxMTOV_error      # list of maximum TOV mass errors
        self.EOSpath = EOSpath                  # path of EOS data files
        self.Neos = Neos                        # number of EOS files

    def __repr__(self):
        return self.__class__.__name__ + ' maxTOV of {} and error of {}'\
            .format(self.maxMTOV, self.maxMTOV_error)

    def maxTOVmass_logweight(self, maxMTOV, maxMTOV_error, EOSpath, Neos):
        
        logweight = []
        for EOSIdx in range(1,Neos+1):
            EOSdata = '{0}/{1}.dat'.format(EOSpath, EOSIdx)
            R, M = np.loadtxt(EOSdata, usecols=[0,1], unpack=True)
            mMax = M[-1]

            logl = 0.
            for i in range(len(maxMTOV)):
                mass = maxMTOV[i]
                error = maxMTOV_error[i]
                logl += scipy.stats.norm.logsf(mMax, loc=mass, scale=error)
            logweight.append(logl)
        
        logweight -= scipy.special.logsumexp(logweight)
        maxTOVmass_weight = np.exp(logweight)
        np.savetxt('maxTOVmass_weight.dat', np.c_[maxTOVmass_weight])

        return maxTOVmass_weight

class NICERConstraint(object):

    def __init__(self, NICER_path, EOSpath, Neos):
        self.NICER_path = NICER_path            # path with NICER data files
        self.EOSpath = EOSpath                  # path of EOS data files
        self.Neos = Neos                        # number of EOS files

    def __repr__(self):
        return self.__class__.__name__ + ' with the observational NICER data taken from {}'\
            .format(self.NICER_path)
   
    def NICER_logweight(self, NICER_path, EOSpath, Neos):
        
        #NICER I 
        NICER_radius, NICER_mass = np.loadtxt('{0}/J0030_3spot_RM.txt'.format(NICER_path), skiprows=1, usecols=[0, 1], unpack=True) 
        NICER_radius = NICER_radius[::100]
        NICER_mass = NICER_mass[::100]
        NICER_KDE = scipy.stats.gaussian_kde((NICER_radius,NICER_mass))

        #NICER II
        NICER_radius, NICER_mass = np.loadtxt('{0}/NICER_x_XMM_J0740_XPSI_STU_NSX_FIH_radius_mass.txt'.format(NICER_path), skiprows=1, usecols=[0, 1], unpack=True) 
        NICER_radius = NICER_radius[::42]
        NICER_mass = NICER_mass[::42]
        NICER_KDE_XPSI = scipy.stats.gaussian_kde((NICER_radius,NICER_mass))

        NICER_radius, NICER_mass = np.loadtxt('{0}/NICER+XMM_J0740_RM.txt'.format(NICER_path), skiprows=1, usecols=[0, 1], unpack=True) 
        NICER_radius = NICER_radius[::160]
        NICER_mass = NICER_mass[::160]
        NICER_KDE_Maryland = scipy.stats.gaussian_kde((NICER_radius,NICER_mass))

        logweight = []

        for EOSIdx in tqdm(range(1, Neos + 1)):
            EOSdata = '{0}/{1}.dat'.format(EOSpath, EOSIdx)
            R, M = np.loadtxt(EOSdata, usecols=[0,1], unpack=True)
            mMax = M[-1]
    
            # NICER I
            R_M_interp = scipy.interpolate.interp1d(M, R, kind='linear')
            Mrange = np.linspace(1., 2.0, num=100)
            Mrange = Mrange[Mrange < mMax]
            NICER_logpdf = scipy.special.logsumexp(NICER_KDE.logpdf((R_M_interp(Mrange),Mrange)))

            # NICER II
            Mrange = np.linspace(1.5, 2.5, num=100)
            Mrange = Mrange[Mrange < mMax]
            NICER_XPSI_logpdf = scipy.special.logsumexp(NICER_KDE_XPSI.logpdf((R_M_interp(Mrange),Mrange)))
            NICER_Maryland_logpdf = scipy.special.logsumexp(NICER_KDE_Maryland.logpdf((R_M_interp(Mrange),Mrange)))
    
            NICERII_logpdf = scipy.special.logsumexp([NICER_XPSI_logpdf, NICER_Maryland_logpdf])  # adding the likelihood from XPSI and Maryland (therefore, an average)
    
            logweight.append(NICER_logpdf + NICERII_logpdf)   

        logweight -= scipy.special.logsumexp(logweight)
        NICER_weight = np.exp(logweight)
        np.savetxt('NICER_weight.dat', np.c_[NICER_weight])

        return NICER_weight


class JointConstraint(object):

    def __init__(self, EOSpath, Neos, pulsar_masses, pulsar_masses_error, pulsar_weight, maxMTOV, maxMTOV_error, maxMTOV_weight, NICER_path, NICER_weight):
        
        self.EOSpath = EOSpath                          # path of EOS data files
        self.Neos = Neos                                # number of EOS files
        self.pulsar_masses = pulsar_masses              # list of pulsar masses
        self.pulsar_masses_error = pulsar_masses_error  # list of errors of pulsar masses
        self.pulsar_weight = pulsar_weight              # pulsar weights computed with JointConstraint.pulsar_logweight()
        self.maxMTOV = maxMTOV                          # list of maximum TOV masses
        self.maxMTOV_error = maxMTOV_error              # list of maximum TOV mass errors
        self.maxMTOV_weight = maxMTOV_weight            # maximum TOV mass weights computed with JointConstraint.maxTOVmass_logweight()
        self.NICER_path = NICER_path                    # path with NICER data files
        self.NICER_weight = NICER_weight                # NICER weights computed with JointConstraint.NICER_logweight()
    
    def __repr__(self):
        return self.__class__.__name__ + ' for pulsar, maximum TOV mass and NICER weights computation. NICER data taken from {}'\
            .format(self.NICER_path)
    

    def pulsar_logweight(self, pulsar_masses, pulsar_masses_error, EOSpath, Neos):
        
        logweight = []
        for EOSIdx in range(1,Neos+1):
            EOSdata = '{0}/{1}.dat'.format(EOSpath, EOSIdx)
            R, M = np.loadtxt(EOSdata, usecols=[0,1], unpack=True)
            mMax = M[-1]

            logl = 0.
            for i in range(len(pulsar_masses)):
                mass = pulsar_masses[i]
                error = pulsar_masses_error[i]
                logl += scipy.stats.norm.logcdf(mMax, loc=mass, scale=error)
            logweight.append(logl)

        return logweight

    def maxTOVmass_logweight(self, maxMTOV, maxMTOV_error, EOSpath, Neos):
        
        logweight = []
        for EOSIdx in range(1,Neos+1):
            EOSdata = '{0}/{1}.dat'.format(EOSpath, EOSIdx)
            R, M = np.loadtxt(EOSdata, usecols=[0,1], unpack=True)
            mMax = M[-1]

            logl = 0.
            for i in range(len(maxMTOV)):
                mass = maxMTOV[i]
                error = maxMTOV_error[i]
                logl += scipy.stats.norm.logsf(mMax, loc=mass, scale=error)
            logweight.append(logl)

        return logweight

    def NICER_logweight(self, NICER_path, EOSpath, Neos):
        
        #NICER I 
        NICER_radius, NICER_mass = np.loadtxt('{0}/J0030_3spot_RM.txt'.format(NICER_path), skiprows=1, usecols=[0, 1], unpack=True) 
        NICER_radius = NICER_radius[::100]
        NICER_mass = NICER_mass[::100]
        NICER_KDE = scipy.stats.gaussian_kde((NICER_radius,NICER_mass))

        #NICER II
        NICER_radius, NICER_mass = np.loadtxt('{0}/NICER_x_XMM_J0740_XPSI_STU_NSX_FIH_radius_mass.txt'.format(NICER_path), skiprows=1, usecols=[0, 1], unpack=True) 
        NICER_radius = NICER_radius[::42]
        NICER_mass = NICER_mass[::42]
        NICER_KDE_XPSI = scipy.stats.gaussian_kde((NICER_radius,NICER_mass))

        NICER_radius, NICER_mass = np.loadtxt('{0}/NICER+XMM_J0740_RM.txt'.format(NICER_path), skiprows=1, usecols=[0, 1], unpack=True) 
        NICER_radius = NICER_radius[::160]
        NICER_mass = NICER_mass[::160]
        NICER_KDE_Maryland = scipy.stats.gaussian_kde((NICER_radius,NICER_mass))

        logweight = []

        for EOSIdx in tqdm(range(1, Neos + 1)):
            EOSdata = '{0}/{1}.dat'.format(EOSpath, EOSIdx)
            R, M = np.loadtxt(EOSdata, usecols=[0,1], unpack=True)
            mMax = M[-1]
    
            # NICER I
            R_M_interp = scipy.interpolate.interp1d(M, R, kind='linear')
            Mrange = np.linspace(1., 2.0, num=100)
            Mrange = Mrange[Mrange < mMax]
            NICER_logpdf = scipy.special.logsumexp(NICER_KDE.logpdf((R_M_interp(Mrange),Mrange)))

            # NICER II
            Mrange = np.linspace(1.5, 2.5, num=100)
            Mrange = Mrange[Mrange < mMax]
            NICER_XPSI_logpdf = scipy.special.logsumexp(NICER_KDE_XPSI.logpdf((R_M_interp(Mrange),Mrange)))
            NICER_Maryland_logpdf = scipy.special.logsumexp(NICER_KDE_Maryland.logpdf((R_M_interp(Mrange),Mrange)))
    
            NICERII_logpdf = scipy.special.logsumexp([NICER_XPSI_logpdf, NICER_Maryland_logpdf])  # adding the likelihood from XPSI and Maryland (therefore, an average)
    
            logweight.append(NICER_logpdf + NICERII_logpdf)   

        return logweight

    def total_logweight(self, EOSpath, Neos, pulsar_weight, maxMTOV_weight, NICER_weight):
        #eos weighting with all information: pulsar, mtov, NICER
        total_weight = list(map(add, pulsar_weight, maxMTOV_weight))
        total_weight = list(map(add, total_weight, NICER_weight))
        total_weight -= scipy.special.logsumexp(total_weight)
        weight = np.exp(total_weight)
        np.savetxt('joint_weight.dat', np.c_[weight])
        
        return weight

    def pulsar_mtov_weight(self, EOSpath, Neos, pulsar_weight, maxMTOV_weight):
        # eos weitghting with pulsar and mtov information, no NICER weighting
        total_weight = list(map(add, pulsar_weight, maxMTOV_weight))
        total_weight -= scipy.special.logsumexp(total_weight)
        weight = np.exp(total_weight)
        np.savetxt('pulsar_mtov_weight.dat', np.c_[weight])
        
        return weight

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
