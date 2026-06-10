## file to store constants used throughout the code. We try to default to the standards in astropy.constants

from astropy import constants as const
from astropy import units as u
from astropy import cosmology
from bilby.gw import cosmology as bilby_cosmo

# helpers
mc2 = const.M_sun*const.c**2
G_per_c2 = const.G/const.c**2
seconds_a_day = 24*3600


## fundamental constants
msun_cgs = const.M_sun.cgs.value
c_cgs = const.c.cgs.value
c_SI = const.c.si.value
c_kms = c_SI / 1000.
G_in_ns_units = (const.G).to(u.km**3 /u.solMass/ u.second**2).value
h = const.h.cgs.value
kb = const.k_B.cgs.value

## distance 
Mpc = const.pc.cgs.value * 1e6
D = 10 * const.pc.cgs.value  # ref distance for absolute magnitude

# simple conversion factors
sigSB = const.sigma_sb.cgs.value
arad = 4 * sigSB / c_cgs
eV_per_h_SI = const.e.si.value / const.h.si.value

## solar references
particle_mass = const.m_p/const.M_sun  #get proton mass in Msun
geom_msun_km = (const.M_sun*G_per_c2).to(u.km).value # geometrised Msun is 1.476625038050125 km
msun_to_ergs = mc2.cgs.value
MeV_per_fm3_to_Msun_per_km3 = 1e54/((mc2).to(u.MeV).value) # 1 MeV/fm**3 is 8.9653E-7 Msun/km**3

## pulsar timing constants
msun_s = (const.M_sun * const.G / const.c**3).value # geometrised Msun in seconds
msun_mus = msun_s * 1e6 # Msun in microseconds, used for pulsar timing
einstein_factor = (const.G*const.M_sun/const.c**3).value**(2/3)

default_cosmology = cosmology.Planck18
def set_cosmology(cosmology_input=None):
    """Set the cosmology for the NMMA package.

    Parameters
    ----------
    cosmology_input: astropy.cosmology.Cosmology or str
        The cosmology to be used. If a string is provided, it should correspond to a valid astropy cosmology name.
        Default is astropy's Planck18 cosmology.
    """
    global COSMOLOGY
    if cosmology_input is None:
        cosmology_input = default_cosmology
    bilby_cosmo.set_cosmology(cosmology_input)

    COSMOLOGY = bilby_cosmo.DEFAULT_COSMOLOGY
    return COSMOLOGY

def get_cosmology():
    """Get the current cosmology used in the NMMA package.

    Returns
    -------
    astropy.cosmology.Cosmology
        The current cosmology.
    """
    return COSMOLOGY

global COSMOLOGY
COSMOLOGY = set_cosmology()