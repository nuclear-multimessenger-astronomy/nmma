## file to store constants used throughout the code. We try to default to the standards in astropy.constants

from astropy import constants as const
from astropy import units as u
from astropy import cosmology

# helpers
mc2 = const.M_sun*const.c**2
G_per_c2 = const.G/const.c**2
default_cosmology = cosmology.Planck18


## fundamental constants
msun_cgs = const.M_sun.cgs.value
c_cgs = const.c.cgs.value
c_SI = const.c.si.value
c_kms = c_SI / 1000.
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
