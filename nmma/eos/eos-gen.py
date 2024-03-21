import numpy as np
import scipy

def eos_from_nep (S0_val, L_val, nsat_val = 0.16, 
	Esat_val = -16.0, Ksat_val = 220.0, Qsat_val = 0.0, Zsat_val = 0.0,
	Ksym_val = -100.0, Qsym_val = 0.0, Zsym_val = 0.0,
	crust_path="BPS.dat"):
	
	# Load crust EOS
	# will load an array with n, p, eps
	crust_EOS = np.loadtxt(crust_path)

	# Define general parameters
	m_neutron = 939.565 #in MeV
	xval      = 0.02    #change this later!!!

	# Define remaining empirical parameters
	# Symmetric matter:
	nsat = nsat_val
	Esat = Esat_val
	Ksat = Ksat_val
	Qsat = Qsat_val
	Zsat = Zsat_val

	#Symmetry energy:
	Ssym = S0_val
	Lsym = L_val 
	Ksym = Ksym_val
	Qsym = Qsym_val
	Zsym = Zsym_val

	# Energy/Particle for symmetric nuclear matter
	def EA_SNM (n):
		xexp = (n-nsat)/(3.*nsat)
		return(Esat + Ksat * xexp**2/2. + Qsat * xexp**3/6. + Zsat * xexp**4/24.)

	# Symmetry energy
	def EA_sym (n):
		xexp = (n-nsat)/(3.*nsat)
		return(Ssym + Lsym * xexp + Ksym * xexp**2/2. + Qsym * xexp**3/6. + Zsym * xexp**4/24.)

	# Symmetry energy
	def EA_beta (n, x):
		return(EA_SNM (n) + EA_sym (n) * (1-2.*x))

	# Generate outer-core EOS
	# will make an array with n, p, eps
	n_values       = np.arange(0.1, 1.6, 0.002)
	EOS_array      = np.zeros((len(n_values), 3))
	EOS_array[:,0] = n_values # n
	EOS_array[:,2] = n_values*(m_neutron + EA_beta (n_values, xval)) # eps

	# function E/A(n)
	EA_beta_inter        = scipy.interpolate.UnivariateSpline(n_values, EA_beta (n_values, xval), k=3)
	EA_beta_inter_derive = EA_beta_inter.derivative()
	EOS_array[:,1]       = n_values**2 * EA_beta_inter_derive(n_values)

	return (np.concatenate((crust_EOS,EOS_array)))

#Just for testing
print(eos_from_nep (32, 60))