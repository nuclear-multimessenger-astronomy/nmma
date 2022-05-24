import numpy as np
import pandas as pd
import bilby

# load posterior file, e.g. from https://zenodo.org/record/6106130#.YoysIHVBwUG 
eos_post = np.loadtxt('./posterior_probability_files/Astro/15nsat_cse_uniform_R14/posterior_probability.txt')

npts = 150000
Neos = 5000
nparams = 3

############# [mass1,    mass2,   DL]
params_low =  [1.001398, 1.001398, 1]
params_high = [2.2,      2.2,     75]

# 1) create dummy EOS samples with eos_post from nature paper
EOS_raw = np.arange(0, Neos)  # the gwem_resampling will add one to this
EOS_samples = np.random.choice(EOS_raw, p=eos_post, size=npts, replace=True)

# 2) generate samples for masses and distance
mass_1 = np.random.uniform(params_low[0], params_high[0], size=npts)
mass_2 = np.random.uniform(params_low[1], params_high[1], size=npts)

mass_1, mass_2 = np.maximum(mass_1, mass_2), np.minimum(mass_1, mass_2)
mass_ratio = mass_2 / mass_1  # mass ratio q < 1 convention is used
chirp_mass = bilby.gw.conversion.component_masses_to_chirp_mass(mass_1, mass_2)

lum_distance = np.random.uniform(params_low[2], params_high[2], size=npts)

# 3) create pandas dataframe
dataset = pd.DataFrame({'mass_1': mass_1, 'mass_2': mass_2, 'chirp_mass': chirp_mass, 'mass_ratio': mass_ratio, 'luminosity_distance': lum_distance, 'EOS': EOS_samples})

# 4) save GWsamples.dat file
dataset.to_csv('GWsamples.dat', index=False, sep=' ')
