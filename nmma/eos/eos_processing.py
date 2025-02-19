import numpy as np
from glob import glob
import os
import json
from ast import literal_eval

def setup_eos_generator(args):
    eos_model_type = args.micro_eos_model.lower()
    if eos_model_type == 'nep-5':
        return NEP5EoSGenerator(args.emulator_metadata)
    ## add more models
    else:
        raise ValueError(f"Unknown eos model type: {eos_model_type}")

class EoSGenerator(object):
    def __init__(self, emulator_path, backend ='jax', eos_parameters=None):
        
        # load the emulator
        try:
            os.environ["KERAS_BACKEND"] = backend
            import keras as k
            if k.backend.backend() != backend:
                ## reimport keras with the correct backend
                import sys
                keys = list(sys.modules.keys())
                for key in keys:
                    if 'keras' in key:
                        del sys.modules[key]        
                print('KERAS_BACKEND reset to ', os.environ['KERAS_BACKEND'])
                import keras as k
            self.emulator =  k.saving.load_model(emulator_path, custom_objects=None, compile= False)

            # formatting function to convert the input parameters to the expected format
            self.format_conv= k.ops.convert_to_tensor
        
        except:
            import pickle
            with open(emulator_path, 'rb') as f:
                self.emulator =  pickle.load(f)
            self.format_conv = np.array
            

        ## set the parameter-keys to be passed to the emulator
        if eos_parameters is None:
            self.eos_parameters = self.identify_eos_parameters()
        else:
            self.eos_parameters = eos_parameters

    def identify_eos_parameters(self):
        """Identify the parameters for the EoS model"""
        #This should be implemented in the subclass
        return None

    def emulate_macro_eos(self, converted_parameters):
        eos_params = np.array([converted_parameters[par] for par in self.eos_parameters]).T
        eos_params = np.atleast_2d(eos_params)
        eos_params = self.format_conv(eos_params)
        return self.emulator.predict(eos_params, verbose=0) 

    
    def adjust_format(self, predictions):
        """Adjust the format of the predictions to the expected format: A n-tuple of three 1-D arrays for radius, mass lambdas, respectively"""
        #This should be implemented in the subclass
        return predictions
    
    def generate_macro_eos(self, converted_parameters):
        predictions = self.emulate_macro_eos(converted_parameters)
        return self.adjust_format(predictions)

    
class NEP5EoSGenerator(EoSGenerator):
    def __init__(self, metadata):
        try:
            with open(metadata, 'r') as f:
                meta_dict = json.load(f)
        except TypeError:
            meta_dict = metadata
        except FileNotFoundError:
            meta_dict = literal_eval(metadata)

        emulator_path = meta_dict['emulator_path']
        backend = meta_dict.get('backend', 'jax')
        super().__init__(emulator_path, backend, meta_dict.get('eos_parameters', None))

        n_mass_samples = meta_dict.get('n_mass_samples', 40)
        self.set_mass_construction(n_mass_samples)

        
    def equal_distance_masses(self, mtov):
        return np.linspace(1, mtov, self.n_mass_samples, axis =-1)
    def disjoint_masses(self, mtov):
        mass_range_low = np.linspace(1,2.*np.ones_like(mtov),self.mass_samples_low, axis=-1)
        mass_range_high = np.linspace(mtov, 2., self.mass_samples_high, endpoint=False, axis=-1)
        mass_range_high = mass_range_high[..., ::-1]
        return np.concatenate([mass_range_low, mass_range_high], axis=-1)  

    def set_mass_construction(self, n_mass_samples):
        if isinstance(n_mass_samples, int):
            self.n_mass_samples = n_mass_samples
            self.decompose_mass_data = self.equal_distance_masses
        elif isinstance(n_mass_samples, (tuple, list)):
            self.mass_samples_low, self.mass_samples_high = n_mass_samples
            self.n_mass_samples = self.mass_samples_low + self.mass_samples_high
            self.decompose_mass_data = self.disjoint_masses

    def identify_eos_parameters(self):
        return ['K_sat', 'L_sym', 'K_sym', '3n_sat', '5n_sat']
    
    def adjust_format(self, predictions):
        rad_data, lam_data, mtov_data = np.split(predictions, [self.n_mass_samples, 2*self.n_mass_samples], axis=-1)
        mass_range = np.squeeze(self.decompose_mass_data(mtov_data), axis=0)
        lam_data = 10**lam_data
        
        return np.stack([rad_data, mass_range, lam_data], axis=1)
        


def load_eos_files(eos_data, Neos):
    if isinstance(eos_data, str):
        eos_data = sorted(glob(os.path.join(eos_data, "*.dat")))
    if Neos:
        assert len(eos_data) ==Neos, f"{eos_data} does not contain {Neos} eos-files"
    else:
        Neos = len(eos_data)
    return eos_data, Neos


def load_weights(weights):
    if isinstance(weights, str):
        return np.loadtxt(weights)
    else:
        return weights

def load_macro_characteristics_from_tabulated_eos_set(eos_data, Neos, masses_for_char_radii= None, masses_for_char_lambdas=None):
    """utility to get MTOV and other characteristic properties for a set of tabulated EOS
    -----------
    Parameters:

    eos_data: str or list of strings
        if string: path to directory with eos_files, else list of eos_files to be read
    Neos: int
        Number of equations of state to consider
    masses_for_char_radii: int or tuple-like, default: None
        mass(es) at which characteristic radii should be evaluated
    masses_for_char_lambdas: int or tuple-like, default: None
        mass(es) at which characteristic tidal deformabilities should be evaluated

    --------
    Returns:
        output: list
            A list containing a 1d-array with the TOV-masses and optionally arrays 
            with the characteristic radii and tidal deformabilities.
    """
    ####SETUP
    do_rads= False
    do_lams=False
    mtovs = np.empty(Neos)
    if masses_for_char_radii is not None:
        do_rads=True
        masses_for_char_radii = np.atleast_1d(masses_for_char_radii)
        radii = np.empty_like(Neos, len(masses_for_char_radii))
    if masses_for_char_lambdas is not None:
        do_lams=True
        masses_for_char_lambdas = np.atleast_1d(masses_for_char_lambdas)
        lambdas = np.empty_like(Neos, len(masses_for_char_lambdas))
    eos_data, Neos = load_eos_files(eos_data, Neos)

    ### Main Loop
    for i, eos_file in enumerate(eos_data):
        m, r, lam = np.loadtxt(eos_file, usecols=[1, 0, 2], unpack=True)
        mtovs[i] = m[-1]
        if do_rads:
            radii[i] = np.interp(masses_for_char_radii, m, r)
        if do_lams:
            lambdas[i] = np.interp(masses_for_char_lambdas, m, lam)

    output = [mtovs]
    if do_rads:
        output.append(np.squeeze(radii))
    if do_lams:
        output.append(np.squeeze(lambdas))



    
def load_tabulated_macro_eos_set_to_dict(eos_data, weights=None, Neos=None):
    eos_files, Neos = load_eos_files(eos_data, Neos)
    weights = load_weights(weights)
    
    EOS_data = {}
    for EOSIdx, eos_file in enumerate(eos_files):
        m, r, lam  = np.loadtxt(eos_file, usecols=[1,0, 2], unpack=True)
        EOS_data[EOSIdx+1] = {"R": r, "M": m, "Lambda": lam}
        if weights is not None:
            EOS_data[EOSIdx+1]["weight"] = weights[EOSIdx]
    
    return EOS_data, weights, Neos