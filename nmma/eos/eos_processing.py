import numpy as np
from glob import glob
import os
import json
import joblib
from ast import literal_eval
import keras as k

def setup_eos_generator(args):
    try:
        with open(args.emulator_metadata, 'r') as f:
            meta_dict = json.load(f)
    except TypeError:
        meta_dict = args.emulator_metadata
    except FileNotFoundError:
        meta_dict = literal_eval(args.emulator_metadata)

    eos_model_type = args.micro_eos_model.lower()
    
    if eos_model_type == 'nep':
        return NEPEoSGenerator(meta_dict)
    elif eos_model_type == 'nep-5':
        return NEP5EoSGenerator(meta_dict)
    
    elif eos_model_type == 'lec':
        return LECEoSGenerator(meta_dict)
    elif eos_model_type == 'lec-7':
        return LEC7EoSGenerator(meta_dict)
    ## add more models
    else:
        raise ValueError(f"Unknown eos model type: {eos_model_type}")

class EoSGenerator(object):
    def __init__(self, emulator_path, eos_parameters=None):
        
        # load the emulator
        try:
            self.emulator =  k.saving.load_model(emulator_path, custom_objects=None, compile= False)

            #NOTE: in the main samling loop, the tensorflow-predict method is much slower
            #  than pure __call__ as it tries to cater to larger batches!
            # We therefore need to set the predict method in 
            # accordance with the appropriate backend
            if k.backend.backend() == 'tensorflow':
                self.predict = self.tensorflow_predict
            elif k.backend.backend() == 'jax':
                self.predict = self.jax_predict
        except:
            import pickle
            with open(emulator_path, 'rb') as f:
                self.emulator =  pickle.load(f)
            self.predict = self.pickle_predict

            

        ## set the parameter-keys to be passed to the emulator
        if eos_parameters is None:
            self.eos_parameters = self.identify_eos_parameters()
        else:
            self.eos_parameters = eos_parameters

    
    def pickle_predict(self, x):
        return self.emulator.predict(x)
    
    def jax_predict(self, x):
        return self.emulator.predict(x, verbose = 0)

    def tensorflow_predict(self, x):
        return self.emulator(x)


    
    def generate_macro_eos(self, converted_parameters):
        predictions = self.emulate_macro_eos(converted_parameters)
        return self.adjust_format(predictions)

    def emulate_macro_eos(self, converted_parameters):
        eos_params = self.assemble_eos_params(converted_parameters)
        return self.predict(eos_params) 
    
    def assemble_eos_params(self, converted_parameters):
        """Assemble the parameters for the EoS model into a 2D-array for the emulator"""
        eos_params = np.array([converted_parameters[par] for par in self.eos_parameters]).T
        return np.atleast_2d(eos_params)

    def adjust_format(self, predictions):
        """Adjust the format of the predictions to the expected format: A n-tuple of three 1-D arrays for radius, mass lambdas, respectively"""
        #This should be implemented in the subclass
        return predictions

    def identify_eos_parameters(self):
        """Identify the parameters for the EoS model"""
        #This should be implemented in the subclass
        return None

class NEPEoSGenerator(EoSGenerator):
    def __init__(self, metadata):

        emulator_path = metadata['emulator_path']
        if metadata.get('backend', False):
            assert k.backend.backend() == metadata['backend'], f"Keras Backend mismatch: {k.backend.backend()} vs {metadata['backend']}. please set the environment variable KERAS_BACKEND to {metadata['backend']}"
        super().__init__(emulator_path, metadata.get('eos_parameters', None))

        n_mass_samples = metadata.get('n_mass_samples', 40)
        self.set_mass_construction(n_mass_samples)  
    
    def set_mass_construction(self, n_mass_samples):
        """Set the mass construction"""
        if isinstance(n_mass_samples, int):
            # if this is a single integer, use equally spaced masses
            self.n_mass_samples = n_mass_samples
            self.decompose_mass_data = self.equal_distance_masses
        elif isinstance(n_mass_samples, (tuple, list)):
            # iterable containing mass points for fixed-distance lower end, variably spaced upper end and optionally mass value at which these methods will be concatenated; if not given, the default is 2.0
            try:
                self.mass_samples_low, self.mass_samples_high, self.split_value = n_mass_samples
            except ValueError:
                self.mass_samples_low, self.mass_samples_high = n_mass_samples
                self.split_value= 2.0
            self.n_mass_samples = self.mass_samples_low + self.mass_samples_high
            self.decompose_mass_data = self.disjoint_masses

    def disjoint_masses(self, mtov):
        """ Helper function when using split mass construction. 
        Some predicted TOV-masses may be lower than the split value 
        for concatenation and would lead to unexpected behaviour. 
        In that case we fall back to equally spaced mass arrays. 
        However, this should not happen for EoSs with a physically 
        reasonable TOV-mass and should only be seen as a graceful fallback.
        """
        return np.where(mtov > self.split_value, self.properly_disjoint_masses(mtov), self.equal_distance_masses(mtov))
    
    def equal_distance_masses(self, mtov):
        "Get mass array(s) from 1 to mtov of length n_mass_samples"
        mass_range= np.linspace(1, mtov, self.n_mass_samples, axis =-1)
        try: 
            mass_range = np.squeeze(mass_range, axis=1)
        except ValueError:
            pass
        return mass_range
    
    def properly_disjoint_masses(self, mtov):
        mass_range_low = np.linspace(1, self.split_value*np.ones_like(mtov),self.mass_samples_low, axis=-1)
        mass_range_high= np.linspace(mtov, self.split_value, self.mass_samples_high, endpoint=False, axis=-1)
        mass_range_high= mass_range_high[..., ::-1]
        mass_range= np.concatenate([mass_range_low, mass_range_high], axis=-1)  
        try: 
            mass_range = np.squeeze(mass_range, axis=1)
        except ValueError:
            pass
        return mass_range
    
  
    def adjust_format(self, predictions):
        rad_data, lam_data, mtov_data = np.split(predictions, [self.n_mass_samples, 2*self.n_mass_samples], axis=-1)
        mass_range = self.decompose_mass_data(mtov_data)

        lam_data = 10**lam_data
        
        return np.stack([rad_data, mass_range, lam_data], axis=1)
        

class NEP5EoSGenerator(NEPEoSGenerator):
    def identify_eos_parameters(self):
        return ['K_sat', 'L_sym', 'K_sym', '3n_sat', '5n_sat']

class LECEoSGenerator(EoSGenerator):
    def __init__(self, metadata):
        self.feature_scaler = joblib.load(metadata['feature_scaler'])
        self.lambda_scaler  = joblib.load(metadata['lambda_scaler'])
        self.radius_scaler  = joblib.load(metadata['radius_scaler'])

        self.mass_emulator  = joblib.load(metadata['mass_emulator'])
        self.radius_emulator= joblib.load(metadata['radius_emulator'])
        self.lambda_emulator= joblib.load(metadata['lambda_emulator'])

        self.n_mass_samples = metadata.get('n_mass_samples', 30)
        self.eos_parameters = self.identify_eos_parameters()
        
    def predict(self, converted_parameters):
        # Scale the input features
        scaled_features = self.feature_scaler.transform(converted_parameters)

        # Make predictions using the emulators
        mass_prediction = self.mass_emulator.predict(scaled_features)
        radius_prediction = self.radius_emulator.predict(scaled_features)
        lambda_prediction = self.lambda_emulator.predict(scaled_features)

        # Inverse transform the predictions
        radius_prediction = self.radius_scaler.inverse_transform(radius_prediction)
        lambda_prediction = self.lambda_scaler.inverse_transform(lambda_prediction)

        return (mass_prediction, radius_prediction, lambda_prediction)
    

    def assemble_eos_params(self, converted_parameters):
        """Assemble the parameters for the EoS model into a 2D-array for the emulator"""
        eos_params = np.array([
            converted_parameters[par] + converted_parameters.get(f"{par}_shift", 0)
            for par in self.eos_parameters
        ]).T
        return np.atleast_2d(eos_params)
    
    def equal_distance_masses(self, mtov):
        "Get mass array(s) from 1 to mtov of length n_mass_samples"
        mass_range= np.linspace(1, mtov, self.n_mass_samples, axis =-1)
        try: 
            mass_range = np.squeeze(mass_range, axis=1)
        except ValueError:
            pass
        return mass_range
    
    def adjust_format(self, predictions):
        mass_data, rad_data, lam_data = predictions
        return np.stack([rad_data, mass_data, 10**lam_data], axis=1)
    
class LEC7EoSGenerator(LECEoSGenerator):
    def identify_eos_parameters(self):
        return ['d11', 'd22', 'd3', 'd4', 'd6', 'd7']
    
    

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


#FIXME this should be used by conversion!
def load_tabulated_macro_eos_set_to_list(eos_data, weights=None, Neos=None):
    eos_files, Neos = load_eos_files(eos_data, Neos)
    weights = load_weights(weights)
    
    EOS_data = [np.loadtxt(eos_file, usecols=[1,0, 2], unpack=True) for eos_file in eos_files]
    
    return EOS_data, weights, Neos