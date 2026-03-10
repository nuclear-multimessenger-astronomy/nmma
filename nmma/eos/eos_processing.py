import numpy as np
from glob import glob
import os
import shutil
import json
import joblib
from ast import literal_eval
import keras as k
from ..core.conversion import radii_from_qur, EOS_to_ns_parameters, EOS_to_system_parameters

def setup_eos_generator(args):
    if isinstance(args, dict):
        meta_dict = args
        eos_model_type = meta_dict['micro_eos_model'].lower()
    else:
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
    elif eos_model_type == 'lec-13':
        return LEC13EoSGenerator(meta_dict)
    ## add more models
    else:
        raise ValueError(f"Unknown eos model type: {eos_model_type}")

class EoSGenerator:
    eos_parameters = None
    def __init__(self, emulator_path, eos_parameters=None, n_mass_samples = 30):
        
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
        if eos_parameters:
            self.eos_parameters = eos_parameters
        
        self.set_mass_construction(n_mass_samples) 

    
    def pickle_predict(self, x):
        return self.emulator.predict(x)
    
    def jax_predict(self, x):
        return self.emulator.predict(x, verbose = 0)

    def tensorflow_predict(self, x):
        return self.emulator(x)

    
    def set_mass_construction(self, n_mass_samples):
            
        self.n_mass_samples = n_mass_samples
        self.decompose_mass_data = self.equal_distance_masses

    def equal_distance_masses(self, mtov):
        "Get mass array(s) from 1 to mtov of length n_mass_samples"
        mass_range= np.linspace(1, mtov, self.n_mass_samples, axis =-1)
        try: 
            mass_range = np.squeeze(mass_range, axis=1)
        except ValueError:
            pass
        return mass_range


    
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
    eos_parameters = ['K_sat', 'L_sym', 'K_sym', '3n_sat', '5n_sat']

class LECEoSGenerator(EoSGenerator):
    def __init__(self, metadata):
        self.feature_scaler = joblib.load(metadata['feature_scaler'])
        self.lambda_scaler  = joblib.load(metadata['lambda_scaler'])
        self.radius_scaler  = joblib.load(metadata['radius_scaler'])

        self.mass_emulator  = joblib.load(metadata['mass_emulator'])
        self.radius_emulator= joblib.load(metadata['radius_emulator'])
        self.lambda_emulator= joblib.load(metadata['lambda_emulator'])

        self.set_mass_construction(metadata.get('n_mass_samples', 30))
        
    def predict(self, converted_parameters):
        # Scale the input features
        scaled_features = self.feature_scaler.transform(converted_parameters)

        # Make predictions using the emulators
        mass_prediction = self.mass_emulator.predict(scaled_features)
        radius_prediction = self.radius_emulator.predict(scaled_features)
        lambda_prediction = self.lambda_emulator.predict(scaled_features)


        return (mass_prediction, radius_prediction, lambda_prediction)
    

    # def assemble_eos_params(self, converted_parameters):
    #     """Assemble the parameters for the EoS model into a 2D-array for the emulator"""
    #     eos_params = np.array([
    #         converted_parameters[par] + converted_parameters.get(f"{par}_shift", 0)
    #         for par in self.eos_parameters
    #     ]).T
    #     return np.atleast_2d(eos_params)
    
    
    def adjust_format(self, predictions):
        mass_data, rad_data, lam_data = predictions

        # Inverse transform the predictions
        mass_array = self.decompose_mass_data(mass_data)
        radius_array = self.radius_scaler.inverse_transform(rad_data)
        lambda_array = self.lambda_scaler.inverse_transform(lam_data)
        return np.stack([radius_array, mass_array, 10**lambda_array], axis=1)
    
class LEC7EoSGenerator(LECEoSGenerator):
    eos_parameters = ['d11', 'd22', 'd3', 'd4', 'd6', 'd7']

class LEC13EoSGenerator(LECEoSGenerator):
    eos_parameters = ['d11', 'd22', 'd3', 'd4', 'd6', 'd7', 
                      'ksat','qsat', 'zsat', 'cssq1', 'cssq2', 'cssq3', 'cssq4']
    
class EoSConverter:
    def __init__(self, args, method=None):
        if method is None:
            if getattr(args, 'eos_file', None) or getattr(args, 'eos_data', None):
                method = "tabulated"
            elif getattr(args, 'emulator_metadata', None):
                method = "emulated"

        self.parameter_conversion = self.full_eos_conversion
        # Case 1: eos is generated from emulator on the fly
        if method == "emulated":
            self.tov_emulator = setup_eos_generator(args)
            self.macro_conversion = self.tov_emulator.generate_macro_eos
        
        elif method == "tabulated":
            #case 2: we use a single eos
            if getattr(args, 'eos_file', None):
                self.eos_data = [np.loadtxt(args.eos_file, usecols = [0,1,2]).T]
                self.macro_conversion = self.single_eos_from_ram
                return
            
            # case 3 : we use multiple eos
            if os.path.isdir(args.eos_data):
                if getattr(args, 'Neos', None) is None:
                    eos_files  = [f"{args.eos_data}/{f}" for f in os.listdir(args.eos_data)]
                else:
                    eos_files = [f"{args.eos_data}/{j+1}.dat" for j in range(args.Neos)]
            else:
                eos_files = glob(args.eos_data)
                if getattr(args, 'Neos', None):
                    assert args.Neos == len(eos_files), 'Number of EOS files found does not match Neos'
              
            self.Neos = len(eos_files)  
            # Case 3a: precomputed eos data is loaded to ram
            if args.eos_to_ram:
                self.eos_data = [np.loadtxt(f, usecols = [0,1,2]).T  for f in eos_files]
                self.macro_conversion = self.eos_from_ram

            # Case 3b: eos are loaded directly from file
            else:
                eos_dir = os.path.dirname(eos_files[0])
                for i, f in enumerate(eos_files):
                    if not os.path.samefile(f, os.path.join(eos_dir, f"{i+1}.dat")):   
                        shutil.copy(f, os.path.join(eos_dir, f"{i+1}.dat"))
                self.eos_data = eos_dir
                self.macro_conversion = self.eos_direct_load

        #case 4: no eos conversion, just QURs
        elif method == "qur":
            self.parameter_conversion = radii_from_qur
        else:
            raise ValueError(f"Unknown EoS conversion method: {method}")
        
    def __call__(self, parameters):
        return self.parameter_conversion(parameters)


    def eos_direct_load(self, converted_parameters):
        EOSID = np.atleast_1d(converted_parameters["EOS"]).astype(int)
        return [np.loadtxt(f"{self.eos_data}/{j+1}.dat", usecols = [0,1,2]).T for j in EOSID]

    def eos_from_ram(self, converted_parameters):
        EOSID = np.atleast_1d(converted_parameters["EOS"]).astype(int)
        return [self.eos_data[i] for i in EOSID]
    
    def single_eos_from_ram(self, _):
        return self.eos_data

    def full_eos_conversion(self, parameters):
        parameters =self.compute_macro_parameters(parameters)
        return self.system_props_from_eos(parameters) 
    
    def compute_macro_parameters(self, parameters):
        eos_macro_keys = ["TOV_mass", "TOV_radius", "R_14", "R_16"]
        eos_data = self.macro_conversion(parameters)

        if len(eos_data) ==1:
            radii, masses, lambdas = eos_data[0]
            for key, val in zip(eos_macro_keys, 
            EOS_to_ns_parameters(radii, masses, lambdas)
            ):
                parameters[key] = val
        else:
            radii, masses, lambdas = map(list, zip(*eos_data))
            TOV_mass_list, TOV_radius_list, R_14_list, R_16_list = [], [], [], []
            for rad, mass, lam in zip(radii, masses, lambdas):
                TOV_mass, TOV_radius, R_14, R_16 = EOS_to_ns_parameters(rad, mass, lam)
                TOV_mass_list.append(TOV_mass)
                TOV_radius_list.append(TOV_radius)
                R_14_list.append(R_14)
                R_16_list.append(R_16) 
            for key, _list in zip(eos_macro_keys, [
                TOV_mass_list, TOV_radius_list, R_14_list, R_16_list
            ]):
                parameters[key] = np.array(_list)
        
        self.macro_parameters = {'radii': radii, 'masses': masses, 'lambdas': lambdas}
        return parameters
    
    def system_props_from_eos(self, converted_parameters):
        system_keys = ["lambda_1", "lambda_2", "radius_1", "radius_2"]
        
        m1_source = converted_parameters["mass_1_source"]
        m2_source = converted_parameters["mass_2_source"]
        radii, masses, lambdas = self.macro_parameters.values()

        if isinstance(radii, np.ndarray): # single eos case
            for key, val_array in zip(system_keys, 
            EOS_to_system_parameters(radii, masses, lambdas, m1_source, m2_source)
            ):
                converted_parameters[key] = val_array
        else:
            lambda_1_list, lambda_2_list, radius_1_list, radius_2_list = [], [], [], []
            for i, rad in enumerate(radii):
                (lambda_1, lambda_2, radius_1, radius_2 ) = EOS_to_system_parameters(
                    rad, masses[i], lambdas[i], m1_source[i],  m2_source[i] )
                    
                lambda_1_list.append(lambda_1)
                lambda_2_list.append(lambda_2)
                radius_1_list.append(radius_1)
                radius_2_list.append(radius_2)

            for key, _list in zip(system_keys, [
                lambda_1_list, lambda_2_list, radius_1_list, radius_2_list
            ]):
                converted_parameters[key] = np.array(_list)

        return converted_parameters


    
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