import numpy as np
import json
import subprocess
import sys
import os
import nmma

import numpy as np
import pandas as pd
from astropy.time import Time

from nmma.em.model import SimpleKilonovaLightCurveModel, GRBLightCurveModel, SVDLightCurveModel, KilonovaGRBLightCurveModel, GenericCombineLightCurveModel, SupernovaLightCurveModel, ShockCoolingLightCurveModel


def lightcurveInjectionTest(model_name, model_lightcurve_function):
    '''
    compares the creation of a lightcurve injection from command line with light_curve_generation and through calling the relevant function directly
    
    Parameters:
    -----------
    - model_name: string
    Name of model prior to test (e.g. 'nugent-hyper'). Must be included in ./priors/ directory
    - model_lightcurve_function: function
    One of the nmma.em.model functions (e.g. nmma.em.model.SupernovaLightCurveModel) that corresponds to the associated model name
    '''
    print('current working directory: ', os.getcwd()) ## assumes run in root nmma folder, will need to modify if this is not true
    data_directory = './nmma/tests/data/'
    test_directory = os.path.join(data_directory, model_name)
    os.makedirs(test_directory, exist_ok=True)
    def create_injection_from_command_line(model_name):
        '''
        Creates the injection file from command line using nmma_create_injection
        
        Parameters:
        ------------
        - model_name: string
        Name of model prior to test (e.g. 'nugent-hyper'). Must be included in ./priors/ directory
        '''
        prior_path = os.path.join('./priors/', model_name + '.prior')
        assert os.path.exists(prior_path), 'prior file does not exist'
        injection_name = os.path.join(test_directory,model_name + '_injection.json')
        command_array = ['nmma_create_injection', 
                '--prior-file', prior_path,
                '--eos-file ../nmma/example_files/eos/ALF2.dat', 
                '--binary-type BNS',
                '--n-injection', '1', ## only generates one lightcurve, could be changed to generate more
                '--original-parameters',
                '--extension json',
                '--filename', injection_name
                ]
        command_string = ' '.join(command_array)
        subprocess.run(command_string, shell=True)
        assert os.path.exists(injection_name), 'injection file does not exist'
        return injection_name
    def create_lightcurve_from_command_line(model_name, injection_file):
        '''
        Creates the lightcurve file from command line using light_curve_generation
        
        Parameters:
        ------------
        - model_name: string
        Name of model prior to test (e.g. 'nugent-hyper'). Must be included in ./priors/ directory
        - injection_file: string
        path to injection file created by create_injection_from_command_line
        '''
        prior_path = os.path.join('./priors/', model_name + '.prior')
        output_directory = test_directory
        command_line_lightcurve_label = model_name + '_command_line_lightcurve.dat'
        time_series = np.arange(0.01, 20.0, 0.5)
        command_array = ['light_curve_generation',
               '--injection', injection_file,
               '--label', command_line_lightcurve_label,
               '--model', model_name,
               '--svd-path', '../nmma/svdmodels',
               '--tmin', str(time_series[0]),
               '--tmax', str(time_series[-1]),
               '--dt', '0.5',
               '--outdir', output_directory
               #'--injection-detection-limit', '21.5',
               ]
        command_string = ' '.join(command_array)
        subprocess.run(command_string, shell=True)
        command_line_lightcurve_file = os.path.join(output_directory, command_line_lightcurve_label)
        assert os.path.exists(command_line_lightcurve_file), 'command line lightcurve file does not exist'
        return command_line_lightcurve_file
    def get_parameters_from_injection_file(injection_file):
        '''
        read the parameters from the injection file
        
        Parameters:
        ------------
        - injection_file: string
        path to the injection file created by create_injection_from_command_line
        '''
        assert os.path.exists(injection_file), 'injection file does not exist'
        with open(injection_file, 'r') as f:
            injection = json.load(f)
        injection_content = injection['injections']['content']
        injection_dictionary = {key:index[0] for key, index in injection_content.items()}
        return injection_dictionary
    def create_lightcurve_from_function(model_name, injection_file, model_lightcurve_function):
        '''
        create lightcurve using associated LightcurveModel function
        
        Parameters:
        ------------
        - model_name: string
        name of model prior to test (e.g. 'nugent-hyper'). Must be included in ./priors/ directory
        - injection_file: string
        path to injection file
        - model_lightcurve_function: function
        associated nmma LightcurveModel function
        '''
        assert os.path.exists(injection_file), 'injection file does not exist'
        lightcurve_parameters = get_parameters_from_injection_file(injection_file)
        time_series = np.arange(0.01, 20.0, 0.5)
        lightcurve_model = model_lightcurve_function(sample_times=time_series, model=model_name)
        lightcurve_from_function = lightcurve_model.generate_lightcurve(sample_times=time_series,parameters=lightcurve_parameters)[1]
        ## need to adjust magnitudes to be absolute
        absolute_magnitude_conversion = lambda magnitude, distance: magnitude + 5 * np.log10(distance * 1e6 / 10.0)
        luminosity_distance = lightcurve_parameters['luminosity_distance']
        adjusted_lightcurve_from_function = {key: absolute_magnitude_conversion(magnitude, luminosity_distance) for key, magnitude in lightcurve_from_function.items()}
        return adjusted_lightcurve_from_function
    
    def cleanup_files():
        '''
        deletes test files directory
        '''
        os.remove(test_directory)
        assert not os.path.exists(test_directory), 'test directory has not been deleted'
        
    
    injection_file = create_injection_from_command_line(model_name)
    create_lightcurve_from_command_line(model_name, injection_file)
    
    ## if all of the above works, then we don't need the files anymore
    cleanup_files()
    
lightcurveInjectionTest('nugent-hyper', SupernovaLightCurveModel)


        
        