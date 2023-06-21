import numpy as np
import pandas as pd
import json
import subprocess
import os
import shutil

from nmma.em.model import (
    SimpleKilonovaLightCurveModel,
    GRBLightCurveModel,
    SVDLightCurveModel,
    KilonovaGRBLightCurveModel,
    GenericCombineLightCurveModel,
    SupernovaLightCurveModel,
    ShockCoolingLightCurveModel,
)


def lightcurveInjectionTest(model_name, model_lightcurve_function):
    """
    compares the creation of a lightcurve injection from command line with light_curve_generation and through calling the relevant function directly

    Parameters:
    -----------
    - model_name: string
    Name of model prior to test (e.g. 'nugent-hyper'). Must be included in ./priors/ directory
    - model_lightcurve_function: function
    One of the nmma.em.model functions (e.g. nmma.em.model.SupernovaLightCurveModel) that corresponds to the associated model name
    """
    print("running lightcurve injection test for ", model_name)
    print(
        "current working directory: ", os.getcwd()
    )  # assumes run in root nmma folder, will need to modify if this is not true
    data_directory = "./nmma/tests/data/"
    test_directory = os.path.join(data_directory, model_name)
    os.makedirs(test_directory, exist_ok=True)

    def create_injection_from_command_line(model_name):
        """
        Creates the injection file from command line using nmma_create_injection

        Parameters:
        ------------
        - model_name: string
        Name of model prior to test (e.g. 'nugent-hyper'). Must be included in ./priors/ directory
        
        Returns:
        ---------
        - injection_name: string
        path to the injection file created by nmma_create_injection
        """
        
        prior_path = os.path.join("./priors/", model_name + ".prior")
        assert os.path.exists(prior_path), "prior file does not exist"
        injection_name = os.path.join(test_directory, model_name + "_injection.json")
        command_array = [
            "nmma_create_injection",
            "--prior-file",
            prior_path,
            "--eos-file ../nmma/example_files/eos/ALF2.dat",
            "--binary-type BNS",
            "--n-injection",
            "1",  # only generates one lightcurve, could be changed to generate more
            "--original-parameters",
            "--extension json",
            "--filename",
            injection_name,
        ]
        command_string = " ".join(command_array)
        subprocess.run(command_string, shell=True)
        assert os.path.exists(injection_name), "injection file does not exist"
        return injection_name

    def create_lightcurve_from_command_line(model_name, injection_file):
        """
        Creates the lightcurve file from command line using light_curve_generation

        Parameters:
        ------------
        - model_name: string
        Name of model prior to test (e.g. 'nugent-hyper'). Must be included in ./priors/ directory
        - injection_file: string
        path to injection file created by create_injection_from_command_line
        
        Returns:
        ----------
        - command_line_lightcurve_file: string
        path to the lightcurve file created by light_curve_generation
        """
        # prior_path = os.path.join("./priors/", model_name + ".prior")
        output_directory = test_directory
        command_line_lightcurve_label = model_name + "_command_line_lightcurve"
        time_series = np.arange(0.01, 20.0, 0.5) ## could potentially make this a parameter
        command_array = [
            "light_curve_generation",
            "--injection",
            injection_file,
            "--label",
            command_line_lightcurve_label,
            "--model",
            model_name,
            "--svd-path",
            "../nmma/svdmodels",
            "--tmin",
            str(time_series[0]),
            "--tmax",
            str(time_series[-1]),
            "--dt",
            "0.5",
            "--outdir",
            output_directory
            # '--injection-detection-limit', '21.5',
        ]
        command_string = " ".join(command_array)
        subprocess.run(command_string, shell=True)
        command_line_lightcurve_file = os.path.join(
            output_directory, f"{command_line_lightcurve_label}.dat"
        )
        assert os.path.exists(
            command_line_lightcurve_file
        ), "command line lightcurve file does not exist"
        return command_line_lightcurve_file
    
    def get_lightcurve_from_file(lightcurve_file): 
        """
        Load the generated lightcurve file from calling the create_lightcurve_from_command_line
        
        Parameters:
        ------------
        - lightcurve_file: string
        Path to the lightcurve model to be loaded
        
        Returns:
        ----------
        lightcurve_file_dictionary: dictionary
        dictionary of lightcurve recovered from file. Only measurements, no time values (for now)
        """
        lightcurve_dataframe = pd.read_csv(lightcurve_file, sep=' ', header=None, names=['t', 'filter', 'mag', 'mag_unc'])
        
        lightcurve_filters = lightcurve_dataframe['filter'].unique()
        lightcurve_dictionary = {
            filter_name: lightcurve_dataframe[lightcurve_dataframe['filter'] == filter_name]['mag'].values
            for filter_name in lightcurve_filters
        }
        return lightcurve_dictionary

    def get_parameters_from_injection_file(injection_file):
        """
        read the parameters from the injection file

        Parameters:
        ------------
        - injection_file: string
        path to the injection file created by create_injection_from_command_line
        
        Returns:
        ----------
        - injection_dictionary: dictionary
        Dictionary of parameters from injection file
        """
        assert os.path.exists(injection_file), "injection file does not exist"
        with open(injection_file, "r") as f:
            injection = json.load(f)
        injection_content = injection["injections"]["content"]
        injection_dictionary = {
            key: index[0] for key, index in injection_content.items()
        }
        return injection_dictionary

    def create_lightcurve_from_function(
        model_name, injection_file, model_lightcurve_function
    ):
        """
        create lightcurve using associated LightcurveModel function

        Parameters:
        ------------
        - model_name: string
        name of model prior to test (e.g. 'nugent-hyper'). Must be included in ./priors/ directory
        - injection_file: string
        path to injection file
        - model_lightcurve_function: function
        associated nmma LightcurveModel function
        
        Returns:
        ----------
        - lightcurve_from_function: dictionary
        dictionary of lightcurve generated via functions
        """
        assert os.path.exists(injection_file), "injection file does not exist"
        lightcurve_parameters = get_parameters_from_injection_file(injection_file)
        time_series = np.arange(0.01, 20.0, 0.5)
        lightcurve_model = model_lightcurve_function(
            sample_times=time_series, model=model_name
        )
        lightcurve_from_function = lightcurve_model.generate_lightcurve(
            sample_times=time_series, parameters=lightcurve_parameters
        )[1]
        # need to adjust magnitudes to be absolute
        absolute_magnitude_conversion = (
            lambda magnitude, distance: magnitude + 5 * np.log10(distance * 1e6 / 10.0)
        )
        luminosity_distance = lightcurve_parameters["luminosity_distance"]
        adjusted_lightcurve_from_function = {
            key: absolute_magnitude_conversion(magnitude, luminosity_distance)
            for key, magnitude in lightcurve_from_function.items()
        }
        return adjusted_lightcurve_from_function
    
    def compare_lightcurves(lightcurve_from_function, lightcurve_from_command_line, acceptance_threshold=0.01):
        """
        Compare the values of the lightcurves generated from the function and command line to look for differences
        
        Parameters:
        ------------
        - lightcurve_from_function: dictionary
        Dictionary of lightcurve generated from function (keys: filters, values: list of magnitudes)
        - lightcurve_from_command_line: dictionary
        Dictionary of lightcurve generated from command line (keys: filters, values: list of magnitudes)
        - acceptance_threshold: float
        Percent difference between values that is acceptable. Value must be between 0 and 1. Default is 0.01 (1%)
        
        Returns:
        None
        """
        assert acceptance_threshold >= 0 and acceptance_threshold <= 1, "acceptance threshold must be between 0 and 1"
        time_series = np.arange(0.01, 20.0, 0.5)
        filters_from_function = lightcurve_from_function.keys()
        filters_from_command_line = lightcurve_from_command_line.keys()
        
        assert filters_from_function == filters_from_command_line, "filters from function and command line do not match"
        ## above line could be an issue if the function and command line don't have the same filter order, though this could
        ## be a potential issue to fix if it comes up
        
        ## goes filter by filter and checks that each time value matches
        for filter_name in filters_from_function:
            for time_index, time_value in enumerate(time_series):
                ## should the values be rounded to some number of decimal places?
                function_value = lightcurve_from_function[filter_name][time_index]
                command_line_value = lightcurve_from_command_line[filter_name][time_index]
                # assert function_value == command_line_value, f"lightcurve values for {filter_name} at t={time_value} do not match. \nFunction Value: {function_value}\nCommand Line Value: {command_line_value}"
                
                ## maybe a better way to do this is to check that the values are within some percentage tolerance of each other
                fractional_difference = np.abs(function_value - command_line_value) / function_value
                assert fractional_difference <= acceptance_threshold, f"lightcurve tolerance for {filter_name} at t={time_value} exceeds {round(100*acceptance_threshold)}% (Measured Value: {round(100*fractional_difference)}%). \nFunction Value: {function_value}\nCommand Line Value: {command_line_value}"
        print("Lightcurve values match within tolerance")
        
    def cleanup_files():
        """
        deletes test files directory
        """
        shutil.rmtree(test_directory)
        assert not os.path.exists(test_directory), "test directory has not been deleted"

    injection_file = create_injection_from_command_line(model_name)
    command_line_lighcurve_path = create_lightcurve_from_command_line(model_name, injection_file)
    command_line_lighcurve_dictionary = get_lightcurve_from_file(command_line_lighcurve_path)
    function_lightcurve_dictionary = create_lightcurve_from_function(model_name, injection_file, model_lightcurve_function)
    
    compare_lightcurves(function_lightcurve_dictionary, command_line_lighcurve_dictionary)

    ## if all of the above works, then we don't need the files anymore
    cleanup_files()

## add additional tests here
lightcurve_model_names = ["nugent-hyper"]
lightcurve_model_functions = [SupernovaLightCurveModel]

for model_name, model_lightcurve_function in zip(lightcurve_model_names, lightcurve_model_functions):
    lightcurveInjectionTest(model_name, model_lightcurve_function)