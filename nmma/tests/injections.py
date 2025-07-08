from argparse import Namespace
import bilby.core
import numpy as np
import json
import os
import shutil

from nmma.em.model import (
    SimpleKilonovaLightCurveModel,
    SupernovaLightCurveModel,
    ShockCoolingLightCurveModel,
    GRBLightCurveModel,
    SVDLightCurveModel,
)

from ..em import em_parsing, lightcurve_handling as lch
from ..em.io import read_lightcurve_file
from ..joint import create_injection
from ..joint.conversion import distance_modulus_nmma


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
    workingDir=os.path.dirname(__file__)
    dataDir = os.path.join(workingDir, 'data')
    test_directory = os.path.join(dataDir, model_name)
    priorDir=os.path.join(workingDir, '../../priors/')
    svdmodels=os.path.join(workingDir, '../../svdmodels/')
    if os.path.isdir(test_directory):
        shutil.rmtree(test_directory)
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

        if model_name == "nugent-hyper":
            prior_path = os.path.join( priorDir, "sncosmo-generic" + ".prior")
        elif model_name == "TrPi2018":
            prior_path = os.path.join( dataDir, "TrPi2018_pinned_parameters" + ".prior") #pinning the parameter svalues in the prior file
        else:
            prior_path = os.path.join( priorDir, model_name + ".prior")
        assert os.path.exists(prior_path), "prior file does not exist"
        injection_name = os.path.join(test_directory, model_name + "_injection.json")

        args = Namespace(
            prior_file=prior_path,
            injection_file=None,
            reference_frequency=20,
            aligned_spin=False,
            filename=injection_name,
            extension="json",
            n_injection=1,
            trigger_time=0,
            gps_file=None,
            deltaT=0.2,
            post_trigger_duration=2,
            duration=4,
            generation_seed=42,
            grb_resolution=5,
            eos_file="example_files/eos/ALF2.dat",
            eject=True,
            detections_file=None,
            indices_file=None,
            original_parameters=True,
            repeated_simulations=0,
        )

        create_injection.main(args)

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

        args = em_parsing.parsing_and_logging(em_parsing.lightcurve_parser, [])
        non_default_args = dict(
            injection=injection_file,
            label=command_line_lightcurve_label,
            em_model=model_name,
            svd_path=svdmodels,
            em_tmin=0.01,
            em_tmax=20.0,
            em_tstep=0.5,
            filters="sdssu",
            outdir=output_directory,
            interpolation_type="sklearn_gp",
            em_error=0.0,
        )
        args.__dict__.update(non_default_args)

        lch.lcs_from_injection_parameters(args)

        command_line_lightcurve_file = os.path.join(
            output_directory, f"{command_line_lightcurve_label}_0.dat"
        )
        assert os.path.exists(
            command_line_lightcurve_file
        ), "command line lightcurve file does not exist"

        return read_lightcurve_file(command_line_lightcurve_file)

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
            injection = json.load(f, object_hook=bilby.core.utils.decode_bilby_json)
        injection_content = injection["injections"]
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
        time_series = np.arange(0.01, 20.0 + 0.5, 0.5)
        init_kwargs = dict(
            model=model_name,
            filters=["sdssu"],
        )
        if model_name == "Ka2017":
            init_kwargs['interpolation_type'] = "sklearn_gp"
        lightcurve_model = model_lightcurve_function(**init_kwargs)
        obs_times, lightcurve_from_function = lightcurve_model.gen_detector_lc(
            sample_times=time_series, parameters=lightcurve_parameters
        )
        lightcurve_from_function["t"] = obs_times

        return lightcurve_from_function

    def compare_lightcurves(lightcurve_from_function, lightcurve_from_command_line):
        """
        Compare the values of the lightcurves generated from the function and command line to look for differences

        Parameters:
        ------------
        - lightcurve_from_function: dictionary
        Dictionary of lightcurve generated from function (keys: filters, values: list of magnitudes)
        - lightcurve_from_command_line: dictionary
        Dictionary of lightcurve generated from command line (keys: filters, values: list of magnitudes)

        Returns:
        None
        """
        filters_from_function = lightcurve_from_function.keys()
        filters_from_command_line = lightcurve_from_command_line.keys()

        assert set(filters_from_function) == set(
            filters_from_command_line
        ), "filters from function and command line do not match"
        # goes filter by filter and checks that each array matches
        for filter_name in filters_from_function:
            assert all(
                np.isclose(
                    lightcurve_from_function[filter_name][
                        ~np.isnan(lightcurve_from_function[filter_name])
                    ],
                    lightcurve_from_command_line[filter_name][
                        ~np.isnan(lightcurve_from_command_line[filter_name])
                    ],
                    rtol=1e-3,
                )
            ), f"lightcurve tolerance for {filter_name} exceeded"

    def cleanup_files():
        """
        deletes test files directory
        """
        shutil.rmtree(test_directory)
        assert not os.path.exists(test_directory), "test directory has not been deleted"
    
    injection_file = create_injection_from_command_line(model_name)
    command_line_lightcurve_dictionary = create_lightcurve_from_command_line(
        model_name, injection_file
    )
    function_lightcurve_dictionary = create_lightcurve_from_function(
        model_name, injection_file, model_lightcurve_function
    )

    compare_lightcurves(
        function_lightcurve_dictionary, command_line_lightcurve_dictionary
    )

    # if all of the above works, then we don't need the files anymore
    cleanup_files()


def test_injections():
    lightcurve_models = {
        "nugent-hyper": SupernovaLightCurveModel,
        "salt2": SupernovaLightCurveModel,
        "Me2017": SimpleKilonovaLightCurveModel,
        "Piro2021": ShockCoolingLightCurveModel,
        "TrPi2018": GRBLightCurveModel,
        "Ka2017": SVDLightCurveModel,
    }
    for model_name, model_lightcurve_function in lightcurve_models.items():
        lightcurveInjectionTest(model_name, model_lightcurve_function)

def test_validate_lightcurves():
    print("validate_lightcurve test")

    ## initialize args, check a file that is known to have 3 observations in the ztf g filter and 1 in the ztf r filter. All detections occur within 9 days of the original observation.
    args = Namespace(
        data='example_files/candidate_data/ZTF20abwysqy.dat',
        filters="ztfg",
        min_obs=3,
        cutoff_time=0,
        verbose=True,
    )
    assert lch.validate_lightcurve(**vars(args)) == True, "Test for 3 observations in the ztf g filter failed"

    args.filters = "ztfr"
    args.min_obs = 1
    assert lch.validate_lightcurve(**vars(args)) == True, "Test for 1 observation in the ztf r filter failed"

    args.filters = "ztfg,ztfr"
    assert lch.validate_lightcurve(**vars(args)) == True, "Test for  passing multiple filters failed"

    args.filters = ""
    args.min_obs = 0
    assert lch.validate_lightcurve(**vars(args)) == True, "Test for automatic filter selection failed"

    args.cutoff_time = 1
    args.min_obs = 1
    assert lch.validate_lightcurve(**vars(args)) == False, "Test for setting cutoff time failed"