from argparse import Namespace
import numpy as np
from pathlib import Path
import shutil
import pytest

from nmma.em import em_parsing, lightcurve_handling as lch
from nmma.em.io import load_em_observations
from nmma.em.model import single_model_from_mapping
from nmma.core.utils import read_injection_file
from nmma.core.parsing import nmma_base_parsing
from nmma.joint import injection_handling, joint_parsing

DATADIR = Path(__file__).parent.parent / "data"
BASEDIR = DATADIR.parent.parent.parent
PRIORDIR = BASEDIR / "priors/"
OUTDIR = DATADIR / "outdir"


@pytest.fixture(autouse=True)
def cleanup_outdir():
    yield
    if Path(OUTDIR).exists():
        shutil.rmtree(OUTDIR, ignore_errors=True)


def lightcurveInjectionTest(model_name):
    """
    compares the creation of a lightcurve injection from command line with light_curve_generation and through calling the relevant function directly
    Parameters:
    -----------
    - model_name: string
    Name of model prior to test (e.g. 'nugent-hyper'). Must be included in ./priors/ directory
    """
    print("running lightcurve injection test for ", model_name)

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
            prior_path = PRIORDIR / ("sncosmo-generic.prior")
        elif model_name == "TrPi2018":
            prior_path = DATADIR / ("TrPi2018_pinned_parameters.prior")
        else:
            prior_path = PRIORDIR / (model_name + ".prior")
        assert prior_path.exists(), "prior file does not exist"
        injection_name = OUTDIR / model_name / "injection.json"

        args = nmma_base_parsing(joint_parsing.injection_parsing, [])
        non_default_args = dict(
            prior_file=str(prior_path),
            simple_setup=True,
            injection_file=injection_name,
            post_processing=["ejecta"],
            n_injection=1,
            eos_file="example_files/eos/ALF2.dat",
            original_parameters=True,
        )

        for key, value in non_default_args.items():
            setattr(args, key, value)
        injection_handling.generate_injection(args)

        assert injection_name.exists(), "injection file does not exist"
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
        output_directory = OUTDIR / model_name
        command_line_lightcurve_label = model_name + "_command_line"

        args = em_parsing.nmma_base_parsing(em_parsing.lightcurve_parser, [])
        non_default_args = dict(
            injection_file=str(injection_file),
            label=command_line_lightcurve_label,
            em_model=model_name,
            svd_path=BASEDIR / "nmma_models" / "svdmodels",
            filters="sdssu",
            outdir=output_directory,
            interpolation_type="tensorflow",
            injection_error_budget=0.0,
            ignore_timeshift=True,
        )
        args.__dict__.update(non_default_args)
        lch.lcs_from_injection_parameters(args)

        command_line_lightcurve_file = (
            output_directory / f"{command_line_lightcurve_label}_0_lc.json"
        )

        assert (
            command_line_lightcurve_file.exists()
        ), "command line lightcurve file does not exist"

        return load_em_observations(command_line_lightcurve_file)

    def create_lightcurve_from_function(model_name, injection_file):
        """
        create lightcurve using associated LightcurveModel function
        Parameters:
        ------------
        - model_name: string
        name of model prior to test (e.g. 'nugent-hyper'). Must be included in ./priors/ directory
        - injection_file: string
        path to injection file
        Returns:
        ----------
        - lightcurve_from_function: dictionary
        dictionary of lightcurve generated via functions
        """
        assert injection_file.exists(), "injection file does not exist"
        injection_dict = read_injection_file(injection_file)
        lightcurve_parameters = {k: v[0] for k, v in injection_dict.items()}
        init_kwargs = dict(model=model_name, filters=["sdssu"])
        if model_name == "Ka2017":
            init_kwargs["interpolation_type"] = "sklearn_gp"
        model_class = single_model_from_mapping(model_name)
        lightcurve_model = model_class(**init_kwargs)
        lc_params = lightcurve_model.parameter_conversion(lightcurve_parameters)
        _, func_lc = lightcurve_model.gen_detector_lc(lc_params)
        # lightcurve_from_function["t"] = obs_times

        return func_lc

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
            cli_mags = lightcurve_from_command_line[filter_name]["mag"]
            gen_mags = lightcurve_from_function[filter_name]
            assert all(
                np.isclose(
                    cli_mags[~np.isnan(cli_mags)],
                    gen_mags[~np.isnan(gen_mags)],
                    rtol=1e-3,
                )
            ), f"lightcurve tolerance for {filter_name} exceeded"

    injection_file = create_injection_from_command_line(model_name)
    cl_lc_dict = create_lightcurve_from_command_line(model_name, injection_file)
    func_lc_dict = create_lightcurve_from_function(model_name, injection_file)

    compare_lightcurves(func_lc_dict, cl_lc_dict)


def test_injections():
    for model_name in ["salt2", "nugent-hyper", "Me2017", "Piro2021", "TrPi2018"]:
        lightcurveInjectionTest(model_name)


def test_validate_lightcurves():
    print("validate_lightcurve test")

    # initialize args, check a file that is known to have 3 observations in the ztf g filter and 1 in the ztf r filter. All detections occur within 9 days of the original observation.
    args = Namespace(
        data_file="example_files/candidate_data/ZTF20abwysqy.dat",
        filters=["ztfg"],
        min_obs=3,
        cutoff_time=0,
        verbose=True,
    )
    assert lch.validate_lightcurve(
        **vars(args)
    ), "Test for 3 observations in the ztf g filter failed"

    args.filters = ["ztfr"]
    args.min_obs = 1
    assert lch.validate_lightcurve(
        **vars(args)
    ), "Test for 1 observation in the ztf r filter failed"

    args.filters = ["ztfg", "ztfr"]
    assert lch.validate_lightcurve(
        **vars(args)
    ), "Test for  passing multiple filters failed"

    args.filters = None
    args.min_obs = 0
    assert lch.validate_lightcurve(
        **vars(args)
    ), "Test for automatic filter selection failed"

    args.cutoff_time = 1
    args.min_obs = 1
    assert not lch.validate_lightcurve(
        **vars(args)
    ), "Test for setting cutoff time failed"
