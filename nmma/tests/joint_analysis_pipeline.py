from argparse import Namespace
import os
import pytest
import numpy as np
import shutil

from nmma.em import model, em_parsing, lightcurve_handling as lch
from nmma.joint import utils
WORKING_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(WORKING_DIR, "data")

def merge_namespaces(*namespaces):
    joint_ns = Namespace()
    for ns in namespaces:
        try:
            joint_ns.__dict__.update(ns.__dict__)
        except AttributeError:
            joint_ns.__dict__.update(ns)
    return joint_ns

em_args = em_parsing.parsing_and_logging(em_parsing.multi_wavelength_analysis_parser, [])
main_args = Namespace(
    label="injection",
    prior="priors/Bu2019lm.prior",
    em_tmin=0.1,
    em_tmax=10.0,
    em_tstep=0.5,
    em_error_budget=0.1,
    bestfit=True,
    filters="ztfr",
    error_budget="0",
    remove_nondetections=True,
    plot=True,
)
em_model_args = Namespace(
    em_model="Bu2019nsbh",
    interpolation_type="tensorflow",
    svd_path=DATA_DIR,
)

injection_args = Namespace(
    injection_file=f"{DATA_DIR}/Bu2019lm_injection.json",
    injection_outfile="outdir/lc.csv",
)

samling_args = Namespace(
    nlive=64,
    local_only=True
)

em_prior_args = Namespace(
    Ebv_max=0.0
)


@pytest.fixture(scope="module")
def args():
    return merge_namespaces(
        em_args,
        main_args,
        em_model_args,
        injection_args, 
        samling_args, 
        em_prior_args
    )

@pytest.fixture(autouse=True)
def cleanup_outdir(args):
    yield
    if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir, ignore_errors=True)

def test_injection_creation(args):
    
    inj_model = model.create_light_curve_model_from_args(args.em_model, args)   
    injection_df = utils.read_injection_file(args)
    injection_parameters = injection_df.iloc[args.injection_num].to_dict()
    data, injection_parameters = lch.make_injection(injection_parameters, args, injection_model=inj_model) 
    

    assert np.isclose(data['ztfr'][12],[4.42478024e+04, 2.03533187e+01, 1.00000000e-01]).all()
    
def test_data_generation(args):
    pass