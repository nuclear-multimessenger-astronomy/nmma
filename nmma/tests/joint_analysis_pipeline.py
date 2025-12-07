from argparse import Namespace
import os
import pytest
import numpy as np
import shutil

from nmma.em import em_parsing as emp, lightcurve_handling as lch, model, analysis as ema
from nmma.eos.eos_parsing import tabulated_eos_parsing
from nmma.eos.eos_likelihood import tabulated_eos_setup
from nmma.joint.base import multi_analysis_loop
from nmma.joint.base_parsing import parsing_and_logging
from nmma.joint.joint_likelihood import MultiMessengerLikelihood
from nmma.joint import utils
from nmma.pbilby import generation
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

em_args = parsing_and_logging(
    (emp.multi_wavelength_analysis_parser,tabulated_eos_parsing), [])
main_args = Namespace(
    label="injection",
    prior_file="priors/Bu2019lm.prior",
    em_tmin=0.1,
    em_tmax=10.0,
    injection_em_tmax=9.0,
    em_tstep=0.5,
    em_error_budget=0.1,
    bestfit=True,
    filters="ztfr",
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
eos_args = Namespace(
        eos_data= f"{DATA_DIR}/eos_macro",
        eos_to_ram=True,
        upper_mtov={'upper_dummy':{'mass':2.23,'error':0.02 } }, 
        lower_mtov={'lower_dummy':{'mass':2.17,'error':0.02 } } 
)
gw_args = Namespace(
    frequency_domain_source_model = 'binary_neutron_star_frequency_sequence',
    waveform_approximant = 'IMRPhenomXAS_NRTidalv3',
    likelihood_type = 'MBGravitationalWaveTransient',
    reference_frequency = 30,
    duration = 256,
    detectors = ['H1', 'L1', 'V1'],
    psd_dict = '{"H1":"/home/hrose/Dokumente/40_nuclear/10_data/GW170817/h1_psd.txt", "L1":"/home/hrose/Dokumente/40_nuclear/10_data/GW170817/l1_psd.txt", "V1":"/home/hrose/Dokumente/40_nuclear/10_data/GW170817/v1_psd.txt"}'
)
@pytest.fixture(scope="module")
def args():
    return merge_namespaces(
        em_args,
        main_args,
        em_model_args,
        injection_args, 
        samling_args, 
        em_prior_args,
        eos_args,
        gw_args
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
    
    assert np.isclose([arr[10] for arr in data['ztfr'].values()],
                      [4.4248125e+04, 2.05459274e+01, 1.00000000e-01]).all()
    
def test_single_thread_setup(args):
    def setup(args):
        priors, em_lhood, injection_parameters = ema.analysis_setup(args)
        eos_priors, eos_lhood, _ = tabulated_eos_setup(args)
        injection_parameters['EOS'] = 7  # set EOS used in injection
        priors.update(eos_priors)
        combined_likelihood = MultiMessengerLikelihood([em_lhood, eos_lhood], priors, args, connected_params=False)
        # combined_likelihood.parameter_conversion = combined_likelihood.identity_conversion
        return priors, combined_likelihood,  injection_parameters
    multi_analysis_loop(args, setup)

def test_data_generation(args):

    generation_parser = generation.create_nmma_generation_parser()
    generation.generate_runner(generation_parser, **vars(args))
    