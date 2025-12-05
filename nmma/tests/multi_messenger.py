import os
import pytest
import shutil

from ..em import analysis, em_parsing 
from ..eos.eos_likelihood import tabulated_eos_setup
from ..eos.eos_parsing import tabulated_eos_parsing
from ..joint.base import multi_analysis_loop
from ..joint.base_parsing import parsing_and_logging
from ..joint.joint_likelihood import MultiMessengerLikelihood

WORKING_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(WORKING_DIR, "data")
os.environ["WORKING_DIR"] = WORKING_DIR


@pytest.fixture(autouse=True)
def cleanup_outdir(args):
    yield
    if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir, ignore_errors=True)


@pytest.fixture(scope="module")
def args():
    args = parsing_and_logging(
        (em_parsing.multi_wavelength_analysis_parser,tabulated_eos_parsing), [])
    non_default_args = dict(
        em_model="Bu2019nsbh",
        interpolation_type="tensorflow",
        svd_path=DATA_DIR,
        label="injection",
        prior_file="priors/Bu2019lm.prior",
        em_tmin=0.1,
        em_tmax=14.0,
        injection_em_tmax=12.0,
        em_tstep=0.5,
        bestfit=True,
        filters="ztfr",
        Ebv_max=0.0,
        nlive=64,
        sampler="pymultinest",
        injection_file=f"{DATA_DIR}/Bu2019lm_injection.json",
        injection_outfile="outdir/lc.csv",
        plot=True,
        eos_data= f"{DATA_DIR}/eos_macro",
        eos_to_ram=True,
        upper_mtov={'upper_dummy':{'mass':2.23,'error':0.02 } }, 
        lower_mtov={'lower_dummy':{'mass':2.17,'error':0.02 } } 
        
    )
    for key, value in non_default_args.items():
        setattr(args, key, value)

    return args

def test_simple_model(args):
    def setup(args):
        priors, em_lhood, injection_parameters = analysis.analysis_setup(args)
        eos_priors, eos_lhood, _ = tabulated_eos_setup(args)
        injection_parameters['EOS'] = 7  # set EOS used in injection
        priors.update(eos_priors)
        combined_likelihood = MultiMessengerLikelihood([em_lhood, eos_lhood], priors, args, connected_params=False)
        # combined_likelihood.parameter_conversion = combined_likelihood.identity_conversion
        return priors, combined_likelihood,  injection_parameters
    multi_analysis_loop(args, setup)