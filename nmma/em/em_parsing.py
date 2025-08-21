import argparse
import os
from ..utils.models import refresh_models_list
from .utils import DEFAULT_FILTERS
from bilby.core.utils import setup_logger
from ..joint.base_parsing import (nmma_base_parsing, base_analysis_parsing, 
    base_injection_parsing, pipe_inj_parsing, nonefloat, noneint, nonestr )



def em_time_parsing(parser):   
    em_time_parser = parser.add_argument_group(
        title="EM analysis time arguments", description="Specify EM analysis sample times"
    ) 
    em_time_parser.add( "--em-tmin","--kilonova-tmin","--tmin", type=nonefloat, 
        help="Days to be started analysing from the trigger time. Default subject to model", )
    em_time_parser.add("--em-tmax","--kilonova-tmax","--tmax", type=nonefloat, 
        help="Days to be stoped analysing from the trigger time. Default subject to model", )
    em_time_parser.add("--em-nsteps", type=int, default=150,
        help="Number of steps to be used for light curve evaluation (default: 150)", )
    em_time_parser.add("--em-timescale", default='log', 
        help="Timescale for the light curve evaluation. "
        "Default: log, further options: linear, geometric")
    em_time_parser.add( "--em-tstep","--dt", type=nonefloat, default=None, 
        help="Time step (in days) for light curve initial evaluation, will overwrite nsteps. Legacy option kept for backward compatibility", )
    return parser

def basic_em_only_parsing(parser):
    parser.add_argument("-o", "--outdir", default="outdir",  help="Path to the output directory")
    parser.add_argument("--label", default ="em_transient", help="Label for the run")
    parser.add_argument("--plot", action='store_true', help="create characteristic plot")
    parser.add_argument("--verbose", action='store_true', help="print out log likelihoods" )
    return parser


def basic_em_only_analysis_parsing(parser):

    parser.add_argument("--config", help="Name of the configuration file containing parameter values.")
    parser.add_argument("--trigger-time", type=float,
        help="Trigger time, format will be inferred but can be but can be explicitly adjusted with --time-format, not required if injection set is provided")
    parser.add_argument("--light-curve-data", "--data", help="Path to data in [time filter magnitude error] format, time format will be inferred, but can be explicitly adjusted with --time-format. If not given, will try to generate data from the injection file.")
    parser.add_argument("--time-format", 
        help="Time format of the light curve data, e.g. isot, mjd, see https://docs.astropy.org/en/stable/time/#time-format")
    parser.add_argument("--prior", help="Path to the prior file")
    parser.add_argument("--skip-sampling", action='store_true', 
        help="If analysis has already run, skip bilby sampling and compute results from checkpoint files. Combine with --plot to make plots from these files.")
    parser.add_argument("--bestfit", action='store_true',
        help="Save the best fit parameters and magnitudes to JSON")
    parser.add_argument("--sampler", default="pymultinest",
        help="Sampler to be used (default: pymultinest)")
    parser.add_argument("--sampler-kwargs", default="{}", 
        help="Additional kwargs (e.g. {'evidence_tolerance':0.5}) for bilby.run_sampler, put a double quotation marks around the dictionary")
    parser.add_argument("--soft-init", action='store_true',
        help="To start the sampler softly (without any checking, default: False)")
    parser.add_argument("--cpus", type=int, default=1,
        help="Number of cores to be used, only needed for dynesty (default: 1)")
    parser.add_argument("-n","--nlive", type=int, default=2048, help="Number of live points (default: 2048)")
    parser.add_argument("--reactive-sampling", action='store_true',
        help="To use reactive sampling in ultranest (default: False)")
    return parser

def multi_wavelength_parsing(parser):
    em_input_parser = parser.add_argument_group(
        title="EM analysis input arguments", description="Specify EM analysis inputs"
    )
    em_input_parser.add_argument("--detection-limit", 
        help="Detection limit per filter, optimally as a dict, e.g., {'r':22, 'g':23}, put a double quotation mark around the dictionary")
    em_input_parser.add("--em-error-budget", "--kilonova-error",  default="1.0", 
        help="Additional statistical error (mag) to be introduced in each filter, can be passed as list or dict. " \
        "(default: 1 for all filters). Will only be used if em_syserr is not given in prior")
    em_input_parser.add_argument("--systematics-file", help="Path to systematics configuration file")
    return parser
    

def em_model_parsing(parser):
    em_model_parser = parser.add_argument_group(
        title="EM model arguments", description="Specify EM model properties"
    )
    em_model_parser.add("--filters", nargs = "*", help="A list of filters to use (e.g. g,r,i)."
        "If none is provided, will use all the filters available")
    em_model_parser.add("--em-transient-class", nargs = "*",
        help="Name of the model-type to be used, can be a comma-seperated list for joint lightcurve models" )
    em_model_parser.add_argument("--em-model", "--kilonova-model","--model", 
        help="Name of the transient model to be used")
    em_model_parser.add_argument("--interpolation-type", "--gptype", 
        default="keras", help="Interpolation library to be used for EM "\
            "transient model. Default: keras, further options: tensorflow, sklearn_gp, api_gp" )
    # Using Fiesta Surrogate
    em_model_parser.add("--surrogate-dir", type=nonestr, help="Path to the Fiesta surrogate directory")
    
    em_model_parser.add_argument("--refresh-models-list", action ="store_true",
        help="Refresh the list of models available on Gitlab")
    em_model_parser.add_argument("--local-only","--local-model-only", action='store_true',
        help="only look for local svdmodels (ignore Gitlab)")
    em_model_parser.add_argument("--absolute", action='store_true',
        help="Use Absolute Magnitude?")
        
    em_model_parser.add_argument( "--svd-path",  default="svdmodels",
         help="Path to the SVD directory with {model}.joblib")
    em_model_parser.add_argument(
        "--svd-mag-ncoeff","--svd-ncoeff", type=int,  default=10, 
        help="Number of eigenvalues to be taken for mag evaluation (default: 10)")
    em_model_parser.add_argument("--svd-lbol-ncoeff", type=int, default=10,
        help="Number of eigenvalues to be taken for lbol evaluation (default: 10)")
    em_model_parser.add_argument( "--xlim", default="0,14", nargs="*",
        help="Start and end time for light curve plot (default: 0-14)")
    em_model_parser.add_argument("--ylim", default="22,16", nargs="*",
        help="Upper and lower magnitude limit for light curve plot (default: 22,16)")

    return parser

def data_processing_parsing(parser):
    parser.add_argument( "--data-path", 
        help="Path to the directory of light curve files")
    parser.add_argument("--format","--data-file-type", default="bulla",
        help="Format of light curve files [bulla, standard, ztf, hdf5]", )
    parser.add_argument("--data-type", default="photometry", choices=["photometry", "spectroscopy"],
        help="Data type for interpolation [photometry or spectroscopy]")
    parser.add_argument( "--ignore-bolometric", action='store_false',
        help="ignore bolometric light curve files (ending in _Lbol.file_extension)")
    parser.add_argument("--lmin",type=float,default=3000.0,
        help="Minimum wavelength to analyze (default: 3000)",)
    parser.add_argument("--lmax", type=float, default=10000.0,
        help="Maximum wavelength to analyze (default: 10000)",)

    return parser


def ml_training_parsing(parser):
    parser.add_argument("--nepochs","--tensorflow-nepochs", type=int, default=15,
        help="Number of epochs for ml training (default: 15)",)
    parser.add_argument("--ncpus", default=1, type=int,
        help="number of cpus to be used, only support for sklearn_gp")
    parser.add_argument( "--data-time-unit", default="days",
        help="Time unit of input data (days, hours, minutes, or seconds)")
    parser.add_argument( "--use-UnivariateSpline", action='store_true',
        help="using UnivariateSpline to mitigate the numerical noise in the grid")
    parser.add_argument(
        "--UnivariateSpline-s", default=2, type=int,
        help="s-value to be used for UnivariateSpline")
    parser.add_argument("--random-seed", type=int, default=42,
        help="random seed to set during training",)
    parser.add_argument(
        "--axial-symmetry", action='store_true',
        help="add training samples based on the fact that there is axial symmetry")
    parser.add_argument(
        "--continue-training", action='store_true',
        help="Continue training an existing model",)
    return parser

def em_injection_parsing(parser):
    lc_injection_parser = parser.add_argument_group(
        title="Lightcurve injection arguments", description="Specify lightcurve injections"
    )
    lc_injection_parser.add("--injection-model-args", type = nonestr, 
        help="Additional arguments for the injection model, given like a python-dict " \
        "e.g. --injection_args='{\"param1\": 0.5, \"param2\": 1.0}'. All other parameters " \
        "needed to create the injection should be specified in the --injection-file.")
    
    lc_injection_parser.add_argument("--injection-model", 
        help="Name of the transient model to be used for injection (default: the same as model used for recovery)")

    lc_injection_parser.add_argument("--train-stats", action='store_true', 
        help="Creates a file too.csv to derive statistics")
    lc_injection_parser.add_argument( "--prompt-collapse", action='store_true',
        help="If the injection simulates prompt collapse and therefore only dynamical")
    parser.add_argument("--ignore-timeshift", action='store_true',
        help="If you want to ignore the timeshift parameter in an injection file.")
    lc_injection_parser.add_argument("--injection-error-budget","--photometric-error-budget",  type=float, default=0.1,
        help="Photometric error (mag) on the injected lightcurve (default: 0.1)")

    #photometry modifiers
    lc_injection_parser.add_argument("--ztf-sampling", help="Use realistic ZTF sampling", action='store_true')
    lc_injection_parser.add_argument("--ztf-uncertainties", help="Use realistic ZTF uncertainties", action='store_true')
    lc_injection_parser.add_argument( "--ztf-ToO", type=nonestr, choices=[None,"180", "300"], 
        help="Adds realistic ToO obeservations during the first one or two days. Sampling depends on exposure time specified. Valid values are 180 (<1000sq deg) or 300 (>1000sq deg). Won't work w/o --ztf-sampling")

    lc_injection_parser.add_argument("--rubin-ToO-type", type=nonestr, 
        choices=[None,"platinum","gold","gold_z","silver","silver_z"], 
        help="Type of ToO observation based on the strategy presented in arxiv.org/abs/2111.01945.")
    
    return parser

def em_only_injection_parsing(parser):
    parser = base_injection_parsing(parser)
    parser.add_argument("--injection", metavar="PATH", 
        help="Legacy:Path to the injection json file")
    parser.add_argument("--injection-num", type=int, default = 0,
        help="The injection number to be taken from the injection set")
    


    return parser



def skymap_parsing(parser):
    parser.add_argument("--fits-file", 
        help="Fits file output from Bayestar, to be used for constructing dL-iota prior")
    parser.add_argument(
        "--cosiota-node-num", default=10,
        help="Number of cos-iota nodes used in the Bayestar fits (default: 10)")

    parser.add_argument("--ra", type=float,
        help="Right ascension of the sky location; to be used together with fits file")
    parser.add_argument("--dec",type=float,
        help="Declination of the sky location; to be used together with fits file")
    parser.add_argument("--dL", type=float,
        help="Distance of the location; to be used together with fits file")
    return parser

def grb_parsing(parser):
    grb_input_parser = parser.add_argument_group(
        title="GRB analysis input arguments", description="Specify GRB analysis inputs"
    )

    ## Using Afterglowpy
    grb_input_parser.add( "--jet-type", type=int, default=0,
        help="GRB jet type (default: 0 (Gaussian))",)
    grb_input_parser.add("--grb-resolution", type=float, default=5,
        help="The upper bound on the ratio between thetaWing and thetaCore (default: 5)", )

    return parser

def modified_em_prior_parsing(parser):
    mod_em_prior_parser = parser.add_argument_group(
        title="EM Prior modification arguments", description="Specify modifications for the EM priors"
    )
    mod_em_prior_parser.add_argument("--conditional-gaussian-prior-thetaObs", action='store_true',
        help="The prior on the inclination is against to a gaussian prior centered at zero with sigma = thetaCore / N_sigma")
    mod_em_prior_parser.add_argument("--conditional-gaussian-prior-N-sigma", 
        default=1,type=float, help="The input for N_sigma; to be used with conditional-gaussian-prior-thetaObs set to True")
    mod_em_prior_parser.add_argument("--use-Ebv", action='store_true',
        help="If using the Ebv extinction during the inference")
    mod_em_prior_parser.add_argument("--Ebv-max", type=float, default=0.5724,
        help="Maximum allowed value for Ebv (default:0.5724)")
    mod_em_prior_parser.add_argument(
        "--fetch-Ebv-from-dustmap",action='store_true',
        help="Fetching Ebv from dustmap, to be used as fixed-value prior")
    return parser

def em_analysis_parsing(parser):
    parser = em_time_parsing(parser)
    parser = em_model_parsing(parser)
    parser = grb_parsing(parser)
    parser = modified_em_prior_parsing(parser)
    parser = multi_wavelength_parsing(parser)
    parser = em_injection_parsing(parser)
    return parser

def multi_wavelength_analysis_parser(parser):
    parser.description="Inference on transient parameters from multi-wavelength observations."
    parser.add_help=True
    
    parser = basic_em_only_parsing(parser)
    parser = base_analysis_parsing(parser)
    parser = em_analysis_parsing(parser)
    parser = basic_em_only_analysis_parsing(parser)
    parser = em_only_injection_parsing(parser)
    parser = skymap_parsing(parser)

    # specific arguments
    parser.add_argument("--remove-nondetections", action='store_true',
        help="remove non-detections from fitting analysis")
    
    return parser



def bolometric_parser(parser):
    
    parser.description="Inference on astronomical transient parameters with bolometric luminosity data."
    parser.add_help=True
    
    parser = basic_em_only_parsing(parser)
    parser = em_time_parsing(parser)
    parser = base_analysis_parsing(parser)
    parser = basic_em_only_analysis_parsing(parser)
    parser = modified_em_prior_parsing(parser)
    #FIXME: add injection to bol_ analysis, this currently does not work
    # parser = injection_parsing(parser)
    
    # specific arguments
    parser.add_argument("--error-budget", type=float, default=0.1,
        help="Bolometric error (default: 10 percent of relative error)")
    parser.add_argument("--em-model","--model", type=str,
        help="Name of the transient model to be used")
    
    return parser

def svd_training_parser(parser):
    parser.description="Train SVD models on kilonova light curves"
    parser = basic_em_only_parsing(parser)
    parser = em_time_parsing(parser)
    parser = em_model_parsing(parser)
    parser = data_processing_parsing(parser)
    parser = ml_training_parsing(parser)

    return parser

def svd_model_benchmark_parser(parser):
    parser.description="Benchmark the performance of SVD surrogate models"

    parser = em_model_parsing(parser)
    parser = data_processing_parsing(parser)

    parser.add_argument("--data-time-unit", default="days",
        help="Time unit of input data (days, hours, minutes, or seconds)" )
    parser.add_argument( "--tmin", type=nonefloat,
        help="Days to be started considering from the trigger time (default: set by model)" )
    parser.add_argument("--tmax", type=nonefloat, 
        help="Days to be stoped considering from the trigger time (default: set by model)" )
    parser.add_argument( "--ncpus", type=int, default=1,
        help="Number of CPU to be used (default: 1)" )
    parser.add_argument( "--outdir", default="benchmark_output",
        help="Path to the output directory" )
    parser.add_argument("--plot", action='store_false',
        help="Create histogram plots (default: True)")

    return parser

def benchmark_plots_parser():
    # this is so miniscule, just keep it like this
    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--outdir", default="benchmark_output",
        help="path to the output directory")
    parser.add_argument( "--search-pattern", default="*",
        help="type of interpolation", )
    return parser.parse_args()

def lc_validation_parser(parser):
    parser.description="Validation that a lightcurve meets a minimum number of observations within a set time."
    parser.add_help = True

    parser = basic_em_only_parsing(parser)

    parser.add_argument("--light-curve-data", "--data", type=str,
        help="Path to the data file in [time(isot) filter magnitude error] format")
    parser.add_argument("--filters", nargs="*",
        help="Comma separated list of filters to validate against. If not provided, all filters in the data will be used.",
    )
    parser.add_argument("--min-obs", default=3, type=int,
        help="Minimum number of observations required in each filter.",
    )
    parser.add_argument(
        "--cutoff-time", default=0., type=float,
        help="Cutoff time (relative to the first data point) that the minimum observations must be in. If not provided, the entire lightcurve will be evaluated",
    )
    return parser


def lightcurve_parser(parser):
    parser.description="Create lightcurves from injection parameters."

    parser = basic_em_only_parsing(parser)
    parser = em_time_parsing(parser)
    parser = grb_parsing(parser)
    parser = em_model_parsing(parser)
    parser = multi_wavelength_parsing(parser)

    parser = em_only_injection_parsing(parser)
    parser = em_injection_parsing(parser)
    
    return parser


def multi_lc_parser(parser):
    parser.add_argument( "--filters", nargs = "*", default=DEFAULT_FILTERS,
        help="filters for photometric lcs; must be from the bandpasses listed at" \
        " https://sncosmo.readthedocs.io/en/stable/bandpass-list.html")
    parser.add_argument("--modeldir", default="model",
        help="directory where data files are located" )
    parser.add_argument("--file-type", type=nonestr,
        help = "Source-file type to be handled, e.g. 'kasen', 'lanl' ")
    parser.add_argument( "--lcdir", default="lcs",
        help="output directory for generated lightcurves" )
    parser.add_argument("--doLbol", action='store_true', 
        help="extract bolometric lightcurves" )
    parser.add_argument( "--doAB", action='store_true',
        help="extract photometric lightcurves")
    parser.add_argument('--doSmoothing', action='store_true',
        help='Employ Savitzky-Golay filter for smoothing')
    parser.add_argument("--dMpc", type=float, default=1e-5,
        help="distance in Mpc, default is 10 pc to get lightcurves in Absolute Mag" )
    parser.add_argument("--z", type=float, 
        help="redshift, if provided it dominates over dMpc")

    return parser

def lc_grid_parser(parser):
    parser.description="Resample a grid of light curves, either by downsampling or fragmenting it."

    parser.add_argument("--gridpath", help="Path to grid files" )
    parser.add_argument("--base-dirname", default="lcs_grid", help="Base name of directory to save new grid(s)")
    parser.add_argument("--base-filename", default="lcs", help="Base name of file to save new grid(s)")
    parser.add_argument("--factor", type=int, default=10, help="Factor for downsampling or fragmenting (default: 10)")
    parser.add_argument("--downsample","--do-downsample", action="store_true",
        help="If set, downsample grid by --factor and save results")
    parser.add_argument("--fragment", "--do-fragment", action="store_true",
        help="If set, fragment grid into --factor chunks and save results")
    parser.add_argument("--shuffle", action="store_true", help="Whether to shuffle input before action")
    parser.add_argument("--remove", action="store_true", help="Delete all created directories in the end?")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for numpy")
    return parser

def lc_marginalisation_parser(parser):
    parser.description="Summary analysis for nmma injection file"

    parser = basic_em_only_parsing(parser)
    parser = em_time_parsing(parser)
    parser = em_model_parsing(parser)
    parser = grb_parsing(parser)

    # specific arguments
    parser.add_argument(
        "--template-file",  help="The template file to be used"
    )
    parser.add_argument("--hdf5-file", help="The hdf5 file to be used")
    parser.add_argument("--coinc-file",  help="The coinc xml file to be used")
    parser.add_argument("-g", "--gps", type=int, default=1187008882)
    parser.add_argument("-s", "--skymap")
    parser.add_argument("--eos-data", "--eos-dir",  
        help="EOS file directory in (radius [km], mass [solar mass], lambda)")
    parser.add_argument("-e", "--eos-weights", "--gw170817-eos", type=str)
    parser.add_argument("-n", "--Nmarg", type=int, default=100)
    parser.add_argument("--generation-seed", type=int, default=42, help="Injection generation seed (default: 42)")
    return parser

def multi_config_parser(parser):
    parser.description="Multi config analysis script for NMMA."
    
    parser.add_argument("--config", type=str,
        help="Name of the configuration file containing parameter values." )
    parser.add_argument("--parallel", action="store_true", default=False,
        help="To run multiple configs in parallel" )
    parser.add_argument("-p", "--process", type=int, help=(
        "No of processess each configuration should have, if --parallel is set then process will be divided equally among all configs,"
        "else each config will run sequentially with given no of process. Strictly required if  --process-per-config is not given" ))
    parser.add_argument("--process-per-config", type=int, help=(
            "If multiple configurations are given, how many MPI process should be assigned to each configuration. In the yaml"
            " file, indicate the number of process for each configuration with the key 'process-per-config'. If not given, all"
            " configurations will be run depending on the state and value of --parallel and --process. This takes precedence"
            " over --process"
        ), )

    return parser

def slurm_parser(parser):
    parser.description="Create lightcurve files from nmma injection file"
    parser = pipe_inj_parsing(parser)
    
    parser.add_argument("--injection-file","--injection",
        required=True, help="The bilby injection json file to be used")
    parser.add_argument("-o", "--outdir", default="outdir", 
        help="Path to the output directory")
    parser.add_argument("--analysis-file", required=True,
        help="The analysis bash script to be replicated")
    parser.add_argument("--lightcurves-per-job", type=int, default=100,
        help="Number of light curves per job")
    ### Dummy for intermediate setup until we decide on whether to abandon or extend these routines
    parser.add_argument("--simple-setup", default=True, choices=[True])
    return parser


def parsing_and_logging(parser_func, args= None):

    if not isinstance(args, argparse.Namespace):
        args = nmma_base_parsing(parser_func, args)

    if getattr(args, 'sampler', None) == "pymultinest":
        if len(args.outdir) > 64:
            raise ValueError("output directory name is longer than 64 characters" )

    if getattr(args, 'refresh_model_list', False):
        refresh_models_list(args.svd_path)

    setup_logger(outdir=args.outdir, label=args.label)
    os.makedirs(args.outdir, exist_ok=True)
    print('Setting up logger and storage directory')
    return args
