import argparse
from ..utils.models import refresh_models_list
from bilby.core import utils
from ..pbilby.parser.shared import StoreBoolean

    
def parsing_and_logging(parser_func, args= None):
    if args is None:
        parser = parser_func()
        args = parser.parse_args()

    if getattr(args, 'sampler', None) == "pymultinest":
        if len(args.outdir) > 64:
            print(
                "WARNING: output directory name is too long, it should not be longer than 64 characters"
            )
            exit()



    if getattr(args, 'refresh_model_list', False):
        refresh_models_list(
            models_home=args.svd_path if args.svd_path not in [None, ""] else None
        )

    utils.setup_logger(outdir=args.outdir, label=args.label)
    utils.check_directory_exists_and_if_not_mkdir(args.outdir)
    print('Setting up logger and storage directory')
    return args

def em_analysis_parser(**kwargs):
    add_help = kwargs.get("add_help", True)
    parser = argparse.ArgumentParser(
        description="Inference on kilonova ejecta parameters.",
        add_help=add_help,
    )

    parser = basic_em_parsing(parser)
    parser = grb_parsing(parser)
    parser = basic_em_analysis_parsing(parser)
    parser = em_model_parsing(parser)
    parser = injection_parsing(parser)
    parser = sampling_parsing(parser)
    parser = ztf_parsing(parser)
    parser = rubin_parsing(parser)
    parser = skymap_parsing(parser)
    parser = photometry_augmentation_parsing(parser)
    parser = modified_em_prior_parsing(parser)

    # specific arguments
    parser.add_argument("--filters", type=str,
        help="A comma seperated list of filters to use (e.g. g,r,i). If none is provided, will use all the filters available")
    parser.add_argument(
        "--em-transient-error","--photometric-error-budget",  type=float, default=0.1,
        help="Photometric error (mag) (default: 0.1)")
    parser.add_argument("--error-budget", type=str, default="1.0",
        help="Additional systematic error (mag) to be introduced (default: 1)")
    parser.add_argument("--remove-nondetections", 
        action=StoreBoolean, default=False,
        help="remove non-detections from fitting analysis")
    
    ##FIXME use common string_to_dict function from bilby pipe
    parser.add_argument("--detection-limit", type=str, default=None,
        help="Dictionary for detection limit per filter, e.g., {'r':22, 'g':23}, put a double quotation mark around the dictionary")
    parser.add_argument(
        "--prompt-collapse",
        help="If the injection simulates prompt collapse and therefore only dynamical",
        action=StoreBoolean,
    )
    ##FIXME: re-implement in EM-Likelihood! 
    parser.add_argument("--systematics-file", default=None,
        help="Path to systematics configuration file",
    )
    return parser



def bolometric_parser(**kwargs):
    add_help = kwargs.get("add_help", True)

    parser = argparse.ArgumentParser(
        description="Inference on astronomical transient parameters with bolometric luminosity data.",
        add_help=add_help,
    )
    parser = basic_em_parsing(parser)
    parser = basic_em_analysis_parsing(parser)
    parser = modified_em_prior_parsing(parser)
    parser = sampling_parsing(parser)
    #FIXME: add injection to bol_ analysis
    # parser = injection_parsing(parser)
    
    # specific arguments
    parser.add_argument("--error-budget", type=float, default=0.1,
        help="Bolometric error (default: 10 percent of relative error)")
    
    return parser

def svd_training_parser():
    parser = argparse.ArgumentParser(description="Train SVD models on kilonova light curves")
    parser = basic_em_parsing(parser)
    parser = em_model_parsing(parser)
    parser = data_processing_parsing(parser)
    parser = ml_training_parsing(parser)

    ##specific arguments
    parser.add_argument("--filters", type=str,
        help="A comma seperated list of filters to use (e.g. g,r,i). If none is provided, will use all the filters available",)
    return parser


def lc_validation_parser(**kwargs):
    parser = argparse.ArgumentParser(
        description="Validation that a lightcurve meets a minimum number of observations within a set time.",
        add_help=True,
    )
    parser.add_argument("--data", type=str, required=True,
        help="Path to the data file in [time(isot) filter magnitude error] format")
    parser.add_argument("--filters", type=str,
        help="Comma separated list of filters to validate against. If not provided, all filters in the data will be used.",
    )
    parser.add_argument("--min-obs", default=3, type=int,
        help="Minimum number of observations required in each filter.",
    )
    parser.add_argument(
        "--cutoff-time", default=0., type=float,
        help="Cutoff time (relative to the first data point) that the minimum observations must be in. If not provided, the entire lightcurve will be evaluated",
    )
    parser.add_argument(
        "--silent",action=StoreBoolean,default=False,
        help="Suppress output")
    return parser


def lightcurve_parser():
    parser = argparse.ArgumentParser(description="Create lightcurves from injection parameters.")

    parser = basic_em_parsing(parser)
    parser = grb_parsing(parser)
    parser = em_model_parsing(parser)
    parser = injection_parsing(parser)
    parser = ztf_parsing(parser)
    parser = rubin_parsing(parser)
    parser = photometry_augmentation_parsing(parser)
    
    ##specific arguments
    parser.add_argument("--filters",type=str, default="ztfr,ztfg,ztfi",
        help="A comma seperated list of filters to use (e.g. sdssu,2massh,2massj). If none is provided, will use all the default filters")
    parser.add_argument("--em-transient-error","--photometric-error-budget", type=float, default=0.0,
        help="Photometric error (mag) (default: 0.0)")
    return parser


def lc_marginalisation_parser():
    parser = argparse.ArgumentParser(
        description="Summary analysis for nmma injection file"
    )

    parser = basic_em_parsing(parser)
    parser = em_model_parsing(parser)
    parser = grb_parsing(parser)
    parser = injection_parsing(parser)

    # specific arguments
    parser.add_argument(
        "--template-file", type=str, help="The template file to be used"
    )
    parser.add_argument("--hdf5-file", type=str, help="The hdf5 file to be used")
    parser.add_argument("--coinc-file", type=str, 
        help="The coinc xml file to be used")
    parser.add_argument("-g", "--gps", type=int, default=1187008882)
    parser.add_argument("-s", "--skymap", type=str,)
    parser.add_argument("--eos-dir", type=str, required=True, 
        help="EOS file directory in (radius [km], mass [solar mass], lambda)",)
    parser.add_argument("-e", "--eos-weights", "--gw170817-eos", type=str)
    parser.add_argument("-n", "--Nmarg", type=int, default=100)
    parser.add_argument("--filters", type=str, default="u,g,r,i,z,y,J,H,K",
        help="A comma seperated list of filters to use (e.g. g,r,i). If none is provided, will use all the filters available")
    parser.add_argument("--plot", action=StoreBoolean, default=False, 
        help="add best fit plot")
    parser.add_argument("--verbose", action=StoreBoolean, default=False,
        help="print out log likelihoods",
    )
    return parser

def slurm_parser(**kwargs):
    parser = argparse.ArgumentParser(
        description="Create lightcurve files from nmma injection file"
    )
    parser.add_argument("--injection","--injection-file",
        type=str, required=True,
        help="The bilby injection json file to be used")
    parser.add_argument("-o", "--outdir", type=str, default="outdir", 
        help="Path to the output directory")
    parser.add_argument("--analysis-file", type=str, required=True,
        help="The analysis bash script to be replicated")
    parser.add_argument("--lightcurves-per-job", type=int, default=100,
        help="Number of light curves per job")
    parser.add_argument("--prior-file", type=str, 
        help="The prior file from which to generate injections")
    return parser

def basic_em_parsing(parser):
    parser.add_argument("--model", type=str,
        help="Name of the transient model to be used")
    parser.add_argument("-o", "--outdir", type=str, default="outdir", 
        help="Path to the output directory")
    parser.add_argument("--label", type=str, default ="em_transient", 
        help="Label for the run")

    parser.add_argument( "--tmin", type=float, default=0.0,
        help="Analysis start time in days after the trigger (default: 0)")
    parser.add_argument("--tmax", type=float, default=14.0,
        help="Analysis stop time in days after the trigger. (default:14)")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step in day (default: 0.1)")
    return parser

def basic_em_analysis_parsing(parser):

    parser.add_argument("--config", type=str, 
        help="Name of the configuration file containing parameter values.")
    parser.add_argument("--trigger-time", type=float,
        help="Trigger time in modified julian day, not required if injection set is provided")
    parser.add_argument("--data", type=str, help="Path to data in [time(isot) filter magnitude error] format")
    parser.add_argument("--prior", type=str, help="Path to the prior file")

    parser.add_argument("--log-space-time", action=StoreBoolean, 
        default=False, help="Create the sample_times to be uniform in log-space")
    parser.add_argument("--n-tstep", type=int, default=50,
        help="Number of time steps (used with --log-space-time, default: 50)")

    parser.add_argument("--bilby-zero-likelihood-mode",
        action=StoreBoolean, default=False, help="enable prior run")
    parser.add_argument("--verbose",  action=StoreBoolean, default=False,
        help="print out log likelihoods")
    parser.add_argument("--skip-sampling", action=StoreBoolean, default=False, 
        help="If analysis has already run, skip bilby sampling and compute results from checkpoint files. Combine with --plot to make plots from these files.")
    parser.add_argument("--bestfit", action=StoreBoolean, default=False,
        help="Save the best fit parameters and magnitudes to JSON")
    parser.add_argument("--plot", action=StoreBoolean, default=False, 
        help="add best fit plot")
    return parser

def em_model_parsing(parser):
    parser.add_argument("--interpolation-type", "--gptype", type=str, 
        default="sklearn", help="SVD interpolation scheme.")
    parser.add_argument("--refresh-models-list", type=bool, default=False,
        help="Refresh the list of models available on Gitlab")
    parser.add_argument("--local-only", action=StoreBoolean, default=False,
        help="only look for local svdmodels (ignore Gitlab)")
    parser.add_argument("--absolute", action=StoreBoolean, default=False, 
        help="Use Absolute Magnitude?")
        
    parser.add_argument( "--svd-path", type=str,  default="svdmodels",
         help="Path to the SVD directory with {model}.joblib")
    parser.add_argument(
        "--svd-mag-ncoeff","--svd-ncoeff", type=int,  default=10, 
        help="Number of eigenvalues to be taken for mag evaluation (default: 10)")
    parser.add_argument("--svd-lbol-ncoeff", type=int, default=10,
        help="Number of eigenvalues to be taken for lbol evaluation (default: 10)")
    parser.add_argument( "--xlim", type=str, default="0,14", nargs="*",
        help="Start and end time for light curve plot (default: 0-14)")
    parser.add_argument("--ylim", type=str, default="22,16", nargs="*",
        help="Upper and lower magnitude limit for light curve plot (default: 22,16)")

    return parser

def data_processing_parsing(parser):
    parser.add_argument( "--data-path", type=str,
        help="Path to the directory of light curve files")
    parser.add_argument("--data-file-type", type=str, default="bulla",
        help="Type of light curve files [bulla, standard, ztf, hdf5]", )
    parser.add_argument(
        type=str, default="photometry", choices=["photometry", "spectroscopy"],
        help="Data type for interpolation [photometry or spectroscopy]")
    parser.add_argument(
        "--ignore-bolometric", action=StoreBoolean, default=False,
        help="ignore bolometric light curve files (ending in _Lbol.file_extension)")
    parser.add_argument("--lmin",type=float,default=3000.0,
        help="Minimum wavelength to analyze (default: 3000)",)
    parser.add_argument("--lmax", type=float, default=10000.0,
        help="Maximum wavelength to analyze (default: 10000)",)

    return parser


def ml_training_parsing(parser):
    parser.add_argument(
        "--nepochs","--tensorflow-nepochs", type=int, default=15,
        help="Number of epochs for tensorflow training (default: 15)",)
    parser.add_argument("--ncpus", default=1, type=int,
        help="number of cpus to be used, only support for sklearn_gp")
    parser.add_argument(
        "--data-time-unit", type=str, default="days",
        help="Time unit of input data (days, hours, minutes, or seconds)")
    parser.add_argument(
        "--use-UnivariateSpline", action=StoreBoolean, default=False,
        help="using UnivariateSpline to mitigate the numerical noise in the grid")
    parser.add_argument(
        "--UnivariateSpline-s", default=2, type=int,
        help="s-value to be used for UnivariateSpline")
    parser.add_argument("--plot", action=StoreBoolean, default=False,)
    parser.add_argument("--random-seed", type=int, default=42,
        help="random seed to set during training",)
    parser.add_argument(
        "--axial-symmetry", action=StoreBoolean, default=False,
        help="add training samples based on the fact that there is axial symmetry")
    parser.add_argument(
        "--continue-training", action=StoreBoolean, default=False,
        help="Continue training an existing model",)
    return parser



def injection_parsing(parser):
    parser.add_argument("--injection", metavar="PATH", type=str, 
        help="Path to the injection json file")
    parser.add_argument("--injection-model", type=str,
        help="Name of the kilonova model to be used for injection (default: the same as model used for recovery)")
    parser.add_argument("--injection-num", type=int,
        help="The injection number to be taken from the injection set")
    parser.add_argument("--generation-seed", type=int, default=42, help="Injection generation seed (default: 42)")
    parser.add_argument("--increment-seeds", action=StoreBoolean,
        help="Change seed for every injection")
    parser.add_argument("--injection-outfile",type=str, help="Path to the output injection lightcurve")
    parser.add_argument("--outfile-type",type=str,default="csv", help="Type of output files, json or csv.")
    parser.add_argument("--ignore-timeshift", action=StoreBoolean, default=False,
        help="If you want to ignore the timeshift parameter in an injection file.")
    parser.add_argument("--injection-detection-limit", metavar="mAB", type=str, default=None,
        help="The highest mAB to be presented in the injection data set, any mAB higher than this will become a non-detection limit. Should be comma delimited list same size as injection set.")
    parser.add_argument("--train-stats", action=StoreBoolean,
        help="Creates a file too.csv to derive statistics")

    return parser

def sampling_parsing(parser):
    parser.add_argument("--sampler", type=str, default="pymultinest",
        help="Sampler to be used (default: pymultinest)")
    parser.add_argument("--sampler-kwargs", default="{}", type=str, 
        help="Additional kwargs (e.g. {'evidence_tolerance':0.5}) for bilby.run_sampler, put a double quotation marks around the dictionary")
    parser.add_argument("--soft-init", action=StoreBoolean, default=False, 
        help="To start the sampler softly (without any checking, default: False)")
    parser.add_argument("--cpus", type=int, default=1,
        help="Number of cores to be used, only needed for dynesty (default: 1)")
    parser.add_argument("--nlive", type=int, default=2048, help="Number of live points (default: 2048)")
    parser.add_argument("--reactive-sampling", action=StoreBoolean, default=False,
        help="To use reactive sampling in ultranest (default: False)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")
    parser.add_argument("--sample-over-Hubble", 
        action=StoreBoolean, default=False,
        help="To sample over Hubble constant and redshift")

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
    parser.add_argument( "--grb-resolution", type=float, default=5,
        help="The upper bound on the ratio between thetaWing and thetaCore (default: 5)")
    parser.add_argument(  "--jet-type", type=int,  default=0,
        help="Jet type to used used for GRB afterglow light curve (default: 0)")
    parser.add_argument("--joint-light-curve", action=StoreBoolean,
        help="Flag for using both kilonova and GRB afterglow light curve")
    parser.add_argument("--with-grb-injection", action=StoreBoolean, 
        help="If the injection has grb included")
    
    return parser

def ztf_parsing(parser):
    parser.add_argument("--ztf-sampling", help="Use realistic ZTF sampling", action=StoreBoolean)
    parser.add_argument("--ztf-uncertainties", help="Use realistic ZTF uncertainties", action=StoreBoolean)
    parser.add_argument( "--ztf-ToO", type=str, choices=["180", "300"], 
        help="Adds realistic ToO obeservations during the first one or two days. Sampling depends on exposure time specified. Valid values are 180 (<1000sq deg) or 300 (>1000sq deg). Won't work w/o --ztf-sampling")
    return parser

def rubin_parsing(parser):
    parser.add_argument( "--rubin-ToO", action=StoreBoolean,
        help="Adds ToO obeservations based on the strategy presented in arxiv.org/abs/2111.01945.")
    parser.add_argument("--rubin-ToO-type", type=str, 
        choices=["platinum","gold","gold_z","silver","silver_z"], 
        help="Type of ToO observation. Won't work w/o --rubin-ToO")
    return parser


def photometry_augmentation_parsing(parser):
    parser.add_argument( "--photometry-augmentation", action=StoreBoolean,
        help="Augment photometry to improve parameter recovery")
    parser.add_argument( "--photometry-augmentation-seed", metavar="seed", 
        type=int, default=0,
        help="Optimal generation seed (default: 0)")
    parser.add_argument("--photometry-augmentation-N-points", type=int,
        default=10, help="Number of augmented points to include")
    parser.add_argument("--photometry-augmentation-filters", type=str,
        help="A comma seperated list of filters to use for augmentation (e.g. g,r,i). If none is provided, will use all the filters available")
    parser.add_argument("--photometry-augmentation-times", type=str,
        help="A comma seperated list of times to use for augmentation in days post trigger time (e.g. 0.1,0.3,0.5). If none is provided, will use random times between tmin and tmax")
    return parser

def modified_em_prior_parsing(parser):
    parser.add_argument("--conditional-gaussian-prior-thetaObs", 
        action=StoreBoolean, default=False,
        help="The prior on the inclination is against to a gaussian prior centered at zero with sigma = thetaCore / N_sigma")
    parser.add_argument("--conditional-gaussian-prior-N-sigma", 
        default=1,type=float, help="The input for N_sigma; to be used with conditional-gaussian-prior-thetaObs set to True")
    parser.add_argument("--use-Ebv", action=StoreBoolean, default=False,
        help="If using the Ebv extinction during the inference")
    parser.add_argument("--Ebv-max", type=float, default=0.5724,
        help="Maximum allowed value for Ebv (default:0.5724)")
    parser.add_argument(
        "--fetch-Ebv-from-dustmap",action=StoreBoolean, default=False,
        help="Fetching Ebv from dustmap, to be used as fixed-value prior")
    return parser