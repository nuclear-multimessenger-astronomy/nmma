import argparse
from ..utils.models import refresh_models_list
from bilby.core import utils

    
def parsing_and_logging(parser_func, args= None):
    if args is None:
        parser = parser_func()
        args = parser.parse_args()

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
    parser = em_analysis_meta_parsing(parser)
    parser = grb_parsing(parser)
    parser = sampling_parsing(parser)
    parser = injection_parsing(parser)
    parser = skymap_parsing(parser)
    parser = ztf_parsing(parser)
    parser = rubin_parsing(parser)
    parser = photometry_augmentation_parsing(parser)
    parser = modified_em_prior_parsing(parser)

    parser.add_argument("--config", type=str, help="Name of the configuration file containing parameter values.")
    parser.add_argument("--trigger-time", type=float,
        help="Trigger time in modified julian day, not required if injection set is provided")
    parser.add_argument("--data", type=str, help="Path to data in [time(isot) filter magnitude error] format")
    parser.add_argument("--prior", type=str, help="Path to the prior file")
    parser.add_argument(
        "--photometric-error-budget",
        type=float,
        default=0.1,
        help="Photometric error (mag) (default: 0.1)",
    )
    parser.add_argument(
        "--filters",
        type=str,
        help="A comma seperated list of filters to use (e.g. g,r,i). If none is provided, will use all the filters available",
    )
    parser.add_argument(
        "--error-budget",
        type=str,
        default="1.0",
        help="Additional systematic error (mag) to be introduced (default: 1)",
    )
    parser.add_argument(
        "--injection-model",
        type=str,
        help="Name of the kilonova model to be used for injection (default: the same as model used for recovery)",
    )
    parser.add_argument(
        "--remove-nondetections",
        action="store_true",
        default=False,
        help="remove non-detections from fitting analysis",
    )
    parser.add_argument(
        "--detection-limit",
        metavar="DICT",
        type=str,
        default=None,
        help="Dictionary for detection limit per filter, e.g., {'r':22, 'g':23}, put a double quotation marks around the dictionary",
    )
    parser.add_argument(
        "--prompt-collapse",
        help="If the injection simulates prompt collapse and therefore only dynamical",
        action="store_true",
    )
    parser.add_argument(
        "--train-stats",
        help="Creates a file too.csv to derive statistics",
        action="store_true",
    )
    parser.add_argument(
        "--bilby-zero-likelihood-mode",
        action="store_true",
        default=False,
        help="enable prior run",
    )

    parser.add_argument(
        "--skip-sampling",
        help="If analysis has already run, skip bilby sampling and compute results from checkpoint files. Combine with --plot to make plots from these files.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--sample-over-Hubble",
        action="store_true",
        default=False,
        help="To sample over Hubble constant and redshift",
    )
    parser.add_argument(
        "--bestfit",
        help="Save the best fit parameters and magnitudes to JSON",
        action="store_true",
        default=False,
    )
    return parser


def lightcurve_parser():
    parser = argparse.ArgumentParser(description="Create lightcurves from injection parameters.")
    parser = em_analysis_meta_parsing(parser)
    parser = injection_parsing(parser)
    parser = grb_parsing(parser)
    parser = ztf_parsing(parser)
    parser = rubin_parsing(parser)
    parser = photometry_augmentation_parsing(parser)

    parser.add_argument("--filters",type=str, default="ztfr,ztfg,ztfi",
        help="A comma seperated list of filters to use (e.g. sdssu,2massh,2massj). If none is provided, will use all the default filters")
    parser.add_argument(
        "--photometric-error-budget",
        type=float,
        default=0.0,
        help="Photometric error (mag) (default: 0.0)",
    )
    parser.add_argument(
        "--train-stats",
        help="Creates a file too.csv to derive statistics",
        action="store_true",
    )
    parser.add_argument(
        "--increment-seeds",
        help="Change seed for every injection",
        action="store_true",
    )

    return parser

def lc_marginalisation_parser():
    parser = argparse.ArgumentParser(
        description="Summary analysis for nmma injection file"
    )
    parser = em_analysis_meta_parsing(parser)
    parser = injection_parsing(parser)
    parser.add_argument(
        "--template-file", type=str, help="The template file to be used"
    )
    parser.add_argument("--hdf5-file", type=str, help="The hdf5 file to be used")
    parser.add_argument("--coinc-file", type=str, help="The coinc xml file to be used")
    parser.add_argument("-g", "--gps", type=int, default=1187008882)
    parser.add_argument("-s", "--skymap", type=str,)
    parser.add_argument("--eos-dir", type=str, required=True, 
                        help="EOS file directory in (radius [km], mass [solar mass], lambda)",)
    parser.add_argument("-e", "--eos-weights", "--gw170817-eos", type=str)
    parser.add_argument("-n", "--Nmarg", type=int, default=100)
    parser.add_argument(
        "--filters",
        type=str,
        help="A comma seperated list of filters to use (e.g. g,r,i). If none is provided, will use all the filters available",
        default="u,g,r,i,z,y,J,H,K",
    )
    return parser


def em_analysis_meta_parsing(parser):
    parser.add_argument("--model", type=str, required=True, help="Name of the kilonova model to be used")
    parser.add_argument("--joint-light-curve", action="store_true",
        help="Flag for using both kilonova and GRB afterglow light curve")
    parser.add_argument("--interpolation-type", "--gptype", type=str, default="sklearn",
        help="SVD interpolation scheme.")
    parser.add_argument("--refresh-models-list", type=bool, default=False,
        help="Refresh the list of models available on Gitlab")
    parser.add_argument("--local-only", action="store_true", default=False,
        help="only look for local svdmodels (ignore Gitlab)")
    parser.add_argument("--absolute", action="store_true", default=False, help="Use Absolute Magnitude?")
    
    parser.add_argument( "--tmin", type=float, default=0.0,
        help="Days to be started analysing from the trigger time (default: 0)")
    parser.add_argument("--tmax", type=float, default=14.0,
        help="Days to be stoped analysing from the trigger time (default: 14)")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step in day (default: 0.1)")
    parser.add_argument("--log-space-time", action="store_true", default=False, 
        help="Create the sample_times to be uniform in log-space")
    parser.add_argument("--n-tstep", type=int, default=50,
        help="Number of time steps (used with --log-space-time, default: 50)",
    )
    
    parser.add_argument( "--svd-path", type=str,  default="svdmodels", help="Path to the SVD directory with {model}.joblib")
    parser.add_argument("--svd-mag-ncoeff", type=int,  default=10, 
        help="Number of eigenvalues to be taken for mag evaluation (default: 10)")
    parser.add_argument("--svd-lbol-ncoeff", type=int, default=10,
        help="Number of eigenvalues to be taken for lbol evaluation (default: 10)")
    
    parser.add_argument("--outdir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--label", type=str, required=True, help="Label for the run")
    parser.add_argument("--plot", action="store_true", default=False, help="add best fit plot")
    parser.add_argument( "--xlim", type=str, default="0,14", nargs="*",
        help="Start and end time for light curve plot (default: 0-14)")
    parser.add_argument("--ylim", type=str, default="22,16", nargs="*",
        help="Upper and lower magnitude limit for light curve plot (default: 22,16)")
    parser.add_argument("--verbose",  action="store_true", default=False, help="print out log likelihoods")

    return parser

def injection_parsing(parser):
    parser.add_argument("--injection", metavar="PATH", type=str, help="Path to the injection json file")
    parser.add_argument("--injection-num", type=int,
        help="The injection number to be taken from the injection set")
    parser.add_argument("--generation-seed", type=int, default=42, help="Injection generation seed (default: 42)")
    parser.add_argument("--injection-outfile",type=str, help="Path to the output injection lightcurve")
    parser.add_argument("--outfile-type",type=str,default="csv", help="Type of output files, json or csv.")
    parser.add_argument("--with-grb-injection", help="If the injection has grb included", action="store_true")
    parser.add_argument("--ignore-timeshift", action="store_true", default=False,
        help="If you want to ignore the timeshift parameter in an injection file.")
    parser.add_argument("--injection-detection-limit", metavar="mAB", type=str, default=None,
        help="The highest mAB to be presented in the injection data set, any mAB higher than this will become a non-detection limit. Should be comma delimited list same size as injection set.")

    return parser

def sampling_parsing(parser):
    parser.add_argument("--sampler", type=str, default="pymultinest",
        help="Sampler to be used (default: pymultinest)")
    parser.add_argument("--sampler-kwargs", default="{}", type=str, 
        help="Additional kwargs (e.g. {'evidence_tolerance':0.5}) for bilby.run_sampler, put a double quotation marks around the dictionary")
    parser.add_argument("--soft-init", action="store_true", default=False, 
        help="To start the sampler softly (without any checking, default: False)")
    parser.add_argument("--cpus", type=int, default=1,
        help="Number of cores to be used, only needed for dynesty (default: 1)")
    parser.add_argument("--nlive", type=int, default=2048, help="Number of live points (default: 2048)")
    parser.add_argument("--reactive-sampling", action="store_true", default=False,
        help="To use reactive sampling in ultranest (default: False)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")

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
    
    return parser

def ztf_parsing(parser):
    parser.add_argument("--ztf-sampling", help="Use realistic ZTF sampling", action="store_true")
    parser.add_argument("--ztf-uncertainties", help="Use realistic ZTF uncertainties", action="store_true")
    parser.add_argument( "--ztf-ToO", type=str, choices=["180", "300"], 
        help="Adds realistic ToO obeservations during the first one or two days. Sampling depends on exposure time specified. Valid values are 180 (<1000sq deg) or 300 (>1000sq deg). Won't work w/o --ztf-sampling")
    return parser

def rubin_parsing(parser):
    parser.add_argument( "--rubin-ToO", action="store_true",
        help="Adds ToO obeservations based on the strategy presented in arxiv.org/abs/2111.01945.")
    parser.add_argument("--rubin-ToO-type", type=str, choices=["BNS", "NSBH"], 
        help="Type of ToO observation. Won't work w/o --rubin-ToO")
    return parser


def photometry_augmentation_parsing(parser):
    parser.add_argument( "--photometry-augmentation", action="store_true",
        help="Augment photometry to improve parameter recovery")
    parser.add_argument( "--photometry-augmentation-seed", metavar="seed", type=int, default=0,
        help="Optimal generation seed (default: 0)")
    parser.add_argument("--photometry-augmentation-N-points", type=int,default=10,
        help="Number of augmented points to include")
    parser.add_argument("--photometry-augmentation-filters", type=str,
        help="A comma seperated list of filters to use for augmentation (e.g. g,r,i). If none is provided, will use all the filters available")
    parser.add_argument("--photometry-augmentation-times", type=str,
        help="A comma seperated list of times to use for augmentation in days post trigger time (e.g. 0.1,0.3,0.5). If none is provided, will use random times between tmin and tmax")
    return parser

def modified_em_prior_parsing(parser):
    parser.add_argument("--conditional-gaussian-prior-thetaObs", action="store_true", default=False,
        help="The prior on the inclination is against to a gaussian prior centered at zero with sigma = thetaCore / N_sigma")
    parser.add_argument("--conditional-gaussian-prior-N-sigma", default=1,type=float,
        help="The input for N_sigma; to be used with conditional-gaussian-prior-thetaObs set to True")
    parser.add_argument("--use-Ebv", action="store_true", default=False,
        help="If using the Ebv extinction during the inference")
    parser.add_argument("--Ebv-max", type=float, default=0.5724,
        help="Maximum allowed value for Ebv (default:0.5724)")
    return parser