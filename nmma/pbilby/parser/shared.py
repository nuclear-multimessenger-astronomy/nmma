import argparse

import bilby
import bilby_pipe
from numpy import inf

logger = bilby.core.utils.logger

from ..._version import __version__


class StoreBoolean(argparse.Action):
    """argparse class for robust handling of booleans with configargparse

    When using configargparse, if the argument is setup with
    action="store_true", but the default is set to True, then there is no way,
    in the config file to switch the parameter off. To resolve this, this class
    handles the boolean properly.

    """

    def __call__(self, parser, namespace, value, option_string=None):
        value = str(value).lower()
        if value in ["true"]:
            setattr(namespace, self.dest, True)
        else:
            setattr(namespace, self.dest, False)


def _create_base_nmma_parser(sampler="dynesty"):
    base_parser = argparse.ArgumentParser("base", add_help=False)
    base_parser.add(
        "--version",
        action="version",
        version=f"%(prog)s={__version__}\nbilby={bilby.__version__}",
    )
    if sampler in ["all", "dynesty"]:
        base_parser = _add_dynesty_settings_to_parser(base_parser)
    base_parser = _add_em_settings_to_parser(base_parser)
    base_parser = _add_eos_settings_to_parser(base_parser)
    base_parser = _add_Hubble_settings_to_parser(base_parser)
    base_parser = _add_misc_settings_to_parser(base_parser)
    return base_parser


def _create_base_nmma_gw_parser(sampler="dynesty"):
    base_parser = argparse.ArgumentParser("base", add_help=False)
    base_parser.add(
        "--version",
        action="version",
        version=f"%(prog)s={__version__}\nbilby={bilby.__version__}",
    )
    if sampler in ["all", "dynesty"]:
        base_parser = _add_dynesty_settings_to_parser(base_parser)
    base_parser = _add_eos_settings_to_parser(base_parser)
    base_parser = _add_Hubble_settings_to_parser(base_parser)
    base_parser = _add_misc_settings_to_parser(base_parser)
    return base_parser


def _add_em_settings_to_parser(parser):
    em_input_parser = parser.add_argument_group(
        title="EM analysis input arguments", description="Specify EM analysis inputs"
    )
    em_input_parser.add(
        "--light-curve-data", type=str, help="Path to the observed light curve data"
    )
    em_input_parser.add(
        "--with-grb",
        action="store_true",
        help="Flag for including GRB afterglow in analysis",
    )
    em_input_parser.add(
        "--grb-jettype",
        type=int,
        default=0,
        help="GRB jet type (default: 0 (Gaussian))",
    )
    em_input_parser.add(
        "--grb-resolution",
        type=float,
        default=5,
        help="The upper bound on the ratio between thetaWing and thetaCore (default: 5)",
    )
    em_input_parser.add(
        "--kilonova-model", type=str, help="Name of the kilonova model to be used"
    )
    em_input_parser.add(
        "--kilonova-model-svd", type=str, help="Path to the kilonova model's SVD data"
    )
    em_input_parser.add(
        "--kilonova-interpolation-type",
        type=str,
        help="Interpolation method to be used for KN model (sklearn_gp or tensorflow)"
    )
    em_input_parser.add(
        "--svd-mag-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for mag evaluation (default: 10)",
    )
    em_input_parser.add(
        "--svd-lbol-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for lbol evaluation (default: 10)",
    )
    em_input_parser.add(
        "--filters",
        type=str,
        default=None,
        help="A comma seperated list of filters to use (e.g. g,r,i)."
        "If none is provided, will use all the filters available",
    )
    em_input_parser.add(
        "--with-grb-injection",
        action="store_true",
        help="Flag for including GRB afterglow in injection",
    )
    em_input_parser.add(
        "--kilonova-injection-model",
        type=str,
        help="Name of the kilonova model to be used for injection",
    )
    em_input_parser.add(
        "--kilonova-injection-svd",
        type=str,
        help="Path to the kilonova model's SVD data for injection",
    )
    em_input_parser.add(
        "--injection-svd-mag-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for mag evaluation (default: 10)",
    )
    em_input_parser.add(
        "--injection-svd-lbol-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for lbol evaluation (default: 10)",
    )
    em_input_parser.add(
        "--injection-detection-limit",
        type=float,
        default=28,
        help="Above which the injection light curve is expected to be not detectable (default: 28)",
    )
    em_input_parser.add(
        "--kilonova-trigger-time",
        type=float,
        help="Time for the kilonova trigger (in MJD)",
    )
    em_input_parser.add(
        "--kilonova-tmin",
        type=float,
        default=0.0,
        help="Days to be started analysing from the trigger time (default: 0)",
    )
    em_input_parser.add(
        "--kilonova-tmax",
        type=float,
        default=14.0,
        help="Days to be stoped analysing from the trigger time (default: 14)",
    )
    em_input_parser.add(
        "--kilonova-tstep",
        type=float,
        default=0.1,
        help="Time step (in days) for light curve initial evalution (default: 0.1)",
    )
    em_input_parser.add(
        "--kilonova-error",
        type=float,
        default=1.0,
        help="Additionaly statistical error (mag) to be introdouced (default: 1)",
    )

    return parser


def _add_eos_settings_to_parser(parser):
    eos_input_parser = parser.add_argument_group(
        title="EOS input arguments", description="Specify EOS inputs"
    )
    eos_input_parser.add("--binary-type", type=str, help="The binary is BNS or NSBH")
    eos_input_parser.add(
        "--with-eos",
        action="store_true",
        default=True,
        help="Flag for sampling over EOS (default:True)",
    )
    eos_input_parser.add("--eos-data", type=str, help="Path to the EOS directory")
    eos_input_parser.add("--Neos", type=int, help="Number of EOSs to be used")
    eos_input_parser.add(
        "--eos-weight", type=str, help="Path to the precalculated EOS weighting"
    )

    return parser


def _add_Hubble_settings_to_parser(parser):
    H0_input_parser = parser.add_argument_group(
        title="Hubble input arguments", description="Specify Hubble inputs"
    )
    H0_input_parser.add(
        "--with-Hubble",
        action="store_true",
        default=False,
        help="Flag for sampling over Hubble constants (default:False)",
    )
    H0_input_parser.add(
        "--Hubble-weight", type=str, help="Path to the precalculated Hubble weighting"
    )

    return parser


def _add_dynesty_settings_to_parser(parser):
    dynesty_group = parser.add_argument_group(title="Dynesty Settings")
    dynesty_group.add_argument(
        "-n", "--nlive", default=1000, type=int, help="Number of live points"
    )
    dynesty_group.add_argument(
        "--dlogz",
        default=0.1,
        type=float,
        help="Stopping criteria: remaining evidence, (default=0.1)",
    )
    dynesty_group.add_argument(
        "--n-effective",
        default=inf,
        type=float,
        help="Stopping criteria: effective number of samples, (default=inf)",
    )
    dynesty_group.add_argument(
        "--dynesty-sample",
        default="acceptance-walk",
        type=str,
        help="Dynesty sampling method (default=acceptance-walk). "
        "Note, the dynesty rwalk method is overwritten by parallel bilby for an optimised version ",
    )
    dynesty_group.add_argument(
        "--dynesty-bound",
        default="live",
        type=str,
        help="Dynesty bounding method (default=multi)",
    )
    dynesty_group.add_argument(
        "--walks",
        default=100,
        type=int,
        help="Minimum number of walks, defaults to 100",
    )
    dynesty_group.add_argument(
        "--proposals",
        type=bilby_pipe.utils.nonestr,
        action="append",
        default=None,
        help="The jump proposals to use, the options are 'diff' and 'volumetric'",
    )
    dynesty_group.add_argument(
        "--maxmcmc",
        default=5000,
        type=int,
        help="Maximum number of walks, defaults to 5000",
    )
    dynesty_group.add_argument(
        "--nact",
        default=2,
        type=int,
        help="Number of autocorrelation times to take, defaults to 2",
    )
    dynesty_group.add_argument(
        "--naccept",
        default=60,
        type=int,
        help="The average number of accepted steps per MCMC chain, defaults to 60",
    )
    dynesty_group.add_argument(
        "--min-eff",
        default=10,
        type=float,
        help="The minimum efficiency at which to switch from uniform sampling.",
    )
    dynesty_group.add_argument(
        "--facc", default=0.5, type=float, help="See dynesty.NestedSampler"
    )
    dynesty_group.add_argument(
        "--enlarge", default=1.5, type=float, help="See dynesty.NestedSampler"
    )
    dynesty_group.add_argument(
        "--n-check-point",
        default=1000,
        type=int,
        help="Steps to take before attempting checkpoint",
    )
    dynesty_group.add_argument(
        "--max-its",
        default=10**10,
        type=int,
        help="Maximum number of iterations to sample for (default=1.e10)",
    )
    dynesty_group.add_argument(
        "--max-run-time",
        default=1.0e10,
        type=float,
        help="Maximum time to run for (default=1.e10 s)",
    )
    dynesty_group.add_argument(
        "--fast-mpi",
        default=False,
        type=bool,
        help="Fast MPI communication pattern (default=False)",
    )
    dynesty_group.add_argument(
        "--mpi-timing",
        default=False,
        type=bool,
        help="Print MPI timing when finished (default=False)",
    )
    dynesty_group.add_argument(
        "--mpi-timing-interval",
        default=0,
        type=int,
        help="Interval to write timing snapshot to disk (default=0 -- disabled)",
    )
    dynesty_group.add_argument(
        "--nestcheck",
        default=False,
        action="store_true",
        help=(
            "Save a 'nestcheck' pickle in the outdir (default=False). "
            "This pickle stores a `nestcheck.data_processing.process_dynesty_run` "
            "object, which can be used during post processing to compute the "
            "implementation and bootstrap errors explained by Higson et al (2018) "
            "in “Sampling Errors In Nested Sampling Parameter Estimation”."
        ),
    )
    dynesty_group.add_argument(
        "--rejection-sample-posterior",
        default=True,
        action=StoreBoolean,
        help=(
            "Whether to generate the posterior samples by rejection sampling the "
            "nested samples or resampling with replacement"
        ),
    )
    return parser


def _add_slurm_settings_to_parser(parser):
    slurm_group = parser.add_argument_group(title="Slurm Settings")
    slurm_group.add_argument(
        "--nodes", type=int, default=1, help="Number of nodes to use (default 1)"
    )
    slurm_group.add_argument(
        "--ntasks-per-node",
        type=int,
        default=2,
        help="Number of tasks per node (default 2)",
    )
    slurm_group.add_argument(
        "--time",
        type=str,
        default="24:00:00",
        help="Maximum wall time (defaults to 24:00:00)",
    )
    slurm_group.add_argument(
        "--mem-per-cpu",
        type=str,
        default="2G",
        help="Memory per CPU (defaults to 2GB)",
    )
    slurm_group.add_argument(
        "--extra-lines",
        type=str,
        default=None,
        help="Additional lines, separated by ';', use for setting up conda env or module imports",
    )
    slurm_group.add_argument(
        "--slurm-extra-lines",
        type=str,
        default=None,
        help="additional slurm args (args that need #SBATCH in front) of the form arg=val separated by sapce",
    )
    return parser


def _add_misc_settings_to_parser(parser):
    misc_group = parser.add_argument_group(title="Misc. Settings")
    misc_group.add_argument(
        "--bilby-zero-likelihood-mode", default=False, action="store_true"
    )
    misc_group.add_argument(
        "--sampling-seed",
        type=bilby_pipe.utils.noneint,
        default=None,
        help="Random seed for sampling, parallel runs will be incremented",
    )
    misc_group.add_argument(
        "-c", "--clean", action="store_true", help="Run clean: ignore any resume files"
    )
    misc_group.add_argument(
        "--no-plot",
        action="store_true",
        help="If true, don't generate check-point plots",
    )
    misc_group.add_argument(
        "--do-not-save-bounds-in-resume",
        default=True,
        action="store_true",
        help=(
            "If true, do not store bounds in the resume file. This can make "
            "resume files large (~GB)"
        ),
    )
    misc_group.add_argument(
        "--check-point-deltaT",
        default=3600,
        type=float,
        help="Write a checkpoint resume file and diagnostic plots every deltaT [s]. Default: 1 hour.",
    )
    misc_group.add_argument(
        "--rotate-checkpoints",
        action="store_true",
        help="If true, backup checkpoint before overwriting (ending in '.bk').",
    )
    return parser
