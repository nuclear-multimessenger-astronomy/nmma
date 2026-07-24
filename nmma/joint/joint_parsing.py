from ..core.parsing import nonestr, base_injection_parsing, pipe_inj_parsing
from ..em.em_parsing import em_analysis_parsing
from ..eos.eos_parsing import tabulated_eos_parsing, eos_parsing
from ..gw.gw_parsing import gw_injection_parsing


def injection_parsing(parser):

    parser.description = (
        "Create a file of nmma-injections, processing a bilby injection if needed"
    )
    parser = base_injection_parsing(parser)
    parser = pipe_inj_parsing(parser)
    parser = tabulated_eos_parsing(parser)
    parser = eos_parsing(parser)
    parser = em_analysis_parsing(parser)
    parser = gw_injection_parsing(parser)
    parser = joint_likelihood_parsing(parser)

    # NMMA-added options
    parser.add_argument(
        "--max-redraws",
        type=int,
        default=10,
        help=(
            "The maximum number of times to redraw the injection "
            "if additional constraints are not satisfied"
        ),
    )
    parser.add_argument(
        "--simple-setup",
        action="store_true",
        help="avoid various complications of the code for tests and post-processing",
    )
    parser.add_argument(
        "--original-parameters",
        action="store_true",
        help="Whether to only store parameters given by injection prior",
    )
    parser.add_argument(
        "--post-processing",
        nargs="*",
        default=[],
        help="Postprocessing steps to apply to the injection data."
        "Appropriate args need to be specified to. These can be "
        "   - 'snr' to append the SNR, requires args for waveform generator"
        "   - 'lightcurve' to compute and store an associated lightcurve",
    )
    parser.add_argument(
        "--tests",
        nargs="*",
        default=[],
        help="Tests to apply to the injection data. These can be "
        "   - 'snr' to calculate the SNR, "
        "   - 'detection_limit' to apply the detection limit, "
        "   - 'ejecta' to check for ejecta, "
        "   - 'peak_magnitude' to check for peak magnitude, or "
        "   - 'eos' to check for EOS."
        "If not given, no tests are applied.",
    )
    parser.add_argument(
        "-o", "--outdir", default="outdir", help="Path to the output directory"
    )
    parser.add_argument(
        "--lc-label",
        type=nonestr,
        help="optional label for lightcurve-files to be generated;"
        "default derives from injection-file",
    )

    parser.add_argument(
        "--peak-magnitude",
        type=nonestr,
        help="Accept injection only if its peak magnitude matches some setting."
        "If 'any', the lightcurve has to pass the detection limit in any filter."
        "If 'all', the lightcurve has to pass the detection limit in all filters."
        "Can also be a dict of filters and magnitudes that each have to be reached.",
    )
    parser.add_argument(
        "--eos-file", help="EOS file (radius [km], mass [M_sun], lambda)."
    )
    parser.add_argument(
        "--binary-type",
        type=nonestr,
        choices=[None, "BNS", "NSBH"],
        default=None,
        help="FIXME Weizmann: restores nmma 0.2.3's --binary-type/--eject behaviour, "
        "removed when injection creation switched to the (redraw-based) --tests/--post-processing "
        "system, which cannot filter injections whose masses come from an external "
        "--gw-injection-file (a fixed mass can never be redrawn away from failing a test). "
        "Requires --eos-file. If given, the corresponding ejecta formula (BNS or NSBH) is "
        "applied uniformly to every injection, and any injection whose resulting ejecta mass "
        "is not finite (i.e. the assumed binary type isn't physically consistent with this EOS) "
        "is dropped -- a one-shot filter, applied once, not a redraw.",
    )
    # FIXME this is potentially misleading when used in conjunction with full analysis
    parser.add_argument(
        "--cosmology",
        help="Name of the cosmology to be used, see astropy.cosmology for available cosmologies (implicit default: Planck18)",
    )
    parser.add_argument(
        "--population-model",
        type=nonestr,
        default="uniform",
        help="The population model to be used for injections (default: uniform)",
    )

    # Parameters for legacy injection file used in conjunction
    parser.add_argument(
        "--gw-injection-file",
        type=nonestr,
        help="The xml injection file or bilby injection json file to be used (optional)",
    )
    parser.add_argument(
        "--reference-frequency",
        type=float,
        default=20,
        help="The reference frequency in the provided gw injection file (default: 20)",
    )

    return parser


def joint_likelihood_parsing(parser):
    parser.description = (
        "Set up a joint NMMA likelihood from provided messengers and analysis modifiers"
    )
    parser.add_argument(
        "--ejecta-conversion",
        action="store_true",
        help="Whether to set up ejecta conversions automatically if ejecta parameters are present",
    )
    return parser
