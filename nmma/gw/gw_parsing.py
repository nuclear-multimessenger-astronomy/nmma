
import bilby_pipe
from nmma.joint.base_parsing import StoreBoolean, nonestr

def gw_parsing(parser):
    gw_input_parser = parser.add_argument_group(
        title="GW input arguments", description="Specify GW inputs"
    )
    gw_input_parser.add(
        "--with-gw",
        action=StoreBoolean,
        default=True,
        help="Flag for sampling over GW parameters (default:True)",
    )

    ## Multibanding kwargs
    gw_input_parser.add("--reference-chirp-mass", type=bilby_pipe.utils.nonefloat, 
        help="The reference chirp mass for multibanding gw likelihood.")

    ## Relative Binning kwargs
    #  This is already defined in bilby-pipe -->
    # gw_input_parser.add("--fiducial-parameters", 
    #     type=bilby_pipe.utils.nonestr, default=None, help="A dict of fiducial parameters, to be read by the GW-likelihood")
    # gw_input_parser.add("--update-fiducial-parameters", 
    #     type=StoreBoolean, default=False, help="Flag to update the fiducial parameters from maximum likelihood")
    # gw_input_parser.add("--epsilon", type=float, 
    #     help ="Tunable parameter which limits the differential phase change in each bin when setting up the bin range. See https://arxiv.org/abs/1806.08792")
    return parser

def gw_injection_parsing(parser):
    """Parser for the gw injection arguments."""
    parser.add_argument("--compute-snr", action=StoreBoolean, default=False,
        help="Flag to compute the SNR of the injections (default: False)")
    parser.add_argument("--snr-threshold", type=float, default=8.0,
        help="SNR threshold for the injections (default: 8.0)")
    parser.add_argument("--gw-detectors", type=str, default="ET,CE",
        help="Comma-separated list of GW detectors to use (default: ET,CE)")
    parser.add_argument("--waveform-arguments", type=nonestr,
        help="Additional arguments to pass to the waveform generator, e.g. 'waveform_arguments={\"waveform_approximant\": \"IMRPhenomXPHM\"}'")
    
    return parser