
from .base_parsing import StoreBoolean, nonestr
from ..eos.eos_parsing import tabulated_eos_parsing, eos_parsing
from ..gw.gw_parsing import gw_parsing, gw_injection_parsing

def injection_parsing(parser):
    
    parser.description="Create a file of nmma-injections, processing a bilby injection if needed"

    parser = bilby_pipe_injection_parsing(parser)
    parser = tabulated_eos_parsing(parser)
    parser = eos_parsing(parser)
    parser = gw_injection_parsing(parser)
    
    ### NMMA-added options
    parser.add_argument("--max-redraws", type=int, default=10,
        help=("The maximum number of times to redraw the injection "
              "if additional constraints are not satisfied"), )
    parser.add_argument("--original-parameters", action="store_true",
        help="Whether to only store parameters given by injection prior",
    )
    
    parser.add_argument("--require-ejecta", action=StoreBoolean)
    
    
    parser.add_argument("--eos-file", type=str, 
        help="EOS file in (radius [km], mass [solar mass], lambda). If n" )
    parser.add_argument("--eos-model-type", type=str, 
        help="model type read by the eos emulator")


    
    ### Parameters for legacy injection file used in conjunction
    parser.add_argument( "--gw-injection-file", type=str, 
        help="The xml injection file or bilby injection json file to be used (optional)" )
    parser.add_argument( "--reference-frequency", type=float, default=20,
        help="The reference frequency in the provided injection file (default: 20)", )
    

    return parser

def bilby_pipe_injection_parsing(parser):
    ### bilby-pipe injectionCreator parameters
    parser.add_argument( "--prior-file", type=nonestr,
        help="The prior file from which to generate injections. Alternatively, a prior-dict must be given.")
    parser.add_argument("--prior-dict", type=nonestr, 
        help=("A prior dict to use for generating injections. If not given, the prior file is used. "))
    parser.add_argument("-n", "--n-injection", type=int, default=20,
        help=("The number of injections to generate: not required" 
              "if --gps-file or injection file is also given"), )
    
    parser.add_argument("-f", "--filename", type=str, default="injection")
    parser.add_argument("-e", "--extension", type=str, default="json",
        choices=["json", "dat, csv"], help="Injection file format", )
    
    parser.add_argument( "-s", "--generation-seed", default=42, type=int,
        help="Random seed used during data generation (default: 42)" )

    # bilby-pipe Optional Time parameters
    parser.add_argument( "-t", "--trigger-time", type=int, default=0,
        help=("The trigger time to use for setting a geocent_time prior "
            "(default=0). Ignored if a geocent_time prior exists in the "
            "prior_file or --gps-file is given." ), )
    parser.add_argument( "--deltaT", type=float, default=0.2,
        help=( "The symmetric width (in s) around the trigger time to"
            " search over the coalesence time. Ignored if a geocent_time prior"
            " exists in the prior_file" ), )
    

    parser.add_argument( "-g", "--gps-file", type=str, default=None,
        help=( "A list of gps start times to use for setting a geocent_time prior"
            ". Note, the trigger time is obtained from "
            " start_time + duration - post_trigger_duration." ), )
    parser.add_argument("--duration", type=float, default=4, help=("The segment "
            "duration (default=4s), used only in conjunction with --gps-file" ), )
    parser.add_argument( "--post-trigger-duration", type=float, default=2,
        help=("The post trigger duration (default=2s), used only in conjunction "
            "with --gps-file" ), )
    return parser
