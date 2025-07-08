import bilby_pipe
from nmma.joint.base_parsing import StoreBoolean

def tabulated_eos_parsing(parser):
    tab_eos_input_parser = parser.add_argument_group(
        title="Tabulated EOS input arguments", description="Specify tabulated EOS inputs"
    )
    tab_eos_input_parser.add( "--with-tabulated-eos", action=StoreBoolean, default=False,
        help="Flag for sampling over tabulated EOS (default:False)",
    )
    tab_eos_input_parser.add( "--eos-to-ram", action=StoreBoolean, default=False,
        help="Depending on cluster architecture, it can be faster to load all EOS files directly to RAM"
    )
    tab_eos_input_parser.add("--eos-data",  help="Path to the EOS directory" )
    tab_eos_input_parser.add("--Neos", type=int, help="Number of EOSs to be used")
    tab_eos_input_parser.add("--eos-weight", help="Path to the precalculated EOS weighting", )
    return parser


def eos_parsing(parser):
    
    eos_input_parser = parser.add_argument_group(
        title="EOS input arguments", description="Specify EOS inputs"
    )

    eos_input_parser.add(
        "--with-eos",
        action=StoreBoolean,
        default=False,
        help="Flag for sampling over nuclear empirical parameters from which we generate an EOS (default:False)"
    )

    eos_input_parser.add(
        "--eos-crust-file", help="Path to data file for eos crust, to be used if --eos-from-neps is set to True"
    )

    eos_input_parser.add('--emulator-metadata', type=str, 
        help='dict or path to dict with metadata for the TOV emulator. Must include "emulator_path", can contain further metadata')
    eos_input_parser.add('--micro-eos-model', type=str, default= 'nep-5',      
        help='The micro EOS model to use. ') ## FIXME: add model_selection
    
        ### args to set up eos likelihood evaluation based on constraints
    eos_input_parser.add(
        "--eos-constraint-json", type = bilby_pipe.utils.nonestr,
        help="path to .json-file from which eos-constraints are read and/or to which they should be stored. Can be appended with additional constraints."
    )

    ### setup LowerMTOVConstraint
    eos_input_parser.add( "--lower-mtov", type=bilby_pipe.utils.nonestr,
        help= "dict with additional lower mtov limits to consider, using style: {'name':{'mass':mass_val,'error':gaussian_error_val [, 'arxiv':'arxiv_id']},...}"
    )
    eos_input_parser.add( "--lower-mtov-name", type=bilby_pipe.utils.nonestr, action = 'append',
        help= "list of identifiers for further lower-mtov-values to consider"
    )
    eos_input_parser.add( "--lower-mtov-mass", type=bilby_pipe.utils.nonestr, action = 'append',
        help= "list of additional lower mtov limits to consider"
    )
    eos_input_parser.add( "--lower-mtov-error",type=bilby_pipe.utils.nonestr, action = 'append',
        help= "list of additional mtov limit errors to consider"
    )
    eos_input_parser.add( "--lower-mtov-arxiv",type=bilby_pipe.utils.nonestr, action = 'append',
        help= "list of arxiv-ids for additional lower mtov limits to consider"
    )
    
    ### setup UpperMTOVConstraint
    eos_input_parser.add( "--upper-mtov", type=bilby_pipe.utils.nonestr,
        help= "dict with additional upper mtov limits to consider, using style: {'name':{'mass':mass_val,'error':gaussian_error_val [, 'arxiv':'arxiv_id']},...}"
    )
    eos_input_parser.add( "--upper-mtov-name", type=bilby_pipe.utils.nonestr, action = 'append',
        help= "list of identifiers for further upper-mtov-values to consider"
    )
    eos_input_parser.add( "--upper-mtov-mass", type=bilby_pipe.utils.nonestr, action = 'append',
        help= "list of additional upper mtov limits to consider"
    )
    eos_input_parser.add( "--upper-mtov-error",type=bilby_pipe.utils.nonestr, action = 'append',
        help= "list of additional mtov limit errors to consider"
    )
    eos_input_parser.add( "--upper-mtov-arxiv",type=bilby_pipe.utils.nonestr, action = 'append',
        help= "list of arxiv-ids for additional upper mtov limits to consider"
    )

    ### setup MassRadiusConstraint
    eos_input_parser.add( "--mass-radius", type=bilby_pipe.utils.nonestr,
        help= "dict with additional mass-radius constraints to consider, using style: {'name':{'file_path':path_to_R_M_posterior,[, 'arxiv':'arxiv_id']},...}"
    )
    eos_input_parser.add(  "--mass-radius-name", type=bilby_pipe.utils.nonestr, action = 'append',
        help= "list of identifiers for further mass-radius-posteriors to consider"
    )
    eos_input_parser.add("--MR-posterior-file-path",type=bilby_pipe.utils.nonestr,action='append',
        help= "list of files with additional radius-mass posteriors to consider"
    )
    eos_input_parser.add( "--mass-radius-arxiv", type=bilby_pipe.utils.nonestr, action = 'append',
        help= "list of arxiv-ids for additional R-M posteriors to consider"
    )
    
    return parser
