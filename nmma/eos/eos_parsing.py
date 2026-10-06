from ..core.parsing import single_messenger_analysis_parsing, yaml_parse


def tabulated_eos_parsing(parser):
    """Add CLI args for sampling over a directory of precomputed EOS
    files: ``--eos-data``, ``--Neos``, ``--eos-weight``, ``--eos-to-ram``.

    Consumed by ``setup_tabulated_eos_priors``/``EoSConverter(..., "tabulated")``.

    Parameters
    ----------
    parser: configargparse.ArgParser

    Returns
    -------
    configargparse.ArgParser
        The same parser, with the "Tabulated EOS input arguments" group added.
    """
    tab_eos_input_parser = parser.add_argument_group(
        title="Tabulated EOS input arguments",
        description="Specify tabulated EOS inputs",
    )

    tab_eos_input_parser.add(
        "--eos-to-ram",
        action="store_true",
        help="Depending on cluster architecture, it can be faster to load all EOS files directly to RAM",
    )
    tab_eos_input_parser.add("--eos-data", help="Path to the EOS directory")
    tab_eos_input_parser.add("--Neos", type=int, help="Number of EOSs to be used")
    tab_eos_input_parser.add(
        "--eos-weight", help="Path to the precalculated EOS weighting"
    )
    return parser


def eos_parsing(parser):
    """Add CLI args for on-the-fly EOS generation (``--emulator-metadata``,
    ``--micro-eos-model``) and for the astrophysical constraints
    (``--lower-mtov*``, ``--upper-mtov*``, ``--mass-radius*``) evaluated
    against it or a tabulated EOS.

    Each constraint kind takes either one ``--<kind>`` dict flag, or
    parallel ``--<kind>-name/-mass|file-path/-error/-arxiv/-plot-kwargs``
    list flags — see ``read_constraint_from_args``.

    Parameters
    ----------
    parser: configargparse.ArgParser

    Returns
    -------
    configargparse.ArgParser
        The same parser, with the "EOS input arguments" group added.
    """
    eos_input_parser = parser.add_argument_group(
        title="EOS input arguments", description="Specify EOS inputs"
    )

    # eos_input_parser.add( "--eos-crust-file",
    #     help="Path to data file for eos crust" )

    eos_input_parser.add(
        "--emulator-metadata",
        help='dict or path to dict with metadata for the TOV emulator. Must include "emulator_path", can contain further metadata',
    )
    eos_input_parser.add(
        "--micro-eos-model", default="nep-5", help="The micro EOS model to use."
    )

    ### args to set up eos likelihood evaluation based on constraints
    eos_input_parser.add(
        "--eos-constraint-json",
        help="path to .json-file from which eos-constraints are read and/or to which they should be stored. Can be appended with additional constraints.",
    )

    ### setup LowerMTOVConstraint
    eos_input_parser.add(
        "--lower-mtov",
        type=yaml_parse,
        help="dict with additional lower mtov limits to consider, using style: {'name':{'mass':mass_val,'error':gaussian_error_val [, 'arxiv':'arxiv_id']},...}",
    )
    eos_input_parser.add(
        "--lower-mtov-name",
        nargs="*",
        help="list of identifiers for further lower-mtov-values to consider",
    )
    eos_input_parser.add(
        "--lower-mtov-mass",
        nargs="*",
        type=float,
        help="list of additional lower mtov limits to consider",
    )
    eos_input_parser.add(
        "--lower-mtov-error",
        nargs="*",
        type=float,
        help="list of additional mtov limit errors to consider",
    )
    eos_input_parser.add(
        "--lower-mtov-arxiv",
        nargs="*",
        help="list of arxiv-ids for additional lower mtov limits to consider",
    )
    eos_input_parser.add(
        "--lower-plot-kwargs",
        type=yaml_parse,
        help="dict with additional plot kwargs for mass-radius posteriors",
    )

    ### setup UpperMTOVConstraint
    eos_input_parser.add(
        "--upper-mtov",
        type=yaml_parse,
        help="dict with additional upper mtov limits to consider, using style: {'name':{'mass':mass_val,'error':gaussian_error_val [, 'arxiv':'arxiv_id']},...}",
    )
    eos_input_parser.add(
        "--upper-mtov-name",
        nargs="*",
        help="list of identifiers for further upper-mtov-values to consider",
    )
    eos_input_parser.add(
        "--upper-mtov-mass",
        type=float,
        nargs="*",
        help="list of additional upper mtov limits to consider",
    )
    eos_input_parser.add(
        "--upper-mtov-error",
        type=float,
        nargs="*",
        help="list of additional mtov limit errors to consider",
    )
    eos_input_parser.add(
        "--upper-mtov-arxiv",
        nargs="*",
        help="list of arxiv-ids for additional upper mtov limits to consider",
    )
    eos_input_parser.add(
        "--upper-plot-kwargs",
        type=yaml_parse,
        help="dict with additional plot kwargs for mass-radius posteriors",
    )

    ### setup MassRadiusConstraint
    eos_input_parser.add(
        "--mass-radius",
        type=yaml_parse,
        help="dict with additional mass-radius constraints to consider, using style: {'name':{'file_path':path_to_R_M_posterior,[, 'arxiv':'arxiv_id']},...}",
    )
    eos_input_parser.add(
        "--mass-radius-name",
        nargs="*",
        help="list of identifiers for further mass-radius-posteriors to consider",
    )
    eos_input_parser.add(
        "--mass-radius-file-path",
        "--mass-radius-posterior",
        nargs="*",
        help="list of files with additional radius-mass posteriors to consider",
    )
    eos_input_parser.add(
        "--mass-radius-arxiv",
        nargs="*",
        help="list of arxiv-ids for additional R-M posteriors to consider",
    )
    eos_input_parser.add(
        "--mass-radius-plot-kwargs",
        type=yaml_parse,
        help="dict with additional plot kwargs for mass-radius posteriors",
    )
    return parser


def eos_analysis_parsing(parser):
    parser = single_messenger_analysis_parsing(parser)
    parser = tabulated_eos_parsing(parser)
    parser = eos_parsing(parser)
    return parser
