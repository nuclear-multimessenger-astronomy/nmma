from nmma.core.parsing import yaml_parse


def gw_parsing(parser):
    """Add GW-likelihood input arguments to the parser.

    Currently adds the multibanding `--reference-chirp-mass` argument.
    Relative-binning arguments (`--fiducial-parameters`,
    `--update-fiducial-parameters`, `--epsilon`) are left commented out
    since they are already defined by bilby_pipe.

    Parameters
    ----------
    parser : configargparse.ArgumentParser
        Parser to extend.

    Returns
    -------
    configargparse.ArgumentParser
        The same parser, with the "GW input arguments" group added.
    """

    gw_input_parser = parser.add_argument_group(
        title="GW input arguments", description="Specify GW inputs"
    )

    ## Multibanding kwargs
    gw_input_parser.add(
        "--reference-chirp-mass",
        type=float,
        help="The reference chirp mass for multibanding gw likelihood.",
    )

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
    """Add GW injection arguments to the parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser or configargparse.ArgumentParser
        Parser to extend.

    Returns
    -------
    argparse.ArgumentParser or configargparse.ArgumentParser
        The same parser, with `--gw-detectors` and `--waveform-arguments`
        added.
    """
    ### FIXME: The help text says "comma-separated," but nargs="*" actually means space-separated tokens. I tested both: --gw-detectors ET,CE (comma-separated, as the help text instructs) produces ['ET,CE'] — a single bogus string containing a literal comma, not two detector names — while --gw-detectors ET CE H1 (space-separated) correctly produces ['ET', 'CE', 'H1']. Anyone following the help text as written will silently get a broken one-element detector list with no error raised.
    parser.add_argument(
        "--gw-detectors",
        default=["ET", "CE"],
        nargs="*",
        help="Comma-separated list of GW detectors to use (default: ET,CE)",
    )
    parser.add_argument(
        "--waveform-arguments",
        type=yaml_parse,
        default={},
        help='Additional arguments to pass to the waveform generator, e.g. \'waveform_arguments={"waveform_approximant": "IMRPhenomXPHM"}\'',
    )

    return parser
