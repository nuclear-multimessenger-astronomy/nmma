from . import maximum_mass_constraint_utils as utils
import pandas as pd
import numpy as np
import bilby

import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser(
        description="Inference on the maximum mass constraint of the EOS when a joint posterior for binary components, ejecta and EOS is provided."
    )
    parser.add_argument("--outdir", metavar="PATH", type=str, required=True)
    parser.add_argument(
        "--joint-posterior",
        metavar="PATH",
        type=str,
        required=True,
        help="Posterior file with chirp mass, eta_star, EOSID, log10_mdisk, log10_mej_dyn as columns.",
    )
    parser.add_argument(
        "--prior",
        metavar="PATH",
        type=str,
        required=True,
        help="Prior specification for chirp mass, eta_star, log10_mdisk, log10_mej_dyn. If use_M_kepler is True, prior file must also contain ratio_R and delta.",
    )
    parser.add_argument(
        "--eos-path-macro",
        metavar="PATH",
        type=str,
        required=True,
        help="EOS folder with EOS files in [R, M, Lambda, p_central] format.",
    )
    parser.add_argument(
        "--eos-path-micro",
        metavar="PATH",
        type=str,
        required=True,
        help="EOS folder with EOS files in [n, energy density, pressure, speed of sound squared] format.",
    )
    parser.add_argument(
        "--use-M-Kepler",
        action="store_true",
        help="If set, it is assumed that the BNS remnant collapsed to a black hole above the Kepler limit.",
    )
    parser.add_argument(
        "--nlive", metavar="nlive", type=int, required=False, default=1024
    )

    return parser


def maximum_mass_resampling(args):
    try:
        os.makedirs(args.outdir + "/pm/")
    except:
        pass

    posterior_samples = pd.read_csv(args.joint_posterior, header=0, delimiter=" ")
    prior = bilby.gw.prior.PriorDict(args.prior)
    Neos = len(next(os.walk(args.eos_path_macro))[2])

    if args.use_M_Kepler:
        n_dims = 7
        if len(prior.keys()) != n_dims - 1:
            raise Exception(
                "If use_M_kepler is True, you need to provide a prior fo ratio_R and delta to be used in the quasi-universal relation."
            )

    else:
        n_dims = 5

    pymulti_kwargs = dict(
        outputfiles_basename=args.outdir + "/pm/",
        n_dims=n_dims,
        n_live_points=args.nlive,
        verbose=True,
        resume=True,
        seed=42,
        importance_nested_sampling=False,
        use_MPI=True,
    )

    solution = utils.PostmergerInference(
        prior,
        posterior_samples,
        Neos,
        args.eos_path_macro,
        args.eos_path_micro,
        args.use_M_Kepler,
        **pymulti_kwargs
    )

    samples = solution.samples.T
    posterior = dict()

    posterior["chirp_mass"] = samples[0]
    posterior["eta_star"] = samples[1]
    posterior["EOS"] = samples[2]
    posterior["log10_mdisk"] = samples[3]
    posterior["log10_mej_dyn"] = samples[4]

    if args.use_M_Kepler:
        posterior["ratio_R"] = samples[5]
        posterior["delta"] = samples[6]

    posterior = pd.DataFrame.from_dict(posterior)
    posterior.to_csv(args.outdir + "/posterior_samples.dat", sep=" ", index=False)


def main(args=None):

    if args is None:
        parser = get_parser()
        args = parser.parse_args()

        maximum_mass_resampling(args)

    else:
        maximum_mass_resampling(args)


if __name__ == "__main__":
    main()
