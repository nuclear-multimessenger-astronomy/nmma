from . import gwem_resampling_utils as sampler_functions
import pandas as pd
import numpy as np
import bilby
import argparse
import os


def main():

    parser = argparse.ArgumentParser(
        description="Inference on binary source parameters with kilonova ejecta posterior and GW source posterior given."
    )
    parser.add_argument("--outdir", metavar="PATH", type=str, required=True)
    parser.add_argument(
            "--GWsamples", 
            metavar="PATH", 
            type=str, 
            required=True,
            help="If no posterior files are available, use gwsamples_creation.py to generate dummy GWsamples."
    )
    parser.add_argument(
            "--EMsamples", 
            metavar="PATH", 
            type=str, 
            required=True,
            help="posterior samples file from a previous Bayesian inference run on EM signals (e.g. Kilonova inference or Kilonova+GRB inference.")
    parser.add_argument(
            "--EOSpath", 
            metavar="PATH", 
            type=str, 
            required=True,
            help="Path of EOS folder, e.g. 15nsat_cse_uniform_R14 (located: https://zenodo.org/record/6106130#.YoysIHVBwUG)"
    )
    parser.add_argument(
            "--Neos", 
            metavar="Neos", 
            type=int, 
            required=True,
            help="Number of EOS files used for the inference."
    )
    parser.add_argument(
            "--nlive", 
            metavar="nlive", 
            type=int, 
            required=False, 
            default=1024
    )
    parser.add_argument(
        "--GWprior",
        metavar="PATH",
        type=str,
        required=True,
        help="Prior file used for the GW analysis",
    )
    parser.add_argument(
        "--EMprior",
        metavar="PATH",
        type=str,
        required=True,
        help="Prior file used for the EM eos analysis",
    )
    parser.add_argument(
        "--total-ejecta-mass",
        action="store_true",
        help="To run with total ejecta mass, if not activated, the two ejecta are consider seperately",
    )
    args = parser.parse_args()

    # read the GW samples
    GWsamples = pd.read_csv(args.GWsamples, header=0, delimiter=" ")
    # down sample
    weights = np.ones(len(GWsamples))
    weights /= np.sum(weights)
    GWsamples = GWsamples.sample(
        frac=30000 / len(GWsamples), weights=weights, random_state=42
    )

    # read the EM samples
    EMsamples = pd.read_csv(args.EMsamples, header=0, delimiter=" ")

    # read the prior files
    GWprior = bilby.gw.prior.PriorDict(args.GWprior)
    EMprior = bilby.gw.prior.PriorDict(args.EMprior)

    try:
        os.makedirs(args.outdir + "/pm/")
    except Exception:
        pass
    pymulti_kwargs = dict(
        outputfiles_basename=args.outdir + "/pm/",
        n_dims=5,
        n_live_points=args.nlive,
        verbose=True,
        resume=True,
        seed=42,
        importance_nested_sampling=False,
    )

    if args.total_ejecta_mass:
        solution = sampler_functions.TotalEjectaMassInference(
            GWsamples,
            EMsamples,
            GWprior,
            EMprior,
            args.Neos,
            args.EOSpath,
            **pymulti_kwargs
        )
    else:
        solution = sampler_functions.EjectaMassInference(
            GWsamples,
            EMsamples,
            GWprior,
            EMprior,
            args.Neos,
            args.EOSpath,
            **pymulti_kwargs
        )

    samples = solution.samples.T
    posterior_samples = dict()
    posterior_samples["chirp_mass"] = samples[0]
    posterior_samples["mass_ratio"] = samples[1]
    posterior_samples["EOS"] = samples[2]
    posterior_samples["alpha"] = samples[3]
    posterior_samples["zeta"] = samples[4]

    posterior_samples = pd.DataFrame.from_dict(posterior_samples)
    posterior_samples.to_csv(
        "{0}/posterior_samples.dat".format(args.outdir), sep=" ", index=False
    )

    sampler_functions.corner_plot(posterior_samples, solution, args.outdir)


if __name__ == "__main__":
    main()
