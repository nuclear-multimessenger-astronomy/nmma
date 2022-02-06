import os
import argparse
import json
import pandas as pd

import bilby
from bilby_pipe.create_injections import InjectionCreator


def main():

    parser = argparse.ArgumentParser(description="Slurm files from nmma injection file")
    parser.add_argument(
        "--prior-file",
        type=str,
        required=True,
        help="The prior file from which to generate injections",
    )
    parser.add_argument(
        "--injection-file",
        type=str,
        required=True,
        help="The bilby injection json file to be used",
    )
    parser.add_argument(
        "--analysis-file",
        type=str,
        required=True,
        help="The analysis bash script to be replicated",
    )
    parser.add_argument("-o", "--outdir", type=str, default="outdir")
    args = parser.parse_args()

    # load the injection json file
    if args.injection_file:
        if args.injection_file.endswith(".json"):
            with open(args.injection_file, "rb") as f:
                injection_data = json.load(f)
                datadict = injection_data["injections"]["content"]
                dataframe_from_inj = pd.DataFrame.from_dict(datadict)
        else:
            print("Only json supported.")
            exit(1)

    if len(dataframe_from_inj) > 0:
        args.n_injection = len(dataframe_from_inj)

    # create the injection dataframe from the prior_file
    injection_creator = InjectionCreator(
        prior_file=args.prior_file,
        prior_dict=None,
        n_injection=args.n_injection,
        default_prior="PriorDict",
        gps_file=None,
        trigger_time=0,
        generation_seed=0,
    )
    dataframe_from_prior = injection_creator.get_injection_dataframe()

    # combine the dataframes
    dataframe = pd.DataFrame.merge(
        dataframe_from_inj,
        dataframe_from_prior,
        how="outer",
        left_index=True,
        right_index=True,
    )

    for index, row in dataframe.iterrows():
        with open(args.analysis_file, "r") as file:
            analysis = file.read()

        outdir = os.path.join(args.outdir, str(index))
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        priors = bilby.gw.prior.PriorDict(args.prior_file)
        priors.to_file(outdir, label="injection")
        priorfile = os.path.join(outdir, "injection.prior")
        injfile = os.path.join(outdir, "lc.csv")

        analysis = analysis.replace("PRIOR", priorfile)
        analysis = analysis.replace("OUTDIR", outdir)
        analysis = analysis.replace("INJOUT", injfile)
        analysis = analysis.replace("INJNUM", str(index))
        analysis_file = os.path.join(outdir, "inference.sh")

        fid = open(analysis_file, "w")
        fid.write(analysis)
        fid.close()


if __name__ == "__main__":
    main()
