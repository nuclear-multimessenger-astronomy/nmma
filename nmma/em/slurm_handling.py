import os
from .em_parsing import slurm_parser, parsing_and_logging
import json

import numpy as np

import bilby

import os
import json
import pandas as pd

import bilby
from bilby_pipe.create_injections import InjectionCreator


def main():
    args = parsing_and_logging(slurm_parser)

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
        gpstimes=None,
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
        os.makedirs(outdir, exist_ok=True)

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




def lc_creation():
    args = parsing_and_logging(slurm_parser)

    with open(args.injection, "r") as f:
        injection_dict = json.load(f, object_hook=bilby.core.utils.decode_bilby_json)

    logdir = os.path.join(args.outdir, "logs")
    os.makedirs(logdir, exist_ok=True)

    number_jobs = int(
        np.ceil(len(injection_dict["injections"]) / args.lightcurves_per_job)
    )

    for ii in range(number_jobs):
        with open(args.analysis_file, "r") as file:
            analysis = file.read()

        injection_min, injection_max = ii * number_jobs, (ii + 1) * number_jobs

        analysis = analysis.replace(
            "INJRANGE", "%d,%d" % (injection_min, injection_max)
        )
        analysis_file = os.path.join(args.outdir, "inference_%d.sh" % ii)

        fid = open(analysis_file, "w")
        fid.write(analysis)
        fid.close()

    fid.close()


if __name__ == "__main__":
    main()