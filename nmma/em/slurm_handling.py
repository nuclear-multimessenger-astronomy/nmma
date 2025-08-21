import os
import numpy as np
from .em_parsing import slurm_parser, parsing_and_logging
from ..joint.injection_handling import NMMAInjectionCreator
from ..joint.utils import read_injection_file


def main():
    args = parsing_and_logging(slurm_parser)
    injection_creator = NMMAInjectionCreator(args)
    dataframe = injection_creator.generate_prelim_dataframe()

    

    for index, row in dataframe.iterrows():
        with open(args.analysis_file, "r") as file:
            analysis = file.read()

        outdir = os.path.join(args.outdir, str(index))
        os.makedirs(outdir, exist_ok=True)

        injection_creator.priors.to_file(outdir, label="injection")
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

    logdir = os.path.join(args.outdir, "logs")
    os.makedirs(logdir, exist_ok=True)

    injection_df = read_injection_file(args)
    number_jobs = int(
        np.ceil(len(injection_df) / args.lightcurves_per_job)
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