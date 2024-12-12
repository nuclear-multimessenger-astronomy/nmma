import argparse

def joint_postprocess_parser(parser):

    parser.add_argument("--outdir", metavar="PATH", type=str, required=True)
    parser.add_argument("--cred-interval", type=float, default=0.95, help="Credible interval to be calculated (default: 0.95)")
    parser.add_argument("--seed", type=int, required=False, default=42, help="Random seed (default: 42)")
    parser.add_argument("-d", "--detections-file", "--detectable", type=str, required=False)
    parser.add_argument("--Nevent", type=int, required=False)
    parser.add_argument("--N-reordering", type=int, default=100, help="Number of reodering realisation to be comsidered (default: 100)")
    parser.add_argument("--N-posterior-samples", type=int, required=False, default=10000, help="Number of posterior samples to be drawn during the resampling (default: 10000)")

def R14_parser():
    parser = argparse.ArgumentParser(
            description="Calculate the trend of estimated R14 with GW+EM input"
        )
    parser = joint_postprocess_parser(parser)
    parser.add_argument("--label", metavar="NAME", type=str, required=True)
    parser.add_argument("--R14-true", type=float, default=11.55,
        help="The true value of Neutron stars's raduis (default:11.55)")
    parser.add_argument("--gwR14trend",type=str,required=True,
        help="Path to the R14trend GW  posterior samples directory, the  file are expected to be in the format of GW_R14trend.dat ")
    parser.add_argument("--GWEMsamples", type=str, required=True,
        help="Path to the GWEM posterior samples directory, the samples files are expected to be in the format of posterior_samples_{i}.dat",)
    parser.add_argument("--Neos", type=int, required=True, help="Number of EOS")
    parser.add_argument("--EOS-prior", type=str, required=False,
        help="Path to the EOS prior file, if None, assuming fla prior across EOSs")
    parser.add_argument("--EOSpath", type=str, required=True, help="The EOS data")
    parser.add_argument(
        "--pdet", type=str, required=False,
        help="Path to the probability of detection as a function of maximum mass (for correcting selection bias")
    

def Hubble_parser(): 
    parser = argparse.ArgumentParser(description="Calculate the combination and seperate trend of estimated Hubble constant with GW and EM input")
    parser = joint_postprocess_parser(parser)
    parser.add_argument("--output-label", metavar="NAME", type=str, required=True)
    parser.add_argument("--GWsamples", metavar='PATH', type=str, required=True, help="Path to the GW posterior samples directory, the samples files are expected to be in the format of posterior_samples_{i}.dat")
    parser.add_argument("--EMsamples", metavar='PATH', type=str, required=True, help="Same as the GW samples but for EM samples")
    parser.add_argument("--injection", metavar='PATH', type=str, required=True)
    parser.add_argument("--inject-Hubble", metavar='H0', type=float, required=True)
    parser.add_argument("--N-prior-samples", type=int, required=False, default=10000, help="Number of prior samples to be used for resampling (default: 10000)")
    parser.add_argument("--p-value-threshold", type=float, required=False, help="p-value threshold used to remove badly recoved injections")
    return parser

def resampling_parser():
    parser = argparse.ArgumentParser(
        description="Inference on binary source parameters with kilonova ejecta posterior and GW source posterior given.")
    parser.add_argument("--outdir", metavar="PATH", type=str, required=True)
    parser.add_argument("--GWsamples", metavar="PATH", type=str, required=True,
            help="posterior samples file from a previous Bayesian inference run on GW signals.")
    parser.add_argument("--EMsamples", metavar="PATH", type=str, required=True,
            help="posterior samples file from a previous Bayesian inference run on EM signals (e.g. Kilonova inference or Kilonova+GRB inference.")
    parser.add_argument("--EOSpath",metavar="PATH",type=str, required=True,
            help="Path of EOS folder, e.g. 15nsat_cse_uniform_R14 (located: https://zenodo.org/record/6106130#.YoysIHVBwUG)")
    parser.add_argument("--Neos", type=int,required=True,help="Number of EOS files used for the inference.")
    parser.add_argument("--nlive", type=int, required=False, default=1024, help="live number")
    parser.add_argument("--GWprior", type=str, required=True, help="Prior file used for the GW analysis")
    parser.add_argument("--EMprior",type=str,required=True,help="Prior file used for the EM eos analysis")
    parser.add_argument("--total-ejecta-mass",action="store_true",
            help="To run with total ejecta mass, if not activated, the two ejecta are consider seperately")
    parser.add_argument("--withNSBH", action="store_true", 
            help="Compute GW-EM-resampling for NSBH source, else: for BNS source.")
    return parser

def condor_parser():
    parser = resampling_parser()
    parser.add_argument("-d", "--detections-file", type=str)
    parser.add_argument("--condor-dag-file",type=str, required=True, help="The condor dag file to be created")
    parser.add_argument("--condor-sub-file",type=str, required=True, help="The condor sub file to be created")
    parser.add_argument("--bash-file", type=str, required=True, help="The bash file to be created")
    return parser