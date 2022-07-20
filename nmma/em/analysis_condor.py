import os
from subprocess import check_output
import argparse

def main():

    parser = argparse.ArgumentParser(
        description="Inference on kilonova ejecta parameters."
    )
    parser.add_argument(
        "--binary-type", 
        type=str, 
        required=False,
        help="Either BNS or NSBH"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the kilonova model to be used"
    )
    parser.add_argument(
        "--interpolation_type",
        type=str,
        help="SVD interpolation scheme.",
        default="sklearn_gp",
    )
    parser.add_argument(
        "--svd-path",
        type=str,
        help="Path to the SVD directory, with {model}_mag.pkl and {model}_lbol.pkl",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Path to the output directory"
    )
    parser.add_argument("--label", type=str, required=True, help="Label for the run")
    parser.add_argument(
        "--trigger-time",
        type=float,
        help="Trigger time in modified julian day, not required if injection set is provided",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the data file in [time(isot) filter magnitude error] format",
    )
    parser.add_argument(
        "--prior",
        type=str,
        required= True,
        help="The prior file from which to generate injections",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=0.0,
        help="Days to start analysing from the trigger time (default: 0)",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=14.0,
        help="Days to stop analysing from the trigger time (default: 14)",
    )
    parser.add_argument(
        "--dt", type=float, default=0.1, help="Time step in day (default: 0.1)"
    )
    parser.add_argument(
        "--photometric-error-budget",
        type=float,
        default=0.1,
        help="Photometric error (mag) (default: 0.1)",
    )
    parser.add_argument(
        "--svd-mag-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for mag evaluation (default: 10)",
    )
    parser.add_argument(
        "--svd-lbol-ncoeff",
        type=int,
        default=10,
        help="Number of eigenvalues to be taken for lbol evaluation (default: 10)",
    )
    parser.add_argument(
        "--filters",
        type=str,
        help="A comma seperated list of filters to use (e.g. g,r,i). If none is provided, will use all the filters available",
    )
    parser.add_argument(
        "--Ebv-max",
        type=float,
        default=0.5724,
        help="Maximum allowed value for Ebv (default:0.5724)",
    )
    parser.add_argument(
        "--grb-resolution",
        type=float,
        default=5,
        help="The upper bound on the ratio between thetaWing and thetaCore (default: 5)",
    )
    parser.add_argument(
        "--jet-type",
        type=int,
        default=0,
        help="Jet type to used used for GRB afterglow light curve (default: 0)",
    )
    parser.add_argument(
        "--error-budget",
        type=str,
        default="1.0",
        help="Additional systematic error (mag) to be introduced (default: 1)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="pymultinest",
        help="Sampler to be used (default: pymultinest)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=1,
        help="Number of cores to be used, only needed for dynesty (default: 1)",
    )
    parser.add_argument(
        "--nlive", type=int, default=2048, help="Number of live points (default: 2048)"
    )
    parser.add_argument(
        "--seed",
        metavar="seed",
        type=int,
        default=42,
        help="Sampling seed (default: 42)",
    )
    parser.add_argument(
        "--injection", metavar="PATH", type=str, help="Path to the injection json file"
    )
    parser.add_argument(
        "--injection-num",
        metavar="eventnum",
        type=int,
        help="The injection number to be taken from the injection set",
    )
    parser.add_argument(
        "--injection-detection-limit",
        metavar="mAB",
        type=str,
        help="The highest mAB to be presented in the injection data set, any mAB higher than this will become a non-detection limit. Should be comma delimited list same size as injection set.",
    )
    parser.add_argument(
        "--injection-outfile",
        metavar="PATH",
        type=str,
        help="Path to the output injection lightcurve",
    )
    parser.add_argument(
        "--remove-nondetections",
        action="store_true",
        default=False,
        help="remove non-detections from fitting analysis",
    )
    parser.add_argument(
        "--detection-limit",
        metavar="DICT",
        type=str,
        default=None,
        help="Dictionary for detection limit per filter, e.g., {'r':22, 'g':23}, put a double quotation marks around the dictionary",
    )
    parser.add_argument(
        "--with-grb-injection",
        help="If the injection has grb included",
        action="store_true",
    )
    parser.add_argument(
        "--prompt-collapse",
        help="If the injection simulates prompt collapse and therefore only dynamical",
        action="store_true",
    )
    parser.add_argument(
        "--ztf-sampling", help="Use realistic ZTF sampling", action="store_true"
    )
    parser.add_argument(
        "--ztf-uncertainties",
        help="Use realistic ZTF uncertainties",
        action="store_true",
    )
    parser.add_argument(
        "--ztf-ToO",
        type=str, 
        nargs='+',
        choices=["180", "300"],
        help="Adds realistic ToO observations during the first one or two days. Sampling depends on exposure time specified. Valid values are 180 (<1000sq deg) or 300 (>1000sq deg). Won't work w/o --ztf-sampling",
    )
    parser.add_argument(
        "--train-stats",
        help="Creates a file too.csv to derive statistics",
        action="store_true",
    )    
    parser.add_argument(
        "--rubin-ToO",
        help="Adds ToO obeservations based on the strategy presented in arxiv.org/abs/2111.01945.",
        action="store_true",
    )
    parser.add_argument(
        "--rubin-ToO-type",
        help="Type of ToO observation. Won't work w/o --rubin-ToO",
        type=str,
        choices=["BNS", "NSBH"],
    )
    parser.add_argument(
        "--xlim",
        type=str,
        default="0,14",
        help="Start and end time for light curve plot (default: 0-14)",
    )
    parser.add_argument(
        "--ylim",
        type=str,
        default="22,16",
        help="Upper and lower magnitude limit for light curve plot (default: 22-16)",
    )
    parser.add_argument(
        "--generation-seed",
        metavar="seed",
        type=int,
        nargs='+', 
        default=42, 
        help="Injection generation seed (default: 42), this can take a list",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False, help="add best fit plot"
    )
    parser.add_argument(
        "--bilby_zero_likelihood_mode",
        action="store_true",
        default=False,
        help="enable prior run",
    )
    parser.add_argument(
        "--photometry-augmentation-seed",
        metavar="seed",
        type=int,
        default=0,
        help="Optimal generation seed (default: 0)",
    )
    parser.add_argument(
        "--photometry-augmentation-N-points",
        help="Number of augmented points to include",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--photometry-augmentation-filters",
        type=str,
        help="A comma seperated list of filters to use for augmentation (e.g. g,r,i). If none is provided, will use all the filters available",
    )
    parser.add_argument(
        "--photometry-augmentation-times",
        type=str,
        help="A comma seperated list of times to use for augmentation in days post trigger time (e.g. 0.1,0.3,0.5). If none is provided, will use random times between tmin and tmax",
    ) 
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="print out log likelihoods",
    )
    parser.add_argument(
        "--condor-dag-file",
        type=str,
        required=True,
        help="The condor dag file to be created"
    )
    parser.add_argument(
        "--condor-sub-file",
        type=str,
        required=True,
        help="The condor sub file to be created"
    )
    parser.add_argument(
        "--bash-file", type=str, required=True, help="The bash file to be created"
    )
    args = parser.parse_args()
    
    binary_type_str = {'Bu2019lm':'BNS', 'Bu2019nsbh': 'NSBH'}
    prior_str = {'BNS':'Bu2019lm.prior', 'NSBH':'Bu2019nsbh.prior'}
    
    inject_file = (args.injection) 
    binary_type= binary_type_str[args.model]
    prior_file = args.prior #prior_str[binary_type]
        
    logdir = os.path.join(args.outdir, f"{binary_type}/logs".format())
    if not os.path.isdir(logdir):
        os.makedirs(logdir)  
    
    light_curve_analysis = (
        check_output(["which", "light_curve_analysis"]).decode().replace("\n", "")
    )
    seed_list = args.generation_seed
    exposure  = args.ztf_ToO
    injection_num     = args.injection_num 
    number_jobs = injection_num*len(seed_list)*len(exposure)
    
    job_number = 0 
    
    fid =  open(args.condor_dag_file, "w")
    fid1 = open(args.bash_file, "w")
     
    for ii in range(1, int(injection_num)+1):
        for exp in exposure:
            for seed  in seed_list:
               
                outdir = os.path.join(args.outdir, f"{binary_type}/{ii}_{exp}_{seed}".format())
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)
               
                inject_outdir = os.path.join(args.outdir, f"{binary_type}/{ii}_{exp}_{seed}".format())
                if not os.path.isdir(inject_outdir):
                    os.makedirs(inject_outdir)
            
                fid.write("JOB %d %s\n" % (job_number, args.condor_sub_file))
                fid.write("RETRY %d 3\n" % (job_number))
                fid.write(
                    'VARS %d jobNumber="%d" OUTDIR="%s" INJOUT="%s" INJNUM="%d"  EXPOSURE="%s" SEED="%d"\n'
                    % (job_number, job_number, outdir, inject_outdir, ii, exp, seed)
                )
                fid.write("\n\n")
                job_number = job_number + 1
               
                if args.interpolation_type:
                    fid1.write(
                        f"{light_curve_analysis} --model {args.model} --svd-path {args.svd_path} --interpolation_type {args.interpolation_type} --outdir {outdir} --label {args.label} --prior {prior_file} --tmin 0 --tmax 20 --dt 0.5 --error-budget 1 --nlive 512 --Ebv-max 0 --injection {inject_file} --injection-num {ii} --injection-detection-limit 25.0,25.0,25.3 --injection-outfile {inject_outdir}/lc.csv --generation-seed {seed} --filters g,r,i --plot --remove-nondetections --ztf-uncertainties --ztf-sampling --ztf-ToO {exp}\n".format()
                    )
                     
                else:
                    fid1.write(
                        f"{light_curve_analysis} --model {args.model} --svd-path {args.svd_path} --outdir {outdir} --label {args.label} --prior {prior_file} --tmin 0 --tmax 20 --dt 0.5 --error-budget 1 --nlive 512 --Ebv-max 0 --injection {inject_file} --injection-num {ii} --injection-detection-limit 25.0,25.0,25.3 --injection-outfile {inject_outdir}/lc.csv --generation-seed {seed} --filters g,r,i --plot --remove-nondetections --ztf-uncertainties --ztf-sampling --ztf-ToO {exp}\n".format()
                    )
    fid.close()
    fid1.close()
    
    fid = open(args.condor_sub_file, "w")
    fid.write("executable = %s\n"%light_curve_analysis)
    fid.write(f"output = {logdir}/out.$(jobNumber)\n")
    fid.write(f"error = {logdir}/err.$(jobNumber)\n")
    
    if args.interpolation_type:
        fid.write(
              f"arguments = --model {args.model} --svd-path {args.svd_path} --interpolation_type {args.interpolation_type} --outdir $(OUTDIR) --label {args.label} --prior {prior_file} --tmin 0 --tmax 20 --dt 0.5 --error-budget 1 --nlive 512 --Ebv-max 0 --injection {inject_file} --injection-num $(INJNUM) --injection-detection-limit 25.0,25.0,25.3 --injection-outfile $(INJOUT)/lc.csv --generation-seed $(SEED) --filters g,r,i --plot --remove-nondetections --ztf-uncertainties --ztf-sampling --ztf-ToO $(EXPOSURE)\n".format()
        ) 
    else:
        fid.write(
            f"arguments = --model {args.model} --svd-path {args.svd_path} --outdir $(OUTDIR) --label {args.label} --prior {prior_file} --tmin 0 --tmax 20 --dt 0.5 --error-budget 1 --nlive 512 --Ebv-max 0 --injection {inject_file} --injection-num $(INJNUM) --injection-detection-limit 25.0,25.0,25.3 --injection-outfile $(INJOUT)/lc.csv --generation-seed $(SEED) --filters g,r,i --plot --remove-nondetections --ztf-uncertainties --ztf-sampling --ztf-ToO $(EXPOSURE)\n".format()
    )
    fid.write('requirements = OpSys == "LINUX"\n')
    fid.write("request_memory = 8192\n")
    fid.write("request_disk = 500 MB\n")
    fid.write("request_cpus = 1\n")
    fid.write("accounting_group = ligo.dev.o3.burst.allsky.stamp\n") 
    fid.write("notification = nevers\n")
    fid.write("getenv = true\n")
    fid.write("log = /local/%s/light_curve_analysis.log\n" % os.environ["USER"])
    fid.write("+MaxHours = 24\n")
    fid.write("universe = vanilla\n")
    fid.write("queue 1\n")

if __name__ == "__main__":
   main() 
