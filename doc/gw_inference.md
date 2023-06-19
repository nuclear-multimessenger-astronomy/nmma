## Inference of gravitational-wave signals

A Bayesian analysis of a gravitational-wave signal which is not accompanied by electromagnetic signals can be performed within nmma following two main steps: 

Setting up a `config.ini` file and running the command

	nmma_gw_generation config.ini

Perform the analysis or parameter estimation using:

    nmma_gw_analysis <name_of_analysis>_data_dump.pickle

Below, we provide an example of a gravitational-wave inference setup using observational data of GW170817 and another example for an injection based analysis. 

### Observed GW signals

In this example, we use GW170817 as an example. First of all, a `config.ini` file needs to be created and adapted it to this specific observation. 
An example is shown below:

    ################################################################################
    ## Data generation arguments
    ################################################################################
    
    trigger_time = 1187008882.43
    
    ################################################################################
    ## Detector arguments
    ################################################################################
    
    detectors = [H1, L1, V1]
    psd_dict = {H1=data/GW170817/h1_psd.txt, L1=data/GW170817/l1_psd.txt, V1=data/GW170817/v1_psd.txt}
    channel_dict = {H1=LOSC-STRAIN, L1=LOSC-STRAIN, V1=LOSC-STRAIN}
    data_dict = {H1=data/GW170817/H-H1_LOSC_CLN_16_V1-1187007040-2048.gwf, L1=data/GW170817/L-L1_LOSC_CLN_16_V1-1187007040-2048.gwf, V1=data/GW170817/V-V1_LOSC_CLN_16_V1-1187007040-2048.gwf}
    duration = 128
    
    ################################################################################
    ## Job submission arguments
    ################################################################################
    
    label = GW170817
    outdir = outdir
    
    ################################################################################
    ## Likelihood arguments
    ################################################################################
    
    distance-marginalization=False
    phase-marginalization=False
    time-marginalization=False
    
    ################################################################################
    ## Prior arguments
    ################################################################################
    
    prior-file = GW170817.prior
    
    ################################################################################
    ## Waveform arguments
    ################################################################################
    
    frequency-domain-source-model = lal_binary_neutron_star
    waveform_approximant = IMRPhenomPv2_NRTidalv2
    binary-type=BNS
    
    ################################################################################
    ## EOS arguments
    ################################################################################
    
    with-eos=True 
    eos-data=./eos/eos_sorted 
    Neos=15000
    eos-weight=./eos/EOS_sorted_weight.dat

The `trigger time` is the time of the observed event. With regard to detector arguments, one has to specify which detectors should be used. For example, `detectors = [H1, L1, V1]` stand for LIGO detectors Hanford, Livingston and Virgo. For each detector, the noise power spectral density of gravitational wave detector needs to be provided. Within `data_dict`, one needs to provide the GW170817 data measured within each detector.

With regard to likelihood arguments, we can specify if want a marginal likelihood that has been integrated over the parameter space, e.g. for distance, phase, or time. The prior file should be tailored to the observed event, in this case GW170817 but also with regard to the GW model that is used for the parameter estimation. The GW model can be specified in `waveform_approximant`. Here, we use the phenomenological model `IMRPhenomPv2_NRTidalv2` for a precessing binary neutron star model, see [Dietrich et al.](https://pure.mpg.de/rest/items/item_3058536/component/file_3058537/content).

For inferring the source properties of GW170817, we will sample over a set of EOSs that were computed with Chiral Effective Field Theory. The EOS set `15nsat_cse_natural_R14` was used in the study of [Huth et al.](https://www.nature.com/articles/s41586-022-04750-w#data-availability) and can be downloaded there. In order to make use of the EOS set during the sampling, a pre-routine is required. Within this step, one can include different constraints on the NS EOS such as measurement of NICER or pulsar measurements which is reflected in the `EOS_sorted_weight.dat` and one needs to sort the EOS files `eos_sorted` in order to reduce sampling time.   

Once the `config.ini` file is set, the genertation can be run with `nmma_gw_generation config.ini` which will create a directory `outdir`. The submit file for the inference can be found under `outdir/data/<name_of_analysis>_data_dump.pickle` in the example above it would be `GW170817_data_dump.pickle`. For the analysis of GW signals, it is recommended to run the inference (`nmma_gw_analysis <name_of_analysis>_data_dump.pickle`) on larger clusters. An exemplary submit script is shown below: 

	#!/bin/bash
	#SBATCH -p <queue_name>
	#SBATCH --job-name=GW170817
	#SBATCH --nodes=17
	#SBATCH --ntasks-per-node=48
	#SBATCH --time=48:00:00
	#SBATCH -o outdir/log_data_analysis/GW170817.log
	#SBATCH -e outdir/log_data_analysis/GW170817.err
	#SBATCH -D ./
	#SBATCH --export=NONE
	#SBATCH --no-requeue
	#SBATCH --account=<account_name>
	#SBATCH --mail-type=BEGIN,END
	#SBATCH --mail-user=<email>
	
	module load slurm_setup
	module load python/3.8
	source <path_to_environment>/bin/activate

	export SLURM_EAR_LOAD_MPI_VERSION="intel"    #for Intel MPI
	export MKL_NUM_THREADS="1"
	export MKL_DYNAMIC="FALSE"
	export OMP_NUM_THREADS=1
	export MPI_PER_NODE=48

	mpiexec -n $SLURM_NTASKS nmma_gw_analysis outdir/data/GW170817_data_dump.pickle --nlive 2048 --maxmcmc 10000 --nact 10 --no-plot --label GW170817 --outdir outdir/result --sampling-seed 1234

The final posterior samples for the observed event GW170817 can be found under `outdir/result/`. Note that settings might differ from cluster to cluster and also the installation of NMMA might be changed (conda vs. python installation).


### Injected GW signals

For synthetic signals, the `config.ini` file needs to be slightly adapted. First of all, some injection specific flags need to be provided which are listed below:

	################################################################################
	## Injection arguments
	################################################################################
	
	injection = True
	n-simulation = 1
	injection-file = ./O4_injections_mdyninj_1e-5.json
	injection_numbers=[0]

The `injection = True` flag enables parameter estimation with injected signals and `n-simulation = 1` initiates inference for one synthetic signal. An `injection-file` needs to be provided in order to specify for which system the inference should be run. The creation of injected signals is shown [here](./data_inj_obs.html). The `injection_numbers= [0]` uses in this case only the first signal in the injection file. 

Moreover, other flags which are related to an observed event should be commented out such as trigger time and provided observational data. The rest remains the same as shown above for the case of an observed GW event. 



