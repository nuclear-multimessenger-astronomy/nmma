## Multi-messenger inference

A joint inference on gravitational-wave and electromagnetic signals requires NMMA to run on a supercomputer cluster because large memory space are required and need to be shared across many CPU cores. Here, we consider a full joint inference on the binary neutron star merger observed on 17th August 2017. 

In order to run a multi-messenger inference, we need to follow to main steps:

	nmma_generation config.ini

Perform the analysis or parameter estimation using:

    nmma_analysis <name_of_analysis>_data_dump.pickle

First of all, we set up the `config.ini` file and provide all required data and information.

**Observational data**

Firstly, all observational data is required that means observational data from single observed events:
- GW170817,
- GRB170817A,
- AT2017gfo.

These observational data need to be provided in the `config.ini` file for the joint inference.

**Prior**

Moreover, a prior on all observed messengers is required and needs to be tailored to the models used in the inference. Here, we use the GRB afterglow light curve model `TrPi2018` from afterglowpy and the kilonova model `Bu2019lm`. For the gravitational-wave signal, we assume the model `IMRPhenomPv2_NRTidalv2` for a precessing neutron star binary. A prior for the joint inference can be found [here](https://github.com/nuclear-multimessenger-astronomy/nmma/tree/main/example_files/prior), called `GW170817_AT2017gfo_GRB170817A.prior`.

**Electroamagnetic data and models**

In order to not only sample on gravitational-wave data, we provide further electromagnetic signal related flags. The flag `with-grb=True` will turn on the sampling on a GRB data. As NMMA currently only includes one GRB model, this model does not need to be further specified. If `with-grb=False`, a joint inference of GW+KN data is possible, excluding the GRB part. With regard to the kilonova model, we need to provide a specific model under `kilonova-model`, its respective reduced model grid (if applicable) under `kilonova-model-svd` and a `kilonova-interpolation-type` which can be either `sklearn_gp` or `tensorflow`. The `light-curve-data` flag should include both GRB and kilonova data if a joint inference on GW-GRB-KN is desired (meaning use: `with-grb=True`) or should just include the kilonova data if a GW-KN inference is targeted (meaning use: `with-grb=False`). The kilonova start/end time and time steps apply to both the GRB and kilonova model which will generate light curves during the inference to match the observed data provided. 

**Including EOS information**

NMMA enables to include nuclear information by using equations-of-state (EOS) and sample over the EOS during the inference. In order to include a set of EOSs, each EOS.dat file needs to include information on Mass, Radius and Tidal deformability. For the example shown in the `config.ini` file below, we see that `Neos = 5000` meaning that we include 5000 EOS.dat files each containing information on mass, radius and tidal deformability. We also see that a constraint from NICER measurements has been folded in and thus the `eos-weight` reflects this in a weighting. The EOS set should be sorted according to this weighting in order to reduce runtime for the sampling on the EOSs. 

**Running the config.ini generation**

In order to prepare the joint inference, a `config.ini` file is required which specifies all kind of models, observational data and inference settings. An example adjusted to the observed BNS merger can be found below:

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
    ## Calibration arguments
    ################################################################################
    
    calibration-model = CubicSpline
    spline-calibration-nodes = 10
    spline-calibration-envelope-dict = {H1:data/GW170817/Feb-20-2018_O2_LHO_GPSTime_1187008882_C02_RelativeResponseUncertainty_FinalResults.txt, L1:data/GW170817/Feb-20-2018_O2_LLO_GPSTime_1187008882_C02_RelativeResponseUncertainty_FinalResults.txt, V1:data/GW170817/V_calibrationUncertaintyEnvelope_magnitude5p1percent_phase40mraddeg20microsecond.txt}
    
    ################################################################################
    ## Job submission arguments
    ################################################################################

    label = GW170817-AT2017gfo-GRB170817A
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
    
    prior-file = GW170817_AT2017gfo_GRB170817A.prior
    
    ################################################################################
    ## Waveform arguments
    ################################################################################
    
    frequency-domain-source-model = lal_binary_neutron_star
    waveform_approximant = IMRPhenomPv2_NRTidalv2
    
    ################################################################################
    ## EM arguments
    ################################################################################
        
    binary-type=BNS
    light-curve-data=data/AT2017gfo-GRB170817A/AT2017gfo_GRB170817A.dat
    kilonova-model=Bu2019lm
    kilonova-model-svd=data/AT2017gfo-GRB170817A/svdmodels_reduced
    svd-mag-ncoeff=10
    svd-lbol-ncoeff=10
    kilonova-trigger-time=57982.5285236896
    kilonova-tmin=0.1
    kilonova-tmax=950
    kilonova-error=1
    kilonova-tstep=0.1
    kilonova-interpolation-type=sklearn_gp
    grb-resolution=12
    with-grb=True
    
    ################################################################################
    ## EOS arguments
    ################################################################################
    
    with-eos=True 
    eos-data=eos/with_NICER_J0740/EOS_024_uniform_5k_sorted
    Neos=5000
    eos-weight=eos/with_NICER_J0740/EOS_sorted_weight.dat


The joint inference generation can be performed by running:
    
    nmma_gw_generation config.ini

This will generate a `GW170817-AT2017gfo-GRB170817A_data_dump.pickle` file under `outdir/data/` which need to be provided for the joint inference function `nmma_analysis`. 

**Running the analysis**

As detailed above, running the analysis with the command `nmma_analysis outidr/data/GW170817-AT2017gfo-GRB170817A_data_dump.pickle` requires computational resources on a larger cluster. Below we show an example script for job submission called `jointinf.pbs` on a German cluster:

    #!/bin/bash
    #PBS -N <name of simulation>
    #PBS -l select=16:node_type=rome:mpiprocs=128
    #PBS -l walltime=24:00:00
    #PBS -e ./outdir/log_data_analysis/err.txt
    #PBS -o ./outdir/log_data_analysis/out.txt
    #PBS -m abe
    #PBS -M <email adress>
    
    module load python
    module load mpt
    module load mpi4py
    source <provide path to venv>

    export MPI_UNBUFFERED_STDIO=true
    export MPI_LAUNCH_TIMEOUT=240
    
    cd $PBS_O_WORKDIR
    mpirun -np 512 omplace -c 0-127:st=4 nmma_analysis <absolute path to folder>/outdir/data/GW170817-AT2017gfo-GRB170817A_data_dump.pickle --nlive 1024 --nact 10 --maxmcmc 10000 --sampling-seed 20210213 --no-plot --outdir <absolute path to outdir/result folder>

Note that settings might differ from cluster to cluster and also the installation of NMMA might be changed (conda vs. python installation). 
