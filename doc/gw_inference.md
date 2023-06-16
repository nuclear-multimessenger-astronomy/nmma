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
    eos-data=./eos/eos_IST_sorted 
    Neos=9501
    eos-weight=./eos/EOS_IST_sorted_weight.dat

The `trigger time` is the time of the observed event. With regard to detector arguments, one has to specify which detectors should be used. For example, `detectors = [H1, L1, V1]` stand for LIGO detectors Hanford, Livingston and Virgo. For each detector, the noise power spectral density of gravitational wave detector needs to be provided. Within `data_dict`, one needs to provide the GW170817 data measured within each detector.

With regard to likelihood arguments, we can specify if want a marginal likelihood that has been integrated over the parameter space, e.g. for distance, phase, or time. The prior file should be tailored to the observed event, in this case GW170817 but also with regard to the GW model that is used for the parameter estimation. The GW model can be specified in `waveform_approximant`. Here, we use the phenomenological model `IMRPhenomPv2_NRTidalv2` for a precessing binary neutron star model.

For inferring the source properties of GW170817, we will sample over a set of EOSs that were computed with Chiral Effective Field Theory. The EOS set 15nsat_cse_natural_R14 was used in the study of Huth et al. and can be downloaded there. In order to

### Inejcted GW signals
