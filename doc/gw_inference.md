## Inference of gravitational-wave signals

A Bayesian analysis of a gravitational-wave signal which is not accompanied by electromagnetic signals can be performed within nmma following two main steps: 

1) setting up a `config.ini` file and running the command

    nmma_gw_generation config.ini

3) perform the analysis or parameter estimation using:

    nmma_gw_analysis <name_of_analysis>_data_dump.pickle

Below, we provide an example of a gravitational-wave inference setup using observational data of GW170817 and another example for an injection based analysis. 
