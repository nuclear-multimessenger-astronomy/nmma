## Contrain neutron stars Equation of State (EoS) 

### create an output 

First of all, you need to create an output directory, this output will host all the data that will be used to constrain the EoS.
    
	mkdir -p ./output


### generate EoS injection.json

Running this command line will generate  a json file (injection.json)  with the BILBY processing of compact binary merging events (BNS, NSBH, optional), here it's BNS. This injection contents a simulation of the parameters : luminosity_distance, log10_mej_wind, KNphi, inclination_EM, KNtimeshift, geocent_time.
At the end of the run, when the process is ok,  an injection.json will be find at  ./output directory.

	nmma_create_injection --prior-file ./priors/Bu2019lm.prior --eos-file ./example_files/eos/ALF2.dat --binary-type BNS -f ./output/injection --n-injection 100 --original-parameters --extension json


### lightcurve posterior 

EMdata will house the posteriors of Electromagnetic data,  the lc.csv (./example_files/csv_lightcurve/outdir/macroeventID, where macroeventID in range(0, 100) )lightcurves.
The next command line provide some EM posteriors in this example we have a simulation of a sample of 100 events and and 26 of them are detectable by the ZTF. The result can be find at  ./output/EMdata 

	for macroeventID in {0..99}
	
	do
	  mkdir -p ./output/EMdata/outdir/$macroeventID/
	  light_curve_analysis --model Bu2019lm --svd-path ./svdmodels --gptype tensorflow --outdir ./output/EMdata/outdir/$macroeventID --label injection_Bu2019lm --prior ./priors/Bu2019lm.prior --tmin 0 --tmax 7 --dt 0.5 --error-budget 1.0 --nlive 256 --Ebv-max 0 --injection ./output/injection.json --injection-num $macroeventID --injection-detection-limit 22,22,22 --injection-outfile ./output/EMdata/outdir/$macroeventID/lc.csv --generation-seed 42 --filters g,r,i --ztf-sampling --ztf-uncertainties --plot --remove-nondetections --optimal-augmentation --optimal-augmentation-filters u,g,r,i,z,y,J,H,K --optimal-augmentation-N 100
	done
	  


### EoS from GW + EM 

This allows to generate the EoS by combining the data of the events (EMdata) and those coming from the gravitational (GWdata).The 26 detectable events are {0,  3,  5,  7,  8, 10, 12, 13, 14, 15, 17, 19, 21, 22, 23, 24, 26, 27, 28, 31, 32,34, 36, 37, 38, 39}.
A GW_EMdata files includind the combination of  GW and EM EoS data can be get at the direction of ./output . 

	for macroeventID in 0 3 5 7 8 10 12 13 14 15 17 19 21 22 23 24 26 27 28 31 32 34 36 37 38 39

	do
	  mkdir -p ./output/GW_EMdata/$macroeventID/
	  gwem_resampling --outdir ./output/GW_EMdata/$macroeventID --EMsamples ./output/EMdata/outdir/$macroeventID/injection_Bu2019lm_posterior_samples.dat --GWsamples  ./output/GWdata/outdir/inj_PhD_posterior_samples_$macroeventID.dat --EOS ./example_files/eos/eos_sorted --nlive 8192 --GWprior ./priors/aligned_spin.priors --EMprior ./priors/EM.prior --total-ejecta-mass --Neos 5000

	done


### After all that plot EOS

There are two pythons files (R14_trend_generate.py, R14_trend_plot.py)  on ./examples_files/post-processing , which can use to visualize the EOS data.

First of all create a foder to put the final data about EoS:

	mkdir -p ./output/Figures

Then got to :
	
	cd ./example_files/post-processing
	
The first file to run is :

	python R14_trend_generate.py

Normally when everything is ok you should be at are actully ./example_files/post-processing.
You should obtain  GW_EM_R14trend_ZTF.dat  at  ../../output/Figures.
Then please stay at the same location, that is mean here : ./example_files/post-processing and run the plot 

	python R14_trend_plot.py

	
This return a EoS plot, R14_trend_GW_EM_ZTF.pdf, at the ../../output/Figures 
