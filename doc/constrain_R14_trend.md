## Contrain neuton stars Equation of State (EoS) 

### create an output 

First of all, you need to creat an output directory, this output will host all the data that will be used to constrain the EoS.
    
	mkdir -p ./nmma/output


### gennerate EoS injection.json

Run this command line will genarate  a json file (injection.json)  with the BILBY processing of compact binary merging events (BNS, NSBH, optional), here it's BNS. This injection contents a simulation of the parameters : luminosity_distance, log10_mej_wind, KNphi, inclination_EM, KNtimeshift, geocent_time.
At the end of the run, when the process is ok,  an injection.json will be find at  ./nmma/output directory.

	nmma_create_injection --prior-file ./nmma/example_files/prior/ZTF_kn.prior --eos-file ./nmma/example_files/eos/ALF2.dat --binary-type BNS -f ./nmma/output/injection --n-injection 100 --original-parameters  --extension json


### create the direction of the EMdata 

EMdata will house the posteriors of Electromagnetic data,  the lc.csv (./nmma/example_files/csv_lightcurve/outdir/macroeventID, where macroeventID in range(0, 100) )lightcurves.

	mkdir -p ./nmma/output/EMdata

### lightcurve posterior 

The next command line provide some EM posterios in this example we have a simulation of a sample of 100 events and and 26 of them are detectable by the ZTF. 

	for macroeventID in {0..99}

	do
	 light_curve_analysis --model Bu2019lm --svd-path ./nmma/svdmodels --gptype tensorflow  --outdir ./nmma/output/EMdata/outdir/$macroeventID --label injection_Bu2019lm --prior ./nmma/example_files/prior/ZTF_kn.prior --tmin 0 --tmax 7 --dt 0.5 --error-budget 1.0 --nlive 256 --Ebv-max 0 --injection ./nmma/output/injection.json --injection-num $macroeventID --injection-detection-limit 22,22,22 --injection-outfile ./nmma/example_files/csv_lightcurve/outdir/$macroeventID/lc.csv --generation-seed 42 --filters g,r,i  --ztf-sampling --ztf-uncertainties --plot --remove-nondetections --optimal-augmentation --optimal-augmentation-filters u,g,r,i,z,y,J,H,K --optimal-augmentation-N 100
    
	done



### EoS from GW + EM 

This allows to generate the EoS by combining the data of the events (EMdata) and those coming from the gravitational (GWdata).The 26 detectable events are {0,  3,  5,  7,  8, 10, 12, 13, 14, 15, 17, 19, 21, 22, 23, 24, 26, 27, 28, 31, 32,34, 36, 37, 38, 39}.
A GW_EMdata files includind the combination of  GW and EM EoS data can be get at the direction of ./nmma/output . 

	for macroeventID in {0,  3,  5,  7,  8, 10, 12, 13, 14, 15, 17, 19, 21, 22, 23, 24, 26, 27, 28, 31, 32,34, 36, 37, 38, 39}

	do

	 gwem_resampling --outdir ./nmma/output/GW_EMdata/$macroeventID --EMsamples ./nmma/output/EMdata/outdir/$macroeventID/injection_Bu2019lm_posterior_samples.dat --GWsamples  ./nmma/output/GWdata/outdir/inj_PhD_posterior_samples_$macroeventID.dat --EOS ./nmma/example_files/eos_sorted --nlive 8192 --GWprior ./nmma/priors/aligned_spin.priors --EMprior ./nmma/priors/EM.prior --total-ejecta-mass --Neos 5000

	done


### After all that plot EOS

There are two pythons files (R14_trend_generate.py, R14_trend_plot.py)  on ./nmma/examples_files, which can use to visualize the EOS data.

First of all create a foder to put the final data about EoS:

	mkdir -p ./nmma/output/Figures

The first file to run is :

	python R14_trend_generate.py

you should obtain  GW_EM_R14trend_ZTF.dat  at ./nmma/output/Figures an then:

	python R14_trend_plot.py

This return a EoS plot.
