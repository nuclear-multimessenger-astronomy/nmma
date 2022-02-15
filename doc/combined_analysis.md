## Perform combined analyses

NMMA is capable of performing combined analyses to constrain the neutron star equation of state (EOS) and Hubble Constant. In the following, we will take as an example the EOS analysis.

### Generate a simulation set

First of all, you need to create an output directory, this output will host all the data that will be used to constrain the EOS.

	mkdir -p ./output

Running the following command line will generate a json file (injection.json)  with the BILBY processing of compact binary merging events. We take here binaries of type BNS, NSBH is also an option. This injection contents a simulation set of parameters : luminosity_distance, log10_mej_wind, KNphi, inclination_EM, KNtimeshift, geocent_time for the Bu2019lm model. This creates an injection.json file in the ./output directory.

	nmma_create_injection --prior-file ./priors/Bu2019lm.prior --eos-file ./example_files/eos/ALF2.dat --binary-type BNS -f ./output/injection --n-injection 100 --original-parameters --extension json


### lightcurve posterior

EMdata will house the posteriors of the electromagnetic data you will produce: in particular the lc.csv (./example_files/csv_lightcurve/outdir/macroeventID, where macroeventID in range(0, 100)) lightcurves. We now compute posteriors using NMMA on this simulated set of 100 events, of which we assume a fraction is detectable by ZTF. The result can be find at  ./output/EMdata

	for macroeventID in {0..99}

	do
	  mkdir -p ./output/EMdata/outdir/$macroeventID/
	  light_curve_analysis --model Bu2019lm --svd-path ./svdmodels --gptype tensorflow --outdir ./output/EMdata/outdir/$macroeventID --label injection_Bu2019lm --prior ./priors/Bu2019lm.prior --tmin 0 --tmax 7 --dt 0.5 --error-budget 1.0 --nlive 256 --Ebv-max 0 --injection ./output/injection.json --injection-num $macroeventID --injection-detection-limit 22,22,22 --injection-outfile ./output/EMdata/outdir/$macroeventID/lc.csv --generation-seed 42 --filters g,r,i --ztf-sampling --ztf-uncertainties --plot --remove-nondetections --optimal-augmentation --optimal-augmentation-filters u,g,r,i,z,y,J,H,K --optimal-augmentation-N 100
	done


### Download GW posteriors

The gravitational wave samples can be can be downloaded at https://zenodo.org/record/6045029#.YgZzwITMKV5. At this link there are simulated posteriors for a number of gravitational-wave waveform models, here, we take the PhenDNRTv2 files.
This only  concern  the PhenDNRTv2 files on this link. These we can directly download by using this command line:

Create an outdir directory to put GW data that you will upload.

	mkdir -p ./output/GWdata/outdir

Go to the GWdata directory

	cd ./output/GWdata/outdir

Running the next command line:

	xargs -n 1 curl -# -O < ../../../example_files/zenodo/gw_posteriors.txt

or use this one :

	for url in `cat ../../../example_files/zenodo/gw_posteriors.txt`
	do
	  curl -#  -O $url
	done


The gw_posteriors.txt contains a list of all links to the PhenDNRTv2 files. When the download is complete, return to the main directory

        cd ../../..

### Download the EOS repository

All of the NMMA EOS simulation sets are kept in a separate github repository here:
https://github.com/diettim/NMMA

For this particular example, we have put the simulation set in Zenodo here:
https://zenodo.org/record/6094691#.YgwA8YTMI5k

### EoS from GW + EM

This command line combines the EOS measurements for each simulation. As stated above, we assume only a fraction is detectable by ZTF (based on simulations of the associated kilonova brightnesses). The indices of the 26 detectable events are {0,  3,  5,  7,  8, 10, 12, 13, 14, 15, 17, 19, 21, 22, 23, 24, 26, 27, 28, 31, 32,34, 36, 37, 38, 39}.

	for macroeventID in 0 3 5 7 8 10 12 13 14 15 17 19 21 22 23 24 26 27 28 31 32 34 36 37 38 39

	do
	  mkdir -p ./output/GW_EMdata/$macroeventID/
	  gwem_resampling --outdir ./output/GW_EMdata/$macroeventID --EMsamples ./output/EMdata/outdir/$macroeventID/injection_Bu2019lm_posterior_samples.dat --GWsamples  ./output/GWdata/outdir/inj_PhD_posterior_samples_$macroeventID.dat --EOS ./example_files/eos/eos_sorted --nlive 8192 --GWprior ./priors/aligned_spin.priors --EMprior ./priors/EM.prior --total-ejecta-mass --Neos 5000

	done


### A combined EOS analysis

We provide a helper function to combine the EOS results.

First of all create a foder to put the final data about EoS:

	mkdir -p ./output/Figures

Then run the last one command line

	 combined_EOS --outdir ./output/Figures --label ZTF --gwR14trend ./example_files/ --GWEMsamples ./output/GW_EMdata --detections-file ./example_files/csv_lightcurve/detectable.txt --EOS-prior ./example_files/eos/EOS_sorted_weight.dat --EOSpath ./example_files/eos/eos_sorted  --pdet ./example_files/eos/pdet_of_Mmax.dat --R14_true 11.55 --Neos 5000 --seed 42  --cred-interval 0.95


This should return a EoS plot, R14_trend_GW_EM_ZTF.pdf, and  GW_EM_R14trend_ZTF.dat  at  ./output/Figures.
