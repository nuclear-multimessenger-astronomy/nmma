## Perform combined analyses

NMMA is capable of performing combined analyses to constrain the neutron star equation of state (EOS) and Hubble Constant. In the following, we will take as an example the EOS analysis.

### Generate a simulation set

First of all, you need to create an output directory, this output will host all the data that will be used to constrain the EOS.

	mkdir -p ./output

Running the following command line will generate a json file (injection.json)  with the BILBY processing of compact binary merging events. We take here binaries of type BNS, NSBH is also an option. This injection contents a simulation set of parameters : luminosity_distance, log10_mej_wind, KNphi, inclination_EM, timeshift, geocent_time for the Bu2019lm model. This creates an injection.json file in the ./output directory.

	nmma_create_injection --prior-file ./priors/Bu2019lm.prior --eos-file ./example_files/eos/ALF2.dat --binary-type BNS -f ./output/injection --n-injection 100 --original-parameters --extension json
