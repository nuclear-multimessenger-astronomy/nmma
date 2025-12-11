# Light Curves Simulation Using the **injections.dat** from the **[observing-scenarios-simulations]**

This document provides a detailed description of how to use the **injections.dat** file, which contains gravitational wave parameters simulated for generating light curves, specifically kilonova light curves.

## 1. Creating a JSON Injection File: GW Conversion to KN

Running the following command will generate a JSON file (`injection_Bu2019lm.json`) with the BILBY processing of compact binary merger events. Here, we consider binaries of type BNS (Binary Neutron Stars), although NSBH (Neutron Star-Black Hole) is also an option. This injection contains a simulation set of parameters: `luminosity_distance`, `log10_mej_wind`, `KNphi`, `inclination_EM`, `timeshift`, `geocent_time` for the Bu2019lm model.


### Command to Run [nmma-create-injection]:

Run this command to create the JSON injection file:


    nmma-create-injection --prior-file ./priors/Bu2019lm.prior --injection-file ./example_files/sim_events/bns_O4_injections.dat --eos-file ./example_files/eos/ALF2.dat --binary-type BNS --extension json -f ./outdir/injection_Bu2019lm --generation-seed 42 --aligned-spin


To get a "dat" format, replace --extension json with ``--extension dat``.
You can add the ``--eject`` flag, which is better as it removes events that do not generate enough ejecta or are too heavy and will become quickly a black hole. Adding the ``--eject`` flag requires adding alpha, ratio_zeta, and ratio_epsilon in the Bu2019lm.prior file:


    alpha = Gaussian(mu=0., sigma=4e-4, name='alpha', latex_label='$\\alpha$')
    ratio_zeta = Uniform(minimum=0., maximum=1.0, name='zeta', latex_label='$\\zeta$')
    ratio_epsilon = 0.04



## 2. Simulating Light Curve Posteriors : Run the [lightcurve-analysis]

#### Run this command to simulate the light curves for the ZTF telescope:


    lightcurve-analysis --model Bu2019lm --svd-path ./svdmodels --interpolation-type sklearn_gp --outdir ./outdir/BNS/0 --label injection_Bu2019lm_0 --prior ./Bu2019lm.prior --tmin 0 --tmax 20 --dt 0.5 --error-budget 1 --nlive 2048 --Ebv-max 0 --injection ./outdir/injection_Bu2019lm.json --injection-num 0 --injection-detection-limit 21.7,21.4,20.9 --injection-outfile ./outdir/BNS/0/lc.csv --generation-seed 42 --filters ztfg,ztfr,ztfi --plot --remove-nondetections --local-only --ztf-ToO 300 --ztf-uncertainties --ztf-sampling --ztf-ToO 300


Here, `--injection-num 0` means to simulate the first event (simulation_id=39) in the JSON file, and `/outdir/BNS/0` is the output directory. For the second event, use `--injection-num 1` and `/outdir/BNS/1`, and so on. The **simulation_id** is the number of the CBC which passed the threshold cut during the observing scenarios simulation, which includes all the subpopulation. Here, `bns_O4_injections.dat` is a subset containing only the BNS events, so **simulation_id** is different from the `--injection-num`.



### Run this command to simulate the light curves for the Vera C. Rubin Observatory (LSST):

    lightcurve-analysis --model Bu2019lm --svd-path ./svdmodels --interpolation-type sklearn_gp --outdir ./outdir/BNS/0 --label injection_Bu2019lm_0 --prior ./Bu2019lm.prior --tmin 0 --tmax 20 --dt 0.5 --error-budget 1 --nlive 2048 --Ebv-max 0 --injection ./outdir/injection_Bu2019lm.json --injection-num 0 --injection-detection-limit 23.9,25.0,24.7,24.0,23.3,22.1 --injection-outfile ./outdir/BNS/0/lc.csv --generation-seed 42 --filters sdssu,ps1__g,ps1__r,ps1__i,ps1__y,ps1__z --plot --remove-nondetections --local-only --rubin-ToO-type BNS --rubin-ToO



# For Deep: Detailed Steps

The latest update of the **[Observing-Scenarios-Simulations]** for the runs O4 and O5 is located in **[10.5281/zenodo.10061254]**, and a published **[paper]** of this work is available.

Download or use this Python script **[downloader]** to do it easily, then unzip it.

The next process is to split the **injections.dat** located in `runs/*/*/farah` into BNS and NSBH. Note that the injections.dat includes all the CBC populations (BNS, NSBH, and BBH). To split them properly, use the **[subpopulation-splitting]** tool and ensure the input directory is correct.

The CBCs masses recorded by the **injections.dat** are the detector masses, so to split them, we need to convert the component masses to their real masses first. The **[subpopulation-splitting]** script is readable and understandable. It also allows adding 'GPS times' and other parameters useful for BILBY. The GPS times are already present in the '.fits' skymap events, adding them might prolong reading times. The formula for `gps_time` is: `gps_time = geocent_end_time + (geocent_end_time_ns * 10^(-9))`, where `geocent_end_time` is in seconds (s) and `geocent_end_time_ns` is in nanoseconds, easily obtainable from the XML files. Set `GPS_TIME = False`.
In `True` means read the GPS times from '.fits' files which is will take so long to process.

In the output, you will have a `subpopulations` folder, which contains the split BNS and NSBH events. This is useful for those who need quick parameters of BNS and NSBH for their light curve simulations or EM counterpart statistical estimation, as shown above.



Install  `gwpy` in your NMMA environment before splitting the BNS and NSBH events, use the following command:

    pip install gwpy


[observing-scenarios-simulations]: https://github.com/lpsinger/observing-scenarios-simulations
[10.5281/zenodo.10061254]: https://zenodo.org/doi/10.5281/zenodo.10061254
[downloader]: https://github.com/weizmannk/ObservingScenariosInsights/blob/main/src/Zenodo_Downloader.py
[subpopulation-splitting]: https://github.com/weizmannk/ObservingScenariosInsights/blob/main/src/Subpopulation_Splitter.py
[paper]: https://doi.org/10.3847/1538-4357/acfcb1

[nmma-create-injection]: https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/nmma/eos/create_injection.py
[lightcurve-analysis]: https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/nmma/em/analysis.py
