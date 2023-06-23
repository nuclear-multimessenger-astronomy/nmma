# Instructions for using **[nmma]** and **[gwemopt]** to simulate KNe detection with ZTF and LSST

First of all we have to install **[nmma]**  and **[gwemopt]** together in the same environmment.

1 **- Install nmma**:

Instructions For  `nmma` click here:  **[NMMA-installation]**

2 **- Install gwemopt**:

Instructions For  `gwemopt` click here:  **[gwemopt-installation]**


3 **- Split Farah(GWTC-3) CBCs data in BNS and NSBH**

In the injections.dat files outcome  **[observing-scenarios-simulations]**, the masses of the CBCs are masses observed in the detectors. To splitting them  into BNS, NSBH and BBH populations, we should convert them in source frame mass.

All CBCs in ``Farah(GWTC-3)`` distribustion  that passed the SNR are recorded here: **[obs-scenarios-data-2022]** . For reasons of weight, we keep only the data necessary to produce the light curves, so the simulation with nmma.
For more data like skymaps file please go to **[zenodo]**. Anyway we need to download it if we need to get skymap files (``.fits`` files )  to run the KNe  detection process with **[gwemopt]**.

Use the following python code for the ``split``:

Click here: **[split_CBC_data_for_nmma.py]**

## **Generate a `simulation set`**

4 **- Then use ``BNS`` or ``NSBH``**  injections.dat  to create injection file with nmma.

To generate a json file (``injection.json``)  with the BILBY processing of compact binary merging events. This injection contents a simulation set of parameters : luminosity_distance, log10_mej_wind, KNphi, inclination_EM, KNtimeshift, geocent_time for the KNe (`Bu2019lm` - BNS or `Bu2019nsbh` - NSBH) model.

**`BNS` type**

    nmma_create_injection --prior-file ./priors/Bu2019lm.prior --injection-file ./injections.dat --eos-file  ./example_files/eos/ALF2.dat --binary-type BNS --original-parameters --extension json --aligned-spin --binary-type BNS --filename ./outdir_BNS/injection_Bu2019lm.json

**`NSBH` type**

    nmma_create_injection --prior-file ./priors/Bu2019nsbh.prior --injection-file ./injections.dat --eos-file  ./example_files/eos/ALF2.dat --binary-type NSBH  --original-parameters --extension json --aligned-spin --filename ./outdir_NSBH/injection_Bu2019nsbh.json

5 **- Generate lightcurve **

This command line concern the ``ZTF`` for ``Rubin`` please replace `--injection-detection-limit 24.1,21.7,21.4,20.9,24.5,23.0,23.2,22.6,22.6 `` by  **--injection-detection-limit 23.9,25.0,24.7,24.0,23.3,22.1,23.2,22.6,22.6**

**`BNS`**

    light_curve_generation --model  Bu2019lm --svd-path ./svdmodels --outdir ./outdir_BNS --label injection_Bu2019lm --dt 0.5 --injection ./outdir_BNS/injection_Bu2019lm.json --injection-detection-limit 24.1,21.7,21.4,20.9,24.5,23.0,23.2,22.6,22.6 --filters sdssu,ps1__g,ps1__r,ps1__i,ps1__z,ps1__y,2massj,2massh,2massks  --absolute --plot --generation-seed 42

**`NSBH`**


light_curve_generation --model  Bu2019lm --svd-path ./svdmodels --outdir ./outdir_NSBH --label injection_Bu2019nsbh --dt 0.5 --injection ./outdir_NSBH/injection_Bu2019nsbh.json --injection-detection-limit 24.1,21.7,21.4,20.9,24.5,23.0,23.2,22.6,22.6 --filters sdssu,ps1__g,ps1__r,ps1__i,ps1__z,ps1__y,2massj,2massh,2massks --absolute --plot --generation-seed 42

6 **- KNe detection process using ``gwemopt``**


* Here we need to use ``skymap`` files from **[zenodo]**  in this example we use  **O4** (``--skymap-dir ./runs/O4/farah/allsky``), but this should be replace by **O5** if you work with **O5** data, then...
* Then the json injection file ``injection_Bu2019lm.json`` for BNS or ``injection_Bu2019nsbh.json`` for NSBH, which contains the IDs of the simulations we need to identify their correspondence in the ``skymap`` data.
* And the direction of the ``config`` foder in  **[gwemopt]** file.

* in **``--telescope ZTF``**, **ZTF** should be replace by ``LSST`` for ``Rubin`` telescope.
* Also the ``--exposuretime 300`` means time of observation (300 secondes), but we should also use ``--exposuretime 180` for 180 secondes.

**`BNS`**

    light_curve_detection --configDirectory ./gwemopt/config --outdir ./ztf_detection/outdir_BNS --injection-file  ./outdir_BNS/injection_Bu2019lm.json  --skymap-dir ./runs/O4/farah/allsky --lightcurve-dir ./outdir_BNS --binary-type BNS --filters sdssu,ps1__g,ps1__r,ps1__i,ps1__z,ps1__y,2massj,2massh,2massks --exposuretime 300 --detections-file ./ztf_detection/bns_lc_skymap_detection.txt --telescope ZTF --tmin 0 --tmax 14 --dt 0.5  --parallel --number-of-cores 10 --generation-seed 42

**`NSBH`**

    light_curve_detection --configDirectory ./gwemopt/config --outdir ./ztf_detection/outdir_NSBH --injection-file  ./outdir_NSBH/injection_Bu2019nsbh.json  --skymap-dir ./runs/O4/farah/allsky --lightcurve-dir ./outdir_NSBH --binary-type NSBH --filters sdssu,ps1__g,ps1__r,ps1__i,ps1__z,ps1__y,2massj,2massh,2massks --exposuretime 300 --detections-file ./ztf_detection/nsbh_lc_skymap_detection.txt --telescope ZTF --tmin 0 --tmax 14 --dt 0.5  --parallel --number-of-cores 10 --generation-seed 42





[nmma]: https://github.com/nuclear-multimessenger-astronomy/nmma
[NMMA-installation]: https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/doc/installation.md
[gwemopt]: https://github.com/mcoughlin/gwemopt
[gwemopt-installation]:https://github.com/mcoughlin/gwemopt/blob/main/README.md
[observing-scenarios-simulations]:https://github.com/lpsinger/observing-scenarios-simulations
[zenodo]:https://doi.org/10.5281/zenodo.7026209
[obs-scenarios-data-2022]:https://github.com/weizmannk/obs-scenarios-data-2022
[split_CBC_data_for_nmma.py]:https://github.com/weizmannk/obs-scenarios-data-2022/blob/main/split_CBC_data_for_nmma.py
