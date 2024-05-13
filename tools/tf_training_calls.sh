# Example tensorflow training calls for different model grids

# model: LANLTS1
# lightcurves: lcs_lanl_TS_wind1
create-svdmodel --model LANLTS1 --svd-path svdmodels_LANLTS1 --interpolation-type tensorflow --tmin 0. --tmax 21.0 --dt 0.1 --data-path lcs_lanl_TS_wind1 --tensorflow-nepochs 100 --outdir output_LANLTS1_tf --plot

# model: LANLTS2
# lightcurves: lcs_lanl_TS_wind2
create-svdmodel --model LANLTS2 --svd-path svdmodels_LANLTS2 --interpolation-type tensorflow --tmin 0. --tmax 21.0 --dt 0.1 --data-path lcs_lanl_TS_wind2 --tensorflow-nepochs 100 --outdir output_LANLTS2_tf --plot

# model: LANLTP1
# lightcurves: lcs_lanl_TP_wind1
create-svdmodel --model LANLTP1 --svd-path svdmodels_LANLTP1 --interpolation-type tensorflow --tmin 0. --tmax 21.0 --dt 0.1 --data-path lcs_lanl_TP_wind1 --tensorflow-nepochs 100 --outdir output_LANLTP1_tf --plot

# model: LANLTP2
# lightcurves: lcs_lanl_TP_wind2
create-svdmodel --model LANLTP2 --svd-path svdmodels_LANLTP2 --interpolation-type tensorflow --tmin 0. --tmax 21.0 --dt 0.1 --data-path lcs_lanl_TP_wind2 --tensorflow-nepochs 100 --outdir output_LANLTP2_tf --plot

# model: Bu2019lm
# lightcurves: lcs_bulla_2019_bns
create-svdmodel --model Bu2019lm --svd-path svdmodels_Bu2019lm --interpolation-type tensorflow --tmin 0. --tmax 21.0 --dt 0.1 --data-path lcs_bulla_2019_bns --tensorflow-nepochs 100 --outdir output_Bu2019lm_tf --plot

# model: Bu2019nsbh
# lightcurves: lcs_bulla_2019_nsbh
create-svdmodel --model Bu2019nsbh --svd-path svdmodels_Bu2019nsbh --interpolation-type tensorflow --tmin 0. --tmax 21.0 --dt 0.1 --data-path lcs_bulla_2019_nsbh --tensorflow-nepochs 100 --outdir output_Bu2019nsbh_tf --plot

# model: Bu2022Ye
# lightcurves: lcs_bulla_2022
create-svdmodel --model Bu2022Ye --svd-path svdmodels_Bu2022Ye --interpolation-type tensorflow --tmin 0. --tmax 21.0 --dt 0.1 --data-path lcs_bulla_2022 --tensorflow-nepochs 100 --outdir output_Bu2022Ye_tf --plot

# model: Bu2023Ye
# lightcurves: lcs_bulla_2023
create-svdmodel --model Bu2023Ye --svd-path svdmodels_Bu2023Ye --interpolation-type tensorflow --tmin 0. --tmax 21.0 --dt 0.1 --data-path lcs_bulla_2023 --tensorflow-nepochs 100 --outdir output_Bu2023Ye_tf --plot

# model: Ka2017 (no smooth)
# lightcurves: lcs_kasen_no_smooth
create-svdmodel --model Ka2017 --svd-path svdmodels_Ka2017_no_smooth --interpolation-type tensorflow --tmin 0. --tmax 21.0 --dt 0.1 --data-path lcs_kasen_no_smooth --tensorflow-nepochs 100 --outdir output_Ka2017_no_smooth_tf --plot

# model: Ka2017 (with smooth)
# lightcurves: lcs_kasen_with_smooth
create-svdmodel --model Ka2017 --svd-path svdmodels_Ka2017_with_smooth --interpolation-type tensorflow --tmin 0. --tmax 21.0 --dt 0.1 --data-path lcs_kasen_with_smooth --tensorflow-nepochs 100 --outdir output_Ka2017_with_smooth_tf --plot

# model: AnBa2022_log
# lightcurves: lcs_collapsar
create-svdmodel --model AnBa2022_log --svd-path svdmodels_AnBa2022_log --interpolation-type tensorflow --tmin 0.0 --tmax 21.0 --dt 0.1 --data-path lcs_collapsar --data-file-type hdf5 --plot --tensorflow-nepochs 100 --data-time-unit seconds --outdir output_AnBa2022_log_tf

# model: AnBa2022_linear
# lightcurves: lcs_collapsar
create-svdmodel --model AnBa2022_linear --svd-path svdmodels_AnBa2022_linear --interpolation-type tensorflow --tmin 0.0 --tmax 21.0 --dt 0.1 --data-path lcs_collapsar --data-file-type hdf5 --plot --tensorflow-nepochs 100 --data-time-unit seconds --outdir output_AnBa2022_linear_tf

# Use svdmodel-benchmark to generate additional performance plots. It takes many of the same arguments as create-svdmodel above, except for --tensorflow-nepochs and --plot.
# (Note that an error is currently raised if --ncpus is > 1 in svdmodel-benchmark, see Issue #125)
