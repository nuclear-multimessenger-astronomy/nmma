export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
lightcurve-analysis-lbol \
	--model Arnett_modified \
	--outdir outdir \
	--label lbol_test \
	--trigger-time 60168.79041667 \
	--data ./23bqun_bbdata.csv \
	--prior ./Arnett_modified.priors \
	--tmin 0.005 \
	--tmax 20 \
	--nlive 1024 \
	--error-budget 0.0001 \
	--seed 123451 \
	--plot \
  --soft-init \
