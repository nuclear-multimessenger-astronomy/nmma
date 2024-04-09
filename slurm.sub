#!/bin/bash
#SBATCH --job-name=lightcurve-analysis.job
#SBATCH --output=slurm_logs/lightcurve-analysis_%A_%a.out
#SBATCH --error=slurm_logs/lightcurve-analysis_%A_%a.err
#SBATCH -p shared
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 10
#SBATCH --gpus 0
#SBATCH --mem 64G
#SBATCH --time=24:00:00
#SBATCH -A umn131
#SBATCH --mail-type=NONE
source activate nmma_env
mpiexec -n 10 -hosts=$(hostname) lightcurve-analysis --model $MODEL --interpolation-type tensorflow --svd-path svdmodels --outdir fritz_outdir/$LABEL --label $LABEL --trigger-time $TT --data $DATA --prior priors/$PRIOR.prior --tmin $TMIN --tmax $TMAX --dt $DT --n-tstep 50 --photometric-error-budget 0.1 --svd-mag-ncoeff 10 --svd-lbol-ncoeff 10 --Ebv-max 0.5724 --grb-resolution 5 --jet-type 0 --error-budget 1.0 --sampler pymultinest --sampler-kwargs {} --cpus 1 --nlive 2048 --seed 42 --xlim 0,14 --ylim 22,16 --generation-seed 42 --photometry-augmentation-seed 0 --photometry-augmentation-N-points 10 --conditional-gaussian-prior-N-sigma 1 --plot $SKIP_SAMPLING

# This script is meant to be placed on SDSC Expanse to assist in running the nmma API service (https://github.com/Theodlz/nmma-api) via skyportal/fritz.
# The script can be generated with nmma by running tools/analysis_slurm.py and providing the desired analysis/compute resource arguments to customize.
# Place the script in the directory where you'll run nmma, making sure that the subdirectories "priors" (containing nmma priors) and "slurm_logs" (to hold log files) exist.
# (It may be easiest to install nmma from source on your HPC resource rather than pip installing.)
