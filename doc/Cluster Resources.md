## Cluster Resources (Expanse)

One might also want to submit bulk jobs while using NMMA. Here, we have 
included an example script for job submission (called as jobscript.sh) in SLURM. This job was submitted on SDSC's
Expanse (ACCESS) cluster:

	#!/bin/bash
	#SBATCH --job-name=gw170817_gp_test.job
	#SBATCH --output=logs/gw170817_gp_test.out
	#SBATCH --error=logs/gw170817_gp_test.err
	#SBATCH -p compute
	#SBATCH --nodes=1
	#SBATCH --ntasks-per-node=10
	#SBATCH --mem=249325M
	#SBATCH --time=00:30:00
	#SBATCH --mail-type=ALL
	#SBATCH --mail-user= your_full_email
	#SBATCH -A <<project*>>
	#SBATCH --export=ALL
	module purge
	module load sdsc
	module load cpu/0.15.4 gcc/10.2.0 intel-mpi/2019.8.254
	source /home/username/anaconda3/bin/activate nmma_env
	export PATH=$PATH:/home/username/anaconda3/lib/
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/username/anaconda3/lib/
	mpiexec -n 10 light_curve_analysis --model Me2017 --outdir outdir --label injection --prior /home/username/nmma/priors/Me2017.prior --tmin 0.1 --tmax 20 --dt 0.5 --error-budget 1 --nlive 512 --Ebv-max 0 --injection /home/username/nmma/injection.json --injection-num 0 --injection-outfile outdir/lc.csv --generation-seed 42 --filters u,g,r,i,z,y,J,H,K --plot --remove-nondetections  

To submit the job, run:
	
	sbatch jobscript.sh

To check the job allotment, you can run:

	squeue -u username


Test runs on other clusters are currently in progress. Further examples on other cluster resources will be subsequently added.