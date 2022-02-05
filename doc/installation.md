## Installation

### anaconda3

First of all, we suggest you install [anaconda](https://docs.anaconda.com/anaconda/install/linux/), in order to avoid conflicts linked to different versions of pythons or pip. Afterwards, update conda:

	conda update --all


### Cloning nmma

Then, you should go ahead and git clone the repository. Note that we link below to the main branch, but suggest making changes on your own fork (please also see our [contributing guide](./contributing.html)).

	git clone git@github.com:nuclear-multimessenger-astronomy/nmma.git

### Create an nmma environment

We suggest to create a conda environment when running nmma, which will allow you to, for example, remove the environment in case things go wrong.

	conda create --name nmma python=3.8
	conda activate nmma

This will pin your python to version 3.8, change the number in case of upgrades. At this point, you can check that things worked here:

	python --version
	pip --version

### Installing dependencies

We first install mpi4py, which turns out to be one of the trickier dependencies for some reason.

	conda install mpi4py
or
	pip install mpi4py

We then suggest installing bilby through conda:

	conda install -c conda-forge bilby

and pip installing parallel_bilby

	pip install parallel_bilby

Another possibility is to do it all in one go with:

	conda install -c conda-forge parallel_bilby

although that command hangs on certain systems.

The next hurdle is pymultinest, and again, we suggest using conda

	conda install -c conda-forge pymultinest

At this point, we should be able to use pip to install nmma's remaining requirements, i.e.

	python setup.py install

or

	pip install .

For reasons that are not totally clearly, some modules like afterglowpy and dust_extinction might not be installed correctly in this step. If not, can simply run, for example,

	pip install afterglowpy

and that should do it. If everything went well, importing nmma and its submodules:

	import nmma
	import nmma.em.analysis
	import nmma.eos.create_injection

should work and you should be good to go.

### Dependency conflicts

Unfortunately, due to the web of package requirements that NMMA depends on, running setup.py does not typically finish without errors the first time through. Experience has shown that in the vast majority of cases, simply pinning versions such as:

	pip install astropy==4.3.1

and then trying again is sufficient for completion of the installation. However, please open issues on GitHub if there appear to be unresolvable conflicts.

### Installation on expanse and other cluster resources

When installation on cluster resources, it is common that all modules required for installing NMMA out of the box are not available. However, most will make it possible to import the required modules (most commonly, these are software like gfortran or mpi).

For example, on XSEDE's Expanse cluster, one can start a terminal session with:

	module load sdsc
	module load openmpi

and follow the instructions above.
