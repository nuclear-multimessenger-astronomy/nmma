# Instructions For NMMA Installation

## Preliminary Steps:

The steps highlighted below are primarily for Linux systems and Windows users are advised to use WSL (preferrably Ubuntu 20.04) for smooth installation.
Ubuntu 20.04 is available for free download on Microsoft Store.

**Installing Anaconda3**

On your Linux/WSL terminal, run the following commands to install anaconda (replace 5.3.1 by the latest version):
```
$ wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
$ bash Anaconda3-5.3.1-Linux-x86_64.sh
```

(For 32-bit installation, skip the ‘_64’ in both commands)

NOTE: If you already have Anaconda3 installed, please make sure that it is updated to the latest version (conda update --all). Also check that you do not have multiple
versions of python installed in usr/lib/ directory as it can cause version conflicts while installing dependencies.

Now do:
```
$ conda update --all
```

**Cloning the NMMA repository**

Fork the NMMA repository given below:


(NMMA Github Repo)[https://github.com/nuclear-multimessenger-astronomy/nmma]


Note that we link above to the main branch, but suggest making changes on your own fork (please also see our [contributing guide](./contributing.html)). Now, after forking, run the following command to clone the repository into your currently directory (by default, in your home directory):
```
$ git clone https://github.com/your_github_username/nmma
```

Change directory to the nmma folder:
```
* $ cd nmma
```

## Main Installation

Create a new environment using this command (environment name is nmma_env in this case):
```
$ conda create --name nmma_env python=3.8
$ conda activate nmma_env
```

NOTE: If this gives an error like: CommandNotFoundError: Your shell has not been properly configured to use 'conda activate', then run:
```
$ source ~/anaconda3/etc/profile.d/conda.sh
```

then proceed with conda activate nmma_env.


Get the latest pip version
```
$ pip install --upgrade pip
```

Check python and pip version like this:
```
$ python --version
$ pip --version
```

Python 3.8 and above and Pip 21.2 and above is ideal for this installation. It is recommended to update these for your installation.

For the moment we advise Linux users to avoid using Python 3.9 and Python 3.10 in their nmma environment; this can generate major problems for the operation. Preferably, use Python 3.8.

Install mpi4py:
```
$ conda install mpi4py
```
OR
```
$ pip install mpi4py
```

Install parallel-bilby:
```
$ conda install -c conda-forge parallel-bilby
```
OR
```
$ pip install parallel-bilby
```

Finally, navigate to where you have downloaded the nmma repository and run the following command to install NMMA:
```
$ python setup.py install
```
OR

```
$ pip install .
```

**Kilonova Models**
In order to run the Bu2019lm model, you will need to download the kilonova pickle files. They can be found at the following links:

- [Bu2019lm_mag.pkl](https://drive.google.com/file/d/1oEgXqTya65laRs-lUzG1LR9labNrZSHa/view)  (3.2 GB)
- [Bu2019lm_lbol.pkl](https://drive.google.com/file/d/1y4VTu-KVFp9rkYzQ1sm32_kZa4uru5jF/view) (0.3 GB)

These will need to be placed in the ```svdmodels/``` folder.

While most utilities in NMMA include a parameter to specify the location of the svdmodels folder, it is recommended to create a [symbolic link](https://manpages.ubuntu.com/manpages/focal/en/man1/ln.1.html) to the svdmodels folder in the nmma directory and the conda installation directory. This can be done by running something along the lines of:

```
ln -sf /Users/[username]/anaconda3/envs/[nmma_env]/lib/[python_version]/site-packages/nmma/em/svdmodels/ /Users/[USERNAME]/nmma/svdmodels
```

where [USERNAME] is your username, [nmma_env] is the name of your nmma environment, and [python_version] is the version of python you are using (e.g. 'python3.8'). This command may need to be modified depending on your specific file system or anaconda configuration. It may also be necessary to run the command with sudo permissions depending on your file system permissions. Once the symbolic link is created, running NMMA utilities that make use of the svdmodels folder will use the files present in the NMMA repository svdmodels folder.

**Known Issues**
1.  In case an error comes up during an NMMA analysis of the form:

ERROR:   Could not load MultiNest library "libmultinest.so"
ERROR:   You have to build it first,
ERROR:   and point the LD_LIBRARY_PATH environment variable to it!

Then, for using the PyMultinest library, it is required to get and compile the Multinest library separately. Instructions for the same are given [here](https://johannesbuchner.github.io/PyMultiNest/install.html)


Use the commands below to install the dependencies given in requirements.txt file which are necessary for NMMA:

```
$ pip install -r requirements.txt
$ python setup.py install
```

2. For those installing on WSL with pip, you may encounter an issue with installing parallel-bilby due to a dependency on python-ligo-lw.
This can be resolved by installing gcc with the following command:

```
 $ sudo apt-get install gcc
```

and attempting to install parallel-bilby again.

Install pymultinest (note this line may not work for arm64 Macs; see specifc instructions below)
```
$ conda install -c conda-forge pymultinest
```

3. Due to the web of package requirements that NMMA depends on, there can be unresolved package versioning issues. These can usually be resolved by pinning the version of the packages that are causing the issue before reattempting the installation. For example, if the installation fails due to an issue with astropy, then running the following command:
```
pip install astropy==4.3.1
```
However, please open a [new issue](https://github.com/nuclear-multimessenger-astronomy/nmma/issues/new) if there appear to be unresolvable conflicts.

4.  There is an issue pip installing `pyfftw` on arm64 Mac systems; see the dedicated section below for a solution. If any package appeared to have an issue installing, you can first check by attempting to install it again using pip:

```
* $ pip install importlib_resources
* $ pip install  extinction
* $ pip install dill
* $ pip install multiprocess
* $ pip install lalsuite
* $ pip install python-ligo-lw
```

NOTE: If everything has gone smoothly, all of these above mentioned "pip install something" commands will show that the requirements have already been satisfied. Otherwise, these will cover the dependencies
if not covered by python setup.py install. Also, if running python setup.py install shows something on the lines of "cannot cythonize without cython", run:

```
* $ conda install -c anaconda cython==0.29.24
```

and reattempt NMMA installation.

**Known arm64 Mac issues**
1. For arm64 Macs (e.g. M1, M2), there is a [documented issue](https://github.com/pyFFTW/pyFFTW/issues/349#issuecomment-1468638458) installing `pyfftw` with pip. To resolve this, use `Homebrew` to run
```
$ brew install fftw
```
then add the following lines to your `.zprofile` or `.bash_profile`:
```
export PATH="/opt/homebrew/bin:$PATH"
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/fftw/lib
export LDFLAGS="-Wl,-S,-rpath,/opt/homebrew/opt/fftw/lib -L/opt/homebrew/opt/fftw/lib"
export CFLAGS="-Wno-implicit-function-declaration -I/opt/homebrew/opt/fftw/include"
```

Close and reopen your terminal and run
```
$ pip install pyfftw
```
You may then need to rerun `pip install -r requirements.txt` to complete the dependency installations.


2. The `osx-arm64` conda-forge channel does not include `pymultinest`. Running `pip install -r requirements.txt` should have installed `pymultinest`, but you will still need to install and compile `Multinest` from the source. Within the `nmma` directory, run:
```
git clone https://github.com/JohannesBuchner/MultiNest
cd MultiNest/build
cmake ..
make
```
Next, add the following lines to your `.zprofile` or `.bash_profile`:
```
export LD_LIBRARY_PATH=$HOME/nmma/MultiNest/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$HOME/nmma/MultiNest/lib:$DYLD_LIBRARY_PATH
```
(NOTE: Modify these paths as appropriate for the location of your `MultiNest` installation. You can also combine the `DYLD_LIBRARY_PATH` lines for `MultiNest` and `fftw` (above) into a single line)


3. There are also issues with `tensorflow` and arm64 Macs. If using `tensorflow`, install it with the following commands:
```
pip install tensorflow-macos
pip install tensorflow-metal
```


**First Test for NMMA**

Run the following commands:

```
$ ipython
import nmma
import nmma.em.analysis
import nmma.eos.create_injection
```

NOTE (Okay, last one!): if everything is ok, it's the end of the installation. But in case it shows that such-and-such modules are absent, feel free to install those modules by visiting their anaconda documentation and install
those with their given commands. In case modules like afterglowpy and dust_extinction are needed, don't hesitate to do it with pip (normally it shouldn't happen), but some modules may not install correctly in case of disturbance.

Please pay special attention to the `import nmma.em.analysis` and make sure that it does not generate any errors.

If there are no errors, it is recommended to attempt the [quickstart tutorial](https://nuclear-multimessenger-astronomy.github.io/nmma/)


## Installation on expanse and other cluster resources

When installation on cluster resources, it is common that all modules required for installing NMMA out of the box are not available. However, most will make it possible to import the required modules (most commonly, these are software like gfortran or mpi).

For example, on XSEDE's Expanse cluster, one can start a terminal session with:

	module load sdsc
	module load openmpi

and follow the instructions above.

NOTE: If "module load openmpi" does not execute directly and it asks for dependencies, one can proceed with:

        module load sdsc
        module load cpu/0.15.4
        module load gcc/9.2.0
        module load openmpi/4.1.1

## Matplotlib fonts

On new Linux installations, we sometimes come across the warning: "findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans". If you do prefer to use 'Times New Roman' for all of your plotting needs, you can install msttcorefonts with:

	sudo apt install msttcorefonts -qq

After removing the matplotlib cache:

	rm ~/.cache/matplotlib -rf

Beautiful fonts should be yours.
