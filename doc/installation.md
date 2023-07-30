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

(For 32-bit installation, skip the ‘\_64’ in both commands)

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

**Python 3.11 is now supported by NMMA!**


Install mpi4py:

```
$ conda install mpi4py
```

> [!IMPORTANT]
> We discourage installing mpi4py with pip. The installation does not work properly due to issues with MPI header files, etc.

Install parallel-bilby:

```
$ conda install -c conda-forge parallel-bilby
```

> [!NOTE]
> Installing parallel-bilby takes quite some time. Please be patient. If you encounter any errors, please check the [parallel-bilby installation guide](https://lscsoft.docs.ligo.org/parallel_bilby/installation) for more details.

> [!NOTE]  
> for those installing on WSL with pip, you may encounter an issue with installing parallel-bilby due to a dependency on python-ligo-lw.
> This can be resolved by installing gcc with the following command:

```
 $ sudo apt-get install gcc
```

and attempting to install parallel-bilby again.

Install pymultinest (note this line may not work for arm64 Macs; see specifc instructions below)

```
$ conda install -c conda-forge pymultinest
```

> [!WARNING]
> In case an error comes up during an NMMA analysis of the form:

```
ERROR: Could not load MultiNest library "libmultinest.so"
ERROR: You have to build it first
ERROR: and point the LD_LIBRARY_PATH environment variable to it!
```

Then, for using the PyMultinest library, it is required to get and compile the Multinest library separately. Instructions for the same are given here:

https://johannesbuchner.github.io/PyMultiNest/install.html

Use the commands below to install the dependencies given in requirements.txt file which are necessary for NMMA:

```
$ pip install -r requirements.txt
$ python setup.py install
```

> [!NOTE]
> There is an issue pip installing `pyfftw` on arm64 Mac systems; see the dedicated section below for a solution. If any package appeared to have an issue installing, you can first check by attempting to install it again using pip:

`$ pip install importlib_resources`

`$ pip install  extinction`

`$ pip install dill`

`$ pip install multiprocess`

`$ pip install lalsuite`

`$ pip install python-ligo-lw`

`$ pip install sncosmo`

`$ pip install scikit-learn`

`$ pip install joblib`

`$ conda install -c conda-forge p-tqdm`

> [!NOTE]
> If everything has gone smoothly, all of these above mentioned "pip install something" commands will show that the requirements have already been satisfied. Otherwise, these will cover the dependencies
> if not covered by python setup.py install. Also, if running python setup.py install shows something on the lines of "cannot cythonize without cython", do:

`$ conda install -c anaconda cython==0.29.24`

and redo `python setup.py install`.

**Known arm64 Mac issues**

1. For arm64 Macs (e.g. M1, M2), there is an issue installing `pyfftw` with pip (see https://github.com/pyFFTW/pyFFTW/issues/349#issuecomment-1468638458). To address, use `Homebrew` to run

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

Unfortunately, due to the web of package requirements that NMMA depends on, running setup.py does not typically finish without errors the first time through. Experience has shown that in the vast majority of cases, simply pinning versions such as:

```
pip install astropy==4.3.1
```

and then trying again is sufficient for completion of the installation. This instruction file will likely cover the issues you might face during your installation. However, please open issues on GitHub if there appear to be unresolvable conflicts.

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
