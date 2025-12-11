The Nuclear Multimessenger Astronomy (NMMA) framework
=====================================================

nmma is a fully featured, Bayesian multi-messenger pipeline targeting
joint analyses of gravitational-wave and electromagnetic data (focusing
on the optical). Using bilby as the back-end, the software is capable of
sampling these data sets using a variety of samplers. It uses chiral
effective field theory based neutron star equation of states when
performing inference, and is also capable of estimating the Hubble Constant.


Instructions For NMMA Installation
----------------------------------

For science users
^^^^^^^^^^^^^^^^^

If you are a science user and want to use NMMA for your analysis, you
can install NMMA using conda as follows:

.. code::

   conda create --name nmma_env python=3.10
   conda install -c conda-forge nmma

If you have an issue, such as ``Solving environment: failed with initial frozen solve``, an option could be:

.. code::

   conda create --name nmma_env python=3.10
   conda install mamba -c conda-forge
   mamba install nmma -c conda-forge


If you are a developer or you want to build NMMA from source, please
refer to the developer section below.

.. note::

   The above may not work for arm64 Macs; see specifc instructions `below <#arm64mac>`_.

For developers and contributors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The steps highlighted below are primarily for Linux systems and Windows
users are advised to use WSL (preferrably Ubuntu 20.04) for smooth
installation. Ubuntu 20.04 is available for free download on Microsoft
Store.

**Installing Anaconda3**

On your Linux/WSL terminal, run the following commands to install
anaconda (replace 5.3.1 by the latest version):

.. code::

   wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
   bash Anaconda3-5.3.1-Linux-x86_64.sh

(For 32-bit installation, skip the ``\_64`` in both commands)

.. note::

   If you already have Anaconda3 installed, please make sure that it is updated to the latest version using `conda update --all`. Also check that you do not have multiple versions of Python installed in `usr/lib/` directory as it can cause version conflicts while installing dependencies.

Now do:

.. code::

   conda update --all

**Cloning the NMMA repository**

Fork the NMMA repository: `NMMA Github Repo <https://github.com/nuclear-multimessenger-astronomy/nmma>`__

Note that we link above to the main branch, but suggest making changes
on your own fork (please also see our `contributing
guide <./contributing.html>`__). Now, after forking, run the following
command to clone the repository into your currently directory (by
default, in your home directory):

.. code::

   git clone https://github.com/your_github_username/nmma

Change directory to the nmma folder:

.. code::

   cd nmma

Create a new environment using this command (environment name is
nmma_env in this case):

.. code::

   conda create --name nmma_env python=3.10
   conda activate nmma_env

.. note::

   If this gives an error like: ``CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'``, then run:

.. code::

   source ~/anaconda3/etc/profile.d/conda.sh

then proceed with ``conda activate nmma_env``.

Get the latest pip version

.. code::

   pip install --upgrade pip

Check python and pip version like this:

.. code::

   python --version
   pip --version

Python 3.10 and above and Pip 21.2 and above is ideal for this
installation. It is recommended to update these for your installation.


Install mpi4py:

.. code::

   conda install mpi4py

.. warning::

   We discourage installing mpi4py with pip. The installation does not work properly due to issues with MPI header files, etc.

Install parallel-bilby:

.. code::

   pip install parallel-bilby

.. note::

   Installing parallel-bilby takes quite some time. Please be patient. If you encounter any errors, please check the `parallel-bilby installation guide <https://lscsoft.docs.ligo.org/parallel_bilby/installation>`__ for more details.


.. note::

   For those installing on WSL with pip, you may encounter an issue with installing parallel-bilby due to a dependency on python-ligo-lw.
   This can be resolved by installing gcc with the following command:

.. code::

   sudo apt-get install gcc

and attempting to install parallel-bilby again.

Install pymultinest (note this line may not work for arm64 Macs; see
specifc instructions below)

.. code::

   conda install conda-forge::pymultinest

.. warning::

   In case an error comes up during an NMMA analysis of the form:

   .. code::

      ERROR: Could not load MultiNest library "libmultinest.so"
      ERROR: You have to build it first
      ERROR: and point the LD_LIBRARY_PATH environment variable to it!

Then, for using the PyMultinest library, it is required to get and
compile the Multinest library separately. Instructions for the same are
given here:

`<https://johannesbuchner.github.io/PyMultiNest/install.html>`__

Use the commands below to install the dependencies given in
requirements.txt file which are necessary for NMMA:

.. code::

   pip install -e .

.. note::

   There is an issue pip installing ``pyfftw`` on arm64 Mac systems; see the dedicated section below for a solution. If any package appeared to have an issue installing, you can first check by attempting to install it again using pip:

.. code::
   pip install keras, tensorflow
   pip install importlib_resources
   pip install  extinction
   pip install dill
   pip install multiprocess
   pip install lalsuite
   pip install python-ligo-lw
   pip install sncosmo
   pip install scikit-learn
   pip install joblib
   conda install -c conda-forge p-tqdm

.. note::

   If everything has gone smoothly, all of these above mentioned "pip install something" commands will show that the requirements have already been satisfied. Otherwise, these will cover the dependencies if not covered by ``pip install .``. Also, if running ``pip install .`` shows something on the lines of "cannot cythonize without cython", do:

.. code::

   conda install -c anaconda cython==0.29.24
   pip install

.. _arm64mac:

For arm64 Macs
^^^^^^^^^^^^^^
Follow the instructions above, but with the following modifications:

Install C compiler and cmake:

.. code::

   conda install -c conda-forge c-compiler
   brew install cmake


**Known arm64 Mac issues**


#. For arm64 Macs (e.g. M1, M2), there is an issue installing ``pyfftw``
   with pip (see
   https://github.com/pyFFTW/pyFFTW/issues/349#issuecomment-1468638458).
   To address, first try running

   .. code::

      conda install -c conda-forge pyfftw

   If this completes successfully, re-run any failed installations and continue. Otherwise, use ``Homebrew`` to run

   .. code::

      brew install fftw

   then add the following lines to your ``.zprofile`` or ``.bash_profile``:

   .. code::

      export PATH="/opt/homebrew/bin:$PATH"
      export DYLD_LIBRARY_PATH=/opt/homebrew/opt/fftw/lib
      export LDFLAGS="-Wl,-S,-rpath,/opt/homebrew/opt/fftw/lib -L/opt/homebrew/opt/fftw/lib"
      export CFLAGS="-Wno-implicit-function-declaration -I/opt/homebrew/opt/fftw/include"

   Close and reopen your terminal and run

   .. code::

      pip install pyfftw

   You may then need to rerun ``pip install -r requirements.txt`` to
   complete the dependency installations.

   .. tip::

      FFTW installation can also be attempted using conda as follows

      .. code::

         conda install -c conda-forge fftw
         DYLD_LIBRARY_PATH=$HOME/anaconda3/envs/nmma_env
         pip install pyfftw
      Replace ``DYLD_LIBRARY`` path by your NMMA virtual environment path if it is not same as give here

#. The ``osx-arm64`` conda-forge channel does not include
   ``pymultinest``. Running ``pip install -r requirements.txt`` should
   have installed ``pymultinest``, but you will still need to install
   and compile ``Multinest`` from the source. Within the ``nmma``
   directory, run:

   .. code::

      git clone https://github.com/JohannesBuchner/MultiNest
      cd MultiNest/build
      cmake ..
      make

   Next, add the following lines to your ``.zprofile`` or
   ``.bash_profile``:

   .. code::

      export LD_LIBRARY_PATH=$HOME/nmma/MultiNest/lib:$LD_LIBRARY_PATH
      export DYLD_LIBRARY_PATH=$HOME/nmma/MultiNest/lib:$DYLD_LIBRARY_PATH

   .. note::

      Modify these paths as appropriate for the location of your ``MultiNest`` installation. You can also combine the ``DYLD_LIBRARY_PATH`` lines for ``MultiNest`` and ``fftw`` (above) into a single line

#. There are also issues with ``tensorflow`` and arm64 Macs. If using
   ``tensorflow``, install it with the following commands:

   .. code::

      pip install tensorflow-macos
      pip install tensorflow-metal



**First Test for NMMA**

Run the following commands:

.. code::

   ipython
   import nmma
   import nmma.em.analysis
   import nmma.eos.create_injection

.. tip::

   (Okay, last one!): if everything is ok, it's the end of the installation. But in case it shows that such-and-such modules are absent, feel free to install those modules by visiting their anaconda documentation and install
   those with their given commands. In case modules like afterglowpy and dust_extinction are needed, don't hesitate to do it with pip (normally it shouldn't happen), but some modules may not install correctly in case of disturbance.

Please pay special attention to the ``import nmma.em.analysis`` and make
sure that it does not generate any errors.

Unfortunately, due to the web of package requirements that NMMA depends
on, running setup.py does not typically finish without errors the first
time through. Experience has shown that in the vast majority of cases,
simply pinning versions such as:

.. code::

   pip install astropy==4.3.1

and then trying again is sufficient for completion of the installation.
This instruction file will likely cover the issues you might face during
your installation. However, please open issues on GitHub if there appear
to be unresolvable conflicts.

Building custom lalsuite from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to install a custom lalsuite version (e.g. with a certain GW template model), then lalsuite needs to be build from scratch. This requires setting up the conda environment through conda-forge

.. code::

   conda create -c conda-forge --prefix=YOUR_PREFIX python=3.10
   conda activate YOUR_PREFIX

and then installing mpi4py first before installing the required packages for the build process (here the second line):

.. code::

   conda install mpi4py
   conda install -c conda-forge "astropy>=1.1.1" autoconf automake bc cfitsio doxygen fftw freezegun c-compiler gsl h5py hdf5 healpy "help2man>=1.37" "ldas-tools-framecpp<3.0.0a0" "libframel<9.0.0a0" ligo-gracedb ligo-segments "lscsoft-glue>=2.0.0" make matplotlib-base metaio "mpmath>=1.0.0" numpy pillow "pkg-config>=0.18.0" pytest "scipy>=0.9.0" six "swig>=3.0.10" zlib

When this is done, you can proceed installing nmma as above (i.e. via requirements.txt etc.) but note that from the dependencies lalsuite is then installed via pip. This installation needs to be removed

.. code::

   pip uninstall lalsuite

Then the build process for the custom lalsuite can be started

.. code::

   cd YOUR_CUSTOM_LALSUITE
   ./00boot
   ./configure --prefix=YOUR_PREFIX --disable-all-lal --enable-swig-python  --enable-lalsimulation --enable-lalframe
   make; make install

It might happen that in the course of this some packages will be downgraded. You can just update them afterwards to a sufficient version, e.g.

.. code::

   pip install astropy>=5.2.2



Installation on expanse and other cluster resources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When installation on cluster resources, it is common that all modules
required for installing NMMA out of the box are not available. However,
most will make it possible to import the required modules (most
commonly, these are software like gfortran or mpi).

For example, on XSEDE’s Expanse cluster, one can start a terminal
session with:

.. code::

   module load sdsc
   module load openmpi

and follow the instructions above.

.. note::

   If ``module load openmpi`` does not execute directly and it asks for dependencies, one can proceed with:

.. code::

   module load sdsc
   module load cpu/0.15.4
   module load gcc/9.2.0
   module load openmpi/4.1.1

Internetless Clusters
^^^^^^^^^^^^^^^^^^^^^

Some cluster resources may not have access to the internet or could block some forms of data transfer. This will cause some difficulties with certain packages and will require some fixes to how NMMA interacts with them and changes to how you call some commands in NMMA. This can also affect how you install NMMA on the cluster. Because the cluster is in some way restricted, it may be that the `conda install nmma -c conda-forge` or the `pip install nmma` commands don't work. Thus to install NMMA, you will need to install from the source. To begin, you will want to copy the NMMA repository from your local to the cluster using an `scp` or `rsync` commmand because likely the git commands will fail. Once NMMA is downloaded to the cluster, navigate to the main directory and run

.. code::

   pip install -r requirements.txt
   pip install .

and check that the installation was successful by running the first check for NMMA given above. After installation, the next most likely difference is how NMMA uses the trained kilonova models during any form of EM analysis. When running any analysis that requires the use of the trained grid of models you will need to add the `--local-only` flag to the analysis command. For example,

.. code::

   lightcurve-analysis \
        --model LANLTP2 \
        --svd-path svdmodels/ \
        --filters ztfg,ztfi,ztfr \
        --ztf-sampling \
        --ztf-uncertainties \
        --ztf-ToO 180 \
        --local-only \
        --interpolation-type tensorflow \
        --outdir outdir/TP2_ztf \
        --label TP2_ztf \
        --prior priors/LANL2022.prior \
        --tmin 0. \
        --tmax 14 \
        --dt 0.1 \
        --error-budget 1 \
        --nlive 1024 \


Matplotlib fonts
^^^^^^^^^^^^^^^^

On new Linux installations, we sometimes come across the warning:
``findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu Sans``.
If you do prefer to use ‘Times New Roman’ for all of your plotting
needs, you can install ``msttcorefonts`` with:

.. code::

   sudo apt install msttcorefonts -qq

After removing the matplotlib cache:

.. code::

   rm ~/.cache/matplotlib -rf

Beautiful fonts should be yours.



Contributing
------------

nmma is released under the MIT license.  We encourage you to
modify it, reuse it, and contribute changes back for the benefit of
others.  We follow standard open source development practices: changes
are submitted as pull requests and, once they pass the test suite,
reviewed by the team before inclusion.  Please also see
`our contributing guide <./contributing.html>`_.

User Guide
----------

.. toctree::
   :maxdepth: 1

   quick-start-guide
   models
   training
   data_inj_obs
   systematics
   fitting
   lfi_analysis
   gw_inference
   joint_inference
   GW-EM-resampling
   combined_analysis
   Cluster_Resources
   contributing
   changelog


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
