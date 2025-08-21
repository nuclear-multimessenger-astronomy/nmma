import mpi4py

from .main import analysis_runner, nmma_analysis

mpi4py.rc.threads = False
mpi4py.rc.recv_mprobe = False

del mpi4py
