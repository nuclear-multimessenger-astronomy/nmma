import mpi4py

from .main import analysis_runner, main_nmma, main_nmma_gw

mpi4py.rc.threads = False
mpi4py.rc.recv_mprobe = False

del mpi4py
