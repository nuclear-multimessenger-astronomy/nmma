# Usa l'immagine ufficiale di Ubuntu 22.04 come base
FROM ubuntu:22.04
# Usa l'immagine di Ubuntu 20.04 con Xfce
#FROM dorowu/ubuntu-desktop-lxde-vnc:focal

#RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 4EB27DB2A3B88B8B
# Aggiorna il repository degli indici dei pacchetti
RUN apt-get update

# Installa le dipendenze di base
RUN apt-get install -y \
    python3 \
    git \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Imposta la directory di lavoro
WORKDIR /work

# Clone a Git repository using the provided credentials
RUN git clone https://github.com/FabioRagosta/nmma/

# Set the working directory to the cloned repository
WORKDIR /work/nmma

RUN apt-get update
RUN apt-get install -y libopenmpi-dev openmpi-bin openmpi-doc
RUN apt install -y python3-mpi4py
# Installa le dipendenze del progetto
RUN pip3 install numpy
RUN pip3 install -r /work/nmma/doc_requirements.txt -r /work/nmma/grb_requirements.txt -r /work/nmma/production_requirements.txt -r /work/nmma/requirements.txt

# Clone e build di MultiNest
RUN git clone https://github.com/JohannesBuchner/MultiNest

WORKDIR /work/nmma/MultiNest

RUN apt-get install -y \
    cmake \
    liblapacke-dev \
    liblapack-dev \
    libblas-dev 

RUN cd build && cmake .. && make

# Imposta una variabile d'ambiente
ENV LD_LIBRARY_PATH=/work/nmma/MultiNest/lib:$LD_LIBRARY_PATH

# Clone di PyMultiNest e installazione
RUN git clone https://github.com/JohannesBuchner/PyMultiNest/ /work/nmma/PyMultiNest
WORKDIR /work/nmma/PyMultiNest
RUN python3 setup.py install --user

# Aggiungi il percorso dell'eseguibile alla variabile PATH
ENV PATH=$PATH:$HOME/.local/bin/

RUN pip3 install nmma
RUN pip3 install --upgrade Flask Jinja2

CMD ["bash"]
