# Use official Ubuntu 22.04 image as base
FROM ubuntu:22.04

# Update the repository for package indexes
RUN apt-get update

# Install dependencies
RUN apt-get install -y \
    python3 \
    git \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /work

# Clone the Git repository
RUN git clone https://github.com/nuclear-multimessenger-astronomy/nmma/

# Set the working directory to the cloned repository
WORKDIR /work/nmma

RUN apt-get update
RUN apt-get install -y libopenmpi-dev openmpi-bin openmpi-doc
RUN apt install -y python3-mpi4py
#install dependencies
RUN pip3 install numpy
RUN pip3 install -r /work/nmma/doc_requirements.txt -r /work/nmma/grb_requirements.txt -r /work/nmma/production_requirements.txt -r /work/nmma/requirements.txt

# Clone and build Multinest
RUN git clone https://github.com/JohannesBuchner/MultiNest

WORKDIR /work/nmma/MultiNest

RUN apt-get install -y \
    cmake \
    liblapacke-dev \
    liblapack-dev \
    libblas-dev 

RUN cd build && cmake .. && make

# Set environment variable
ENV LD_LIBRARY_PATH=/work/nmma/MultiNest/lib:$LD_LIBRARY_PATH

# Clone and install PyMultiNest
RUN git clone https://github.com/JohannesBuchner/PyMultiNest/ /work/nmma/PyMultiNest
WORKDIR /work/nmma/PyMultiNest
RUN python3 setup.py install --user

# Add the executable path to the enviromental variable PATH
ENV PATH=$PATH:$HOME/.local/bin/

RUN pip3 install nmma
RUN pip3 install --upgrade Flask Jinja2

CMD ["bash"]
