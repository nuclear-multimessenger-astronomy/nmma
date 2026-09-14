# --- Build Stage ---
# Use official Ubuntu 22.04 image as base
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Update the repository for package indexes and install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git cmake make g++ gcc gfortran \
    liblapacke-dev liblapack-dev libblas-dev \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /build

# Clone and build Multinest
RUN git clone https://github.com/JohannesBuchner/MultiNest \
    && cd MultiNest/build \
    && cmake .. \
    && make

# Clone the Git repository (with build arguments for forks/branches)
ARG REPO_URL="https://github.com/nuclear-multimessenger-astronomy/nmma.git"
ARG BRANCH="main"
RUN git clone --branch ${BRANCH} ${REPO_URL} /build/nmma

# Set the working directory to the cloned repository
WORKDIR /build/nmma

# Clone and install PyMultiNest, and install dependencies
RUN git clone https://github.com/JohannesBuchner/PyMultiNest/ /build/PyMultiNest \
    && pip3 wheel --no-cache-dir --wheel-dir /build/wheels /build/PyMultiNest .[production,grb] Flask Jinja2

# --- Runtime Stage ---
# Use official Ubuntu 22.04 image as base
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies (runtime only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    liblapack3 libblas3 openmpi-bin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Transfer the compiled MultiNest shared libraries from the builder
COPY --from=builder /build/MultiNest/lib /usr/local/lib/MultiNest

# Set environment variable
ENV LD_LIBRARY_PATH=/usr/local/lib/MultiNest:$LD_LIBRARY_PATH

# Transfer and install the pre-built Python wheels
COPY --from=builder /build/wheels /wheels
RUN pip3 install --no-cache-dir /wheels/* --no-index

# Add the executable path to the enviromental variable PATH
ENV PATH=$PATH:$HOME/.local/bin/

WORKDIR /work
CMD ["bash"]