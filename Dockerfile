# Use NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic utilities
    wget \
    curl \
    git \
    vim \
    htop \
    tmux \
    zip \
    unzip \
    ca-certificates \
    build-essential \
    cmake \
    pkg-config \
    # Python build dependencies
    python3-dev \
    # OpenGL and rendering dependencies (for PyRender)
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    libxi6 \
    libxinerama1 \
    libxcursor1 \
    libgconf-2-4 \
    xvfb \
    # OpenCV dependencies
    libopencv-dev \
    libboost-all-dev \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# install conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    conda update -n base -c defaults conda && \
    conda clean -afy

WORKDIR /workspace

# copy the env file 
COPY environment.yml /tmp/environment.yml

# create the conda env
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy && \
    rm /tmp/environment.yml

# initialize conda shell
RUN conda init bash

# activate the conda environment
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate ssl-probing\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Set up Jupyter config (optional, for notebook access)
RUN /opt/conda/envs/ssl-probing/bin/jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py

# Create necessary directories
RUN mkdir -p /workspace/{data,checkpoints,results,logs}

# Set up virtual display for rendering (PyRender)
ENV DISPLAY=:99

# copy rest of the code
COPY . /workspace/

# expose port for jupyter and tensorboard
EXPOSE 8888 6006

# set entrypoint!
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
