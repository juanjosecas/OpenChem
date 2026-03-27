FROM nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev && \
     rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o ~/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y numpy pyyaml scipy ipython mkl mkl-include && \
     /opt/conda/bin/conda install -y -c rdkit rdkit nox cairo && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

WORKDIR /opt/openchem
COPY . .

RUN pip install --upgrade pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN pip install -e .

WORKDIR /workspace
RUN chmod -R a+w /workspace
