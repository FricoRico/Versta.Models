FROM ubuntu:20.04

MAINTAINER Ricardo Snoek <info@neurora.nl>
LABEL description="Marian 1.10.0 CPU docker container training models using Versta infrastructure"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -yq && \
    apt-get install -yq \
        build-essential \
        git-core \
        pkg-config \
        libtool \
        zlib1g-dev \
        libbz2-dev \
        automake \
        python-dev \
        perl \
        libsparsehash-dev \
        libboost-all-dev \
        libprotoc17 \
        libprotobuf17 \
        protobuf-compiler \
        libprotobuf-dev \
        openssl \
        libssl-dev \
        libgoogle-perftools-dev \
        wget \
        apt-transport-https \
        ca-certificates \
        gnupg \
        software-properties-common \
        cmake \
        vim \
        nano \
        unzip \
        gzip \
        python3-pip \
        php && \
  rm -rf /var/lib/apt/lists/*

RUN wget -qO- 'https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB' | apt-key add - && sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list' && apt-get update && apt-get install -yq intel-mkl-64bit-2020.0-088

# Install tools
RUN pip3 install packaging
RUN pip3 install onnx
RUN pip3 install onnxruntime==1.11.1
RUN pip3 install langid

ENV MARIANPATH /marian-dev
ENV TOOLSDIR /tools

## Install onnxconverter-common
RUN git clone --branch v1.14.0 https://github.com/microsoft/onnxconverter-common.git $MARIANPATH/../onnxconverter-common

## Install Marian
RUN git clone --recursive --branch 1.12.0 https://github.com/marian-nmt/marian-dev $MARIANPATH

COPY ./marian-dev $MARIANPATH

RUN cd $MARIANPATH/src/3rd_party/onnx/protobuf && \
    protoc onnx-ml.proto --cpp_out .

WORKDIR $MARIANPATH

RUN mkdir -p build
WORKDIR $MARIANPATH/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_SENTENCEPIECE=ON -DUSE_MPI=ON -DCOMPILE_CPU=on -DCOMPILE_SERVER=on -DCOMPILE_CUDA=off -DUSE_ONNX=ON && make -j$(nproc)

WORKDIR $MARIANPATH/build