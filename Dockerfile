#Copyright(c) Microsoft Corporation.All rights reserved.
#Licensed under the MIT license.

FROM ubuntu:jammy

RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:git-core/ppa
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y git make cmake g++ libaio-dev libgoogle-perftools-dev libunwind-dev clang-format libboost-dev libboost-program-options-dev libmkl-full-dev libcpprest-dev python3.10

WORKDIR /app
RUN git clone https://github.com/FisherKKK/DiskANN.git
WORKDIR /app/DiskANN
RUN mkdir build
RUN cmake -S . -B build  -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build -- -j
