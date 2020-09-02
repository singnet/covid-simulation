FROM ubuntu:18.04

ARG git_owner="singnet"
ARG git_repo="covid-simulation"
ARG git_branch="master"

ENV PROJECT_DIR=/opt/${git_owner}/${git_repo}
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN mkdir -p ${PROJECT_DIR}

RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    vim \
    curl

RUN cd /tmp && \
    apt update && \
    apt-get install build-essential && \
    wget https://www.imagemagick.org/download/ImageMagick.tar.gz && \
    tar xvzf ImageMagick.tar.gz && \
    cd ImageMagick-* && \
    ./configure && \
    make && \
    make install && \
    ldconfig /usr/local/lib

WORKDIR ${PROJECT_DIR}
