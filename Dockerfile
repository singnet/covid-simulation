FROM ubuntu:18.04

ARG USER_ID
ARG GROUP_ID

ARG git_owner="singnet"
ARG git_repo="covid-simulation"
ARG git_branch="master"

ENV SINGNET_DIR=/opt/${git_owner}
ENV PROJECT_DIR=/opt/${git_owner}/${git_repo}
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONPATH "${PYTONPATH}:${PROJECT_DIR}/covid19_sir"
ENV LD_LIBRARY_PATH /usr/local/lib

RUN mkdir -p ${PROJECT_DIR}

RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    vim \
    curl \
    sqlite3 \
    libsqlite3-0 \
    libsqlite3-dev \
    libtiff5 \
    libtiff-dev \
    libcurl4 \
    libcurl4-nss-dev \
    libspatialindex-c4v5 \
    libspatialindex-dev

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

RUN cd /tmp && \
    wget https://download.osgeo.org/proj/proj-7.2.0.tar.gz && \
    tar xzvf proj-7.2.0.tar.gz && \
    cd proj-7.2.0 && \
    ./configure && \
    make && \
    make install

ADD ./requirements.txt ${SINGNET_DIR}

RUN cd ${SINGNET_DIR} && \
    pip3 install -r requirements.txt

RUN cd /tmp && \
    pip3 install -e git+https://github.com/corvince/mesa-geo.git#egg=mesa-geo

RUN addgroup --gid $GROUP_ID user && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

USER user

WORKDIR ${PROJECT_DIR}
