# Shadow Harmonization for Realisitc Compositing (c)
# by Lucas Valença, Jinsong Zhang, Michaël Gharbi,
# Yannick Hold-Geoffroy and Jean-François Lalonde.
#
# Developed at Université Laval in collaboration with Adobe, for more
# details please see <https://lvsn.github.io/shadowcompositing/>.
#
# Work published at ACM SIGGRAPH Asia 2023. Full open access at the ACM
# Digital Library, see <https://dl.acm.org/doi/10.1145/3610548.3618227>.
#
# This code is licensed under a Creative Commons
# Attribution-NonCommercial 4.0 International License.
#
# You should have received a copy of the license along with this
# work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.

FROM nvidia/cuda:11.5.2-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

# Ubuntu packages
RUN apt-get update && apt-get install -y \
    sudo \
    locales \
    software-properties-common \
    build-essential \
    cmake \
    gdb \
    gfortran \
    wget \
    curl \
    ssh \
    rsync \
    keychain \
    git \
    git-lfs \
    zip \
    unzip \
    vim \
    imagemagick \
    ffmpeg \
    openexr \
    libopenexr-dev \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    cython \
    python-is-python3

RUN apt-get -y autoremove && apt-get -y clean

# Locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Python packages
RUN pip3 install --upgrade pip && pip3 install \
    numpy==1.22 \
    numba==0.56.0 \
    cupy-cuda116==10.6.0 \
    tensorflow-gpu==2.9.1 \
    scipy==1.9.0 \
    scikit-learn==1.1.2 \
    scikit-image==0.19.3 \
    pandas==1.4.3 \
    matplotlib==3.5.3 \
    imageio==2.21.1 \
    imageio-ffmpeg==0.4.7 \
    opencv-python==4.6.0.66 \
    OpenEXR==1.3.8 \
    skylibs==0.7.2 \
    omegaconf==2.2.3 \
    tqdm==4.64.1

RUN pip3 install \
    torch==1.12.1 \
    torchvision==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    lpips==0.1.4

# User setup
RUN useradd -ms /bin/bash user
RUN echo user:pass | chpasswd && usermod -aG sudo user
RUN echo 'root:root' | chpasswd
