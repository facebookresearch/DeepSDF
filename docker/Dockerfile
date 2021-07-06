# Originated from https://github.com/phygevs/DeepSDF/blob/master/Dockerfile

FROM nvidia/cudagl:11.1-base-ubuntu18.04

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
        git \
        vim \
        g++ \
        gcc \
        cmake

WORKDIR /root


RUN apt-get update --fix-missing \
    && apt-get upgrade -y

# Pangolin
RUN git clone https://github.com/stevenlovegrove/Pangolin.git
RUN apt-get install -y \
    libgl1-mesa-dev \
    libglew-dev \
    ffmpeg \
    libavcodec-dev \
    libavutil-dev \
    libavformat-dev \
    libswscale-dev \
    libavdevice-dev \
    libdc1394-22-dev \
    libraw1394-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff5-dev \
    libopenexr-dev \
    libeigen3-dev \
    doxygen \
    libpython3-all-dev \
    libegl1-mesa-dev \
    libwayland-dev \
    libxkbcommon-dev \
    wayland-protocols

RUN apt-get install -y python3.7 python3-pip python3.7-dev
RUN python3.7 -mpip install \
    numpy \
    pyopengl \
    Pillow \
    pybind11

RUN cd Pangolin \
    && git submodule init \
    && git submodule update \
    && mkdir build \
    && cd build \
    && cmake .. \
    && cmake --build . \
    && make install

# nanoflann
RUN apt-get install -y \
    build-essential \
    libgtest-dev \
    cmake \
    libeigen3-dev

RUN git clone https://github.com/jlblancoc/nanoflann.git \
    && cd nanoflann \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && make test \
    && make install \
    && cd .. \
    && mkdir /usr/local/include/nanoflann -p \
    && mkdir /usr/include/nanoflann -p \
    && cp include/nanoflann.hpp /usr/local/include/nanoflann \
    && cp include/nanoflann.hpp /usr/include/nanoflann

# CLI11
RUN git clone https://github.com/CLIUtils/CLI11.git \
    && cd CLI11 \
    && git submodule update --init \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install \
    && GTEST_COLOR=1 CTEST_OUTPUT_ON_FAILURE=1 make test

# variant
RUN git clone https://github.com/mpark/variant.git \
    && mkdir variant/build \
    && cd variant/build \
    && cmake .. \
    && cmake --build . --target install

# anaconda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# main
WORKDIR /workspace/DeepSDF

RUN conda install -y -c pytorch -c nvidia \
    pytorch \
    torchvision \
#    cpuonly \
    cudatoolkit=11.1 \
    scikit-image \
    scipy

RUN conda install -c conda-forge trimesh -y
RUN pip install plyfile

#COPY . .


#RUN mkdir build \
#    && cd build \
#    && cmake .. \
#    && make -j


ENV PANGOLIN_WINDOW_URI headless://

#RUN mkdir data/ShapeNetCore.v2 -p
#RUN mkdir data/ShapeNetCore.v2-DeepSDF -p

#VOLUME /usr/src/sdf/data/ShapeNetCore.v2
#VOLUME /usr/src/sdf/data/ShapeNetCore.v2-DeepSDF


#CMD ["bash", "-c", "./process_whole_dataset.sh"]
