FROM ubuntu:16.04 

MAINTAINER Yannik Behr <y.behr@gns.cri.nz>

RUN apt-get update && apt-get install -y \
    python-pip \
    git \
    libgeos-dev \
    libproj-dev \
    proj-bin \
    python-scipy \
    python-numpy \
    python-dateutil \
    python-pyproj \
    python-matplotlib \
    python-tables \
    python-nose \
    python-pandas 

RUN pip install -I -U pip \
    && pip install cartopy

RUN git clone --depth=1 https://github.com/yannikbehr/spectroscopy.git 

RUN cd spectroscopy \
    && python setup.py install \
    && python setup.py test
 
