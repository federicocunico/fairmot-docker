FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04 as base

ARG DEBIAN_FRONTEND=noninteractive

# System requirements
RUN apt update && apt install -y git wget nano 

# Prepare env
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ENV ENV_NAME=fairmot

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && chmod +x Miniconda3-latest-Linux-x86_64.sh \
    && ./Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 


RUN conda create --name ${ENV_NAME} python=3.7
# Bypass the activate, does not support the variables...
SHELL ["conda", "run", "--no-capture-output", "-n", "fairmot", "/bin/bash", "-c"]

RUN conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.0 -c pytorch


# FairMOT

# You should pre-download this  http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth
# and put it here:
#  /root/.cache/torch/hub/checkpoints/dla34-ba72cf86.pth
WORKDIR /root/.cache/torch/hub/checkpoints/
RUN wget http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth

WORKDIR /fairmot


RUN git clone https://github.com/microsoft/FairMOT . \
    && git checkout 8e6f5c6a82f15bf951f35bbd8592e558938dfd2f

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install cython
RUN pip install -r requirements.txt
RUN pip install opencv-python

RUN git clone https://github.com/ifzhang/DCNv2.git
RUN cd DCNv2 \ 
    && git checkout ab4d98efc0aafeb27bb05b803820385401d9921b \ 
    && ./make.sh

RUN mkdir models

COPY ./FairMOT/models/all_dla34.pth /fairmot/models/all_dla34.pth
COPY ./FairMOT/videos/delete.m4v /fairmot/videos/delete.m4v

COPY ./FairMOT/detect.py /fairmot/detect.py
COPY ./FairMOT/demo_2d.py /fairmot/demo_2d.py
COPY ./start.sh /fairmot/start.sh
RUN chmod +x start.sh



# Last cmd
# SHELL ["conda", "init"]


# CMD ["python3", "detect.py"]
# CMD ["/bin/bash", "start.sh"]
