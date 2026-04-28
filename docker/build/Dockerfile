FROM  pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

RUN apt-get update \
    && apt-get install -y git vim curl wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    timm==0.5.4 \
    jupyter==1.1.1 \
    matplotlib==3.10.8 \
    opencv-python-headless==4.13.0.92 \
    fvcore==0.1.5.post20221221 \
    tensorboardX==2.6.4

WORKDIR /workspace/
