ARG BASE_IMAGE=nvcr.io/nvidia/tensorflow:20.10-tf1-py3
FROM $BASE_IMAGE

RUN apt-get update 
RUN apt-get -y install libgl1-mesa-glx
# RUN /usr/bin/python -m pip install --upgrade pip
RUN pip install scipy==1.3.3
RUN pip install requests==2.22.0
RUN pip install Pillow==6.2.1
RUN pip install h5py==2.9.0
RUN pip install imageio==2.9.0
RUN pip install imageio-ffmpeg==0.4.2
RUN pip install tqdm==4.49.0
RUN pip install click ninja ftfy regex gdown opencv-python
RUN pip install imageio-ffmpeg==0.4.3 pyspng==0.1.0
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install git+https://github.com/openai/CLIP.git

WORKDIR /workspace