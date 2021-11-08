FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
CMD nvidia-smi

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    wget \
    unzip \
    software-properties-common \
 && rm -rf /var/lib/apt/lists/*


WORKDIR /workspace
#RUN wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip && unzip Banana_Linux.zip
#RUN wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip && unzip #Banana_Linux_NoVis.zip
#COPY ./TrainedAgents ./TrainedAgents
#COPY ./*.ipynb ./
#COPY ./*.py ./
#COPY ./*.pth ./
