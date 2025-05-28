# ✅ Use CUDA 12.3 base image
# FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

LABEL org.opencontainers.image.source https://github.com/serengil/deepface

# Set environment variables for CUDA

# Create app directory
RUN mkdir -p /app && chown -R 1001:0 /app
RUN mkdir /app/deepface
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip ffmpeg libsm6 libxext6 libhdf5-dev \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && apt-get install cmake -y \
    && rm -rf /var/lib/apt/lists/* 

RUN python -m pip install --upgrade pip


# 🔁 Copy local cuDNN repo .deb file
COPY cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb .

# ✅ Install cuDNN
RUN dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb && \
    cp /var/cudnn-local-repo-*/cudnn-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y libcudnn8 libcudnn8-dev && \
    rm -f cudnn-local-repo-*.deb


# ✅ Install TensorFlow GPU version
RUN pip install --upgrade pip && \
    pip install tensorflow==2.16.1 tf-keras

# 🔁 Continue with DeepFace setup
COPY ./deepface /app/deepface
COPY ./requirements.txt /app/requirements.txt
COPY ./requirements_local /app/requirements_local.txt
COPY ./package_info.json /app/
COPY ./setup.py /app/
COPY ./README.md /app/
COPY ./entrypoint.sh /app/deepface/api/src/entrypoint.sh


#RUN python -m pip install --upgrade pip

# ✅ Confirm GPU access
RUN python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

# Install remaining Python dependencies
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -r /app/requirements_local.txt

# Install DeepFace
RUN pip install -e .

WORKDIR /app/deepface/api/src
EXPOSE 5000

ENTRYPOINT [ "sh", "entrypoint.sh" ]

##






