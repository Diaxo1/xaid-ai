FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REAL_ESRGAN_DIR=/opt/Real-ESRGAN

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/xinntao/Real-ESRGAN.git /opt/Real-ESRGAN

WORKDIR /opt/Real-ESRGAN

# Keep the CUDA-enabled PyTorch from the base image. Install Real-ESRGAN's
# Python dependencies without reinstalling torch/torchvision from PyPI.
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir basicsr facexlib gfpgan runpod boto3 requests opencv-python-headless ffmpeg-python && \
    pip install --no-cache-dir -e .

WORKDIR /app
COPY handler.py /app/handler.py
COPY runpod_handler.py /app/runpod_handler.py

CMD ["python", "-u", "/app/runpod_handler.py"]
