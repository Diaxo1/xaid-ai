FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REAL_ESRGAN_DIR=/opt/Real-ESRGAN

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/xinntao/Real-ESRGAN.git /opt/Real-ESRGAN
WORKDIR /opt/Real-ESRGAN

RUN pip install --upgrade pip setuptools wheel && \
    pip install basicsr facexlib gfpgan && \
    pip install -r requirements.txt && \
    pip install -e . && \
    pip install runpod boto3 requests opencv-python-headless ffmpeg-python

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]
