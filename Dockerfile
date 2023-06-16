# FROM paddlepaddle/paddle:2.4.2-gpu-cuda11.7-cudnn8.4-trt8.4
FROM nvcr.io/nvidia/paddlepaddle:23.04-py3
MAINTAINER Mike Macpherson <mike@macphunk.net>

# Set environment varibles
ENV USERNAME mike # for metaflow
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND='noninteractive' apt-get install -y --no-install-recommends \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.in .
RUN pip install -U pip setuptools wheel \
    && pip install -U \
    fuzzywuzzy \
    metaflow \
    paddleocr \
    # paddlepaddle-gpu \
    pandas \
    pillow \
    python-levenshtein \
    scipy \
&& echo "Installed python packages."

# Induce PP to pre-download the required OCR models.
COPY init_pp_models.py .
RUN python init_pp_models.py

WORKDIR /app

