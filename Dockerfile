FROM python:3.11-slim

# INSTALL_DIARIZE=1 -> speaker labels (pyannote + CPU torch, image grows by ~1.5 GB)
# GPU=1             -> CUDA libraries for faster-whisper on an NVIDIA card (e.g. the RTX 3090)
ARG INSTALL_DIARIZE=0
ARG GPU=0

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data \
    HF_HOME=/data/models/hf \
    LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

WORKDIR /srv
COPY requirements.txt requirements-diarize.txt ./
RUN pip install -r requirements.txt \
 && if [ "$GPU" = "1" ]; then pip install "nvidia-cublas-cu12" "nvidia-cudnn-cu12==9.*"; fi \
 && if [ "$INSTALL_DIARIZE" = "1" ]; then \
      if [ "$GPU" = "1" ]; then pip install torch torchaudio; \
      else pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio; fi \
      && pip install -r requirements-diarize.txt; \
    fi

COPY app ./app

RUN useradd -m -u 1000 app && mkdir -p /data && chown app:app /data
USER app
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
