FROM python:3.11-slim

# INSTALL_DIARIZE=1 -> speaker labels (pyannote + CPU torch, image grows by ~1.5 GB)
# GPU=1             -> CUDA libraries for faster-whisper on an NVIDIA card (e.g. the RTX 3090)
ARG INSTALL_DIARIZE=0
ARG GPU=0
# INSTALL_TTS=1     -> Piper text-to-speech for spoken answers (~60 MB)
ARG INSTALL_TTS=1
# INSTALL_BROWSER=1 -> headless Chromium, so Watch AI sources can read sites that build their pages with JavaScript
#                      or only answer browsers (~350 MB)
ARG INSTALL_BROWSER=1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data \
    HF_HOME=/data/models/hf \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

WORKDIR /srv
COPY requirements.txt requirements-diarize.txt requirements-tts.txt ./
RUN pip install -r requirements.txt \
 && if [ "$GPU" = "1" ]; then pip install "nvidia-cublas-cu12" "nvidia-cudnn-cu12==9.*"; fi \
 && if [ "$INSTALL_DIARIZE" = "1" ]; then \
      if [ "$GPU" = "1" ]; then pip install torch==2.8.0 torchaudio==2.8.0; \
      else pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchaudio==2.8.0; fi \
      && pip install -r requirements-diarize.txt; \
    fi \
 && if [ "$INSTALL_TTS" = "1" ]; then pip install -r requirements-tts.txt || echo "TTS install failed - continuing without speech"; fi \
 && if [ "$INSTALL_BROWSER" = "1" ]; then (pip install "playwright>=1.45,<2" && playwright install --with-deps chromium \
      && chmod -R a+rx /ms-playwright) || echo "browser install failed - continuing without (JavaScript sites can't be read)"; fi

COPY app ./app

RUN useradd -m -u 1000 app && mkdir -p /data && chown app:app /data
USER app
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
