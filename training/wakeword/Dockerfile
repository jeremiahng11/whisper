# "Hey Oracle" + offline commands for the Idea Saver Round, trained with microWakeWord on computer-generated speech.
# CPU is fine (slow: a few hours for everything); an NVIDIA GPU makes it much faster (docker run --gpus all).
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git wget unzip ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /kit
# microWakeWord (training) and piper-sample-generator (speech) at the versions this kit was written for
RUN git clone https://github.com/kahrendt/microWakeWord /kit/microWakeWord \
    && git -C /kit/microWakeWord checkout 4665173cd35f1cff9a61e06fc427f124766c488e \
    && git clone https://github.com/rhasspy/piper-sample-generator /kit/piper-sample-generator \
    && git -C /kit/piper-sample-generator checkout 2971426a55072f7d22fec416ca7800df8bd23207
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e /kit/microWakeWord -e /kit/piper-sample-generator \
    && pip install --no-cache-dir soundfile librosa tensorboard "datasets>=3,<4" pyyaml requests
COPY phrases.yaml train_wakeword.py /kit/
ENV PYTHONUNBUFFERED=1
VOLUME ["/work", "/out"]
ENTRYPOINT ["python", "/kit/train_wakeword.py"]
