#!/usr/bin/env python3
"""Train "Hey Oracle" and the offline commands for the watch, then (optionally) install them on Oracle Minutes.

    python train_wakeword.py                       # everything in phrases.yaml
    python train_wakeword.py --only wake           # just the wake word
    python train_wakeword.py --quick               # a small test run (minutes, not hours; the model will be poor)

Steps, per model (data that every model shares is downloaded once into /work/data and kept):
  1. speech:   thousands of people saying the phrase (and the near-misses in `not:`), from the Piper LibriTTS
               generator (900 voices, mixed speeds and styles)
  2. sound:    each clip mixed with room echo (MIT impulse responses), background noise and music (AudioSet, FMA),
               turned into the same 40-band features the watch computes
  3. training: microWakeWord's MixedNet, against hours of ordinary speech, dinner parties and silence
               (pre-computed negative features) plus your near-misses
  4. cut-off:  picked from the test results so false wakes stay under `max_faph` per hour
Output in /out: <id>.tflite, <id>_roc.txt, wake.json and oracle_wakeword.zip (what Oracle Minutes takes).
With ORACLE_URL and ORACLE_KEY set, the zip is uploaded and the watch picks it up at its next check.

Note: the background / negative datasets have mixed licences, several non-commercial. Models trained with them are
for personal use; train with your own noise data before putting them in a product.
"""
import argparse
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml

KIT = Path(__file__).resolve().parent
WORK = Path(os.environ.get("WORK_DIR", "/work"))
OUT = Path(os.environ.get("OUT_DIR", "/out"))
DATA = WORK / "data"
GENERATOR_URL = "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt"
NEG_URL = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
NEG_SETS = ["dinner_party.zip", "dinner_party_eval.zip", "no_speech.zip", "speech.zip"]
AUDIOSET_URL = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train09.tar"
FMA_URL = "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip"


def log(*a):
    print("[oracle-wakeword]", *a, flush=True)


def run(cmd, **kw):
    log("$", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def fetch(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["wget", "-q", "--show-progress", "-O", dest.with_suffix(dest.suffix + ".part"), url])
    dest.with_suffix(dest.suffix + ".part").rename(dest)


def to_wav16k(src: Path, dst: Path):
    """Any audio file -> 16 kHz mono 16-bit WAV (ffmpeg)."""
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
                    "-sample_fmt", "s16", str(dst)], check=False)


# ---------------------------------------------------------------- shared data
def prepare_data(quick: bool):
    DATA.mkdir(parents=True, exist_ok=True)
    fetch(GENERATOR_URL, DATA / "en_US-libritts_r-medium.pt")
    # room echo
    rirs = DATA / "mit_rirs"
    if not (rirs / ".done").exists():
        import soundfile as sf
        from datasets import Audio, load_dataset
        rirs.mkdir(exist_ok=True)
        ds = load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
        for i, row in enumerate(ds):
            a = row["audio"]
            raw = a.get("bytes") or open(a["path"], "rb").read()
            tmp = rirs / f"_{i}.bin"
            tmp.write_bytes(raw)
            data, sr = sf.read(str(tmp))
            tmp.unlink()
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != 16000:
                idx = np.round(np.arange(0, len(data), sr / 16000)).astype(int)
                data = data[idx[idx < len(data)]]
            sf.write(str(rirs / f"rir_{i:04d}.wav"), data, 16000, subtype="PCM_16")
        (rirs / ".done").write_text("ok")
    # background noise (AudioSet part) and music (FMA extra small)
    aset = DATA / "audioset_16k"
    if not (aset / ".done").exists():
        tar = DATA / "bal_train09.tar"
        fetch(AUDIOSET_URL, tar)
        raw = DATA / "audioset_raw"
        raw.mkdir(exist_ok=True)
        run(["tar", "-xf", tar, "-C", raw])
        aset.mkdir(exist_ok=True)
        files = sorted(raw.glob("**/*.flac"))
        for f in files[: 300 if quick else None]:
            to_wav16k(f, aset / (f.stem + ".wav"))
        shutil.rmtree(raw, ignore_errors=True)
        (aset / ".done").write_text("ok")
    fma = DATA / "fma_16k"
    if not (fma / ".done").exists():
        z = DATA / "fma_xs.zip"
        fetch(FMA_URL, z)
        raw = DATA / "fma_raw"
        raw.mkdir(exist_ok=True)
        run(["unzip", "-q", "-o", z, "-d", raw])
        fma.mkdir(exist_ok=True)
        files = sorted(raw.glob("**/*.mp3"))
        for f in files[: 150 if quick else None]:
            to_wav16k(f, fma / (f.stem + ".wav"))
        shutil.rmtree(raw, ignore_errors=True)
        (fma / ".done").write_text("ok")
    # ordinary speech / dinner party / silence, already as features
    neg = DATA / "negative_datasets"
    for name in NEG_SETS:
        d = neg / name[:-4]
        if (d / ".done").exists():
            continue
        z = DATA / name
        fetch(NEG_URL + name, z)
        run(["unzip", "-q", "-o", z, "-d", neg])
        (d / ".done").write_text("ok")


# ---------------------------------------------------------------- your voice (from the watch's "Teach my voice")
VOICE = WORK / "my_voice"


def fetch_voice(zip_path: str | None):
    """Your clips: from --voice <zip>, else from Oracle Minutes (ORACLE_URL + ORACLE_KEY). 16 kHz PCM in VOICE/<id>/."""
    data = None
    if zip_path:
        data = Path(zip_path).read_bytes()
    elif os.environ.get("ORACLE_URL") and os.environ.get("ORACLE_KEY"):
        import requests
        r = requests.get(os.environ["ORACLE_URL"].rstrip("/") + "/api/wakeword/samples.zip",
                         headers={"X-API-Key": os.environ["ORACLE_KEY"]}, timeout=120)
        if r.status_code == 200:
            data = r.content
        else:
            log("could not fetch your voice clips:", r.status_code)
    if not data:
        return {}
    shutil.rmtree(VOICE, ignore_errors=True)
    raw = WORK / "my_voice_raw"
    shutil.rmtree(raw, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(raw)
    counts = {}
    for f in sorted(raw.glob("*/*.wav")):
        d = VOICE / f.parent.name
        d.mkdir(parents=True, exist_ok=True)
        to_wav16k(f, d / f.name)                  # the watch sends IMA ADPCM: training wants plain PCM
        counts[f.parent.name] = counts.get(f.parent.name, 0) + 1
    shutil.rmtree(raw, ignore_errors=True)
    log("your voice:", counts)
    return counts


# ---------------------------------------------------------------- one model
def speak(texts: list[str], out: Path, n: int, quick: bool):
    """n clips of the texts (cycled), many voices and speeds."""
    if out.exists() and len(list(out.glob("*.wav"))) >= n:
        return
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    lst = out.with_suffix(".txt")
    lst.write_text("\n".join(texts) + "\n")
    run([sys.executable, "-m", "piper_sample_generator", lst, "--model", DATA / "en_US-libritts_r-medium.pt",
         "--max-samples", n, "--batch-size", 10 if quick else 50, "--max-speakers", 800,
         "--length-scales", "0.8", "0.95", "1.1", "1.3", "--noise-scales", "0.5", "0.667", "0.9",
         "--noise-scale-ws", "0.6", "0.8", "1.0", "--output-dir", out])


def features(clip_dir: Path, out_dir: Path, splits: dict[str, tuple[str, int, int]]):
    """Augmented spectrograms of the clips, as the RaggedMmap folders microWakeWord trains on."""
    from mmap_ninja.ragged import RaggedMmap
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration

    if (out_dir / ".done").exists():
        return
    shutil.rmtree(out_dir, ignore_errors=True)
    clips = Clips(input_directory=str(clip_dir), file_pattern="*.wav", max_clip_duration_s=None,
                  remove_silence=False, random_split_seed=10, split_count=0.1)
    aug = Augmentation(augmentation_duration_s=3.2,
                       augmentation_probabilities={"SevenBandParametricEQ": 0.1, "TanhDistortion": 0.1,
                                                   "PitchShift": 0.1, "BandStopFilter": 0.1, "AddColorNoise": 0.1,
                                                   "AddBackgroundNoise": 0.75, "Gain": 1.0, "RIR": 0.5},
                       impulse_paths=[str(DATA / "mit_rirs")],
                       background_paths=[str(DATA / "fma_16k"), str(DATA / "audioset_16k")],
                       background_min_snr_db=-5, background_max_snr_db=10, min_jitter_s=0.195, max_jitter_s=0.205)
    for folder, (split, repeat, slide) in splits.items():
        d = out_dir / folder
        d.mkdir(parents=True, exist_ok=True)
        gen = SpectrogramGeneration(clips=clips, augmenter=aug, slide_frames=slide, step_ms=10)
        RaggedMmap.from_generator(out_dir=str(d / "wakeword_mmap"),
                                  sample_generator=gen.spectrogram_generator(split=split, repeat=repeat),
                                  batch_size=100, verbose=True)
    (out_dir / ".done").write_text("ok")


def pick_cutoff(roc: Path, max_faph: float, lowest: float = 0.5) -> tuple[float, float, float]:
    """(cutoff, false rejection rate, false accepts per hour): the lowest cut-off whose false wakes are in budget."""
    best = (0.97, 1.0, 99.0)
    rows = []
    for line in roc.read_text().splitlines():
        m = re.match(r"Cutoff ([\d.]+): frr=([\d.]+); faph=([\d.]+)", line)
        if m:
            rows.append(tuple(float(x) for x in m.groups()))
    ok = [r for r in rows if r[2] <= max_faph]
    if ok:
        best = min(ok, key=lambda r: (r[1], -r[0]))           # fewest misses within the false-wake budget
    return max(lowest, min(0.99, best[0])), best[1], best[2]


def train_one(m: dict, quick: bool) -> dict:
    mid = m["id"]
    base = WORK / mid
    n = 400 if quick else int(m.get("samples", 10000))
    steps = 300 if quick else int(m.get("steps", 10000))
    log(f"== {m['label']} ({mid}): {n} samples, {steps} steps")
    speak(m["say"], base / "positive", n, quick)
    speak(m.get("not") or ["hello"], base / "nearmiss", max(200, n // 3), quick)
    features(base / "positive", base / "pos_features",
             {"training": ("train", 2, 10), "validation": ("validation", 1, 10), "testing": ("test", 1, 1)})
    features(base / "nearmiss", base / "near_features",
             {"training": ("train", 1, 10), "validation": ("validation", 1, 10), "testing": ("test", 1, 1)})
    mine, others = VOICE / mid, base / "my_voice_others"
    voice_feats = []
    if mine.exists() and len(list(mine.glob("*.wav"))) >= 5:
        # your own takes: each one many times with different rooms / noise, and weighted up
        shutil.rmtree(base / "my_features", ignore_errors=True)          # new clips may have come in
        features(mine, base / "my_features",
                 {"training": ("train", 25, 10), "validation": ("validation", 3, 10), "testing": ("test", 3, 1)})
        voice_feats.append({"features_dir": str(base / "my_features"), "sampling_weight": 3.0, "penalty_weight": 1.5,
                            "truth": True, "truncation_strategy": "truncate_start", "type": "mmap"})
        # you saying the OTHER phrases: must not trigger this one
        shutil.rmtree(others, ignore_errors=True)
        others.mkdir(parents=True)
        for f in VOICE.glob("*/*.wav"):
            if f.parent.name != mid:
                shutil.copy(f, others / f"{f.parent.name}_{f.name}")
        if len(list(others.glob("*.wav"))) >= 5:
            shutil.rmtree(base / "my_other_features", ignore_errors=True)
            features(others, base / "my_other_features",
                     {"training": ("train", 10, 10), "validation": ("validation", 2, 10), "testing": ("test", 2, 1)})
            voice_feats.append({"features_dir": str(base / "my_other_features"), "sampling_weight": 2.0,
                                "penalty_weight": 2.0, "truth": False, "truncation_strategy": "truncate_start",
                                "type": "mmap"})
        log(f"   with {len(list(mine.glob('*.wav')))} clips of your voice")
    neg = DATA / "negative_datasets"
    cfg = {
        "window_step_ms": 10,
        "train_dir": str(base / "trained"),
        "features": [
            {"features_dir": str(base / "pos_features"), "sampling_weight": 2.0, "penalty_weight": 1.0,
             "truth": True, "truncation_strategy": "truncate_start", "type": "mmap"},
            {"features_dir": str(base / "near_features"), "sampling_weight": 4.0, "penalty_weight": 2.0,
             "truth": False, "truncation_strategy": "truncate_start", "type": "mmap"},
            {"features_dir": str(neg / "speech"), "sampling_weight": 10.0, "penalty_weight": 1.0, "truth": False,
             "truncation_strategy": "random", "type": "mmap"},
            {"features_dir": str(neg / "dinner_party"), "sampling_weight": 10.0, "penalty_weight": 1.0,
             "truth": False, "truncation_strategy": "random", "type": "mmap"},
            {"features_dir": str(neg / "no_speech"), "sampling_weight": 5.0, "penalty_weight": 1.0, "truth": False,
             "truncation_strategy": "random", "type": "mmap"},
            {"features_dir": str(neg / "dinner_party_eval"), "sampling_weight": 0.0, "penalty_weight": 1.0,
             "truth": False, "truncation_strategy": "split", "type": "mmap"},
        ] + voice_feats,
        "training_steps": [steps],
        "positive_class_weight": [1],
        "negative_class_weight": [20],
        "learning_rates": [0.001],
        "batch_size": 128,
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": max(100, steps // 20),
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }
    cfg_path = base / "training_parameters.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    run([sys.executable, "-m", "microwakeword.model_train_eval", f"--training_config={cfg_path}",
         "--train", "1", "--restore_checkpoint", "1", "--test_tf_nonstreaming", "0",
         "--test_tflite_nonstreaming", "0", "--test_tflite_nonstreaming_quantized", "0",
         "--test_tflite_streaming", "0", "--test_tflite_streaming_quantized", "1", "--use_weights", "best_weights",
         "mixednet", "--pointwise_filters", "64,64,64,64", "--repeat_in_block", "1, 1, 1, 1",
         "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]", "--residual_connection", "0,0,0,0",
         "--first_conv_filters", "32", "--first_conv_kernel_size", "5", "--stride", "3"])
    q = base / "trained" / "tflite_stream_state_internal_quant"
    model = q / "stream_state_internal_quant.tflite"
    if not model.exists():
        raise SystemExit(f"training finished but {model} is missing")
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(model, OUT / f"{mid}.tflite")
    roc = q / "tflite_streaming_roc.txt"
    cutoff, frr, faph = (0.97, -1, -1)
    if roc.exists():
        shutil.copy(roc, OUT / f"{mid}_roc.txt")
        cutoff, frr, faph = pick_cutoff(roc, float(m.get("max_faph", 1.0)), 0.85 if mid == "wake" else 0.7)
    log(f"   {mid}: cut-off {cutoff:.2f} -> misses {frr * 100:.1f}% of tries, {faph:.2f} false wakes an hour (test set)")
    return {"id": mid, "label": m["label"], "file": f"{mid}.tflite", "cutoff": round(cutoff, 2), "window": 5,
            "arena": 30000, "test": {"miss_rate": frr, "false_per_hour": faph, "samples": n, "steps": steps}}


def upload(zip_path: Path):
    url, key = os.environ.get("ORACLE_URL", "").rstrip("/"), os.environ.get("ORACLE_KEY", "")
    if not url or not key:
        log("ORACLE_URL / ORACLE_KEY not set: upload", zip_path, "on Oracle Minutes yourself (or with curl)")
        return
    import requests
    r = requests.post(url + "/api/wakeword", headers={"X-API-Key": key},
                      files={"file": ("oracle_wakeword.zip", zip_path.read_bytes(), "application/zip")}, timeout=120)
    if r.status_code != 200:
        raise SystemExit(f"upload failed: HTTP {r.status_code} {r.text[:300]}")
    log("installed on Oracle Minutes, version", r.json().get("version"), "- the watch picks it up at its next check")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phrases", default=str(KIT / "phrases.yaml"))
    ap.add_argument("--only", action="append", help="train only these ids (repeat); the rest are kept from /out")
    ap.add_argument("--quick", action="store_true", help="tiny test run to check the setup")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--voice", help="zip of your voice clips (default: fetched from Oracle Minutes)")
    a = ap.parse_args()
    random.seed(1)
    spec = yaml.safe_load(Path(a.phrases).read_text())["models"]
    if not any(m["id"] == "wake" for m in spec):
        raise SystemExit("phrases.yaml needs a model with id: wake")
    prepare_data(a.quick)
    fetch_voice(a.voice)
    done = {}
    old = OUT / "wake.json"
    if old.exists():
        done = {e["id"]: e for e in json.loads(old.read_text()).get("models", [])}
    for m in spec:
        if a.only and m["id"] not in a.only and m["id"] in done and (OUT / done[m["id"]]["file"]).exists():
            continue
        done[m["id"]] = train_one(m, a.quick)
    models = [done[m["id"]] for m in spec if m["id"] in done]
    (OUT / "wake.json").write_text(json.dumps({"models": models}, indent=1))
    z = OUT / "oracle_wakeword.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUT / "wake.json", "wake.json")
        for e in models:
            zf.write(OUT / e["file"], e["file"])
    log("ready:", z)
    for e in models:
        t = e.get("test", {})
        log(f"   {e['label']:<16} cut-off {e['cutoff']:.2f}  misses {t.get('miss_rate', -1) * 100:5.1f}%  "
            f"false/hour {t.get('false_per_hour', -1):.2f}")
    if not a.no_upload:
        upload(z)


if __name__ == "__main__":
    main()
