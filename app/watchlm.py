"""Train the watch's own small AI model from the knowledge base, in the background on this server.

POST /api/watch/model/train starts it (a separate low-priority process, half the CPU cores, so meetings still get
transcribed); GET /api/watch/model shows progress and, when done, the result: how many of the company questions it
answers right (also ones worded in a way it never saw), its size and version. The watch downloads
GET /api/watch/model/file when the version changes.

Training data, from knowledge.facts(): every fact's question and its other wordings -> its answer, plus "who is"
and topic questions, plus a set of off-topic questions -> "I don't know that yet". One other wording per fact is held
back to test with, so the score shows how it copes with questions put differently.

Run by hand:  python -m app.watchlm train DATA_DIR/watchai   (reads dataset.json there)
"""
import hashlib
import json
import logging
import math
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config import settings

log = logging.getLogger("watchlm")
DONT_KNOW = "I don't know that yet - ask me again when I'm online."
SURE_CUT = 0.6
OFF_TOPIC = ["what is the weather today", "tell me a joke", "what time is it", "play some music", "how are you",
             "what is the capital of france", "who won the football", "what is two plus two", "sing a song",
             "what is the meaning of life", "set an alarm", "how tall is mount everest", "who is the president",
             "what should i eat for lunch", "translate hello", "what is bitcoin worth", "open the camera",
             "call my mum", "what day is it", "good morning"]
_proc: subprocess.Popen | None = None
_lock = threading.Lock()


def wdir() -> Path:
    d = settings.data_dir / "watchai"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_dataset(facts: list[dict], seed: int = 0) -> dict:
    """[{q, a, alts, topic}] -> {"train": [[q, a]], "test": [[q, a]], "facts": n}."""
    rng = random.Random(seed)
    train, test = [], []
    from collections import Counter
    topics = Counter((f.get("topic") or "").lower() for f in facts)
    for f in facts:
        qs = list(dict.fromkeys([f["q"]] + [x for x in (f.get("alts") or []) if x]))
        if len(qs) >= 3:                            # hold one other wording back to test with
            held = qs.pop(rng.randrange(1, len(qs)))
            test.append([held, f["a"]])
        if f.get("topic") and f.get("kind") != "person" and topics[f["topic"].lower()] == 1:   # one answer only
            qs.append(f"tell me about {f['topic']}")
        for q in qs:
            train.append([q, f["a"]])
    for q in OFF_TOPIC:
        train.append([q, DONT_KNOW])
    return {"train": train, "test": test, "facts": len(facts), "sig": facts_sig(facts)}


def facts_sig(facts: list[dict]) -> str:
    """Changes when a question or answer does: the nightly run only trains when there is something new."""
    rows = sorted([f["q"], f["a"], sorted(f.get("alts") or [])] for f in facts)
    return hashlib.sha1(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- the training process
def _status(out: Path, **kw) -> None:
    tmp = out / "status.tmp"
    tmp.write_text(json.dumps({**kw, "at": time.time()}))
    tmp.replace(out / "status.json")


def train(out: Path, max_minutes: float = 120, max_epochs: int = 200, seed: int = 0, config: dict | None = None) -> dict:
    import numpy as np
    from . import watchlm_core as W

    ds = json.loads((out / "dataset.json").read_text())
    pairs = ds["train"]
    vocab = W.Vocab.build([q for q, _ in pairs + ds["test"]], [a for _, a in pairs], max_size=6000)
    cfg = {**W.config_for(ds["facts"], len(vocab.itos)), **(config or {})}   # bigger with more facts
    T = cfg["seq"]
    seqs = []
    for q, a in pairs:
        ids = [W.Q] + vocab.enc_q(q)[: T // 2] + [W.A]
        start = len(ids)
        ids = (ids + vocab.enc_a(a) + [W.EOS])[: T + 1]
        seqs.append((ids, start))
    p = W.init_params(cfg, seed)
    opt = W.Adam(p, lr=2e-3)
    bs = 32
    steps_per_epoch = max(1, math.ceil(len(seqs) / bs))
    total = steps_per_epoch * max_epochs
    rng = np.random.default_rng(seed)
    t0, step, loss_avg, done_early = time.time(), 0, None, False
    log.info("training: %d pairs, vocab %d, %d steps max", len(seqs), len(vocab.itos), total)
    for ep in range(max_epochs):
        order = rng.permutation(len(seqs))
        for b in range(steps_per_epoch):
            batch = [seqs[i] for i in order[b * bs:(b + 1) * bs]]
            L = max(len(s) for s, _ in batch) - 1
            X = np.zeros((len(batch), L), np.int64)
            Y = np.zeros((len(batch), L), np.int64)
            M = np.zeros((len(batch), L), np.float32)
            for r, (s, st) in enumerate(batch):
                n = len(s) - 1
                X[r, :n], Y[r, :n] = s[:-1], s[1:]
                M[r, st - 1:n] = 1.0                    # learn the answer (and the stop), not the question
            warm = 100
            lr = 2e-3 * min(1.0, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total))))
            loss, cache = W.forward(p, cfg, X, Y, M)
            opt.step(p, W.backward(p, cfg, cache), lr)
            step += 1
            loss_avg = loss if loss_avg is None else 0.95 * loss_avg + 0.05 * loss
            if step % 20 == 0:
                el = time.time() - t0
                _status(out, state="training", step=step, steps=total, epoch=ep + 1, loss=round(loss_avg, 4),
                        eta_s=int(el / step * (total - step)), pairs=len(seqs), vocab=len(vocab.itos))
            if (time.time() - t0) / 60 > max_minutes:
                done_early = True
                break
        if done_early or (loss_avg is not None and loss_avg < 0.005 and ep >= 30):
            break
    # the watch runs it in int8: test exactly that
    _status(out, state="testing", step=step, steps=total, loss=round(loss_avg or 0, 4))
    np.savez_compressed(out / "params.npz", **p)       # the float weights, to re-export or keep training later
    q8 = W.quantize(p)
    pq = W.dequantize(q8)
    sample = random.Random(seed).sample(pairs, min(200, len(pairs)))

    def score(items):
        right = 0
        for q, a in items:
            got = W.answer(pq, cfg, vocab, q)[0]
            right += _same(got, a)
        return round(100 * right / max(1, len(items)))

    report = {"trained_on": score(sample), "reworded": score(ds["test"]) if ds["test"] else None,
              "facts": ds["facts"], "pairs": len(pairs), "vocab": len(vocab.itos), "steps": step,
              "minutes": round((time.time() - t0) / 60, 1), "loss": round(loss_avg or 0, 4),
              "examples": [{"q": q, "a": W.answer(pq, cfg, vocab, q)[0]} for q, _ in (ds["test"] or sample)[:6]]}
    tmp = out / "model.owl.tmp"
    # below this certainty the watch marks the answer "not sure" (measured: catches some wrong answers, no right ones)
    meta = {"made": time.strftime("%Y-%m-%d %H:%M"), "facts": ds["facts"], "dont_know": DONT_KNOW, "sure_cut": SURE_CUT}
    size = W.export(tmp, q8, cfg, vocab, meta)
    ver = hashlib.sha1(tmp.read_bytes()).hexdigest()[:12]
    tmp.replace(out / "model.owl")
    report.update(version=ver, size=size, config=cfg, made=meta["made"], facts_sig=ds.get("sig", ""))
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False))
    _status(out, state="done", step=step, steps=total, loss=report["loss"])
    return report


def _same(got: str, want: str) -> bool:
    from .watchlm_core import detok, words
    n = lambda s: detok(words(s)).lower().rstrip(".")   # noqa: E731
    return n(got) == n(want)


# ---------------------------------------------------------------- the server side: start, watch, stop
def state() -> dict:
    out = wdir()
    st = {}
    try:
        st = json.loads((out / "status.json").read_text())
    except (OSError, ValueError):
        pass
    running = _proc is not None and _proc.poll() is None
    if not running and st.get("state") in ("training", "testing", "starting"):
        st = {**st, "state": "stopped" if _proc is None or _proc.returncode in (-9, -15) else "error"}
        try:
            st["error"] = (out / "train.log").read_text()[-600:]
        except OSError:
            pass
    rep = {}
    try:
        rep = json.loads((out / "report.json").read_text())
    except (OSError, ValueError):
        pass
    return {"training": running, "status": st, "model": rep if (out / "model.owl").exists() else None}


def start(facts: list[dict]) -> dict:
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:
            return state()
        if len(facts) < 5:
            raise ValueError("the knowledge base has fewer than 5 facts - let it learn from your meetings first")
        out = wdir()
        (out / "dataset.json").write_text(json.dumps(make_dataset(facts), ensure_ascii=False))
        _status(out, state="starting")
        threads = str(settings.watchai_threads or max(1, (os.cpu_count() or 2) // 2))
        env = {**os.environ, "OPENBLAS_NUM_THREADS": threads, "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads}
        log_f = open(out / "train.log", "w")
        _proc = subprocess.Popen([sys.executable, "-m", "app.watchlm", "train", str(out), str(settings.watchai_minutes)],
                                 cwd=str(Path(__file__).resolve().parent.parent), env=env, stdout=log_f,
                                 stderr=subprocess.STDOUT, preexec_fn=lambda: os.nice(10))
    return state()


def stop() -> None:
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()


def nightly(facts: list[dict]) -> bool:
    """Called by the knowledge sweep: between 02:00 and 05:00 server time, train again if the facts changed since
    the last model (WATCHAI_NIGHTLY=0 turns it off). True if it started."""
    if not settings.watchai_nightly or not 2 <= time.localtime().tm_hour < 5 or len(facts) < 5:
        return False
    st = state()
    if st["training"]:
        return False
    if st["model"] and st["model"].get("facts_sig") == facts_sig(facts):
        return False
    last = (st.get("status") or {}).get("at") or 0
    if time.time() - last < 6 * 3600 and (st.get("status") or {}).get("state") in ("error", "stopped"):
        return False                                  # it failed tonight already: not again and again
    log.info("nightly watch AI training: the knowledge changed")
    start(facts)
    return True


def model_file() -> Path | None:
    p = wdir() / "model.owl"
    return p if p.exists() else None


if __name__ == "__main__":                          # python -m app.watchlm train <dir> [minutes]
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) >= 3 and sys.argv[1] == "train":
        mins = float(sys.argv[3]) if len(sys.argv) > 3 else 120
        print(json.dumps(train(Path(sys.argv[2]), max_minutes=mins), indent=1, ensure_ascii=False))
