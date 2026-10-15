"""The watch's own small AI model, in plain numpy (no PyTorch needed on the server).

Architecture = esp32-ai's tiny decoder-only transformer (github.com/slvDev/esp32-ai, MIT), "baseline" arm:
RMSNorm, rotary positions, causal self-attention, SwiGLU feed-forward, output head tied to the input embedding.
(esp32-ai's per-layer embedding table pays off with a 32k vocabulary; ours is a few thousand company words, so the
plain model is both smaller and faster on the watch.)

Tokens are whole words (your company's own words, names and numbers) plus punctuation, so a short answer is ~15-25
steps on the watch and the C tokenizer there is a simple word lookup.

  sequence:  <q> who is ryan <a> Ryan is the Group CEO . <eos>
  training:  loss only on the answer part; the question is lower-cased (the watch hears it as lower-case words)

File format "OWL1" (what the watch loads), little-endian:
  b"OWL1", u32 header length, header JSON {"config", "vocab", "version", ...} (padded with spaces to 4 bytes),
  then per tensor (fixed order, see tensor_order): float32 vectors as they are; matrices as one float32 scale per row,
  then rows x cols int8, zero-padded to 4 bytes (every float stays 4-byte aligned: the watch's CPU needs that).
"""
import json
import math
import re
import struct

import numpy as np

PAD, UNK, Q, A, EOS = 0, 1, 2, 3, 4
SPECIAL = ["<pad>", "<unk>", "<q>", "<a>", "<eos>"]
_TOK = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\sA-Za-z0-9]")
NO_SPACE_BEFORE = set(".,?!:;%)'") | {"'s"}
NO_SPACE_AFTER = set("($#")
JOINERS = set("-/@")


def words(text: str) -> list[str]:
    # a spaced dash ("yet - ask") is a pause, written back with spaces; "V2-ready" / "10/2026" stay joined
    return _TOK.findall(re.sub(r"\s[-\u2013\u2014]\s", " \u2013 ", text or ""))


def detok(toks: list[str]) -> str:
    out = ""
    for t in toks:
        if not out:
            out = t
        elif t in NO_SPACE_BEFORE or t in JOINERS or out[-1] in JOINERS or out[-1] in NO_SPACE_AFTER:
            out += t
        else:
            out += " " + t
    return out


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def build(cls, questions: list[str], answers: list[str], max_size: int = 8000) -> "Vocab":
        from collections import Counter
        c = Counter()
        for q in questions:
            c.update(w.lower() for w in words(q) if w[0].isascii() and w[0].isalnum())
        for a in answers:
            c.update(words(a))
        itos = SPECIAL + [w for w, _ in c.most_common(max_size - len(SPECIAL))]
        return cls(itos)

    def enc_q(self, q: str) -> list[int]:
        # questions are words only ("Who is Ryan?" = "who is ryan"): heard or typed, punctuation means nothing
        return [self.stoi.get(w.lower(), UNK) for w in words(q) if w[0].isascii() and w[0].isalnum()]

    def enc_a(self, a: str) -> list[int]:
        return [self.stoi.get(w, self.stoi.get(w.lower(), UNK)) for w in words(a)]

    def dec(self, ids: list[int]) -> str:
        return detok([self.itos[i] for i in ids if i > EOS])


# ---------------------------------------------------------------- model
def default_config(vocab_size: int) -> dict:
    return {"vocab": vocab_size, "d": 160, "layers": 6, "heads": 5, "ffn": 432, "seq": 96, "rope_theta": 10000.0}


def weight_bytes(cfg: dict) -> int:
    """What the watch must hold for this model: int8 weights + a float scale per row (the file, about)."""
    d, f, L, V = cfg["d"], cfg["ffn"], cfg["layers"], cfg["vocab"]
    per_layer = 4 * d * d + 3 * d * f + 4 * (3 * d + d + 2 * f + d) + 8 * d
    return V * d + 4 * V + L * per_layer + 4 * d


# bigger models for more facts (they answer reworded questions better), as long as the watch can hold them:
# (d, layers, heads, ffn, from this many facts)
SIZES = [(160, 6, 5, 432, 0), (192, 6, 6, 512, 1500), (224, 6, 7, 608, 4000), (256, 8, 8, 704, 8000)]
WATCH_BUDGET = 4_500_000                           # bytes of PSRAM for the weights (the watch frees the speech model for it)


def config_for(n_facts: int, vocab_size: int, budget: int = WATCH_BUDGET) -> dict:
    best = default_config(vocab_size)
    for d, L, H, f, need in SIZES:
        c = {**best, "d": d, "layers": L, "heads": H, "ffn": f}
        if n_facts >= need and weight_bytes(c) <= budget:
            best = c
    return best


def init_params(cfg: dict, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d, f, L = cfg["d"], cfg["ffn"], cfg["layers"]
    n = lambda *s, std=0.02: (rng.standard_normal(s) * std).astype(np.float32)   # noqa: E731
    res = 0.02 / math.sqrt(2 * L)                     # residual-writing projections start small (GPT-2 style)
    p = {"emb": n(cfg["vocab"], d), "out_norm": np.ones(d, np.float32)}
    for i in range(L):
        p[f"{i}.attn_norm"] = np.ones(d, np.float32)
        p[f"{i}.qkv"] = n(3 * d, d)
        p[f"{i}.proj"] = n(d, d, std=res)
        p[f"{i}.ffn_norm"] = np.ones(d, np.float32)
        p[f"{i}.gate"] = n(f, d)
        p[f"{i}.up"] = n(f, d)
        p[f"{i}.down"] = n(d, f, std=res)
    return p


def tensor_order(cfg: dict) -> list[str]:
    names = ["emb"]
    for i in range(cfg["layers"]):
        names += [f"{i}.attn_norm", f"{i}.qkv", f"{i}.proj", f"{i}.ffn_norm", f"{i}.gate", f"{i}.up", f"{i}.down"]
    return names + ["out_norm"]


def rope_tables(T: int, dh: int, theta: float):
    inv = 1.0 / (theta ** (np.arange(0, dh, 2, dtype=np.float64) / dh))
    fr = np.outer(np.arange(T, dtype=np.float64), inv)
    return np.cos(fr).astype(np.float32), np.sin(fr).astype(np.float32)


def _rope(x, cos, sin):                              # x (B,H,T,dh)
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


def _rope_back(dy, cos, sin):
    h = dy.shape[-1] // 2
    d1, d2 = dy[..., :h], dy[..., h:]
    return np.concatenate([d1 * cos + d2 * sin, -d1 * sin + d2 * cos], -1)


EPS = 1e-6


def _rms(x, g):
    r = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)
    return x * r * g, r


def _rms_back(dy, x, r, g):
    gdy = dy * g
    dg = (dy * x * r).reshape(-1, x.shape[-1]).sum(0)
    dx = r * gdy - x * (r ** 3) * (gdy * x).mean(-1, keepdims=True)
    return dx, dg


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def forward(p: dict, cfg: dict, X: np.ndarray, Y: np.ndarray | None = None, M: np.ndarray | None = None):
    """X (B,T) token ids; Y targets; M loss mask (1 = learn this position). Returns (loss, cache) or logits."""
    B, T = X.shape
    d, H = cfg["d"], cfg["heads"]
    dh = d // H
    cos, sin = rope_tables(T, dh, cfg["rope_theta"])
    mask = np.triu(np.full((T, T), -1e9, np.float32), 1)
    x = p["emb"][X]
    cache = {"X": X, "layers": []}
    for i in range(cfg["layers"]):
        c = {"x0": x}
        a, c["r1"] = _rms(x, p[f"{i}.attn_norm"])
        c["a"] = a
        qkv = a @ p[f"{i}.qkv"].T                      # (B,T,3d)
        q, k, v = [t.reshape(B, T, H, dh).transpose(0, 2, 1, 3) for t in np.split(qkv, 3, -1)]
        qr, kr = _rope(q, cos, sin), _rope(k, cos, sin)
        s = (qr @ kr.transpose(0, 1, 3, 2)) / math.sqrt(dh) + mask
        s -= s.max(-1, keepdims=True)
        P = np.exp(s)
        P /= P.sum(-1, keepdims=True)
        o = (P @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
        c.update(qr=qr, kr=kr, v=v, P=P, o=o)
        x = x + o @ p[f"{i}.proj"].T
        c["x1"] = x
        fi, c["r2"] = _rms(x, p[f"{i}.ffn_norm"])
        g = fi @ p[f"{i}.gate"].T
        u = fi @ p[f"{i}.up"].T
        sg = _sig(g)
        hdn = g * sg * u
        c.update(fi=fi, g=g, u=u, sg=sg, hdn=hdn)
        x = x + hdn @ p[f"{i}.down"].T
        cache["layers"].append(c)
    out, r = _rms(x, p["out_norm"])
    logits = out @ p["emb"].T
    if Y is None:
        return logits
    cache.update(xL=x, rL=r, out=out)
    lg = logits - logits.max(-1, keepdims=True)
    lse = np.log(np.exp(lg).sum(-1))
    logp_t = np.take_along_axis(lg, Y[..., None], -1)[..., 0] - lse
    n = max(1.0, float(M.sum()))
    loss = float(-(logp_t * M).sum() / n)
    probs = np.exp(lg - lse[..., None])
    dlog = probs
    dlog[np.arange(B)[:, None], np.arange(T)[None, :], Y] -= 1.0
    dlog *= (M / n)[..., None]
    cache["dlog"] = dlog.astype(np.float32)
    return loss, cache


def backward(p: dict, cfg: dict, cache: dict) -> dict:
    B, T = cache["X"].shape
    d, H = cfg["d"], cfg["heads"]
    dh = d // H
    cos, sin = rope_tables(T, dh, cfg["rope_theta"])
    gr = {k: np.zeros_like(v) for k, v in p.items()}
    dlog = cache["dlog"]
    gr["emb"] += dlog.reshape(-1, dlog.shape[-1]).T @ cache["out"].reshape(-1, d)
    dout = dlog @ p["emb"]
    dx, gr["out_norm"] = _rms_back(dout, cache["xL"], cache["rL"], p["out_norm"])
    for i in reversed(range(cfg["layers"])):
        c = cache["layers"][i]
        # feed-forward
        dh_ = dx @ p[f"{i}.down"]
        gr[f"{i}.down"] += dx.reshape(-1, d).T @ c["hdn"].reshape(-1, c["hdn"].shape[-1])
        g, u, sg = c["g"], c["u"], c["sg"]
        du = dh_ * g * sg
        dg = dh_ * u * sg * (1 + g * (1 - sg))
        fi2 = c["fi"].reshape(-1, d)
        gr[f"{i}.gate"] += dg.reshape(-1, dg.shape[-1]).T @ fi2
        gr[f"{i}.up"] += du.reshape(-1, du.shape[-1]).T @ fi2
        dfi = dg @ p[f"{i}.gate"] + du @ p[f"{i}.up"]
        d1, dgn = _rms_back(dfi, c["x1"], c["r2"], p[f"{i}.ffn_norm"])
        gr[f"{i}.ffn_norm"] += dgn
        dx = dx + d1
        # attention
        gr[f"{i}.proj"] += dx.reshape(-1, d).T @ c["o"].reshape(-1, d)
        do = (dx @ p[f"{i}.proj"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        P = c["P"]
        dP = do @ c["v"].transpose(0, 1, 3, 2)
        dv = P.transpose(0, 1, 3, 2) @ do
        ds = P * (dP - (dP * P).sum(-1, keepdims=True)) / math.sqrt(dh)
        dqr = ds @ c["kr"]
        dkr = ds.transpose(0, 1, 3, 2) @ c["qr"]
        dq, dk = _rope_back(dqr, cos, sin), _rope_back(dkr, cos, sin)
        dqkv = np.concatenate([t.transpose(0, 2, 1, 3).reshape(B, T, d) for t in (dq, dk, dv)], -1)
        gr[f"{i}.qkv"] += dqkv.reshape(-1, 3 * d).T @ c["a"].reshape(-1, d)
        da = dqkv @ p[f"{i}.qkv"]
        d1, dgn = _rms_back(da, c["x0"], c["r1"], p[f"{i}.attn_norm"])
        gr[f"{i}.attn_norm"] += dgn
        dx = dx + d1
    np.add.at(gr["emb"], cache["X"].reshape(-1), dx.reshape(-1, d))
    return gr


class Adam:
    def __init__(self, params: dict, lr=2e-3, b1=0.9, b2=0.98, wd=0.01):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.lr, self.b1, self.b2, self.wd, self.t = lr, b1, b2, wd, 0

    def step(self, p: dict, g: dict, lr: float | None = None, clip: float = 1.0):
        self.t += 1
        lr = self.lr if lr is None else lr
        norm = math.sqrt(sum(float((x * x).sum()) for x in g.values()))
        sc = min(1.0, clip / (norm + 1e-9))
        for k in p:
            gk = g[k] * sc
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gk * gk
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            if p[k].ndim == 2:
                p[k] -= lr * self.wd * p[k]
            p[k] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        return norm


# ---------------------------------------------------------------- int8 like the watch, and generation
def quantize(p: dict) -> dict:
    """Matrices -> int8 rows + per-row scale, exactly what the watch stores. Vectors stay float32."""
    q = {}
    for k, w in p.items():
        if w.ndim == 2:
            s = np.abs(w).max(1) / 127.0
            s[s == 0] = 1e-8
            q[k] = (np.clip(np.round(w / s[:, None]), -127, 127).astype(np.int8), s.astype(np.float32))
        else:
            q[k] = w.astype(np.float32)
    return q


def dequantize(q: dict) -> dict:
    return {k: (v[0].astype(np.float32) * v[1][:, None]) if isinstance(v, tuple) else v for k, v in q.items()}


def generate(p: dict, cfg: dict, vocab: Vocab, question: str, max_new: int = 48) -> str:
    """Greedy answer (the watch does the same, step by step with a KV cache)."""
    ids = [Q] + vocab.enc_q(question)[: cfg["seq"] // 2] + [A]
    out = []
    for _ in range(max_new):
        if len(ids) >= cfg["seq"]:
            break
        logits = forward(p, cfg, np.array([ids]))[0, -1]
        t = int(np.argmax(logits))
        if t == EOS:
            break
        out.append(t)
        ids.append(t)
    return vocab.dec(out)


def export(path, q: dict, cfg: dict, vocab: Vocab, meta: dict) -> int:
    header = json.dumps({"config": cfg, "vocab": vocab.itos, "special": SPECIAL, **meta}, ensure_ascii=False).encode()
    header += b" " * (-len(header) % 4)
    with open(path, "wb") as f:
        f.write(b"OWL1" + struct.pack("<I", len(header)) + header)
        for name in tensor_order(cfg):
            v = q[name]
            if isinstance(v, tuple):
                b = v[0].tobytes()
                f.write(v[1].astype("<f4").tobytes() + b + b"\0" * (-len(b) % 4))
            else:
                f.write(v.astype("<f4").tobytes())
        return f.tell()


def load(path) -> tuple[dict, dict, Vocab, dict]:
    raw = open(path, "rb").read()
    assert raw[:4] == b"OWL1", "not an OWL1 model"
    n = struct.unpack("<I", raw[4:8])[0]
    h = json.loads(raw[8:8 + n])
    cfg, off, q = h["config"], 8 + n, {}
    d, f, V = cfg["d"], cfg["ffn"], cfg["vocab"]
    shapes = {"emb": (V, d), "out_norm": (d,)}
    for i in range(cfg["layers"]):
        shapes.update({f"{i}.attn_norm": (d,), f"{i}.qkv": (3 * d, d), f"{i}.proj": (d, d), f"{i}.ffn_norm": (d,),
                       f"{i}.gate": (f, d), f"{i}.up": (f, d), f"{i}.down": (d, f)})
    for name in tensor_order(cfg):
        s = shapes[name]
        if len(s) == 2:
            sc = np.frombuffer(raw, "<f4", s[0], off)
            off += 4 * s[0]
            w = np.frombuffer(raw, np.int8, s[0] * s[1], off).reshape(s)
            off += s[0] * s[1] + (-(s[0] * s[1]) % 4)
            q[name] = (w, sc)
        else:
            q[name] = np.frombuffer(raw, "<f4", s[0], off).copy()
            off += 4 * s[0]
    return q, cfg, Vocab(h["vocab"]), h
