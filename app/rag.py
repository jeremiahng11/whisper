"""Find by meaning (RAG): what the company knows, matched to a question by meaning instead of shared words.

Everything the knowledge base has is cut into short passages and turned into vectors with a small embedding model
(Ollama's nomic-embed-text by default - ~270 MB, a few ms per passage; OpenAI's text-embedding-3-small with the
openai backend):
  * every fact ("Who is the CTO of Aleta Planet? Jeremiah Ng ...") and who-is answer,
  * your pages, documents, research and news (passages of ~900 characters),
  * every meeting's minutes (passages).
A question is turned into a vector the same way and the closest passages go to the AI with it - so "who runs the
tech side?" finds "Jeremiah Ng is the CTO" although no word is shared. Only new or changed passages are embedded
(the knowledge sweep keeps the index up to date); the index lives in DATA_DIR/knowledge/rag/.

If the embedding model isn't there (or RAG=0), everything works as before with word matching.
"""
import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path

import httpx

from . import knowledge
from .config import settings

log = logging.getLogger("rag")
PASSAGE = 900
BATCH = 32
_lock = threading.Lock()
_mem: dict = {"at": None, "meta": [], "vecs": None}
status: dict = {"state": "idle", "passages": 0, "error": "", "updated": 0}


def enabled() -> bool:
    return settings.rag and settings.summary_backend.lower() in ("ollama", "openai")


def _dir() -> Path:
    d = knowledge.kdir() / "rag"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- embeddings
def embed(texts: list[str], query: bool = False, timeout: float = 60) -> list[list[float]]:
    """Vectors for texts (a question: query=True). Raises when the embedding model can't be reached."""
    if not texts:
        return []
    b = settings.summary_backend.lower()
    if b == "ollama":
        model = settings.ollama_embed_model
        pre = ("search_query: " if query else "search_document: ") if "nomic" in model else ""
        body = {"model": model, "input": [pre + t for t in texts]}
        r = httpx.post(f"{settings.ollama_url}/api/embed", json=body, timeout=timeout)
        if r.status_code == 404 and settings.ollama_auto_pull:
            from . import summarize
            threading.Thread(target=summarize.ensure_ollama_model, args=(model,), daemon=True).start()
            raise RuntimeError(f"embedding model {model} is being downloaded")
        r.raise_for_status()
        return r.json()["embeddings"]
    if b == "openai":
        headers = {"Authorization": f"Bearer {settings.openai_api_key}"} if settings.openai_api_key else {}
        r = httpx.post(f"{settings.openai_base_url}/embeddings", headers=headers, timeout=timeout,
                       json={"model": settings.openai_embed_model, "input": texts})
        r.raise_for_status()
        return [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
    raise RuntimeError("no embedding backend")


# ---------------------------------------------------------------- what gets indexed
def _split(text: str, size: int = PASSAGE) -> list[str]:
    out, cur = [], ""
    for para in re.split(r"\n\s*\n|\n(?=#)", text):
        para = re.sub(r"[ \t]+", " ", para).strip()
        if not para:
            continue
        while len(para) > size:
            cut = para.rfind(". ", 0, size)
            cut = cut + 1 if cut > size // 2 else size
            if cur:
                out.append(cur)
                cur = ""
            out.append(para[:cut])
            para = para[cut:].strip()
        if len(cur) + len(para) + 1 > size and cur:
            out.append(cur)
            cur = ""
        cur = (cur + "\n" + para).strip()
    if cur:
        out.append(cur)
    return [p for p in out if len(p) >= 40]


def passages() -> list[dict]:
    """[{"h", "kind", "ref", "title", "text"}] - every passage there should be in the index."""
    from . import db, sources, worker
    out = []
    for f in knowledge.facts():
        t = f"{f['q']}? {f['a']}" + (f" ({f.get('meeting') or 'learned'}, {f['date']})" if f.get("date") else "")
        out.append({"kind": "fact", "ref": f["id"], "title": f.get("topic") or "", "text": t})
    for m in sources.listing():
        try:
            txt = (sources.sdir() / f"{m['id']}.txt").read_text()
        except OSError:
            continue
        title = m.get("title") or m.get("query") or ""
        for p in _split(txt[:sources.MAX_TEXT]):
            out.append({"kind": m.get("kind") or "source", "ref": m["id"], "title": title, "text": p})
    for j in db.list_jobs(100000, "done"):
        p = worker.result_dir(j["id"]) / "summary.md"
        try:
            md = p.read_text(encoding="utf-8")
        except OSError:
            continue
        title = (j.get("options") or {}).get("title") or j.get("title") or j.get("filename") or "meeting"
        date = str(j.get("created") or "")[:10]
        for part in _split(md):
            out.append({"kind": "meeting", "ref": j["id"], "title": f"{title} {date}".strip(), "text": part})
    for x in out:
        x["h"] = hashlib.sha1((x["kind"] + "|" + x["title"] + "|" + x["text"]).encode()).hexdigest()[:16]
    return out


# ---------------------------------------------------------------- the index
def _load():
    import numpy as np
    d = _dir()
    try:
        at = (d / "meta.json").stat().st_mtime
    except OSError:
        return [], None
    if _mem["at"] != at:
        try:
            meta = json.loads((d / "meta.json").read_text())
            vecs = np.load(d / "vecs.npy")
            if len(meta) != len(vecs):
                raise ValueError("index out of step")
        except (OSError, ValueError) as e:
            log.warning("rag index unreadable (%s): building again", e)
            return [], None
        _mem.update(at=at, meta=meta, vecs=vecs)
    return _mem["meta"], _mem["vecs"]


def update(max_new: int = 4000) -> int:
    """Embed the passages that are new or changed and drop the ones that are gone. Returns how many were embedded."""
    import numpy as np
    if not enabled():
        return 0
    with _lock:
        status.update(state="indexing", error="")
        try:
            meta, vecs = _load()
            have = {m["h"]: i for i, m in enumerate(meta)}
            want = passages()
            new = [p for p in want if p["h"] not in have][:max_new]
            keep_h = {p["h"] for p in want}
            got = []
            for i in range(0, len(new), BATCH):
                got += embed([p["text"] for p in new[i:i + BATCH]])
            rows, mrows = [], []
            for m in meta:                             # still wanted: keep
                if m["h"] in keep_h:
                    rows.append(vecs[have[m["h"]]])
                    mrows.append(m)
            for p, v in zip(new, got):
                v = np.asarray(v, dtype=np.float32)
                rows.append(v / max(1e-9, float(np.linalg.norm(v))))
                mrows.append({k: p[k] for k in ("h", "kind", "ref", "title", "text")})
            if not new and len(mrows) == len(meta):
                status.update(state="idle", passages=len(meta))
                return 0
            d = _dir()
            arr = np.stack(rows).astype(np.float32) if rows else np.zeros((0, 1), np.float32)
            np.save(d / "vecs.tmp.npy", arr)
            (d / "vecs.tmp.npy").replace(d / "vecs.npy")
            tmp = d / "meta.tmp"
            tmp.write_text(json.dumps(mrows, ensure_ascii=False))
            tmp.replace(d / "meta.json")
            status.update(state="idle", passages=len(mrows), updated=time.time())
            if new:
                log.info("rag: %d passages embedded (%d in all)", len(new), len(mrows))
            return len(new)
        except Exception as e:                       # noqa: BLE001 - the embedding model may not be there yet
            status.update(state="error", error=str(e)[:300])
            raise


def needs_update() -> bool:
    if not enabled():
        return False
    meta, _ = _load()
    return {m["h"] for m in meta} != {p["h"] for p in passages()}


def find(q: str, k: int = 12, timeout: float = 6) -> list[dict]:
    """The passages closest in meaning to the question: [{"score", "kind", "ref", "title", "text"}] ([] when there's
    no index or the embedding model can't be reached quickly)."""
    import numpy as np
    if not enabled():
        return []
    meta, vecs = _load()
    if not meta or vecs is None or not len(vecs):
        return []
    try:
        v = np.asarray(embed([q], query=True, timeout=timeout)[0], dtype=np.float32)
    except Exception as e:                           # noqa: BLE001
        log.info("rag query: %s", e)
        return []
    if v.shape[0] != vecs.shape[1]:
        return []                                    # the embedding model changed: the sweep re-indexes
    v /= max(1e-9, float(np.linalg.norm(v)))
    sims = vecs @ v
    idx = np.argsort(-sims)[: k * 2]
    out, seen = [], set()
    for i in idx:
        m = meta[int(i)]
        key = (m["kind"], m["ref"], m["text"][:60])
        if key in seen:
            continue
        seen.add(key)
        out.append({"score": float(sims[int(i)]), **{x: m[x] for x in ("kind", "ref", "title", "text")}})
        if len(out) >= k:
            break
    return out


def context_for(q: str, limit: int = 3500) -> str:
    hits = [h for h in find(q) if h["score"] >= settings.rag_min_score]
    if not hits:
        return ""
    names = {"fact": "fact", "meeting": "meeting minutes", "url": "web page", "file": "document", "text": "notes",
             "news": "news", "research": "web research"}
    lines, size = [], 0
    for h in hits:
        line = f"- [{names.get(h['kind'], h['kind'])}{': ' + h['title'] if h['title'] else ''}] {h['text']}"
        if size + len(line) > limit:
            break
        lines.append(line)
        size += len(line)
    return "Passages from the company's knowledge closest in meaning to the question (best first):\n" + "\n".join(lines)


def summary() -> dict:
    meta, _ = _load()
    return {"enabled": enabled(), "model": settings.ollama_embed_model if settings.summary_backend.lower() == "ollama"
            else settings.openai_embed_model, "passages": len(meta), **{k: status[k] for k in ("state", "error", "updated")}}
