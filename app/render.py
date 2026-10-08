"""Output formats: Markdown minutes, plain text, SRT subtitles, JSON."""
import re
from datetime import datetime
from typing import Optional


def ts(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def srt_ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def recorded_at(filename: str, fallback: float) -> datetime:
    """Idea Saver names files like 2026-10-08_1535.wav - use that as the recording time."""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})[_ T-]?(\d{2})[:\-]?(\d{2})", filename or "")
    if m:
        try:
            return datetime(*map(int, m.groups()))
        except ValueError:
            pass
    return datetime.fromtimestamp(fallback)


def paragraphs(segments: list[dict], max_len: int = 600, gap: float = 2.5) -> list[dict]:
    """Merge segments into readable paragraphs: new paragraph on speaker change, long pause or length."""
    out: list[dict] = []
    for s in segments:
        p = out[-1] if out else None
        if (p and p["speaker"] == s.get("speaker", "") and s["start"] - p["end"] < gap
                and len(p["text"]) + len(s["text"]) < max_len):
            p["text"] += " " + s["text"].strip()
            p["end"] = s["end"]
        else:
            out.append({"start": s["start"], "end": s["end"], "speaker": s.get("speaker", ""), "text": s["text"].strip()})
    return out


def markdown(job: dict, tr: dict, summary_md: Optional[str], title: str = "") -> str:
    when = recorded_at(job["filename"], job["created_at"])
    mins = tr.get("duration", 0) / 60
    lines = [f"# {title or job.get('title') or job['filename']}", ""]
    meta = [when.strftime("%Y-%m-%d %H:%M"), f"{mins:.0f} min" if mins >= 1 else f"{tr.get('duration', 0):.0f} s"]
    if tr.get("language"):
        meta.append(tr["language"])
    meta.append(f"whisper {tr.get('model', '')}")
    lines += ["*" + " · ".join(meta) + "*", ""]
    if summary_md:
        lines += [summary_md.strip(), ""]
    elif job.get("summary_error"):
        lines += [f"> Summary not available: {job['summary_error']}", ""]
    lines += ["## Transcript", ""]
    for p in paragraphs(tr.get("segments", [])):
        who = f"**{p['speaker']}:** " if p["speaker"] else ""
        lines += [f"`[{ts(p['start'])}]` {who}{p['text']}", ""]
    if not tr.get("segments"):
        lines += ["*(no speech found)*", ""]
    return "\n".join(lines).rstrip() + "\n"


def text(tr: dict) -> str:
    out = []
    for p in paragraphs(tr.get("segments", [])):
        who = f"{p['speaker']}: " if p["speaker"] else ""
        out.append(f"[{ts(p['start'])}] {who}{p['text']}")
    return "\n\n".join(out) + "\n"


def srt(tr: dict) -> str:
    out = []
    for i, s in enumerate(tr.get("segments", []), 1):
        who = f"[{s['speaker']}] " if s.get("speaker") else ""
        out.append(f"{i}\n{srt_ts(s['start'])} --> {srt_ts(s['end'])}\n{who}{s['text'].strip()}\n")
    return "\n".join(out)


def transcript_for_llm(tr: dict) -> str:
    """Compact transcript fed to the summariser (one line per paragraph, with time and speaker)."""
    return "\n".join(
        f"[{ts(p['start'])}] {p['speaker'] + ': ' if p['speaker'] else ''}{p['text']}"
        for p in paragraphs(tr.get("segments", []))
    )
