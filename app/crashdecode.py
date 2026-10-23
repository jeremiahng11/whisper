"""The watch's crash reports, made readable: the addresses in its backtrace turned into function names.

Every firmware pushed to GitHub has a symbol list next to it (app/watch_files/firmware/syms/<version>.sym.gz: the
function addresses from that build's firmware.elf). A crash from that version is decoded here - no toolchain on this
server, and nothing to copy by hand from the watch's Settings > About."""
import bisect
import gzip
import logging
import re
import threading
import urllib.parse
from pathlib import Path
from typing import Optional

from .config import settings

log = logging.getLogger("crashdecode")
_cache: dict[str, Optional[tuple[list[int], list[tuple[int, str]]]]] = {}
_lock = threading.Lock()
BUNDLED = Path(__file__).parent / "watch_files" / "firmware" / "syms"
VER = re.compile(r"\b((?:W1B-)?R?\d+\.\d+\.\d+(?:\.\d+)?)\b")
ADDR = re.compile(r"0x(4[0-3][0-9a-fA-F]{6})\b")


def parse_syms(text: str) -> tuple[list[int], list[tuple[int, str]]]:
    """'4200a1b4 00000032 T loop' lines (nm -n -S) -> sorted starts, (size, name)."""
    rows = []
    for ln in text.splitlines():
        p = ln.split(None, 3)
        if len(p) < 4 or p[2] not in "tTwW":
            continue
        try:
            rows.append((int(p[0], 16), int(p[1], 16), p[3].strip()))
        except ValueError:
            continue
    rows.sort()
    return [r[0] for r in rows], [(r[1], r[2]) for r in rows]


def _fetch(version: str) -> Optional[bytes]:
    name = f"{version}.sym.gz"
    for d in (settings.data_dir / "syms", BUNDLED):
        if (d / name).exists():
            return (d / name).read_bytes()
    repo = settings.firmware_repo
    if not repo or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        return None
    try:
        from .devices import _gh_get
        base = f"https://api.github.com/repos/{repo}/contents/{urllib.parse.quote(settings.firmware_path)}"
        data = _gh_get(f"{base}/syms/{urllib.parse.quote(name)}?ref={urllib.parse.quote(settings.firmware_branch)}",
                       raw=True, timeout=30)
    except Exception as e:                                   # noqa: BLE001 - not there / no internet
        log.info("no symbols for %s: %s", version, e)
        return None
    try:
        (settings.data_dir / "syms").mkdir(parents=True, exist_ok=True)
        (settings.data_dir / "syms" / name).write_bytes(data)
    except OSError:
        pass
    return data


def symbols(version: str):
    with _lock:
        if version in _cache:
            return _cache[version]
    data = _fetch(version)
    table = None
    if data:
        try:
            table = parse_syms(gzip.decompress(data).decode("utf-8", "replace"))
        except (OSError, ValueError) as e:
            log.warning("symbols for %s unreadable: %s", version, e)
    with _lock:
        _cache[version] = table
    return table


def lookup(table, addr: int) -> str:
    starts, info = table
    i = bisect.bisect_right(starts, addr) - 1
    if i < 0:
        return "?"
    size, name = info[i]
    off = addr - starts[i]
    if size and off >= size + 64:                          # past the end of the nearest function: unknown
        return "?"
    return f"{name}+0x{off:x}"


def decode(text: str, fw: str = "") -> dict:
    """{"version", "frames": ["0x420... loop()+0x12", ...]} or {"error"} - the crash's own version first (the one that
    was running when it crashed), else the one that sent it."""
    m = VER.search(text or "")
    version = m.group(1) if m else (fw or "")
    if not version:
        return {"error": "no firmware version in the report"}
    table = symbols(version)
    if not table:
        return {"version": version, "error": f"no symbols for {version} (only firmware pushed from W1B-1.9.3 on has them)"}
    frames = [f"0x{a} {lookup(table, int(a, 16))}" for a in ADDR.findall(text)]
    return {"version": version, "frames": frames}
