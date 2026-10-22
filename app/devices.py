"""Watches as devices: their logs and status (sent every minute or so), and firmware updates they install over WiFi.

Logs:     data_dir/devlogs/<id>.log   (last ~512 KB, each line with the time it arrived)
          data_dir/devlogs/<id>.json  (latest status, last seen, last line, firmware)
Firmware: data_dir/firmware/<board>/<version>.bin + .json (sha256, size, uploaded_at, notes, active)
A firmware file says which board and version it is: the ASCII marker "ORACLEFW|<board>|<version>|" built into it,
so a file for one watch can never be offered to another (the Round's build boot-loops a watch1b)."""
import hashlib
import json
import os
import re
import threading
import time
import logging
from pathlib import Path
from typing import Optional

from .config import settings

log = logging.getLogger("devices")

DEV_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
BOARD = re.compile(r"^[a-z0-9_]{1,20}$")
VERSION = re.compile(r"^[A-Za-z0-9._-]{1,24}$")
MARK = re.compile(rb"ORACLEFW\|([a-z0-9_]{1,20})\|([A-Za-z0-9._-]{1,24})\|")
LOG_MAX = 512 * 1024
FW_MAX = 8 * 1024 * 1024
POST_EVERY_S = 60
_lock = threading.Lock()


# ---------------------------------------------------------------- logs + status
def _logdir() -> Path:
    d = settings.data_dir / "devlogs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def add_log(did: str, text: str, status: Optional[dict]) -> None:
    now = time.time()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime(now))
    lines = [ln.rstrip("\r") for ln in text.split("\n") if ln.strip()]
    d = _logdir()
    lp, mp = d / f"{did}.log", d / f"{did}.json"
    with _lock:
        if lines:
            with open(lp, "a", encoding="utf-8") as f:
                f.write("".join(stamp + ln[:2000] + "\n" for ln in lines))
            if lp.stat().st_size > LOG_MAX + LOG_MAX // 4:          # trim now and then, not on every post
                data = lp.read_bytes()[-LOG_MAX:]
                cut = data.find(b"\n")
                tmp = lp.with_suffix(".tmp")
                tmp.write_bytes(data[cut + 1:] if cut >= 0 else data)
                os.replace(tmp, lp)
        try:
            meta = json.loads(mp.read_text())
        except (OSError, ValueError):
            meta = {}
        meta["id"] = did
        meta["last_seen"] = now
        if lines:
            meta["last_line"] = lines[-1][:300]
        if status:
            meta["status"] = status
            if status.get("fw"):
                meta["fw"] = str(status["fw"])[:30]
            if status.get("board"):
                meta["board"] = str(status["board"])[:20]
        tmp = mp.with_suffix(".jtmp")
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, mp)


def list_devices() -> list[dict]:
    out = []
    for mp in _logdir().glob("*.json"):
        try:
            m = json.loads(mp.read_text())
        except (OSError, ValueError):
            continue
        out.append({"id": m.get("id") or mp.stem, "fw": m.get("fw", ""), "board": m.get("board", ""),
                    "last_seen": m.get("last_seen", 0), "status": m.get("status") or {},
                    "last_line": m.get("last_line", "")})
    out.sort(key=lambda x: x["last_seen"], reverse=True)
    return out


def read_log(did: str, lines: int) -> str:
    lp = _logdir() / f"{did}.log"
    if not lp.exists():
        return ""
    with _lock:
        data = lp.read_bytes()
    rows = data.decode("utf-8", "replace").splitlines()
    return "\n".join(rows[-lines:]) + ("\n" if rows else "")


def clear_log(did: str) -> None:
    with _lock:
        (_logdir() / f"{did}.log").unlink(missing_ok=True)


# ---------------------------------------------------------------- firmware
def _fwdir() -> Path:
    d = settings.data_dir / "firmware"
    d.mkdir(parents=True, exist_ok=True)
    return d


def vkey(v: str) -> tuple:
    """"W1B-1.10.2" -> (1, 10, 2): the numbers after any prefix, for comparing versions."""
    groups = re.findall(r"\d+(?:\.\d+)*", v or "")
    if not groups:
        return ()
    dotted = [g for g in groups if "." in g]
    return tuple(int(x) for x in (dotted or groups)[-1].split("."))


def identify(data: bytes) -> tuple[str, str]:
    """(board, version) of a firmware image, or ValueError saying why it can't be used."""
    if not data or data[0] != 0xE9:
        raise ValueError("not an ESP32 firmware image (.bin from .pio/build/<board>/firmware.bin)")
    m = MARK.search(data)
    if not m:
        raise ValueError("no ORACLEFW marker in this firmware - build it from the Oracle OS / Idea Saver source")
    return m.group(1).decode(), m.group(2).decode()


def save_firmware(data: bytes, notes: str = "") -> dict:
    if len(data) > FW_MAX:
        raise ValueError("firmware bigger than 8 MB")
    board, version = identify(data)
    d = _fwdir() / board
    d.mkdir(parents=True, exist_ok=True)
    meta = {"board": board, "version": version, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "uploaded_at": time.time(), "notes": (notes or "")[:500], "active": True}
    tmp = d / f"{version}.bin.tmp"
    tmp.write_bytes(data)
    os.replace(tmp, d / f"{version}.bin")
    (d / f"{version}.json").write_text(json.dumps(meta))
    prune_firmware(board)
    return meta


KEEP_FW = int(os.getenv("FIRMWARE_KEEP", "2") or 2)        # per board: the newest and the one before (to go back to)


def prune_firmware(board: str = "") -> list[str]:
    """Delete all but the newest KEEP_FW versions of each board's firmware. Returns what was deleted."""
    gone = []
    boards = {m["board"] for m in list_firmware()} if not board else {board}
    for b in boards:
        items = sorted((m for m in list_firmware() if m["board"] == b),
                       key=lambda m: (vkey(m["version"]), m.get("uploaded_at", 0)), reverse=True)
        for m in items[max(1, KEEP_FW):]:
            if delete_firmware(b, m["version"]):
                gone.append(f"{b}/{m['version']}")
                log.info("firmware %s/%s deleted (keeping the newest %d)", b, m["version"], KEEP_FW)
    return gone


def _meta(board: str, version: str) -> Optional[dict]:
    p = _fwdir() / board / f"{version}.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def list_firmware() -> list[dict]:
    out = []
    for p in _fwdir().glob("*/*.json"):
        try:
            m = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if (p.parent / f"{p.stem}.bin").exists():
            out.append(m)
    out.sort(key=lambda m: m.get("uploaded_at", 0), reverse=True)
    return out


def latest(board: str, current: str = "") -> Optional[dict]:
    items = [m for m in list_firmware() if m["board"] == board and m.get("active", True)]
    if not items:
        return None
    best = max(items, key=lambda m: (vkey(m["version"]), m.get("uploaded_at", 0)))
    cur = vkey(current)
    v = best["version"]
    # the watch reads the version from the first number on: "W1B-1.5.8" read as 1 (the 1 of W1B) on W1B-1.5.8 and
    # older - every version looked the same, so none was ever offered. Those get "1.5.8"; "name" has it in full.
    shown = v if re.match(r"^\D*\d+\.", v) else ".".join(str(x) for x in vkey(v)) or v
    return {"version": shown, "name": v, "size": best["size"], "sha256": best["sha256"],
            "url": f"/api/firmware/{board}/{best['version']}.bin",
            "newer": vkey(best["version"]) > cur if cur else True,
            "notes": best.get("notes", "")}


def firmware_path(board: str, version: str) -> Optional[Path]:
    p = _fwdir() / board / f"{version}.bin"
    return p if p.exists() else None


def set_active(board: str, version: str, active: bool) -> Optional[dict]:
    m = _meta(board, version)
    if not m:
        return None
    m["active"] = bool(active)
    (_fwdir() / board / f"{version}.json").write_text(json.dumps(m))
    return m


def delete_firmware(board: str, version: str) -> bool:
    d = _fwdir() / board
    found = (d / f"{version}.bin").exists()
    (d / f"{version}.bin").unlink(missing_ok=True)
    (d / f"{version}.json").unlink(missing_ok=True)
    return found


# ---------------------------------------------------------------- firmware that comes with Oracle Minutes
# app/watch_files/firmware/*.bin: the watch firmware built with this server version. Each new one is added to the
# firmware list when the server starts, so redeploying Oracle Minutes is enough for the watches to be offered the
# update (Settings > Software update on the watch). One you delete here is not added again.
def import_bundled() -> list[str]:
    from pathlib import Path as _P
    src = _P(__file__).parent / "watch_files" / "firmware"
    seen_p = _fwdir() / "bundled_seen.json"
    try:
        seen = set(json.loads(seen_p.read_text()))
    except (OSError, ValueError):
        seen = set()
    added = []
    for f in sorted(src.glob("*.bin")):
        try:
            data = f.read_bytes()
            board, version = identify(data)
        except (OSError, ValueError) as e:
            log.warning("bundled firmware %s: %s", f.name, e)
            continue
        key = f"{board}/{version}"
        if key in seen or _meta(board, version):
            seen.add(key)
            continue
        from .main import VERSION
        save_firmware(data, f"Comes with Oracle Minutes {VERSION}")
        seen.add(key)
        added.append(key)
        log.info("firmware %s added (comes with this server)", key)
    seen_p.write_text(json.dumps(sorted(seen)))
    return added



# ---------------------------------------------------------------- firmware from GitHub (no redeploy needed)
# The firmware .bin files pushed to the repo (FIRMWARE_REPO, FIRMWARE_BRANCH, FIRMWARE_PATH - by default this
# server's own repo, app/watch_files/firmware) are looked at every FIRMWARE_CHECK_MIN minutes, and again when a watch
# checks for updates (at most every 2 minutes). A file that changed on GitHub is downloaded and, if it's a version
# not here yet, added like an upload. So pushing a watch update to GitHub is enough. One you delete isn't added again.
gh_status: dict = {"checked": 0, "error": "", "added": [], "repo": ""}
_gh_lock = threading.Lock()


def _gh_get(url: str, raw: bool = False, timeout: float = 20) -> bytes:
    import urllib.request
    h = {"User-Agent": "OracleMinutes", "Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json"}
    if settings.github_token:
        h["Authorization"] = "Bearer " + settings.github_token
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
        return r.read(FW_MAX + 1)


def github_check(min_age: float = 0) -> list[str]:
    """Look at GitHub for new watch firmware now (unless looked within min_age seconds). Returns what was added."""
    import urllib.error
    import urllib.parse
    repo = settings.firmware_repo
    if not repo or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        return []
    if time.time() - gh_status["checked"] < min_age or not _gh_lock.acquire(blocking=False):
        return []
    added: list[str] = []
    try:
        gh_status.update(checked=time.time(), repo=f"{repo}@{settings.firmware_branch}/{settings.firmware_path}")
        base = f"https://api.github.com/repos/{repo}/contents/{urllib.parse.quote(settings.firmware_path)}"
        ref = "?ref=" + urllib.parse.quote(settings.firmware_branch)
        items = json.loads(_gh_get(base + ref))
        if not isinstance(items, list):
            raise ValueError("that path is a file, not a folder")
        shas_p = _fwdir() / "github_seen.json"
        seen_p = _fwdir() / "bundled_seen.json"           # (shared with the bundled ones: a deleted version stays gone)
        try:
            shas = json.loads(shas_p.read_text())
        except (OSError, ValueError):
            shas = {}
        try:
            seen = set(json.loads(seen_p.read_text()))
        except (OSError, ValueError):
            seen = set()
        for it in items:
            name, sha = it.get("name") or "", it.get("sha") or ""
            if it.get("type") != "file" or not name.endswith(".bin") or shas.get(name) == sha:
                continue
            if (it.get("size") or 0) > FW_MAX:
                continue
            data = _gh_get(f"{base}/{urllib.parse.quote(name)}{ref}", raw=True, timeout=120)
            try:
                board, version = identify(data)
            except ValueError as e:
                log.warning("github firmware %s: %s", name, e)
                shas[name] = sha
                continue
            key = f"{board}/{version}"
            if key not in seen and not _meta(board, version):
                save_firmware(data, f"From GitHub ({repo})")
                added.append(key)
                log.info("firmware %s added from GitHub", key)
            seen.add(key)
            shas[name] = sha
        shas_p.write_text(json.dumps(shas))
        seen_p.write_text(json.dumps(sorted(seen)))
        gh_status.update(error="", added=added or gh_status.get("added") or [])
    except urllib.error.HTTPError as e:
        gh_status["error"] = f"GitHub {e.code}" + (" - private repo? set GITHUB_TOKEN" if e.code == 404 and not settings.github_token else "")
        log.warning("github firmware: %s", gh_status["error"])
    except Exception as e:                                   # noqa: BLE001 - no internet: next time
        gh_status["error"] = str(e)[:200]
        log.warning("github firmware: %s", e)
    finally:
        _gh_lock.release()
    return added


def start_github_watcher() -> None:
    if not settings.firmware_repo or settings.firmware_check_min <= 0:
        return

    def loop():
        time.sleep(20)
        while True:
            github_check()
            time.sleep(max(5.0, settings.firmware_check_min) * 60)
    threading.Thread(target=loop, daemon=True, name="fw-github").start()
