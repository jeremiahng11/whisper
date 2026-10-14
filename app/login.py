"""Sign-in for the web page: ADMIN_ID + ADMIN_PASS (environment), then optionally a passkey - Face ID, Touch ID,
Windows Hello or a phone - so the next sign-in is one touch. The API key keeps working for the watch and scripts.

Sessions are signed cookies (HttpOnly, SameSite=Lax, Secure on https), 30 days. Passkeys are stored in
/data/passkeys.json (public keys only - the private key never leaves your device)."""
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

from .config import settings

log = logging.getLogger("login")
router = APIRouter()
COOKIE = "om_session"
DAYS = 30
_lock = threading.Lock()
_fails: dict[str, list[float]] = {}
_challenges: dict[str, tuple[bytes, float, str]] = {}     # token -> (challenge, expires, kind)


def enabled() -> bool:
    return bool(settings.admin_id and settings.admin_pass)


def _secret() -> bytes:
    if settings.session_secret:
        return settings.session_secret.encode()
    p = settings.data_dir / "session.key"
    with _lock:
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(secrets.token_hex(32))
            os.chmod(p, 0o600)
        return p.read_text().strip().encode()


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_session(user: str) -> str:
    # the password's fingerprint is part of the session: changing ADMIN_PASS signs everyone out
    pw = hashlib.sha256(settings.admin_pass.encode()).hexdigest()[:12]
    body = f"{user}|{int(time.time()) + DAYS * 86400}|{pw}"
    sig = hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()
    return _b64(body.encode()) + "." + sig


def session_user(request: Request) -> Optional[str]:
    tok = request.cookies.get(COOKIE) or ""
    if "." not in tok:
        return None
    b, sig = tok.rsplit(".", 1)
    try:
        body = _unb64(b).decode()
    except Exception:                                       # noqa: BLE001
        return None
    if not hmac.compare_digest(hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest(), sig):
        return None
    user, exp, pw = (body.split("|") + ["", "", ""])[:3]
    if int(exp or 0) < time.time() or pw != hashlib.sha256(settings.admin_pass.encode()).hexdigest()[:12]:
        return None
    return user


def same_origin(request: Request) -> bool:
    """A signed-in browser's changes must come from this site (no other site can post on your behalf)."""
    o = request.headers.get("origin") or request.headers.get("referer") or ""
    if not o:
        return True
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    return o.split("://", 1)[-1].split("/", 1)[0] == host.split(",")[0].strip()


def _site(request: Request) -> tuple[str, str]:
    """(rp_id, origin) of this server as the browser sees it (behind Coolify / Cloudflare)."""
    if settings.public_url:
        u = settings.public_url.rstrip("/")
        return u.split("://", 1)[-1].split("/")[0].split(":")[0], u
    host = (request.headers.get("x-forwarded-host") or request.headers.get("host") or "localhost").split(",")[0].strip()
    # the browser's own address (Origin) is the truth: behind Cloudflare / Coolify the request often arrives as
    # plain http while the page is https. Only an Origin for this same host is believed.
    o = (request.headers.get("origin") or "").rstrip("/")
    if o.split("://", 1)[-1] == host and o.split("://", 1)[0] in ("http", "https"):
        return host.split(":")[0], o
    proto = (request.headers.get("cf-visitor") and "https" in request.headers.get("cf-visitor", "") and "https") or \
        (request.headers.get("x-forwarded-proto") or request.url.scheme).split(",")[0].strip()
    return host.split(":")[0], f"{proto}://{host}"


def _set_cookie(resp: Response, request: Request, user: str) -> None:
    ref = request.headers.get("origin") or request.headers.get("referer") or ""
    secure = ref.startswith("https://") or _site(request)[1].startswith("https")
    resp.set_cookie(COOKIE, make_session(user), max_age=DAYS * 86400, httponly=True, samesite="lax", secure=secure, path="/")


def _ip(request: Request) -> str:
    return (request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for") or
            (request.client.host if request.client else "?")).split(",")[0].strip()


def _too_many(ip: str) -> bool:
    now = time.time()
    with _lock:
        _fails[ip] = [t for t in _fails.get(ip, []) if now - t < 900]
        return len(_fails[ip]) >= 8


def _failed(ip: str) -> None:
    with _lock:
        _fails.setdefault(ip, []).append(time.time())


# ---------------------------------------------------------------- passkeys (Face ID / Touch ID / Windows Hello)
def _keys_path():
    return settings.data_dir / "passkeys.json"


def passkeys() -> list[dict]:
    try:
        return json.loads(_keys_path().read_text()) if _keys_path().exists() else []
    except ValueError:
        return []


def _save_keys(keys: list[dict]) -> None:
    _keys_path().parent.mkdir(parents=True, exist_ok=True)
    _keys_path().write_text(json.dumps(keys, indent=1))
    os.chmod(_keys_path(), 0o600)


def _challenge(kind: str) -> tuple[str, bytes]:
    tok, ch = secrets.token_urlsafe(16), secrets.token_bytes(32)
    now = time.time()
    with _lock:
        for k in [k for k, v in _challenges.items() if v[1] < now]:
            _challenges.pop(k, None)
        _challenges[tok] = (ch, now + 300, kind)
    return tok, ch


def _take_challenge(tok: str, kind: str) -> bytes:
    with _lock:
        v = _challenges.pop(tok or "", None)
    if not v or v[1] < time.time() or v[2] != kind:
        raise HTTPException(400, "the sign-in request expired - try again")
    return v[0]


# ---------------------------------------------------------------- routes
@router.get("/login", response_class=HTMLResponse, include_in_schema=False)
def login_page(request: Request):
    if not enabled() or session_user(request):
        return RedirectResponse("/", 303)
    return (settings_static() / "login.html").read_text(encoding="utf-8")


def settings_static():
    from pathlib import Path
    return Path(__file__).parent / "static"


@router.get("/api/session")
def session_info(request: Request):
    return {"login": enabled(), "user": session_user(request), "passkeys": len(passkeys())}


@router.post("/api/login")
async def login(request: Request):
    if not enabled():
        raise HTTPException(400, "sign-in isn't set up (ADMIN_ID / ADMIN_PASS)")
    ip = _ip(request)
    if _too_many(ip):
        raise HTTPException(429, "too many tries - wait 15 minutes")
    data = await request.json()
    uid, pw = str(data.get("id") or ""), str(data.get("password") or "")
    ok = hmac.compare_digest(uid.lower().encode(), settings.admin_id.lower().encode()) & \
        hmac.compare_digest(pw.encode(), settings.admin_pass.encode())
    if not ok:
        _failed(ip)
        log.warning("failed sign-in from %s", ip)
        raise HTTPException(401, "wrong ID or password")
    resp = Response(json.dumps({"ok": True, "passkeys": len(passkeys())}), media_type="application/json")
    _set_cookie(resp, request, settings.admin_id)
    return resp


@router.post("/api/logout")
def logout():
    resp = Response(json.dumps({"ok": True}), media_type="application/json")
    resp.delete_cookie(COOKIE, path="/")
    return resp


@router.post("/api/passkey/register/options")
def passkey_register_options(request: Request):
    import webauthn
    from webauthn.helpers.structs import (AuthenticatorSelectionCriteria, PublicKeyCredentialDescriptor,
                                          ResidentKeyRequirement, UserVerificationRequirement)
    if not session_user(request):
        raise HTTPException(401, "sign in with your password first")
    rp_id, _ = _site(request)
    tok, ch = _challenge("reg")
    opts = webauthn.generate_registration_options(
        rp_id=rp_id, rp_name="Oracle Minutes", user_name=settings.admin_id, user_display_name=settings.admin_id,
        user_id=hashlib.sha256(settings.admin_id.encode()).digest()[:16], challenge=ch,
        authenticator_selection=AuthenticatorSelectionCriteria(resident_key=ResidentKeyRequirement.PREFERRED,
                                                               user_verification=UserVerificationRequirement.REQUIRED),
        exclude_credentials=[PublicKeyCredentialDescriptor(id=_unb64(k["id"])) for k in passkeys()])
    return {"token": tok, "options": json.loads(webauthn.options_to_json(opts))}


@router.post("/api/passkey/register")
async def passkey_register(request: Request):
    import webauthn
    if not session_user(request):
        raise HTTPException(401, "sign in with your password first")
    data = await request.json()
    rp_id, origin = _site(request)
    try:
        v = webauthn.verify_registration_response(credential=data["credential"], expected_challenge=_take_challenge(data.get("token"), "reg"),
                                                  expected_rp_id=rp_id, expected_origin=origin, require_user_verification=True)
    except HTTPException:
        raise
    except Exception as e:                                  # noqa: BLE001
        raise HTTPException(400, f"couldn't add this passkey: {e}")
    keys = passkeys()
    keys.append({"id": _b64(v.credential_id), "public_key": _b64(v.credential_public_key), "sign_count": v.sign_count,
                 "name": str(data.get("name") or "This device")[:60], "created": int(time.time()), "last_used": None})
    _save_keys(keys)
    return {"ok": True, "passkeys": len(keys)}


@router.post("/api/passkey/login/options")
def passkey_login_options(request: Request):
    import webauthn
    from webauthn.helpers.structs import PublicKeyCredentialDescriptor, UserVerificationRequirement
    if not enabled() or not passkeys():
        raise HTTPException(400, "no passkeys yet - sign in with your password and add one")
    rp_id, _ = _site(request)
    tok, ch = _challenge("auth")
    opts = webauthn.generate_authentication_options(
        rp_id=rp_id, challenge=ch, user_verification=UserVerificationRequirement.REQUIRED,
        allow_credentials=[PublicKeyCredentialDescriptor(id=_unb64(k["id"])) for k in passkeys()])
    return {"token": tok, "options": json.loads(webauthn.options_to_json(opts))}


@router.post("/api/passkey/login")
async def passkey_login(request: Request):
    import webauthn
    ip = _ip(request)
    if _too_many(ip):
        raise HTTPException(429, "too many tries - wait 15 minutes")
    data = await request.json()
    cred = data.get("credential") or {}
    keys = passkeys()
    k = next((k for k in keys if k["id"] == cred.get("id") or k["id"] == cred.get("rawId")), None)
    if not k:
        _failed(ip)
        raise HTTPException(401, "this passkey isn't registered here")
    rp_id, origin = _site(request)
    try:
        v = webauthn.verify_authentication_response(
            credential=cred, expected_challenge=_take_challenge(data.get("token"), "auth"), expected_rp_id=rp_id,
            expected_origin=origin, credential_public_key=_unb64(k["public_key"]),
            credential_current_sign_count=k.get("sign_count", 0), require_user_verification=True)
    except HTTPException:
        raise
    except Exception as e:                                  # noqa: BLE001
        _failed(ip)
        raise HTTPException(401, f"passkey check failed: {e}")
    k["sign_count"], k["last_used"] = v.new_sign_count, int(time.time())
    _save_keys(keys)
    resp = Response(json.dumps({"ok": True}), media_type="application/json")
    _set_cookie(resp, request, settings.admin_id)
    return resp


@router.get("/api/passkeys")
def list_passkeys(request: Request):
    if not session_user(request):
        raise HTTPException(401, "sign in first")
    return {"passkeys": [{"id": k["id"], "name": k.get("name"), "created": k.get("created"), "last_used": k.get("last_used")}
                         for k in passkeys()]}


@router.delete("/api/passkeys/{kid}")
def delete_passkey(kid: str, request: Request):
    if not session_user(request) or not same_origin(request):
        raise HTTPException(401, "sign in first")
    keys = [k for k in passkeys() if k["id"] != kid]
    _save_keys(keys)
    return {"passkeys": len(keys)}
