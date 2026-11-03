import inspect, re, sys
sys.path.insert(0, ".")
import os
os.environ.setdefault("API_KEY", "x"); os.environ.setdefault("TRANSCRIBER", "fake")
from app import main
from fastapi.routing import APIRoute
app = main.app
groups = {}
order = []
def group(path):
    p = path.split("/")
    g = p[2] if len(p) > 2 and p[1] == "api" else (p[1] or "page")
    return {"jobs": "Meetings (jobs)", "watch": "Watch", "voice": "Voice & questions", "devices": "Devices & firmware",
            "knowledge": "Watch AI knowledge", "research": "Research", "sources": "Sources", "people": "People",
            "voices": "Known voices", "wakeword": "Wake word", "live": "Live recording", "health": "Server",
            "background": "Server", "stt": "Voice & questions", "intent": "Voice & questions", "tts": "Voice & questions", "ask": "Voice & questions", "vocab": "Vocabulary & corrections", "corrections": "Vocabulary & corrections", "glossary": "Vocabulary & corrections", "firmware": "Devices & firmware", "me": "Sign-in", "templates": "Meetings (jobs)", "search": "Meetings (jobs)", "recordings": "Uploads", "agenda": "Watch", "auth": "Sign-in", "login": "Sign-in", "passkey": "Sign-in"}.get(g, g.capitalize())
for r in app.routes:
    if not isinstance(r, APIRoute) or not r.include_in_schema:
        continue
    g = group(r.path)
    if g not in groups: groups[g] = []; order.append(g)
    deps = [d.call.__name__ for d in r.dependant.dependencies if getattr(d, "call", None)]
    doc = inspect.getdoc(r.endpoint) or ""
    params = []
    for p in r.dependant.query_params:
        default = "" if p.field_info.is_required() else f" = {p.field_info.default!r}"
        params.append(f"`{p.name}`{default}")
    for m in sorted(r.methods - {"HEAD"}):
        groups[g].append((m, r.path, "auth" in deps, doc, params))
out = [f"# Oracle Minutes API (server {main.VERSION})", "",
       "Base URL: `https://whisper.jeremiah.sg`. Interactive version (try calls in the browser): `/docs`; raw schema: `/openapi.json`.", "",
       "**Authentication.** Endpoints marked 🔒 need either the API key — `Authorization: Bearer <API_KEY>` or `X-API-Key: <API_KEY>` (the watch and scripts) — or a signed-in web session (the page; changes must come from the same site). Others are open.", "",
       "Bodies are JSON unless the endpoint takes an upload (multipart form or raw audio). Errors come back as `{\"detail\": \"...\"}` with a 4xx/5xx status.", ""]
out.append("## Contents\n")
for g in order:
    out.append(f"- [{g}](#{re.sub(r'[^a-z0-9 -]', '', g.lower()).replace(' ', '-')}) ({len(groups[g])})")
out.append("")
for g in order:
    out.append(f"## {g}\n")
    for m, path, a, doc, params in groups[g]:
        out.append(f"### `{m} {path}`{' 🔒' if a else ''}\n")
        if doc: out.append(doc.strip() + "\n")
        if params: out.append("Query: " + ", ".join(params) + "\n")
open("docs/API.md", "w").write("\n".join(out))
print(sum(len(v) for v in groups.values()), "endpoints,", len(order), "groups")
