# Whisper Minutes

Self-hosted speech-to-text + meeting summaries for Coolify.
Upload a recording (from the Idea Saver, a phone, or the web page) → get back a Markdown file with a title,
summary, key points, decisions, action items and the full timestamped transcript.

- **Transcription:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper), `large-v3-turbo` by default, voice-activity filter on
- **Summary:** local LLM through Ollama (bundled, `qwen2.5:7b-instruct`) or any OpenAI-compatible API
- **Speaker labels (optional):** pyannote, "Speaker 1 / Speaker 2"
- **Built for flaky WiFi:** resumable chunked uploads, idempotent recording ids, a queue that survives restarts
- Nothing leaves your server unless you point the summary at a cloud API

## Deploy on Coolify

1. **New Resource → Public/Private Repository** → `jeremiahng11/whisper` → Build Pack **Docker Compose**.
2. **Environment Variables:** set `API_KEY` to a long random string (`openssl rand -hex 32`).
   Everything else has defaults - see the table below.
3. Set a **domain** on the `whisper` service, e.g. `https://whisper.jeremiah.sg` (port 8000).
4. **Deploy.** First start downloads the Whisper model (~1.6 GB) and the Ollama model (~4.7 GB) into volumes,
   so give it a few minutes. `GET /api/health` shows `"model_loaded": true` when ready.
5. Open the domain in a browser, paste the API key, drop in an audio file.

Coolify uses named volumes, so nothing else is needed. (If you switch to a bind mount, make the folder
writable by uid 1000.)

**Cloudflare:** the free plan limits a single request to 100 MB. A 1-hour 16 kHz WAV is ~115 MB, so either
use the resumable upload below (chunks of a few MB - what the Idea Saver does), or turn off the proxy (grey cloud)
for this hostname.

### Speed on the GMKtec K8 Plus (CPU only)

Rough guide for `large-v3-turbo` int8: 1 hour of audio in ~15-30 min, then 1-3 min for the summary.
Set `WHISPER_MODEL=small` for ~4x faster with lower accuracy. With the RTX 3090: build with `GPU=1`,
uncomment the `deploy:` blocks in `docker-compose.yml` (needs the NVIDIA Container Toolkit on the host) -
an hour then takes about 1-2 min.

## Settings

| Variable | Default | What it does |
|---|---|---|
| `API_KEY` | *(required)* | Clients send `Authorization: Bearer <key>` (or `X-API-Key`) |
| `WHISPER_MODEL` | `large-v3-turbo` | `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`, `distil-large-v3` |
| `WHISPER_DEVICE` | `auto` | `cpu` or `cuda` |
| `WHISPER_THREADS` | `0` | CPU threads, 0 = all |
| `DEFAULT_LANGUAGE` | *(auto)* | e.g. `en`. Setting it avoids wrong-language detection on short clips |
| `INITIAL_PROMPT` | | Names and jargon to help spelling, e.g. `Aleta Planet, SkenPay, Coolify, MAS` |
| `SUMMARY_BACKEND` | `ollama` | `ollama`, `openai` (any compatible API: OpenAI, OpenRouter, vLLM, LM Studio...) or `none` |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Pulled automatically on first use |
| `OLLAMA_NUM_CTX` | `16384` | LLM context window; longer transcripts are summarised in parts automatically |
| `SUMMARY_LANGUAGE` | *(same as audio)* | e.g. `English` to always write minutes in English |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` | | Used when `SUMMARY_BACKEND=openai` |
| `DIARIZE` | `0` | Default for speaker labels (per upload with `?diarize=true`) |
| `INSTALL_DIARIZE` | `0` | Build arg: install pyannote. Needs `HF_TOKEN` and accepting the terms of `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` on huggingface.co |
| `KEEP_AUDIO_DAYS` | `30` | Delete audio after N days (results stay). 0 = keep |
| `KEEP_JOBS_DAYS` | `0` | Delete whole jobs after N days. 0 = keep |
| `MAX_UPLOAD_MB` | `2048` | Largest accepted file |
| `CALENDAR_ICS_URLS` | | Private calendar links for the device's Agenda app, comma separated (see below) |
| `CALENDAR_TZ` | `Asia/Singapore` | Time zone for all-day events |
| `SYNC_MAX_KB` | `1024` | Largest text file accepted by notes sync |

## API

All `/api/*` routes except `/api/health` need the key. Interactive docs: `/docs`.

### Simple upload (one request)

```bash
curl -H "Authorization: Bearer $KEY" -H "X-Filename: 2026-10-08_1535.wav" \
     --data-binary @2026-10-08_1535.wav "https://whisper.jeremiah.sg/api/jobs?language=en"
# or multipart:  curl -H "Authorization: Bearer $KEY" -F file=@meeting.m4a https://whisper.jeremiah.sg/api/jobs
```

Query options: `language` (`en`, `zh`, `ms`, ... or empty for auto), `summarize` (`true`/`false`),
`diarize` (`true`/`false`), `title`, `priority=high` (jump the queue - used for short dictation clips). Header `X-Recording-Id: <your id>` makes re-uploads return the same job
instead of a duplicate.

### Resumable upload (what the Idea Saver uses)

```
GET  /api/uploads/{rid}                      -> {"received": 1048576, "job": null}
PUT  /api/uploads/{rid}   X-Offset: 1048576  body = next chunk   -> {"received": 2097152}
                          (wrong offset -> 409 with the size the server has; continue from there)
POST /api/uploads/{rid}/complete?filename=2026-10-08_1535.wav&language=en
                          X-Total-Size: 115200044                -> job (201), or the existing job (200)
```

`rid` = the device's own id for the recording, e.g. `ideasaver1-2026-10-08_1535` (letters, digits, `. _ -`).
If a recording id already has a finished or queued job, every call just returns that job.
If its job failed, uploading again replaces it.

### Results

```
GET /api/jobs/{id}                 status: queued | transcribing | summarizing | done | error, progress 0..1
GET /api/recordings/{rid}          same, looked up by recording id
GET /api/jobs/{id}/result.md       minutes + transcript (also /api/recordings/{rid}/result.md)
GET /api/jobs/{id}/transcript.txt | transcript.srt | transcript.json | summary.md
POST /api/ask   {"messages":[{"role":"user","content":"..."}]}  -> {"id","status":"pending"}
GET  /api/ask/{id}                 -> {"status":"done","answer":"..."}  (Ask AI app on the device)
POST /api/jobs/{id}/summarize      redo the summary (e.g. after changing the LLM)
POST /api/jobs/{id}/retry          retry a failed job
DELETE /api/jobs/{id}
GET /api/health                    model status, queue counts (no key needed)
```

Example `result.md`:

```markdown
# Q4 roadmap sync

*2026-10-08 15:35 · 42 min · en · whisper large-v3-turbo*

### Summary
...
### Action items
- [ ] Speaker 2 - send the revised budget (Friday)

## Transcript

`[00:00]` **Speaker 1:** Okay, let's start with...
```

## Calendar (Agenda app on the Idea Saver)

Add your calendars' **private ICS links** to `CALENDAR_ICS_URLS` and redeploy:

- **Google Calendar:** calendar.google.com → Settings → pick the calendar → *Integrate calendar* → **Secret address in iCal format**.
- **Outlook / Microsoft 365:** outlook.office.com → Settings → Calendar → *Shared calendars* → **Publish a calendar** → pick "Can view all details" → copy the **ICS** link.
- **iCloud:** Calendar app → share the calendar → *Public Calendar* → copy the link (`webcal://` is fine).

`GET /api/agenda?days=7` returns `{"events":[{"title","start","end","all_day","location"}]}` (epoch seconds),
recurring meetings expanded, cancelled ones dropped, duplicates across calendars merged. Links are fetched at most every 5 min.

## Meeting notes

The device sends notes typed during a recording as a JSON body on `POST /api/uploads/{rid}/complete`:
`{"notes": "[0:02] Agreed to move go-live to Nov 3"}`. They are given to the summariser and appear as **My notes** in the minutes.

## Notes sync (two-way backup)

The Idea Saver copies its notes, journal, tasks, reminders and minutes here; the **Notes** panel on the web page
shows and edits them, and edits go back to the device on its next sync. Each file has a revision number:

```
GET    /api/files                         -> {"devices": [...]}
GET    /api/files?device=ID               -> {"files": [{"path","rev","hash","size","deleted","updated_at","updated_by"}]}
GET    /api/files/notes/Inbox.md?device=ID           -> content (headers X-Rev, X-Hash)
PUT    /api/files/notes/Inbox.md?device=ID&base_rev=N   body = content -> {"rev","hash"}; 409 if it changed since rev N
DELETE /api/files/notes/Inbox.md?device=ID&base_rev=N   -> {"rev","deleted":true}
```

`hash` is 32-bit FNV-1a (hex). The last 20 versions of each file are kept in the `file_history` table.
If both sides changed a file, the device keeps its version and saves the server's next to it as "(server copy)".

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q                                   # runs with a fake transcriber + fake LLM, no models needed
API_KEY=dev DATA_DIR=./data uvicorn app.main:app --reload
```
