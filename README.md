# Oracle Minutes

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

With the Docker Compose build pack Coolify uses the named volumes from `docker-compose.yml`, so nothing else is
needed. (If you switch to a bind mount, make the folder writable by uid 1000.)

**Using the Dockerfile build pack instead?** Then Coolify ignores `docker-compose.yml` and `/data` is a temporary
volume: every deploy starts empty (recordings, minutes, Names & terms, remembered voices, the downloaded model).
The web page shows a red warning and `/api/health` says `"storage": "temporary"`. Fix it once:
app → **Persistent Storage** → **+ Add** → *Volume*, name `whisper-data`, destination path `/data` → redeploy.
To keep what is there now, copy it out first (on the server):

```
C=$(docker ps --format '{{.Names}} {{.Image}}' | grep -i whisper | grep -vi ollama | head -1 | cut -d' ' -f1)
docker cp "$C":/data /root/whisper-backup                 # before adding the storage
# add Persistent Storage /data in Coolify, redeploy, then:
C=$(docker ps --format '{{.Names}} {{.Image}}' | grep -i whisper | grep -vi ollama | head -1 | cut -d' ' -f1)
docker cp /root/whisper-backup/. "$C":/data/ && docker restart "$C"
```

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
| `REGION` | (none) | `sg` = Singapore: Whisper and the minutes expect English mixed with Singlish, Mandarin and Malay, and fix mis-heard names like "Sun Jun" -> Shenzhen |
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
| `OLLAMA_MODEL_BEST` | | Bigger model for "Best quality" minutes, e.g. `qwen2.5:14b-instruct` (pulled on first use, ~9 GB). Empty = option off |
| `OPENAI_MODEL_BEST` | | Same, when `SUMMARY_BACKEND=openai` |
| `REDACT` | `1` | Hide NRIC/FIN, card numbers (Luhn-checked), IBANs and bank account numbers in transcripts and notes |
| `VOICE_MATCH` | `0.70` | How similar a voice must be to a known one to be named automatically (0-1) |
| `LIVE_MIN_NEW_S` | `240` | Live uploads: transcribe a part once this many seconds of new audio arrived |
| `TTS` / `TTS_VOICES` | `piper` / en + zh | Spoken answers for the round Idea Saver (see Voice API) |
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

## Better minutes (1.2)

- **Speaker names + voice memory.** Open a recording on the web page, type the names under *Who's speaking?* and save.
  The minutes and action items are rewritten with the names, and each voice is remembered: later recordings name
  those people automatically (badge "recognised"). `POST /api/jobs/{id}/speakers {"names": {"Speaker 1": "San"}}`,
  `GET /api/voices`, `DELETE /api/voices/{name}`. Needs speaker labels installed (`INSTALL_DIARIZE=1`).
- **Number of people.** `?speakers=3` on upload (web: *People*) - fixes one person being split in two.
- **Names & terms.** One per line on the web page (`GET/PUT /api/glossary`). Fed to Whisper (spelling) and to the LLM.
  Attendees from the device's Agenda (or `?attendees=San,Komei`) are added for that meeting.
- **Meeting types.** `?template=` `general`, `standup`, `client`, `one_on_one`, `interview`, `manco` (management committee: updates by department, approvals, risks and compliance, escalations, carried forward), `board` (formal minutes with
  RESOLVED THAT...), `memo`. Redo with another type: `POST /api/jobs/{id}/summarize {"template": "board", "best": true}`.
- **Best quality.** `?best=true` uses `OLLAMA_MODEL_BEST`.
- **Flagged moments.** Notes lines like `[12:03] !! budget` (the device's mark key) make the minutes cover what was
  said there, and appear under *Flagged moments*.
- **Action items as data.** `GET /api/jobs/{id}/actions.json` -> `[{"task","owner","due":"YYYY-MM-DD"}]` (relative dates
  resolved). The device adds them to Tasks and sets reminders for yours.
- **Ask.** `POST /api/ask` with `"recording": "<id>"` (one meeting), `"scope": "meetings"` (searches all of them), or
  `"recording": "<id>", "task": "email"` (follow-up email draft). `GET /api/search?q=` = full-text search.
- **Export.** `GET /api/jobs/{id}/result.docx` (Word); *Print / PDF* on the web page.
- **Redaction** of NRIC/card/account numbers (on by default).
- **Live upload.** The device uploads while it records (`PUT` with `X-Live: 1`); the server transcribes the parts as they
  arrive, so after you stop only the last minutes are left - minutes come much sooner.

## Voice API (round Idea Saver, 1.3)

- `POST /api/stt?language=` body = WAV (PCM or IMA ADPCM, up to ~2 min) -> `{"text","language","duration"}` right away
  (not queued) - used for voice commands, Ask AI and Translate.
- `POST /api/intent {"text","now","tz"}` -> the action to take: `reminder {text, at, repeat}`, `timer {seconds, label}`,
  `task {text}`, `note {text}`, `meeting {title}`, `translate {text, lang}` or `question`. Common phrasings ("remind me at 3
  to...", "timer 10 minutes", "add ... to my tasks") are parsed instantly with rules; the rest goes to the LLM.
- `POST /api/tts {"text","lang"}` -> 16 kHz WAV, spoken by [Piper](https://github.com/rhasspy/piper). Built in by default
  (`INSTALL_TTS=1`); voices download from huggingface.co on first use. `TTS_VOICES=en=en_US-lessac-medium,zh=zh_CN-huayan-medium`
  (any voice name from rhasspy/piper-voices). `TTS=none` turns it off.
- Recordings can be IMA ADPCM WAV (4-bit, 4x smaller) - the round device records that so ~50 min fit in its flash.
* `POST /api/stt?languages=en,zh` detects the language but only picks from the list (stops accented English
  being heard as Malay); `?language=en` forces one.

## AI answers not working? ("Name or service not known")

`GET /api/health` now shows `"llm": "ok"` or the reason the summary/answer model can't be reached.
"Name or service not known" means the container can't find the host in `OLLAMA_URL` (default `http://ollama:11434`,
the `ollama` service in this compose file). Fix one of:
* the `ollama` service isn't running in Coolify - start it (it pulls the model on first use);
* Ollama runs directly on the server - set `OLLAMA_URL=http://host.docker.internal:11434` (the compose file maps that
  name to the host) and start Ollama with `OLLAMA_HOST=0.0.0.0`;
* Ollama runs on another machine - set `OLLAMA_URL=http://<its-ip>:11434`.

## Who said what (1.3.7)

Open a recording on the web page:

* **Who's speaking?** - each speaker has a ▶ button that plays their longest line. Type names and *Save names*.
* **Who said what** - every transcript line with ▶ (play just that line) and a speaker menu. Pick who really said it,
  or *Someone else…* to add a person. With *Remember voices* on, that clip is added to the person's voice print, so
  later meetings name them automatically (and better each time you correct one). Filter to one speaker to check them.
* **Split speakers are joined up:** if the speaker separation split one person over several groups, a group whose
  voice matches someone you tagged becomes theirs (`AUTOTAG_GROUP`, default 0.50), then single lines that sound
  more like them than anyone else follow (`AUTOTAG_MATCH` 0.50, `AUTOTAG_MARGIN` 0.03). *Find their other lines*
  re-runs this for meetings tagged earlier. If voice prints can't be made on the server, the page says why.
* **People you've named before are found everywhere:** every voice you've named is remembered, and every meeting
  is checked against them - new meetings right after they're transcribed (`AUTOTAG_NEW`, default on), older ones in
  the background whenever the server is idle, one a minute, newest first, each time you've named or taught a new
  voice (`AUTOTAG_SWEEP`, default on). Opening a meeting on the web that hasn't been checked yet runs it there and
  then. Their lines get the name (marked *auto*) instead of *?*. A remembered voice that sounds like someone you
  tagged in that meeting never overrides your tag, and two remembered people with near-identical voices are left
  for you to pick.
* **Names & terms fix mis-heard words:** after each transcription, on *Redo minutes* and whenever you save Names &
  terms (all meetings, in the background), words that sound like one of them get your spelling ("C trip" ->
  CTRIP, "Pricilla" -> Priscilla) in the transcript and minutes; the line keeps what was heard (`heard`). Names of
  remembered voices count as Names & terms, and a name typed for a speaker takes the spelling from there (🔗 on
  the web = a name that's in both).
* **Meetings with no speakers:** if speaker separation failed when a meeting was transcribed (1.5.3-1.5.6 had a bug
  that broke it on real servers), the server separates the speakers again in the background when idle, or when
  you open the meeting (`POST /api/jobs/{id}/speakers/redo`). Lines you tagged keep their names; people with
  remembered voices are named, and your tags spread to the same voices. `/api/health` shows `speaker_separation`.
* **Voices lost, tags kept:** if the voices were lost (e.g. `/data` wasn't persistent on an old deploy), the server
  learns them again from the lines you tagged when it starts (`POST /api/voices/relearn` does it on demand).
  `/api/health` shows `voice_prints` - "ok", or why voice prints can't be made.
* **Same name, one speaker:** giving two speakers the same name in *Who's speaking* (any capitals) and saving joins
  them: their lines, tags and voice prints become one speaker, and that voice is looked for in the rest of the meeting.
* **✓ Confirm auto lines:** once you've checked the *auto* lines, one click marks them ✓ (yours - never changed
  automatically again) and adds them to those people's voice prints. Pick a speaker in the filter first to confirm
  only theirs (`POST /api/jobs/{id}/autotag/accept`, `{"name": "..."}` optional).
* **Sign-in:** set `ADMIN_ID` and `ADMIN_PASS` and the web page asks for them (signed 30-day cookie; 8 wrong tries
  = 15 minutes' wait; changing `ADMIN_PASS` signs everyone out). After a password sign-in on a device with Face ID /
  Touch ID / Windows Hello, it offers a passkey for one-touch sign-in next time (Mac: kept in iCloud Keychain, so the
  iPhone can use it; KeePassXC / 1Password can hold it too). Manage devices and sign out under *Account*. The API key
  keeps working for the watch and scripts. Behind a proxy the site address is taken from `X-Forwarded-*`; set
  `PUBLIC_URL=https://whisper.example.com` if passkeys complain about the address.
* **People:** everyone you've named gets a profile, built from their lines across all meetings and updated in the
  background when the server is idle (`PEOPLE_SWEEP`): up to 100 voice samples, spread over meetings (recognition
  uses their closest samples, not only an average - it gets better with every meeting), typical pitch and range,
  speaking pace, the words / phrases / Names & terms they use more than others, Singlish / Malay / Hokkien /
  Chinese mixed in, talk share in team vs external meetings, and an AI note from your own LLM (what they raise and
  own, how they communicate, how to work with them - work and communication only). The minutes writer and Ask get
  these profiles. `GET /api/people`, `GET /api/people/{name}`, `POST /api/people/{name}/refresh`; forgetting a voice
  removes the samples and profile.
* **Dynamics tab:** per meeting - talk share, turns, average and longest turn, questions, cutting in, who answered
  whom, who opened and closed - and an AI read of how it ran (who led, who drove what, open disagreements, and with
  a vendor / partner / client: who spoke for us and what the other side pushed for). Mark a meeting as external there.
  `GET/POST /api/jobs/{id}/dynamics` (`{"external": true}`, `{"write": true}`). The AI read is written by itself when
  the server is idle (recent 30 meetings, one at a time) and rewritten after names or lines change.
* **Built-in vocabularies** (Names page, `VOCAB_PACKS=all|payments,tech,finance|none`): about 280 payments /
  fintech (KYC, AML, PSP, VASP, PayNow, SGQR, chargeback, MDR ...), tech (API, UAT, OAuth, Kubernetes ...) and finance
  & compliance (MAS TRM, PDPA, EBITDA, P&L, STR, PEP ...) terms, spelled right in transcripts and minutes and given to
  the speech model as hints (the ones your meetings use most go first). Spelled-out letters ("k y c", "K.Y.C.")
  become the acronym. Your own Names & terms and corrections come first; everyday words (fast, bin, pin) are hints only.
* **Fix the words, the server learns:** click a line's words in *Who said what*, correct them, press Enter. The
  words you replaced are remembered (`TAPIN` -> `DABIN`, `B to B` -> `B2B`, `see trip` -> `CTRIP`) and fixed in every
  meeting, new ones as they're transcribed, and the right words are given to Whisper as hints. Everyday English
  words and grammar changes aren't learned; *Only this line* in the message undoes the learning of one edit. The list
  is under *Names & voices > Learned corrections* (`GET /api/corrections`, `DELETE /api/corrections/{heard}`), and
  it's in the backup. `POST /api/jobs/{id}/segments/{n}/text {"text": "...", "learn": true}`.
* **Voice check line:** under *Who said what* the page shows the last check of that meeting - who it compared
  with, how many lines it named, how many are still unnamed and why (close call between two voices, sounds like
  nobody you've named, too short). *Match more* accepts closer calls for that meeting (`POST /api/jobs/{id}/autotag?loose=true`);
  *Only unnamed (?)* in the filter lists what's left to tag.
* **Tag once, the voice follows:** tagging a line names its whole speaker group at once (the speaker separation
  already put that voice's lines together - unless you named that group as someone else yourself). Then a voice
  comparison picks up that person's lines that ended up in other groups. Lines you set show ✓, automatic ones
  *auto*. **Me** in the menu tags you (your name is asked once, `GET/PUT /api/me`).
* **Auto-tag:** after you tag a line, every other line of the meeting that clearly sounds like that person gets
  their name too (marked *auto*; lines you tagged are marked *you* and never changed automatically). The first time
  it listens to the whole meeting (a progress count shows), later tags are quick. Tag a couple of lines per person
  for best results; fix a wrong *auto* line by picking the right name, and it re-checks. Tuning: `AUTOTAG_MATCH`
  (default 0.60, higher = stricter), `AUTOTAG_MARGIN` (0.05). `"spread": false` in the API skips it.
* **Two people at once:** lines are split at the word where the speaker changes ("Ship Friday?" / "Yes, if QA
  passes" become two lines), and lines where someone talks over someone else get both names ("Alan + Wei Ling").
  The **+** menu on each line adds or removes a second person yourself. Such lines don't teach voice prints and
  auto-tag leaves them alone (a mixed voice isn't a fair sample of either person). Needs a new transcription
  (*Redo* on old recordings keeps their old lines).
* **Keep what you taught it:** remembered voices and Names & terms live in the database in `/data` - use
  Persistent Storage for `/data` (see Deploy). *Download backup* / *Restore backup* (Names & terms box) saves them to
  one file (`GET`/`POST /api/backup`) so they survive anything.
* Names change in the minutes right away; *Redo minutes* rewrites the wording with the corrected speakers.
* **Clearer audio** (on by default) plays lines with noise reduction, a rumble filter and steady volume, plus 0.3 s
  either side so words aren't cut off; untick it for the original. The speed menu slows playback (0.85x / 0.7x)
  without changing the pitch. The transcript is always made from the original recording.

API: `GET /api/jobs/{id}/clip?start=12.5&end=20` (WAV, at most 60 s) and
`POST /api/jobs/{id}/segments/{line}` with `{"name": "Alan", "learn": true}` or `{"speaker": "Speaker 2"}`.
The watch can name speakers too (meeting list → hold → *Name the speakers*).

## Quiet / far-away voices (1.4.1)

Before Whisper, recordings are levelled (2-second windows; quiet stretches lifted up to +18 dB, silence left alone;
`NORMALIZE=0` turns it off) and the speech detector keeps quieter speech (`VAD_THRESHOLD`, default 0.35; Silero's
default is 0.5 - lower keeps more, too low lets noise through as "words").

## Calendar (Agenda app on the Idea Saver)

Add your calendars' **private ICS links** to `CALENDAR_ICS_URLS` and redeploy:

- **Google Calendar:** calendar.google.com → Settings → pick the calendar → *Integrate calendar* → **Secret address in iCal format**.
- **Outlook / Microsoft 365:** outlook.office.com → Settings → Calendar → *Shared calendars* → **Publish a calendar** → pick "Can view all details" → copy the **ICS** link.
- **iCloud:** Calendar app → share the calendar → *Public Calendar* → copy the link (`webcal://` is fine).

`GET /api/agenda?days=7` returns `{"events":[{"title","start","end","all_day","location","attendees"}]}` (epoch seconds),
recurring meetings expanded, cancelled ones dropped, duplicates across calendars merged. Links are fetched at most every 5 min.


### Teams / Outlook (Microsoft 365) without a public link

If "Publish a calendar" isn't available (often switched off on work accounts), the server can read your calendar
through Microsoft Graph, read-only, signed in as an app:

1. **entra.microsoft.com** → *App registrations* → *New registration*: name `Whisper Minutes calendar`,
   single tenant, no redirect URI → *Register*. Copy the **Application (client) ID** and **Directory (tenant) ID**.
2. **Don't** add `Calendars.Read` under *API permissions* (an Entra grant there means *every* calendar in the
   company). The permission is given in Exchange instead, for your mailbox only (RBAC for Applications - it replaces
   Application Access Policies).
3. *Certificates & secrets* → *New client secret* (e.g. 24 months) → copy the **Value** (shown once).
4. Exchange Online PowerShell (`Connect-ExchangeOnline`, as Exchange admin). Take the IDs from
   *Enterprise applications* → `Whisper Minutes calendar` (the Object ID there, not the one in App registrations):
   ```
   New-ServicePrincipal -AppId <application id> -ObjectId <enterprise app object id> -DisplayName "Whisper Minutes calendar"
   New-ManagementScope -Name "Whisper - my mailbox" -RecipientRestrictionFilter "PrimarySmtpAddress -eq 'you@company.com'"
   New-ManagementRoleAssignment -App <application id> -Role "Application Calendars.Read" -CustomResourceScope "Whisper - my mailbox"
   Test-ServicePrincipalAuthorization -Identity <application id> -Resource you@company.com | Format-Table   # InScope True
   Test-ServicePrincipalAuthorization -Identity <application id> -Resource someone.else@company.com | Format-Table  # False
   ```
   It can take 30 minutes to 2 hours before it works.
5. In Coolify set `MS_TENANT_ID`, `MS_CLIENT_ID`, `MS_CLIENT_SECRET`, `MS_CALENDAR_USER=you@company.com`
   (and `CALENDAR_TZ=Asia/Singapore`), redeploy. `/api/health` shows `"calendars": 1`; errors from Microsoft show
   in `GET /api/agenda` under `errors`.

Meetings come with their real attendee names (rooms left out), cancelled ones are skipped. ICS links and
Microsoft 365 can be used together.

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

## Watch AI: company knowledge (1.12)

Oracle Minutes reads every finished meeting once more and keeps short **facts** (question + answer): decisions,
owners, dates, budgets, terms, and who people are. In the background it reads one new or changed meeting at a time
(`KNOWLEDGE_SWEEP=0` turns that off); **Account > Watch AI > Learn from meetings now** reads them all.

- **Ask** (all meetings) gets the facts that match the question, next to the meeting excerpts. Since 1.13.3 every
  other question too (the watch's "Hey Oracle ...", Ask on the web): when facts, people or meetings match, they go
  along, so "what is Aleta Planet" or "who is Ryan" is answered from your own knowledge.
- **Sources** besides meetings (1.13.5: read like a browser does - sites that answer only browsers, Zendesk Help
  Centers through their article API, and pages built by JavaScript through a headless Chromium; the image includes it,
  `INSTALL_BROWSER=0` leaves it out): a web page (and up to 60 pages under it - a product section), a PDF / Word / text /
  Markdown / CSV file, or pasted text. Read the same way; web pages are fetched again daily and re-read only when
  their text changed. Meeting facts win when both answer the same question.
- **News watch** (1.14): "Follow the news about…" a company, person or topic. Once a day the server searches the
  news (Bing News, else Google News; free, no key), reads the new articles (headless browser where needed), keeps
  60 days of them and turns them into dated facts ("On 14 October 2026 (The Business Times) ...").
- **Research on its own** (1.15, `app/research.py`): research topics ("Sanrio Aleta Adventure") are searched on the
  web (Bing, else DuckDuckGo) and in the news every week; only pages really about the topic are read (all its words),
  and the AI turns them into facts. With *find topics* on (`RESEARCH_AUTO`, default on) it also adds the company,
  product and brand names your facts keep mentioning (not people; a topic you remove isn't found again) and
  researches the questions nobody could answer, with your company's name (`RESEARCH_QUESTIONS` a day, default 4);
  when it finds the answer, the question is linked to it, and the watch model learns it at night. Ask does a quick
  live web search when the knowledge base knows nothing about a question (`RESEARCH_LIVE`, default on).
  API: `GET/PUT /api/research`, `POST/DELETE /api/research/topics`, `POST /api/research/now`.
- **Who is X?** (1.13.6) is written by the AI from everything known about X: lines of your pages and documents that
  name them (a leadership page), facts that mention them, and their meeting profile.
- **Fix / Hide** a fact on the Account page; your fix survives re-reading, a hidden fact is used nowhere.
- **The watch's offline answers**: `GET /api/watch/pack` gives ~150 short spoken questions (letters only, numbers as
  words, for Espressif MultiNet on the watch) with their answers.
- **The watch's own AI model**: **Build watch AI** trains a small transformer (esp32-ai style, ~2-4 MB int8, whole words
  as tokens) on all the facts, here on this server: plain numpy, nothing extra to install, low priority on half the CPU
  cores (`WATCHAI_THREADS`, longest run `WATCHAI_MINUTES`, default 120). The result shows how many questions it
  answers right, including ones worded in a way it never saw. The watch downloads `GET /api/watch/model/file`.
  It also trains again by itself between 02:00 and 05:00 when the facts changed (`WATCHAI_NIGHTLY=0`: only by hand).
  With more facts it trains a bigger model (1,500+ facts: d=192), as long as the watch can hold it (4.5 MB).
- **Learns how you ask** (1.13): every question you ask (Ask on the web or the watch, and the watch's offline
  questions) is kept; when it clearly matches one fact, your wording becomes one of that fact's wordings (the watch
  model trains on it, the watch can hear it) and the facts you ask most come first in the watch pack.
- **Spoken answers offline** (1.13): the pack's answers are read out once with the server's voice (Piper) and the
  watch keeps them (IMA ADPCM, ~8 KB a second) to say the answer without the internet.

| | |
|---|---|
| `GET /api/knowledge?q=` | facts (all, or the best matches for a question) + counts |
| `POST /api/knowledge/learn` `{"all": true}` | read new/changed (or all) meetings in the background |
| `PUT /api/knowledge/{id}` | `{"a": "..."}` fix, `{"hidden": true}` hide, `{}` undo |
| `GET /api/watch/pack?extra=1` | offline answer pack for the watch (+ more questions for its own model) |
| `GET /api/watch/voice/{id}` | a pack answer read out (WAV, made once) |
| `POST /api/watch/asked` | `{"items": [{"id", "q"}]}` questions asked on the watch offline |
| `GET /api/watch/speech` | the watch's offline speech model (Espressif MultiNet7 English, 2.7 MB, once) |
| `GET` / `POST /api/knowledge/sources`, `DELETE`, `POST .../{id}/refresh` | your web pages, files and texts |
| `GET /api/watch/model`, `POST` / `DELETE /api/watch/model/train`, `GET /api/watch/model/file` | the watch model |

Train by hand (same thing the button runs): `python -m app.watchlm train $DATA_DIR/watchai` (reads `dataset.json` there).

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q                                   # runs with a fake transcriber + fake LLM, no models needed
API_KEY=dev DATA_DIR=./data uvicorn app.main:app --reload
```
