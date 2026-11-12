# Oracle Minutes API (server 1.47.0)

Base URL: `https://whisper.jeremiah.sg`. Interactive version (try calls in the browser): `/docs`; raw schema: `/openapi.json`.

**Authentication.** Endpoints marked 🔒 need either the API key — `Authorization: Bearer <API_KEY>` or `X-API-Key: <API_KEY>` (the watch and scripts) — or a signed-in web session (the page; changes must come from the same site). Others are open.

Bodies are JSON unless the endpoint takes an upload (multipart form or raw audio). Errors come back as `{"detail": "..."}` with a 4xx/5xx status.

## Contents

- [Meetings (jobs)](#meetings-jobs) (19)
- [Uploads](#uploads) (6)
- [Voice & questions](#voice--questions) (8)
- [Watch AI knowledge](#watch-ai-knowledge) (12)
- [Server](#server) (3)
- [Research](#research) (5)
- [Watch](#watch) (14)
- [Wake word](#wake-word) (8)
- [Vocabulary & corrections](#vocabulary--corrections) (6)
- [Known voices](#known-voices) (3)
- [Sign-in](#sign-in) (2)
- [Backup](#backup) (2)
- [People](#people) (6)
- [Files](#files) (4)
- [Live recording](#live-recording) (2)
- [Devices & firmware](#devices--firmware) (12)

## Meetings (jobs)

### `POST /api/jobs` 🔒

Send the audio as the raw request body (Content-Type audio/wav etc.) or as multipart field `file`.

Query: `language` = None, `summarize_` = True, `diarize` = None, `title` = None, `priority` = '', `template` = None, `speakers` = None, `best` = None, `redact_` = None, `attendees` = None, `minutes_lang` = None

### `GET /api/jobs` 🔒

Query: `limit` = 50, `status` = None

### `GET /api/jobs/{jid}` 🔒

### `GET /api/jobs/{jid}/{name}` 🔒

Query: `start` = 0.0, `end` = 0.0, `enhance` = True, `analysis` = False

### `POST /api/jobs/{jid}/summarize` 🔒

Write the minutes again, optionally with {"template": "...", "best": true}. Runs in the background.

### `POST /api/jobs/{jid}/title` 🔒

Rename the meeting: {"title": "CTRIP dashboard review"}. Kept when the minutes are redone; "" = back to the
title the minutes were written with.

### `POST /api/jobs/{jid}/retry` 🔒

### `DELETE /api/jobs/{jid}` 🔒

### `POST /api/jobs/{jid}/speakers` 🔒

{"names": {"Speaker 1": "San", "Speaker 2": ""}, "remember": true} - "" goes back to the label.
remember = keep their voices so later recordings are named automatically.

### `POST /api/jobs/{jid}/segments/{idx}` 🔒

Say who really spoke one transcript line: {"name": "Alan", "learn": true} or {"speaker": "Speaker 2"}.
A name used in this meeting goes to that speaker; a new name becomes a new speaker. learn = also add this
clip to that person's voice print, so later meetings recognise them better.
{"also": "Wei Ling"} = a second person talking at the same time ("" = nobody); can be sent with or without a name.

### `POST /api/jobs/{jid}/segments/{idx}/text` 🔒

Fix the words of one line: {"text": "...", "learn": true}. learn = remember the words you changed (heard ->
right) and fix the same mistake in every meeting, and in new ones as they're transcribed.

### `POST /api/jobs/{jid}/fill` 🔒

Nothing was written down between start and end: {"start": 242, "end": 511} transcribes that stretch again
with a more sensitive speech detector, adds what it finds, then runs the voice check on the new lines.

### `POST /api/jobs/{jid}/autotag/accept` 🔒

The "auto" lines are right: mark them ✓ (yours). {"name": "Priscilla"} = only hers.

### `POST /api/jobs/{jid}/autotag` 🔒

Look again for lines that sound like the people you tagged here or anyone remembered from other meetings.

Query: `loose` = None

### `GET /api/jobs/{jid}/speakers` 🔒

### `POST /api/jobs/{jid}/speakers/redo` 🔒

Separate the speakers again (e.g. it failed when the meeting was transcribed). Your tagged lines stay.

### `POST /api/jobs/{jid}/dynamics` 🔒

{"external": true|false} = was this with a vendor / partner / client; {"write": true} = (re)write the AI
read of how the meeting ran (background; poll GET .../dynamics).

### `GET /api/search` 🔒

Search your meetings (not the web): the words in q across every transcript and its minutes, best matches
first, at most 4 per meeting. On this server only (SQLite full-text index) - milliseconds, no Brave / Tavily.
-> {"hits": [{"job_id", "recording_id", "title", "date", "start" (seconds, -1 = in the minutes), "speaker",
"text"}]}. Example: /api/search?q=budget%20Q4&limit=10

Query: `q`, `limit` = 20

### `GET /api/templates`

## Uploads

### `GET /api/uploads/{rid}` 🔒

How many bytes the server already has -> continue the upload from there.

### `PUT /api/uploads/{rid}` 🔒

Append the request body at byte X-Offset. Wrong offset -> 409 with the size the server has.

Live (a meeting transcribed while it is still being recorded): send the pieces with header X-Live: 1. The first
piece starts with a WAV header (16 kHz mono: PCM 16-bit, IMA ADPCM, or the watch's Opus), then keep appending
audio as it is recorded (any size; the watch sends 4-32 KB every few seconds). The server transcribes each new
part as it arrives (GET /api/live shows the transcript so far) and every reply carries "brief": the last line
heard, who said it, who has talked. Nothing new to send (silence skipped)? Send an empty body at the same offset
with X-Live-State (e.g. "quiet", "held up") and optional X-Live-Why, so the page doesn't show "Paused". When the
recording stops: POST /api/uploads/{rid}/complete - the parts already transcribed are kept, so the minutes come
fast. ?language=en fixes the language for the live transcript.

Query: `language` = None

### `POST /api/uploads/{rid}/complete` 🔒

Query: `language` = None, `summarize_` = True, `diarize` = None, `title` = None, `filename` = None, `priority` = '', `template` = None, `speakers` = None, `best` = None, `redact_` = None, `attendees` = None, `minutes_lang` = None

### `DELETE /api/uploads/{rid}` 🔒

The device threw a recording away (a meeting stopped within 10 s - an accidental start): what arrived of it,
and its live transcript, go too. A recording that already became a meeting isn't touched.

### `GET /api/recordings/{rid}` 🔒

Look a job up by the device's own recording id.

### `GET /api/recordings/{rid}/{name}` 🔒

## Voice & questions

### `POST /api/ask` 🔒

{"messages": [...]} -> {"id", "status": "pending"}; poll GET /api/ask/{id}.
Optional: "recording": <recording or job id> (ask about one meeting), "scope": "meetings" (search all
meetings), "task": "email" (draft a follow-up email for "recording"), "best": true, "speak": true (the answer
will be read out: its voice is made as soon as it's ready). ?wait=<s> (up to 15): answer in this reply if it's
ready by then (one request instead of several polls).

Query: `wait` = 0

### `GET /api/ask/{aid}` 🔒

?wait=<s> (up to 15): reply as soon as the answer is ready (or after that long).

Query: `wait` = 0

### `GET /api/voice/pick` 🔒

Which voice answers: {"voice": "eleven" | "cedar" | "coral", "choices": [...]} (Settings > Voice on the watch).

### `POST /api/voice/pick` 🔒

### `POST /api/stt` 🔒

Short clip (WAV, up to ~2 min) -> {"text", "language", "duration"} right away, outside the job queue.
language=en forces one language; languages=en,zh detects but only picks from that list.

Query: `language` = None, `languages` = None

### `POST /api/voice` 🔒

A spoken question in one trip (the watch): body = the recording (WAV); header X-Ask = base64 of JSON
{"messages": [system + the conversation so far]}. The reply is a stream of lines, as each part is ready:
  T {"text", "language"}   what you said (at once - the watch may answer it itself and hang up)
  X {"type", ...}          a reminder / timer / task / note / meeting / translation to do on the watch (then E)
  P {"text"}               the answer so far, a sentence at a time
  B <base64>               its voice: IMA ADPCM blocks (16 kHz, 256-byte blocks), playable as they come
  A {"answer"}             the whole answer
  E {} / E {"error"}       the end
The voice of the first sentence comes while the AI is still writing the rest.

A typed question (no recording) gets the same fast stream: Content-Type application/json, body
{"text": "the question", "messages": [system + the conversation so far]} - then there's no T line's speech to
text, and the rest is as above (instead of /api/ask + /api/tts, which wait for the whole answer, then its whole
voice).

Query: `language` = None, `speak` = 1, `tz` = None, `city` = '', `scope` = '', `intents` = 1

### `POST /api/intent` 🔒

{"text": "remind me at 3 to call San", "now": <epoch>, "tz": <minutes from UTC>} -> an action:
reminder {text, at, repeat} | timer {seconds, label} | task {text} | note {text} | meeting {title} |
translate {text, lang} | question.

### `POST /api/tts` 🔒

{"text": "...", "lang": "en", "format": "adpcm"} -> audio/wav (16 kHz mono; "adpcm" = IMA ADPCM, 4x smaller,
what the round Idea Saver asks for). 503 when no TTS is installed.

## Watch AI knowledge

### `GET /api/knowledge` 🔒

What the knowledge base knows: {"facts": n, "kinds", "meetings_read", "state", ..., "items": [...]}.
?q= returns the facts best matching a question.

Query: `q` = '', `limit` = 500

### `POST /api/knowledge/learn` 🔒

Read the meetings that are new or changed (or all of them with {"all": true}) in the background.

### `GET /api/knowledge/sources` 🔒

Your web pages, files and texts the knowledge base learns from.

### `POST /api/knowledge/sources` 🔒

{"url": "https://...", "crawl": 20} (the page + up to 20 more under it), {"news": "Aleta Planet"} (follow the
news about it: searched and read daily), or {"text": "...", "title": ""};
or a file (PDF, Word, text, Markdown, CSV, HTML) as multipart field "file". Read by the AI in the background.

### `DELETE /api/knowledge/sources/{sid}` 🔒

### `POST /api/knowledge/sources/{sid}/refresh` 🔒

Fetch / read it again now (the AI re-reads it even if the text is the same).

### `PUT /api/knowledge/auto` 🔒

{"on": false} - stop learning by itself (facts from new meetings, who-is, company profile, re-reading pages,
research, watch-model training); {"on": true} - start again. The buttons still work either way.

### `POST /api/knowledge/core` 🔒

Write the company profile (the essentials always on the watch) again now, from everything known.

### `PUT /api/knowledge/{fid}` 🔒

{"a": "the right answer"} or {"hidden": true} (wrong or private fact) or {} (back to what the AI read).

### `POST /api/knowledge/correct` 🔒

"Edit this answer": {"q": "who is ryan", "a": "the right answer"} -> the matching fact gets it (or it becomes
your own question). Oracle on the web and the watch answer with it from then on.

### `POST /api/knowledge/custom` 🔒

Your own question for the watch: {"q": "Who is Oracle AI", "a": "...", "alts": ["other ways to ask"]} - always
on the watch and answered there.

### `DELETE /api/knowledge/custom/{fid}` 🔒

## Server

### `GET /api/background` 🔒

### `PUT /api/background` 🔒

{"mode": "on" | "night" | "off", "from": 1, "to": 7} - everything the server does by itself (learning,
research, watch AI training, voices, profiles, search by meaning): always, only in the quiet hours, or never.
Meetings you upload, the watch's questions and the buttons still work at once.

### `GET /api/health`

Up or not (for Coolify / a monitor) - without the key or a sign-in only {ok, version, model_loaded}: the rest
shows your recent questions, meeting titles and what is running (it was open to anyone).

## Research

### `GET /api/research` 🔒

Topics researched on the web (yours and the ones it found), questions it researched, what's running.

### `PUT /api/research` 🔒

{"auto": false} - stop finding topics and researching unanswered questions by itself.

### `POST /api/research/topics` 🔒

{"topic": "Sanrio Aleta Adventure"} - research it on the web (now, then weekly).

### `DELETE /api/research/topics` 🔒

Stop researching it (a topic it found isn't found again); what it learned goes too.

Query: `topic`

### `POST /api/research/now` 🔒

Find topics, read every topic again and research the questions nobody could answer - now.

## Watch

### `GET /api/watch/pack` 🔒

The watch's offline answers: {"version", "count", "items": [{"id", "q", "say": [phrases], "a", ...}]};
extra=1 also lists more questions (no answers) for the watch's own model: "extra": [{"id", "q", "say"}].

Query: `limit` = 150, `extra` = False

### `GET /api/watch/voice/{fid}` 🔒

A pack answer read out (IMA ADPCM WAV), for the watch to play offline. Made on first request, then kept.

### `POST /api/watch/crash` 🔒

The watch's last crash (task, address, backtrace) after it restarted: {"fw", "text"} - kept to look into.

### `GET /api/watch/crashes` 🔒

The watch's crash reports, newest first.

### `GET /api/watch/weather` 🔒

Today's and tomorrow's weather as spoken sentences, kept on the watch (every 2 hours) so "what's the weather"
is answered even offline.

Query: `city` = ''

### `GET /api/watch/prompts` 🔒

The watch's own spoken words ("Okay.", "Starting the meeting."), in the answers' voice: {"voice", "names"}.

### `GET /api/watch/prompts/{name}` 🔒

### `POST /api/watch/asked` 🔒

The questions asked on the watch while offline: {"items": [{"id": "<fact id>", "q": "who is ryan"}]}.

### `GET /api/watch/speech` 🔒

The watch's offline speech models: ESP-SR's srmodels.bin exactly as Espressif ships it (MultiNet7 English and
what it uses with it, 3.3 MB) - the trimmed copy of 1.12 (MultiNet only) crashed the watch.

### `GET /api/watch/model` 🔒

The watch's own AI model: {"training": bool, "status": {...progress}, "model": {version, size, scores...} | null}.

### `POST /api/watch/model/train` 🔒

Train it again from the knowledge base (background, low priority; minutes to an hour or two).

### `DELETE /api/watch/model/train` 🔒

### `GET /api/watch/model/file` 🔒

### `GET /api/agenda` 🔒

Query: `days` = 7

## Wake word

### `GET /api/wakeword` 🔒

The watch's wake word set ("Hey Oracle" + offline commands): manifest with a version the watch compares.

### `GET /api/wakeword/files/{name}` 🔒

### `POST /api/wakeword` 🔒

Install a trained set: a zip (or several files) with wake.json and the .tflite files it names, as multipart
form fields, or the zip as the raw body. The watch picks it up at its next check (within a few hours, or
Settings > Hey Oracle).

### `POST /api/wakeword/samples/{mid}` 🔒

One clip of you saying a phrase (WAV body), from the watch's "Teach my voice".

### `GET /api/wakeword/samples` 🔒

### `GET /api/wakeword/samples.zip` 🔒

All your clips, for the training kit.

### `DELETE /api/wakeword/samples` 🔒

Query: `phrase` = None

### `DELETE /api/wakeword` 🔒

Back to the starter wake word.

## Vocabulary & corrections

### `GET /api/vocab` 🔒

Built-in vocabularies (payments / tech / finance) and which are on.

### `PUT /api/vocab` 🔒

{"on": ["payments", "tech"]} - the rest are turned off; meetings are re-spelled in the background.

### `GET /api/corrections` 🔒

Words you corrected: {"tapin": "DABIN"} - fixed in every meeting.

### `DELETE /api/corrections/{heard}` 🔒

### `GET /api/glossary` 🔒

### `PUT /api/glossary` 🔒

## Known voices

### `GET /api/voices` 🔒

### `DELETE /api/voices/{name}` 🔒

### `POST /api/voices/relearn` 🔒

Learn voices again from the lines you tagged (for names that have no remembered voice).

## Sign-in

### `GET /api/me` 🔒

Your own name, for "Me" when tagging speakers.

### `PUT /api/me` 🔒

## Backup

### `GET /api/backup` 🔒

Everything you taught the server, in one file: Names & terms and the remembered voices.

### `POST /api/backup` 🔒

Load a backup: Names & terms are merged (new lines added), voices added or replaced by name.

## People

### `GET /api/people` 🔒

Everyone you've named: meetings, talk time, pace, pitch (profiles are updated in the background).

### `GET /api/people/{name}` 🔒

### `GET /api/people/{name}/samples` 🔒

The voice samples kept for this person and the line each came from (to play, and remove wrong ones).

### `POST /api/people/{name}/not-them` 🔒

{"job": "<meeting id>", "i": <line>} - that clip isn't them: not used for them any more (Hear them, their
profile and voice), and the voice samples taken from it go. Their profile is rebuilt in the background.

### `POST /api/people/{name}/relearn` 🔒

Rebuild this person's voice only from lines named as them now (e.g. after fixing lines that weren't them).

### `POST /api/people/{name}/refresh` 🔒

Rebuild this person's profile now (numbers, words, voice, and a new AI note).

## Files

### `GET /api/files` 🔒

Query: `device` = ''

### `GET /api/files/{path:path}` 🔒

Query: `device`

### `PUT /api/files/{path:path}` 🔒

Query: `device`, `base_rev` = 0, `by` = 'device'

### `DELETE /api/files/{path:path}` 🔒

Query: `device`, `base_rev` = 0, `by` = 'device'

## Live recording

### `GET /api/live` 🔒

Recordings streaming in from a watch right now, with the live transcript so far.

### `GET /api/live/{rid}/brief` 🔒

For the watch while it records: the last line heard, who said it, who has talked so far. Small and quick.

## Devices & firmware

### `POST /api/devices/{did}/log` 🔒

The watch's log lines (text/plain) and, in X-Status, its state (battery, WiFi, memory...) - every minute.

### `GET /api/devices` 🔒

### `GET /api/devices/{did}/stability` 🔒

How the watch has been, day by day: restarts, crashes, failed sends, memory running out, questions, slowest.

Query: `days` = 7

### `GET /api/devices/{did}/log` 🔒

Query: `lines` = 500

### `DELETE /api/devices/{did}/log` 🔒

### `POST /api/firmware` 🔒

A firmware .bin for watches to install over WiFi; its board and version are read from the file.

Query: `notes` = ''

### `GET /api/firmware` 🔒

### `POST /api/firmware/github` 🔒

Look at GitHub for new watch firmware now.

### `GET /api/firmware/latest` 🔒

Query: `board`, `current` = ''

### `GET /api/firmware/{board}/{fname}` 🔒

### `POST /api/firmware/{board}/{version}` 🔒

### `DELETE /api/firmware/{board}/{version}` 🔒
