# "Hey Oracle" for the watch

Trains the wake word and the offline commands the Idea Saver Round listens for, using
[microWakeWord](https://github.com/kahrendt/microWakeWord) and computer-generated speech
([piper-sample-generator](https://github.com/rhasspy/piper-sample-generator), ~900 voices). No recordings of you needed.

| id | says | does on the watch |
|---|---|---|
| `wake` | "Hey Oracle" | wakes the watch; online: asks Oracle Minutes hands-free |
| `start_meeting` | "start meeting" | records a meeting (works offline) |
| `voice_memo` | "voice memo" | records a memo |
| `stop_recording` | "stop recording" | stops the meeting / memo |
| `mark_this` | "mark this" | marks the moment in a meeting |

The commands are only listened for in the 5 seconds after "Hey Oracle".

## Run

On any machine with Docker (the Coolify server is fine; ~15 GB of disk for the data):

```bash
ORACLE_KEY=<your Oracle Minutes API key> ./run.sh
```

Several hours on a CPU - run it overnight. At the end it prints, for each phrase, how often it was missed and how
many false wakes per hour it had on the test sounds, and uploads `out/oracle_wakeword.zip` to Oracle Minutes. The
watch picks it up at its next check (every 6 hours, at start-up, or Settings > Hey Oracle). Until then the watch uses
the starter wake word "Hey Jarvis".

Without `ORACLE_KEY` it only builds the zip; upload it yourself:

```bash
curl -H "X-API-Key: $ORACLE_KEY" -F file=@out/oracle_wakeword.zip https://whisper.jeremiah.sg/api/wakeword
```

`DELETE /api/wakeword` goes back to the starter.

## Tuning

Edit `phrases.yaml`: more `samples` / `steps` for a better model, more near-misses in `not:` if something else wakes
it (a colleague's name, a phrase you say a lot), lower `max_faph` for fewer false wakes (more misses). Then
`./run.sh --only wake` to retrain just that one.

## Licence note

The background-noise and negative-speech datasets have mixed licences, several non-commercial. Models trained with
them are for personal use; for a product, train with noise data you're allowed to use commercially.
