#!/usr/bin/env bash
# Train "Hey Oracle" + the offline commands and install them on Oracle Minutes.
#   ORACLE_KEY=<your API key> ./run.sh                 # everything (CPU: several hours; run it overnight)
#   ORACLE_KEY=<your API key> ./run.sh --only wake     # just the wake word
#   ./run.sh --quick --no-upload                       # 10-minute check that the setup works
# Data and half-finished work are kept in ./work, results in ./out - running it again continues / reuses them.
set -euo pipefail
cd "$(dirname "$0")"
docker build -t oracle-wakeword .
mkdir -p work out
docker run --rm -it \
  -v "$PWD/work:/work" -v "$PWD/out:/out" \
  -e ORACLE_URL="${ORACLE_URL:-https://whisper.jeremiah.sg}" -e ORACLE_KEY="${ORACLE_KEY:-}" \
  oracle-wakeword "$@"
