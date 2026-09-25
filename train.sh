#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
  else
    PYTHON=python3
  fi
fi

REPLAY_FILE="${1:-}"
if [[ -z "$REPLAY_FILE" ]]; then
  latest_id=-1
  for candidate in replays/kaggle_replays/*.json; do
    filename="${candidate##*/}"
    episode_id="${filename%.json}"
    if [[ "$episode_id" =~ ^[0-9]+$ ]] && (( 10#$episode_id > latest_id )); then
      latest_id=$((10#$episode_id))
      REPLAY_FILE="$candidate"
    fi
  done
fi

if [[ -z "$REPLAY_FILE" ]]; then
  echo "No timestamped Kaggle replay found in replays/kaggle_replays/." >&2
  exit 2
fi

if [[ ! -f "$REPLAY_FILE" ]]; then
  echo "Replay file not found: $REPLAY_FILE" >&2
  echo "Usage: $0 [replay-json-path] [player-index]" >&2
  exit 2
fi

REPLAY_PLAYER="${2:-}"
if [[ -z "$REPLAY_PLAYER" ]]; then
  REPLAY_PLAYER="$("$PYTHON" - "$REPLAY_FILE" "${KAGGLE_AGENT_NAME:-Hassan Bazzoun-dev}" <<'PY'
import json
import sys

path, our_name = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    info = json.load(stream).get("info", {})
agents = info.get("Agents", [])
names = [agent.get("Name") for agent in agents]
if our_name not in names:
    raise SystemExit(
        f"Could not find {our_name!r} in replay player names {names}. "
        "Set KAGGLE_AGENT_NAME or pass the opponent player index as argument 2."
    )
print(1 - names.index(our_name))
PY
)"
fi

TRIALS="${TRIALS:-50}"
SEARCH_SEED="${SEARCH_SEED:-7}"

if ! "$PYTHON" -c 'import optuna, kaggle_environments' >/dev/null 2>&1; then
  "$PYTHON" -m pip install -r requirements-optimizer.txt
fi

echo "Training against newest replay: $REPLAY_FILE (opponent player $REPLAY_PLAYER)"
"$PYTHON" -m optimizer.tpe_search \
  --trials "$TRIALS" \
  --seed "$SEARCH_SEED" \
  --seed-config optimized_config.json \
  --opponents random starter "replay:${REPLAY_FILE}:${REPLAY_PLAYER}"
