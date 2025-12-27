#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace"
RUNNER_DIR="$ROOT/runner"
ORCH="$ROOT/orchestrate_cycle.py"
ADAPTERS_DIR="$ROOT/adapters"

ENABLED_FILE="$RUNNER_DIR/runner_enabled"
STOP_FILE="$RUNNER_DIR/runner_stop"
LOCK_FILE="$RUNNER_DIR/runner.lock"
STATUS_FILE="$RUNNER_DIR/runner_status.json"
LOG_FILE="$RUNNER_DIR/runner.log"
SEED_FILE="$RUNNER_DIR/random_seed.txt"
ORCH_PID_FILE="$RUNNER_DIR/orchestrate.pid"
ORCH_START_FILE="$RUNNER_DIR/orchestrate_started_at.txt"

KEEP_ADAPTERS="${KEEP_ADAPTERS:-50}"
SLEEP_WHEN_DISABLED_S="${SLEEP_WHEN_DISABLED_S:-15}"
MAX_CONSEC_FAILS="${MAX_CONSEC_FAILS:-5}"
HEARTBEAT_S="${HEARTBEAT_S:-30}"

mkdir -p "$RUNNER_DIR"

if [[ ! -f "$SEED_FILE" ]]; then
  echo "12345" > "$SEED_FILE"
fi

write_status() {
  local mode="${1:-idle}"
  local rc="${2:-0}"
  local msg="${3:-}"
  local consec_fails="${4:-0}"
  local seed
  seed="$(cat "$SEED_FILE" 2>/dev/null || echo "")"

  python3 - "$STATUS_FILE" "$mode" "$rc" "$msg" "$consec_fails" "$seed" "$KEEP_ADAPTERS" <<'PY'
import json, sys, time
status_path = sys.argv[1]
mode = sys.argv[2]
rc = int(sys.argv[3])
msg = sys.argv[4]
consec_fails = int(sys.argv[5])
seed = sys.argv[6]
keep = int(sys.argv[7])

obj = {
  "ts": time.time(),
  "mode": mode,
  "last_rc": rc,
  "message": msg,
  "consec_fails": consec_fails,
  "seed": seed,
  "keep_adapters": keep,
}
with open(status_path, "w", encoding="utf-8") as f:
  json.dump(obj, f, ensure_ascii=False)
PY
}

start_heartbeat() {
  local mode="$1"
  local consec_fails="$2"
  (
    while true; do
      write_status "$mode" 0 "running" "$consec_fails"
      sleep "$HEARTBEAT_S"
    done
  ) &
  echo $!
}

cleanup_adapters() {
  local keep="$KEEP_ADAPTERS"
  [[ "$keep" =~ ^[0-9]+$ ]] || keep=50

  if [[ ! -d "$ADAPTERS_DIR" ]]; then
    return 0
  fi

  python3 - "$ADAPTERS_DIR" "$keep" <<'PY'
import os, re, shutil, sys
root = sys.argv[1]
keep = int(sys.argv[2])
pat = re.compile(r"^qlora_run(\d+)$")

cands = []
for name in os.listdir(root):
    m = pat.match(name)
    if not m:
        continue
    p = os.path.join(root, name)
    if os.path.isdir(p):
        cands.append((int(m.group(1)), p))

cands.sort()
to_delete = cands[:-keep] if keep > 0 else cands
for _, p in to_delete:
    try:
        shutil.rmtree(p)
    except Exception:
        pass
PY
}

run_orchestrate() {
  echo "[$(date -Is)] RUN: CHOOSE_MODE=${CHOOSE_MODE:-} FORCE_CHOICE=${FORCE_CHOICE:-} RANDOM_SEED=${RANDOM_SEED:-}" >> "$LOG_FILE"

  # start orchestrate in background so we can track PID
  python3 "$ORCH" >> "$LOG_FILE" 2>&1 &
  local pid=$!
  echo "$pid" > "$ORCH_PID_FILE"
  date -Is > "$ORCH_START_FILE"

  wait "$pid"
  local rc=$?

  rm -f "$ORCH_PID_FILE" "$ORCH_START_FILE"
  return "$rc"
}

run_self() {
  export CHOOSE_MODE="loss"
  unset FORCE_CHOICE || true
  unset RANDOM_SEED || true
  run_orchestrate
}

run_random() {
  export CHOOSE_MODE="random"
  unset FORCE_CHOICE || true
  local seed
  seed="$(cat "$SEED_FILE" 2>/dev/null || echo 12345)"
  export RANDOM_SEED="$seed"
  run_orchestrate
  python3 - "$SEED_FILE" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
try:
    n = int(p.read_text().strip())
except Exception:
    n = 12345
p.write_text(str(n+1))
PY
  unset RANDOM_SEED || true
}

run_fixed() {
  local ch="$1"
  export CHOOSE_MODE="loss"
  export FORCE_CHOICE="$ch"
  unset RANDOM_SEED || true
  run_orchestrate
  unset FORCE_CHOICE || true
}

main_loop() {
  local consec_fails=0
  write_status "idle" 0 "runner started" 0

  while true; do
    if [[ -f "$STOP_FILE" ]]; then
      write_status "stopped" 0 "stop flag set" "$consec_fails"
      echo "[$(date -Is)] STOP flag detected. Exiting." >> "$LOG_FILE"
      exit 0
    fi

    if [[ ! -f "$ENABLED_FILE" ]]; then
      write_status "paused" 0 "disabled (runner_enabled missing)" "$consec_fails"
      sleep "$SLEEP_WHEN_DISABLED_S"
      continue
    fi

    for mode in self random fixedA fixedB fixedC; do
      if [[ -f "$STOP_FILE" ]]; then
        write_status "stopped" 0 "stop flag set" "$consec_fails"
        exit 0
      fi
      if [[ ! -f "$ENABLED_FILE" ]]; then
        break
      fi

      write_status "$mode" 0 "starting" "$consec_fails"
      hb_pid="$(start_heartbeat "$mode" "$consec_fails")"

      set +e
      if [[ "$mode" == "self" ]]; then
        run_self; rc=$?
      elif [[ "$mode" == "random" ]]; then
        run_random; rc=$?
      elif [[ "$mode" == "fixedA" ]]; then
        run_fixed "A"; rc=$?
      elif [[ "$mode" == "fixedB" ]]; then
        run_fixed "B"; rc=$?
      else
        run_fixed "C"; rc=$?
      fi
      set -e

      if [[ -n "${hb_pid:-}" ]]; then
        kill "$hb_pid" >/dev/null 2>&1 || true
      fi

      if [[ "$rc" -ne 0 ]]; then
        consec_fails=$((consec_fails+1))
        write_status "$mode" "$rc" "orchestrate failed" "$consec_fails"
        echo "[$(date -Is)] ERROR rc=$rc consec_fails=$consec_fails" >> "$LOG_FILE"
        sleep_s=$((30 * (2 ** (consec_fails-1))))
        if [[ "$sleep_s" -gt 600 ]]; then sleep_s=600; fi
        sleep "$sleep_s"
        if [[ "$consec_fails" -ge "$MAX_CONSEC_FAILS" ]]; then
          write_status "error" "$rc" "max consecutive failures reached" "$consec_fails"
          echo "[$(date -Is)] Max consecutive failures reached. Exiting." >> "$LOG_FILE"
          exit 1
        fi
      else
        consec_fails=0
        write_status "$mode" 0 "ok" 0
        cleanup_adapters || true
      fi
    done
  done
}

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "Runner already running (lock busy): $LOCK_FILE"
    exit 0
  fi
fi

main_loop
