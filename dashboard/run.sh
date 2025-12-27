#!/usr/bin/env bash
set -euo pipefail
PORT="${1:-8501}"
HOST="${2:-0.0.0.0}"
LOG="${3:-/workspace/runs/run_log.jsonl}"

export KISE_RUN_LOG="$LOG"

python -m pip -q install -r requirements.txt
streamlit run dashboard.py --server.address "$HOST" --server.port "$PORT"
