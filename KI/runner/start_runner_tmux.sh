\
#!/usr/bin/env bash
set -euo pipefail
ROOT="/workspace"
RUNNER_DIR="$ROOT/runner"
SESSION="${1:-kise_runner}"

mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux or run ./round_robin_runner.sh directly."
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session exists: $SESSION"
  echo "attach: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" "cd $RUNNER_DIR && ./round_robin_runner.sh"
echo "started tmux session: $SESSION"
echo "attach: tmux attach -t $SESSION"
