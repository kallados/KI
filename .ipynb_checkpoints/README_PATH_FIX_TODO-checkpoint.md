# TEMP: Path-adjustment checklist (for moving work to /workspace/KI)

Purpose: ensure every script works when executed from the repo root (`/workspace/KI`) and does **not** depend on the current working directory.

Scope: **path handling only** (relative paths, hardcoded working-dir assumptions, log/artifact locations, dashboard log discovery).

## What to change (rule)
- Derive `REPO_ROOT` from `__file__` (not from `cwd`) for repo-internal files.
- Send runtime outputs (logs, artifacts, checkpoints, datasets) to **external** dirs (e.g. `/workspace/runs`, `/workspace/artifacts`) via:
  - environment variables (preferred), or
  - a single central config/paths module.
- Dashboard must not read logs via relative paths; show resolved path + mtime + line counts.

## Files to review/patch (path-related)

Top-level:
- `orchestrate_cycle.py`
- `choose_source.py`
- `run_cycle.py`
- `train_qlora.py`
- `train_qlora_247.py`
- `eval_cycle0.py`
- `eval_cycle1.py`
- `compare_cycle.py`
- `ask.py`

Dashboard:
- `dashboard/dashboard.py`
- `dashboard/log_utils.py`

Runner package:
- `runner/` (all Python files inside)

Automation / sync:
- `sync_to_KI.sh`
- `auto_sync_push.sh`

## After this temp file
Delete this file and replace it with the real project README.
