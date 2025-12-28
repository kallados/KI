# KI — Self-Choice QLoRA Cycles + Dashboard

Repeated QLoRA training cycles where the model (or a policy) selects the next data source (“self-choice”).  
Each cycle is logged as JSONL and analyzed/visualized in a lightweight dashboard.

Repo (public): https://github.com/kallados/KI  
Default branch (current): `master`

Dashboard (static for now): https://degrees-emails-upcoming-pac.trycloudflare.com

## Workspace ↔ Git mirror workflow

- Development + runs happen in: `/workspace`
- Git mirror used for commits/push: `/workspace/KI`
- Sync script: `sync_to_KI.sh` (rsync `/workspace` → `/workspace/KI` with excludes + `--delete`)

Rule: edit/run in `/workspace` → sync → commit/push from `/workspace/KI`.

## Logs (not in Git)

Logs live in `/workspace/runs/`:
- `run_log.jsonl` (one line per run/cycle; includes `cycle_id`)
- `choice_log.jsonl` (choices/decisions; includes `cycle_id`)

The dashboard reads these logs and joins by `cycle_id`.

## Dashboard data pipeline (expected)

raw files → parsed JSONL → cleaned rows → joined table → filtered view → displayed cycles

Important: the join must be a **left join from `run_log`** so missing `choice_log` rows never drop cycles.

## P0 (current blocker)

Symptom: the dashboard shows fewer cycles than `run_log.jsonl` (e.g., 26 displayed vs. 36 lines).

Primary suspects:
- wrong resolved log path (dashboard reads a different `runs/`)
- parser/clean step drops rows (JSON errors, `cycle_id` type mismatch, dedup, null handling)
- join behaves like inner join / implicitly filters
- UI filters (run_type/status/date) reduce what is displayed without being obvious

Fix strategy (minimal, high signal):
- Add a Debug panel with counts at each pipeline stage:
  - raw → parsed → cleaned → joined → filtered → displayed
- Show resolved log paths + file mtimes + line counts
- Show the missing `cycle_id`s: `run_log` minus displayed
- Enforce `run_log`-left-join semantics in `dashboard/dashboard.py` + `dashboard/log_utils.py`

## GitHub URL patterns (verified)

Branch (tree):
- `https://github.com/kallados/KI/tree/master`

File (blob):
- `https://github.com/kallados/KI/blob/master/<path>`

Blame:
- `https://github.com/kallados/KI/blame/master/<path>`

Raw:
- `https://raw.githubusercontent.com/kallados/KI/refs/heads/master/<path>`
