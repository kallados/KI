#!/usr/bin/env bash
set -euo pipefail

SRC="/workspace/"
DST="/workspace/KI/"

exec 9>/workspace/ki_sync.lock
flock -n 9 || exit 0

rsync -a --delete \
  --exclude='KI/' \
  --exclude='.git/' \
  --exclude='.gitignore' \
  --exclude='.venv-backups/' \
  --exclude='snapshots/' \
  --exclude='*.lock' \
  --exclude='*.pid' \
  --exclude='.hf_home/' \
  --exclude='adapters/' \
  --exclude='runs/' \
  --exclude='data/' \
  --exclude='clean_data/' \
  --exclude='clean_data_norm/' \
  --exclude='clean_data_val/' \
  --exclude='models/' \
  --exclude='__pycache__' \
  --exclude='.ipynb_checkpoints' \
  --exclude='.pytest_cache' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  --exclude='.cache' \
  --exclude='*.bckp' \
  --exclude='workspace_tree.txt' \
  --exclude='workspace_du.txt' \
  --exclude='*.log' \
  --exclude='*.out' \
  "$SRC" "$DST" || {
    rc=$?
    [ "$rc" -eq 24 ] && exit 0
    exit "$rc"
  }
