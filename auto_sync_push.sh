#!/usr/bin/env bash
set -euo pipefail

/workspace/sync_to_KI.sh

cd /workspace/KI
git add -A

# nichts zu committen
git diff --cached --quiet && exit 0

git commit -m "sync $(date -u +%F_%H%MZ)"
git push
