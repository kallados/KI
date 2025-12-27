#!/usr/bin/env python3
import sys
from pathlib import Path
import difflib

RUNS_DIR = Path("/workspace/runs")

def diff_counts(diff_lines):
    plus = minus = 0
    for line in diff_lines:
        if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            plus += 1
        elif line.startswith("-"):
            minus += 1
    return plus, minus

def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 compare_cycle.py <cycleN>")
    n = str(int(sys.argv[1]))

    for i in (1,2,3):
        a = RUNS_DIR / f"cycle0_q{i}.txt"
        b = RUNS_DIR / f"cycle{n}_q{i}.txt"
        out = RUNS_DIR / f"diff_cycle0_vs_cycle{n}_q{i}.diff"

        if not (a.exists() and b.exists()):
            print(f"Q{i}: missing file")
            continue

        a_txt = a.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        b_txt = b.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

        diff = list(difflib.unified_diff(a_txt, b_txt, fromfile=str(a), tofile=str(b), n=3))
        out.write_text("".join(diff), encoding="utf-8")

        adds, dels = diff_counts(diff)
        print(f"Q{i}: +{adds} -{dels}  saved: {out}")

if __name__ == "__main__":
    main()
