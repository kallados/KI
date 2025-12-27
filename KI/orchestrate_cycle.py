#!/usr/bin/env python3
import json, os, time, shutil, hashlib, re, random
from pathlib import Path
import subprocess

ROOT = Path("/workspace")
RUNS = ROOT / "runs"
CHOICE_LOG = RUNS / "choice_log.jsonl"
RUN_LOG = RUNS / "run_log.jsonl"

CLEAN_SRC = ROOT / "clean_data"
CLEAN_NORM_BASE = ROOT / "clean_data_norm"  # cache root
CLEAN_NORM_LATEST = CLEAN_NORM_BASE / "latest"

VAL_BASE = ROOT / "clean_data_val"          # cache root for val sets + manifest
VAL_LATEST = VAL_BASE / "latest"

# --- normalization rules (conservative) ---
RE_NUM_COMMA = re.compile(r"(\d)\s*@,@\s*(\d)")
RE_NUM_DOT   = re.compile(r"(\d)\s*@\.@\s*(\d)")
RE_HEADER    = re.compile(r"@-@_remaining=\d+|@,@_remaining=\d+")

ART_PATTERNS = ["@,@", "@-@", "@.@", "Ġ"]

def read_last_choice():
    if not CHOICE_LOG.exists():
        raise SystemExit("choice_log.jsonl fehlt.")
    lines = CHOICE_LOG.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise SystemExit("choice_log.jsonl leer.")
    return json.loads(lines[-1])

def diff_counts(path: Path):
    plus = minus = 0
    if not path.exists():
        return {"plus": 0, "minus": 0}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+"):
                plus += 1
            elif line.startswith("-"):
                minus += 1
    return {"plus": plus, "minus": minus}

def artifact_scan(base: Path, max_files=5, max_bytes=200_000):
    counts = {p: 0 for p in ART_PATTERNS}
    if not base.exists():
        return {"checked_files": 0, "counts": counts, "files": []}

    files = [p for p in base.rglob("*.txt") if p.is_file()]
    files = sorted(files)[:max_files]

    checked_files = []
    for fp in files:
        try:
            data = fp.read_bytes()[:max_bytes]
        except Exception:
            continue
        text = data.decode("utf-8", errors="ignore")
        for pat in ART_PATTERNS:
            counts[pat] += text.count(pat)
        checked_files.append(str(fp))
    return {"checked_files": len(checked_files), "counts": counts, "files": checked_files}

def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def copy_latest(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)

def update_latest(cycle: int, adapter_path: str, choice: str):
    (RUNS / "latest_cycle.txt").write_text(str(cycle), encoding="utf-8")
    (RUNS / "latest_adapter.txt").write_text(adapter_path or "", encoding="utf-8")
    (RUNS / "latest_choice.txt").write_text(choice or "", encoding="utf-8")
    for q in (1, 2, 3):
        qf = RUNS / f"cycle{cycle}_q{q}.txt"
        if qf.exists():
            copy_latest(qf, RUNS / f"latest_q{q}.txt")
        df = RUNS / f"diff_cycle0_vs_cycle{cycle}_q{q}.diff"
        if df.exists():
            copy_latest(df, RUNS / f"latest_diff_q{q}.diff")

def compute_fingerprint(src: Path) -> str:
    """Hash of relpath + size + mtime_ns for all .txt files (fast-ish)."""
    h = hashlib.sha256()
    files = [p for p in src.rglob("*.txt") if p.is_file()]
    for p in sorted(files):
        st = p.stat()
        rel = str(p.relative_to(src)).encode("utf-8", errors="ignore")
        h.update(rel + b"\0")
        h.update(str(st.st_size).encode() + b"\0")
        mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
        h.update(str(mtime_ns).encode() + b"\0")
    return h.hexdigest()[:16]

def _hardlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def normalize_text(text: str):
    removed_header_lines = 0
    out_lines = []
    for line in text.splitlines(True):
        if RE_HEADER.search(line):
            removed_header_lines += 1
            continue
        out_lines.append(line)
    t = "".join(out_lines)

    comma_repl = 0
    dot_repl = 0
    prev = None
    while prev != t:
        prev = t
        t, n1 = RE_NUM_COMMA.subn(r"\1,\2", t)
        t, n2 = RE_NUM_DOT.subn(r"\1.\2", t)
        comma_repl += n1
        dot_repl += n2

    return t, {"removed_header_lines": removed_header_lines, "repl_num_comma": comma_repl, "repl_num_dot": dot_repl}

def prune_old_versions(base: Path, keep: int = 2, prefix: str = "v_"):
    if not base.exists():
        return
    dirs = []
    for p in base.iterdir():
        if p.name == "latest":
            continue
        if p.is_dir() and p.name.startswith(prefix):
            dirs.append(p)
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in dirs[keep:]:
        shutil.rmtree(p, ignore_errors=True)

def prepare_clean_data_norm():
    """Cached normalization. Returns (used_path, meta)."""
    if not CLEAN_SRC.exists():
        raise SystemExit(f"clean_data not found: {CLEAN_SRC}")

    CLEAN_NORM_BASE.mkdir(parents=True, exist_ok=True)
    fp = compute_fingerprint(CLEAN_SRC)
    version_dir = CLEAN_NORM_BASE / f"v_{fp}"
    manifest = version_dir / "manifest.json"

    built = False
    totals = {"removed_header_lines": 0, "repl_num_comma": 0, "repl_num_dot": 0, "files_changed": 0, "files_total": 0}

    if not version_dir.exists():
        version_dir.mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        built = True
        for src_file in sorted([p for p in CLEAN_SRC.rglob("*") if p.is_file()]):
            rel = src_file.relative_to(CLEAN_SRC)
            dst_file = version_dir / rel
            totals["files_total"] += 1

            if src_file.suffix.lower() != ".txt":
                _hardlink_or_copy(src_file, dst_file)
                continue

            data = src_file.read_text(encoding="utf-8", errors="ignore")
            fixed, stats = normalize_text(data)
            totals["removed_header_lines"] += stats["removed_header_lines"]
            totals["repl_num_comma"] += stats["repl_num_comma"]
            totals["repl_num_dot"] += stats["repl_num_dot"]

            if fixed != data:
                totals["files_changed"] += 1
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                dst_file.write_text(fixed, encoding="utf-8")
            else:
                _hardlink_or_copy(src_file, dst_file)

        manifest.write_text(json.dumps({
            "fingerprint": fp,
            "src": str(CLEAN_SRC),
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "totals": totals,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # update latest symlink
    try:
        if CLEAN_NORM_LATEST.exists() or CLEAN_NORM_LATEST.is_symlink():
            CLEAN_NORM_LATEST.unlink()
        CLEAN_NORM_LATEST.symlink_to(version_dir)
        used = CLEAN_NORM_LATEST
    except Exception:
        (CLEAN_NORM_BASE / "latest_path.txt").write_text(str(version_dir), encoding="utf-8")
        used = version_dir

    prune_old_versions(CLEAN_NORM_BASE, keep=2, prefix="v_")

    meta = {
        "clean_fingerprint": fp,
        "clean_norm_dir": str(version_dir),
        "clean_used": str(used),
        "clean_built": built,
        "clean_totals": totals if built else None,
    }
    return used, meta

def _reservoir_sample_indices_and_lines(file_path: Path, k: int, min_len: int, seed: int):
    """Reservoir sample (idx, line) pairs from eligible lines in a file."""
    if k <= 0:
        return []
    rng = random.Random(seed)
    res = []
    seen = 0
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            if RE_HEADER.search(s):
                continue
            if len(s) < min_len:
                continue
            seen += 1
            if len(res) < k:
                res.append((idx, s))
            else:
                j = rng.randrange(seen)
                if j < k:
                    res[j] = (idx, s)
    return res

def prepare_val_sets(clean_dir: Path):
    """
    Build (cached) held-out val sets and an exclusion manifest.
    - Val lines are sampled deterministically from normalized clean data.
    - Training reads from clean_dir but excludes selected absolute line indices.
    Returns: (val_root: Path, manifest_path: Path, meta: dict)
    """
    VAL_BASE.mkdir(parents=True, exist_ok=True)
    fp = compute_fingerprint(clean_dir)
    val_lines_per_source = int(os.environ.get("VAL_LINES_PER_SOURCE", "2000"))
    val_min_len = int(os.environ.get("VAL_MIN_LEN", "50"))
    seed_base = int(fp, 16) % (2**31)

    version_dir = VAL_BASE / f"v_{fp}"
    manifest_path = version_dir / "val_manifest.json"

    built = False
    meta = {"val_fingerprint": fp, "val_dir": str(version_dir), "val_used": str(VAL_LATEST), "val_built": False}

    if not version_dir.exists():
        version_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        built = True
        exclude = {}
        val_counts = {}
        for src in ("A", "B", "C"):
            src_dir = clean_dir / src
            files = sorted([p for p in src_dir.glob("*.txt") if p.is_file()])
            if not files:
                raise SystemExit(f"no .txt in {src_dir}")

            # allocate k across files (equal split; adjust remainder)
            k_total = val_lines_per_source
            k_each = max(1, k_total // len(files))
            ks = [k_each] * len(files)
            rem = k_total - sum(ks)
            for i in range(rem):
                ks[i % len(ks)] += 1

            val_lines = []
            taken = 0
            for fp_i, k_i in zip(files, ks):
                pairs = _reservoir_sample_indices_and_lines(fp_i, k_i, val_min_len, seed_base + hash((src, fp_i.name)) % 1000003)
                # store exclusions
                rel = f"{src}/{fp_i.name}"
                idxs = sorted([idx for idx, _ in pairs])
                if idxs:
                    exclude[rel] = idxs
                val_lines.extend([line for _, line in pairs])
                taken += len(pairs)

            rng = random.Random(seed_base + hash(src) % 1000003)
            rng.shuffle(val_lines)
            out_dir = version_dir / src
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "val.txt"
            out_file.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
            val_counts[src] = {"lines": len(val_lines), "min_len": val_min_len}

        manifest = {
            "val_fingerprint": fp,
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "clean_dir": str(clean_dir),
            "val_lines_per_source": val_lines_per_source,
            "val_min_len": val_min_len,
            "exclude": exclude,
            "val_counts": val_counts,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # update latest symlink
    try:
        if VAL_LATEST.exists() or VAL_LATEST.is_symlink():
            VAL_LATEST.unlink()
        VAL_LATEST.symlink_to(version_dir)
        used = VAL_LATEST
    except Exception:
        (VAL_BASE / "latest_path.txt").write_text(str(version_dir), encoding="utf-8")
        used = version_dir

    prune_old_versions(VAL_BASE, keep=2, prefix="v_")

    meta = {
        "val_fingerprint": fp,
        "val_dir": str(version_dir),
        "val_used": str(used),
        "val_built": built,
        "manifest": str(manifest_path),
    }
    return used, manifest_path, meta

def main():
    RUNS.mkdir(parents=True, exist_ok=True)

    # 0) normalize (cached)
    raw_scan = artifact_scan(CLEAN_SRC)
    clean_dir, clean_meta = prepare_clean_data_norm()
    norm_scan = artifact_scan(clean_dir)

    # 0b) build val sets (cached) + exclusion manifest
    val_dir, manifest_path, val_meta = prepare_val_sets(clean_dir)

    env_choose = os.environ.copy()
    env_choose["CLEAN_DATA_DIR"] = str(clean_dir)
    env_choose["CLEAN_DATA_VAL_DIR"] = str(val_dir)

    t0 = time.time()

    # 1) choose (loss computed on val dir)
    subprocess.run(["python3", str(ROOT / "choose_source.py")], check=True, env=env_choose)
    j = read_last_choice()
    cycle = j.get("cycle_next")
    choice = j.get("choice")
    adapter_used = j.get("adapter_used")

    if cycle is None:
        raise SystemExit("cycle_next fehlt im letzten choice_log-Eintrag.")

    print("orchestrator: cycle_next =", cycle, "| choice =", choice, "| mode =", j.get("mode"))

    # 2) train/eval (train on clean_dir, exclude val indices)
    env_train = os.environ.copy()
    env_train["CLEAN_DATA_DIR"] = str(clean_dir)
    env_train["CLEAN_DATA_EXCLUDE_JSON"] = str(manifest_path)
    env_train["CYCLE_ID"] = str(cycle)
    if adapter_used:
        env_train["PREV_ADAPTER"] = str(adapter_used)

    subprocess.run(["python3", str(ROOT / "run_cycle.py")], check=True, env=env_train)

    # 3) compare
    subprocess.run(["python3", str(ROOT / "compare_cycle.py"), str(cycle)], check=True, env=os.environ.copy())

    runtime_s = round(time.time() - t0, 3)
    diff = {f"q{q}": diff_counts(RUNS / f"diff_cycle0_vs_cycle{cycle}_q{q}.diff") for q in (1,2,3)}

    entry = {
        "cycle": cycle,
        "choice": choice,
        "adapter_used": adapter_used,
        "losses_or_scores": j.get("losses") or j.get("scores"),
        "diff_counts": diff,
        "artifact_check_raw": raw_scan,
        "artifact_check_norm": norm_scan,
        "clean_meta": clean_meta,
        "val_meta": val_meta,
        "runtime_s": runtime_s,
        "env": {
            "CHOOSE_MODE": os.environ.get("CHOOSE_MODE"),
            "SAMPLE_LINES": os.environ.get("SAMPLE_LINES"),
            "BATCH_SIZE": os.environ.get("BATCH_SIZE"),
            "MAX_LEN": os.environ.get("MAX_LEN"),
            "MAX_NEW_TOKENS": os.environ.get("MAX_NEW_TOKENS"),
            "VAL_LINES_PER_SOURCE": os.environ.get("VAL_LINES_PER_SOURCE"),
            "VAL_MIN_LEN": os.environ.get("VAL_MIN_LEN"),
        },
    }
    append_jsonl(RUN_LOG, entry)
    update_latest(int(cycle), str(adapter_used or ""), str(choice or ""))

    print("orchestrator: done cycle", cycle)

if __name__ == "__main__":
    main()
