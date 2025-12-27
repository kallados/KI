import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _safe_float(x):
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        return []
    return rows


def load_choice_log_map(path: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for e in load_jsonl(path):
        cyc = e.get("cycle_next")
        try:
            cyc_i = int(cyc)
        except Exception:
            continue
        out[cyc_i] = e
    return out


def extract_cycle(entry: Dict[str, Any]) -> Optional[int]:
    for k in ("cycle", "cycle_n", "cycle_num", "n"):
        if k in entry:
            try:
                return int(entry[k])
            except Exception:
                pass
    ap = entry.get("adapter_used") or entry.get("adapter_path") or entry.get("adapter") or ""
    if isinstance(ap, str):
        m = re.search(r"qlora_run(\d+)", ap)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


def extract_choice(entry: Dict[str, Any]) -> Optional[str]:
    for k in ("choice", "chosen", "chosen_source"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().upper()
    ch = entry.get("choose")
    if isinstance(ch, dict):
        v = ch.get("choice")
        if isinstance(v, str) and v.strip():
            return v.strip().upper()
    return None


def infer_run_type(entry: Dict[str, Any], choice_log_entry: Optional[Dict[str, Any]] = None) -> str:
    rt = entry.get("run_type")
    if isinstance(rt, str) and rt.strip():
        return rt.strip().lower()

    if isinstance(choice_log_entry, dict):
        fc = choice_log_entry.get("forced_choice")
        if isinstance(fc, str) and fc.strip():
            return "fixed"

    env = entry.get("env") or entry.get("config") or {}
    if isinstance(env, dict):
        cm = env.get("CHOOSE_MODE")
        if isinstance(cm, str) and cm.strip().lower() == "random":
            return "random"
        if "RANDOM_SEED" in env:
            return "random"

    choose_mode = entry.get("choose_mode")
    if isinstance(choose_mode, str) and choose_mode.strip().lower() == "random":
        return "random"
    if "random_seed" in entry or "RANDOM_SEED" in entry:
        return "random"

    return "self"


def extract_losses(entry: Dict[str, Any]) -> Dict[str, float]:
    losses: Dict[str, float] = {}

    v = entry.get("losses")
    if not isinstance(v, dict):
        v = entry.get("losses_or_scores")
    if isinstance(v, dict):
        for k2, v2 in v.items():
            if isinstance(k2, str) and len(k2) == 1:
                losses[k2.upper()] = _safe_float(v2)

    ch = entry.get("choose")
    if isinstance(ch, dict) and isinstance(ch.get("losses"), dict):
        for k2, v2 in ch["losses"].items():
            if isinstance(k2, str) and len(k2) == 1:
                losses[k2.upper()] = _safe_float(v2)

    for k, v in entry.items():
        if isinstance(k, str):
            m = re.fullmatch(r"loss[_\-]?([A-Za-z])", k)
            if m:
                losses[m.group(1).upper()] = _safe_float(v)

    losses = {k: v for k, v in losses.items() if isinstance(v, (int, float)) and math.isfinite(v)}
    return losses


def _best_and_second(losses: Dict[str, float]) -> Tuple[Optional[str], float, float]:
    if not losses:
        return None, float("nan"), float("nan")
    items = sorted(losses.items(), key=lambda kv: kv[1])
    best_k, best_v = items[0]
    second_v = items[1][1] if len(items) > 1 else float("nan")
    return best_k, float(best_v), float(second_v)


def extract_runtime_s(entry: Dict[str, Any]) -> float:
    for k in ("runtime_s", "runtime_sec", "runtime_seconds", "duration_s", "duration_sec"):
        if k in entry:
            return _safe_float(entry.get(k))
    perf = entry.get("perf") or entry.get("timing")
    if isinstance(perf, dict):
        for k in ("runtime_s", "runtime_sec", "duration_s", "duration_sec"):
            if k in perf:
                return _safe_float(perf.get(k))
    return float("nan")


def extract_diff_parts(entry: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    dc = entry.get("diff_counts") or entry.get("diffs")
    if isinstance(dc, dict):
        for q, v in dc.items():
            if not isinstance(v, dict):
                continue
            plus = _safe_float(v.get("plus"))
            minus = _safe_float(v.get("minus"))
            if math.isfinite(plus) or math.isfinite(minus):
                out[f"diff_{q}"] = abs(plus if math.isfinite(plus) else 0.0) + abs(minus if math.isfinite(minus) else 0.0)
    return out


def extract_diff_total(entry: Dict[str, Any]) -> float:
    for k in ("diff_total", "diff_magnitude", "diff_abs_total"):
        if k in entry:
            return _safe_float(entry.get(k))
    parts = extract_diff_parts(entry)
    if parts:
        return float(sum(parts.values()))
    return float("nan")


def extract_fingerprints(entry: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    cm = entry.get("clean_meta")
    if isinstance(cm, dict) and "clean_fingerprint" in cm:
        out["norm_fingerprint"] = cm.get("clean_fingerprint")

    vm = entry.get("val_meta")
    if isinstance(vm, dict) and "val_fingerprint" in vm:
        out["val_fingerprint"] = vm.get("val_fingerprint")
    if isinstance(vm, dict) and "manifest" in vm:
        out["val_manifest"] = vm.get("manifest")

    for k in (
        "raw_fingerprint","norm_fingerprint","val_fingerprint",
        "data_fingerprint_raw","data_fingerprint_norm",
        "val_manifest_fingerprint","adapter_fingerprint",
        "adapter_path","adapter_used"
    ):
        if k in entry:
            out[k] = entry.get(k)

    return out


def to_dataframe(entries: List[Dict[str, Any]], choice_map: Optional[Dict[int, Dict[str, Any]]] = None) -> pd.DataFrame:
    rows = []
    warnings = set()
    choice_map = choice_map or {}

    for e in entries:
        cycle = extract_cycle(e)
        choice = extract_choice(e)
        losses = extract_losses(e)

        best_choice, min_loss, second_loss = _best_and_second(losses)
        loss_margin = second_loss - min_loss if math.isfinite(second_loss) and math.isfinite(min_loss) else float("nan")
        is_best_choice = (choice == best_choice) if (choice and best_choice) else None

        clog = choice_map.get(cycle) if isinstance(cycle, int) else None
        forced_choice = None
        choice_unforced = None
        forced_overrode = None
        if isinstance(clog, dict):
            fc = clog.get("forced_choice")
            if isinstance(fc, str) and fc.strip():
                forced_choice = fc.strip().upper()
            cu = clog.get("choice_unforced")
            if isinstance(cu, str) and cu.strip():
                choice_unforced = cu.strip().upper()
            if forced_choice and choice_unforced:
                forced_overrode = (forced_choice != choice_unforced)

        run_type = infer_run_type(e, clog)
        runtime_s = extract_runtime_s(e)
        diff_parts = extract_diff_parts(e)
        diff_total = extract_diff_total(e)

        chosen_loss = float("nan")
        regret = float("nan")
        if losses and choice and choice in losses:
            chosen_loss = float(losses[choice])
            regret = chosen_loss - min_loss
        elif losses:
            warnings.add("choice_not_in_losses")
        else:
            warnings.add("missing_losses")

        if cycle is None:
            warnings.add("missing_cycle")
        if choice is None:
            warnings.add("missing_choice")

        row = {
            "cycle": cycle,
            "run_type": run_type,
            "choice": choice,
            "best_choice": best_choice,
            "is_best_choice": is_best_choice,
            "loss_margin": loss_margin,
            "forced_choice": forced_choice,
            "choice_unforced": choice_unforced,
            "forced_overrode": forced_overrode,
            "min_loss": min_loss,
            "chosen_loss": chosen_loss,
            "regret": regret,
            "diff_total": diff_total,
            "runtime_s": runtime_s,
        }
        row.update(diff_parts)

        for k in ("ts", "timestamp", "datetime"):
            if k in e:
                row["ts"] = e.get(k)
                break

        row.update(extract_fingerprints(e))
        rows.append(row)

    df = pd.DataFrame(rows)
    if "ts" in df.columns:
        df["ts_parsed"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    else:
        df["ts_parsed"] = pd.NaT

    df.attrs["warnings"] = sorted(list(warnings))
    return df
