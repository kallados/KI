import os
import json
import time
import signal
import subprocess
import shlex
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import streamlit as st

from log_utils import load_jsonl, load_choice_log_map, to_dataframe

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTO = True
except Exception:
    HAS_AUTO = False

st.set_page_config(page_title="KI-Selbstentfaltung Dashboard", layout="wide")
st.title("KI-Selbstentfaltung – Dashboard (read-only)")
# --- Runner STATE UI ---
RUNNER_DIR = Path("/workspace/runner")
ENABLED_FILE = RUNNER_DIR / "runner_enabled"             # allow new cycles
STOP_FILE = RUNNER_DIR / "runner_stop"                  # stop after cycle
STATUS_FILE = RUNNER_DIR / "runner_status.json"         # heartbeat/status
ORCH_PID_FILE = RUNNER_DIR / "orchestrate.pid"          # current orchestrate PID
ORCH_START_FILE = RUNNER_DIR / "orchestrate_started_at.txt"

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False

def _runner_alive() -> bool:
    # Try pgrep first; fallback to ps+grep.
    try:
        r = subprocess.run(["pgrep", "-f", "round_robin_runner.sh"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return True
    except Exception:
        pass
    try:
        cmd = "ps aux | grep -F 'round_robin_runner.sh' | grep -v grep"
        r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
        return bool(r.stdout.strip())
    except Exception:
        return False

def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _read_pid(p: Path):
    try:
        return int(p.read_text().strip())
    except Exception:
        return None

def _read_iso_ts(p: Path):
    try:
        s = p.read_text().strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None

# UI
st.markdown("---")
st.subheader("Runner")

allow_new = ENABLED_FILE.exists()
stop_flag = STOP_FILE.exists()
runner_alive = _runner_alive()

orch_pid = _read_pid(ORCH_PID_FILE) if ORCH_PID_FILE.exists() else None
orch_alive = _pid_alive(orch_pid) if orch_pid is not None else False
orch_started = _read_iso_ts(ORCH_START_FILE) if ORCH_START_FILE.exists() else None

status = _read_json(STATUS_FILE) if STATUS_FILE.exists() else None
mode = status.get("mode") if isinstance(status, dict) else None
rc = status.get("last_rc") if isinstance(status, dict) else None
fails = status.get("consec_fails") if isinstance(status, dict) else None
seed = status.get("seed") if isinstance(status, dict) else None
msg = status.get("message") if isinstance(status, dict) else ""
status_ts = status.get("ts") if isinstance(status, dict) else None
status_age_s = (time.time() - float(status_ts)) if status_ts else None

# 3-state model:
# RUNNING = orchestrate.pid exists and process alive
# PAUSED  = allow_new == false
# IDLE/STOPPED = not running; if runner_alive -> IDLE else STOPPED
if orch_alive:
    state = "RUNNING"
elif not allow_new:
    state = "PAUSED"
else:
    state = "IDLE" if runner_alive else "STOPPED"

m1, m2, m3, m4 = st.columns(4)
m1.metric("STATE", state)
m2.metric("runner alive", "yes" if runner_alive else "no")
m3.metric("allow new cycles", "yes" if allow_new else "no")
m4.metric("stop after cycle", "yes" if stop_flag else "no")

details = f"mode={mode} rc={rc} fails={fails} seed={seed} msg={msg}"
if status_age_s is not None:
    details += f" | last update {status_age_s:.0f}s ago"
st.caption(details if details.strip() else "status: -")

elapsed_min = None
if orch_alive and orch_started is not None:
    try:
        elapsed_min = (datetime.now(timezone.utc) - orch_started.astimezone(timezone.utc)).total_seconds() / 60.0
    except Exception:
        elapsed_min = None

cc1, cc2, cc3 = st.columns(3)
cc1.metric("orchestrate.pid", str(orch_pid) if orch_pid is not None else "-")
cc2.metric("cycle elapsed", f"{elapsed_min:.1f} min" if elapsed_min is not None else "-")
cc3.metric("running", "yes" if orch_alive else "no")

# Last completed cycle (always from latest_cycle.txt; age from run_log.jsonl mtime if present)
latest_cycle_file = Path("/workspace/runs/latest_cycle.txt")
run_log_default = Path("/workspace/runs/run_log.jsonl")
if latest_cycle_file.exists():
    cyc = latest_cycle_file.read_text().strip()
    try:
        ref = run_log_default if run_log_default.exists() else latest_cycle_file
        age_min = (time.time() - ref.stat().st_mtime) / 60.0
        st.caption(f"Last completed cycle: {cyc} ({age_min:.1f} min ago)")
    except Exception:
        st.caption(f"Last completed cycle: {cyc}")
else:
    st.caption("Last completed cycle: -")

# --- Controls ---
btn_start, btn_allow, btn_pause, btn_stopafter, btn_clearstop, btn_kill = st.columns(6)

if "start_runner_result" not in st.session_state:
    st.session_state["start_runner_result"] = None

def _start_runner_now():
    # Prefer tmux-managed runner (survives notebook refresh).
    tmux_script = RUNNER_DIR / "start_runner_tmux.sh"
    if tmux_script.exists():
        cmd = f"cd {shlex.quote(str(RUNNER_DIR))} && chmod +x start_runner_tmux.sh round_robin_runner.sh && ./start_runner_tmux.sh"
    else:
        # Fallback: run in background and write runner.out
        cmd = (
            f"cd {shlex.quote(str(RUNNER_DIR))} && "
            f"chmod +x round_robin_runner.sh && "
            f"rm -f runner_stop runner.lock && "
            f"touch runner_enabled && "
            f"nohup ./round_robin_runner.sh > {shlex.quote(str(RUNNER_DIR / 'runner.out'))} 2>&1 &"
        )
    r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=15)
    return cmd, r.returncode, r.stdout, r.stderr

if btn_start.button("Start runner"):
    if _runner_alive():
        st.info("Runner läuft bereits.")
    else:
        try:
            cmd, rc_s, out_s, err_s = _start_runner_now()
            st.session_state["start_runner_result"] = {
                "ts": time.time(),
                "rc": int(rc_s),
                "cmd": cmd,
                "stdout": (out_s or "").strip(),
                "stderr": (err_s or "").strip(),
            }
        except Exception as e:
            st.session_state["start_runner_result"] = {"ts": time.time(), "rc": 999, "cmd": "", "stdout": "", "stderr": str(e)}
    st.rerun()

res = st.session_state.get("start_runner_result")
if isinstance(res, dict) and res.get("ts"):
    age_s = time.time() - float(res["ts"])
    label = f"Letzter Start-Versuch: rc={res.get('rc')} ({age_s:.0f}s ago)"
    with st.expander(label, expanded=(res.get("rc") not in (0, None))):
        st.code(res.get("cmd", ""), language="bash")
        if res.get("stdout"):
            st.text("stdout:")
            st.code(res["stdout"])
        if res.get("stderr"):
            st.text("stderr:")
            st.code(res["stderr"])
        if (res.get("rc") == 0) and (not res.get("stderr")):
            st.caption("Wenn trotzdem nichts startet: tmux/runner-Skript Output prüfen: /workspace/runner/runner.out")

if btn_allow.button("Allow cycles"):
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    ENABLED_FILE.write_text("1")
    st.rerun()

if btn_pause.button("Pause cycles"):
    try:
        if ENABLED_FILE.exists():
            ENABLED_FILE.unlink()
    except Exception:
        pass
    st.rerun()

if btn_stopafter.button("Stop after cycle"):
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    STOP_FILE.write_text("1")
    st.rerun()

if btn_clearstop.button("Clear stop"):
    try:
        if STOP_FILE.exists():
            STOP_FILE.unlink()
    except Exception:
        pass
    st.rerun()

if btn_kill.button("Kill current cycle"):
    pid_val = _read_pid(ORCH_PID_FILE) if ORCH_PID_FILE.exists() else None
    if not pid_val:
        st.warning("No active orchestrate.pid")
        st.rerun()
    try:
        os.kill(pid_val, signal.SIGTERM)
        time.sleep(8)
        os.kill(pid_val, 0)
        os.kill(pid_val, signal.SIGKILL)
        st.warning("Killed (SIGKILL)")
    except ProcessLookupError:
        st.success("Stopped (SIGTERM)")
    except Exception as e:
        st.error(str(e))
    st.rerun()


DEFAULT_RUN_LOG = os.environ.get("KISE_RUN_LOG", "/workspace/runs/run_log.jsonl")
DEFAULT_CHOICE_LOG = os.environ.get("KISE_CHOICE_LOG", "/workspace/runs/choice_log.jsonl")

with st.sidebar:
    log_path = st.text_input("run_log.jsonl Pfad", value=DEFAULT_RUN_LOG)

    st.markdown("---")
    auto = st.toggle("Auto-Refresh", value=True)
    interval_s = st.slider("Intervall (Sek.)", 2, 60, 10)
    if auto and not HAS_AUTO:
        st.warning("Auto-Refresh: pip install streamlit-autorefresh")

if auto and HAS_AUTO:
    st_autorefresh(interval=interval_s * 1000, key="kise_refresh")

entries = load_jsonl(log_path)
choice_map = load_choice_log_map(DEFAULT_CHOICE_LOG) if os.path.exists(DEFAULT_CHOICE_LOG) else {}
df = to_dataframe(entries, choice_map) if entries else pd.DataFrame()

if df.empty:
    st.error("Keine Daten geladen.")
    st.stop()

warns = df.attrs.get("warnings", [])
if warns:
    st.warning("Hinweise: " + ", ".join(warns))

# ------------------- Summary: overall + per run_type -------------------
st.subheader("Averages (alle Daten)")

avg_df = df.copy()
avg_df["runtime_min"] = avg_df["runtime_s"] / 60.0

def _n_cycles(x):
    return int(pd.Series(x).nunique())

overall = {
    "run_type": "overall",
    "n_cycles": _n_cycles(avg_df["cycle"]) if "cycle" in avg_df else len(avg_df),
    "mean_regret": float(avg_df["regret"].mean()),
    "best_choice_rate": float(avg_df["is_best_choice"].mean()) if "is_best_choice" in avg_df else float("nan"),
    "mean_loss_margin": float(avg_df["loss_margin"].mean()),
    "mean_diff_total": float(avg_df["diff_total"].mean()),
    "mean_runtime_min": float(avg_df["runtime_min"].mean()),
}

grp = (
    avg_df.groupby("run_type", dropna=False)
    .agg(
        n_cycles=("cycle", _n_cycles),
        mean_regret=("regret", "mean"),
        best_choice_rate=("is_best_choice", "mean"),
        mean_loss_margin=("loss_margin", "mean"),
        mean_diff_total=("diff_total", "mean"),
        mean_runtime_min=("runtime_min", "mean"),
    )
    .reset_index()
)

summary = pd.concat([pd.DataFrame([overall]), grp], ignore_index=True)

summary_show = summary.copy()
summary_show["mean_regret"] = summary_show["mean_regret"].round(4)
summary_show["best_choice_rate"] = (summary_show["best_choice_rate"] * 100.0).round(1)
summary_show["mean_loss_margin"] = summary_show["mean_loss_margin"].round(4)
summary_show["mean_diff_total"] = summary_show["mean_diff_total"].round(1)
summary_show["mean_runtime_min"] = summary_show["mean_runtime_min"].round(1)

st.dataframe(
    summary_show[["run_type","n_cycles","mean_regret","best_choice_rate","mean_loss_margin","mean_diff_total","mean_runtime_min"]],
    use_container_width=True,
    height=200,
)

# ------------------- Quick comparison cards: self vs random -------------------
types = set([t for t in avg_df["run_type"].dropna().unique().tolist() if isinstance(t, str)])
if "self" in types and "random" in types:
    st.subheader("Self vs Random (Delta)")
    s = summary.set_index("run_type")
    self_reg = float(s.loc["self","mean_regret"])
    rnd_reg = float(s.loc["random","mean_regret"])
    self_rate = float(s.loc["self","best_choice_rate"])
    rnd_rate = float(s.loc["random","best_choice_rate"])
    self_diff = float(s.loc["self","mean_diff_total"])
    rnd_diff = float(s.loc["random","mean_diff_total"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Ø Regret (self)", f"{self_reg:.4f}", delta=f"{(rnd_reg-self_reg):.4f} vs random")
    c2.metric("Best-choice Rate (self)", f"{(self_rate*100):.1f}%", delta=f"{((self_rate-rnd_rate)*100):.1f} pp")
    c3.metric("Ø diff_total (self)", f"{self_diff:.1f}", delta=f"{(self_diff-rnd_diff):.1f} vs random")

# ------------------- Filters -------------------
with st.sidebar:
    st.markdown("---")
    rt_opts = ["(alle)"] + sorted([x for x in df["run_type"].dropna().unique().tolist() if isinstance(x, str)])
    ch_opts = ["(alle)"] + sorted([x for x in df["choice"].dropna().unique().tolist() if isinstance(x, str)])
    sel_rt = st.selectbox("run_type", rt_opts, 0)
    sel_ch = st.selectbox("choice", ch_opts, 0)

fdf = df.copy()
if sel_rt != "(alle)":
    fdf = fdf[fdf["run_type"] == sel_rt]
if sel_ch != "(alle)":
    fdf = fdf[fdf["choice"] == sel_ch]

# newest first in table
if "cycle" in fdf.columns:
    fdf = fdf.sort_values("cycle", ascending=False)

# ------------------- Main metrics -------------------
c1, c2, c3, c4 = st.columns(4)
n_cycles = int(fdf["cycle"].nunique()) if "cycle" in fdf else len(fdf)
mean_regret = float(fdf["regret"].mean())
mean_diff = float(fdf["diff_total"].mean())
mean_runtime_min = float((fdf["runtime_s"] / 60.0).mean())

c1.metric("Zyklen", n_cycles)
c2.metric("Ø Regret", f"{mean_regret:.4f}")
c3.metric("Ø diff_total", f"{mean_diff:.1f}")
c4.metric("Ø runtime", f"{mean_runtime_min:.1f} min")

# ------------------- Charts: compare run_types directly -------------------
st.subheader("Vergleich (run_type)")

plot_all = df.sort_values("cycle", ascending=True).dropna(subset=["cycle"])
plot_all["cycle"] = plot_all["cycle"].astype(int)

reg_pivot = plot_all.pivot_table(index="cycle", columns="run_type", values="regret", aggfunc="mean")
diff_pivot = plot_all.pivot_table(index="cycle", columns="run_type", values="diff_total", aggfunc="mean")

a, b = st.columns(2)
a.caption("Regret über Zyklen (mehrere Linien)")
a.line_chart(reg_pivot)
b.caption("diff_total über Zyklen (mehrere Linien)")
b.line_chart(diff_pivot)

# Quantiles table
st.subheader("Verteilung (Quantile)")
q = (
    plot_all.groupby("run_type")
    .agg(
        n_cycles=("cycle", lambda x: int(pd.Series(x).nunique())),
        regret_p50=("regret", "median"),
        regret_p90=("regret", lambda x: x.quantile(0.9)),
        diff_p50=("diff_total", "median"),
        diff_p90=("diff_total", lambda x: x.quantile(0.9)),
    )
    .reset_index()
)
q_show = q.copy()
for col in ["regret_p50","regret_p90"]:
    q_show[col] = q_show[col].round(4)
for col in ["diff_p50","diff_p90"]:
    q_show[col] = q_show[col].round(1)
st.dataframe(q_show, use_container_width=True, height=180)

# ------------------- Table (rounded display) -------------------
st.subheader("Tabelle")
df_show = fdf.copy()
df_show["runtime_min"] = df_show["runtime_s"] / 60.0

for col in ["min_loss", "chosen_loss", "regret", "loss_margin"]:
    if col in df_show.columns:
        df_show[col] = df_show[col].round(4)

df_show["runtime_min"] = df_show["runtime_min"].round(1)

cols = [c for c in [
    "cycle","run_type","choice","best_choice","is_best_choice","loss_margin",
    "forced_choice","choice_unforced","forced_overrode",
    "min_loss","chosen_loss","regret",
    "diff_total","runtime_min",
    "diff_q1","diff_q2","diff_q3",
    "ts_parsed",
    "adapter_used","adapter_path",
    "norm_fingerprint","val_fingerprint","val_manifest",
] if c in df_show.columns]
st.dataframe(df_show[cols], use_container_width=True, height=420)

# ------------------- Export (raw) -------------------
st.subheader("Export")
st.download_button(
    "CSV (gefiltert, Rohwerte)",
    data=fdf.to_csv(index=False).encode("utf-8"),
    file_name="kise_filtered.csv",
    mime="text/csv",
)