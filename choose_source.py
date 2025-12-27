import os, re, glob, json, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
ADAPTERS_DIR = os.environ.get("ADAPTERS_DIR", "/workspace/adapters")
CLEAN_DATA_DIR = os.environ.get("CLEAN_DATA_DIR", "/workspace/clean_data")
CLEAN_DATA_VAL_DIR = os.environ.get("CLEAN_DATA_VAL_DIR", "").strip() or None
OUT_FILE = os.environ.get("OUT_FILE", "/workspace/choice.txt")
RUNS_DIR = os.environ.get("RUNS_DIR", "/workspace/runs")

# mode: "loss" (default) or "score"
CHOOSE_MODE = os.environ.get("CHOOSE_MODE", "loss").strip().lower()

# loss-mode params
SAMPLE_LINES = int(os.environ.get("SAMPLE_LINES", "32"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
MAX_LEN = int(os.environ.get("MAX_LEN", "256"))
MIN_LEN = int(os.environ.get("MIN_LEN", "50"))

# optional baselines
FORCE_CHOICE = os.environ.get("FORCE_CHOICE", "").strip().upper() or None
RANDOM_SEED_BASE = os.environ.get("RANDOM_SEED", "").strip()

# score-mode params
SCORE_PROMPT = os.environ.get(
    "SCORE_PROMPT",
    "Choose next source. Reply with exactly one letter: A, B, or C.\n"
    "A = science\nB = dialog\nC = philosophy\n\nChoice:"
)

RE_SKIP_HEADER = re.compile(r"@-@_remaining=\d+|@,@_remaining=\d+")

def latest_adapter_dir(adapters_dir: str) -> str | None:
    """Return latest valid qlora_runN folder that contains adapter_config.json."""
    pattern = re.compile(r"^qlora_run(\d+)$")
    best_n = -1
    best = None
    if not os.path.isdir(adapters_dir):
        return None
    for name in os.listdir(adapters_dir):
        m = pattern.match(name)
        if not m:
            continue
        n = int(m.group(1))
        p = os.path.join(adapters_dir, name)
        if not os.path.isdir(p):
            continue
        if not os.path.isfile(os.path.join(p, "adapter_config.json")):
            continue
        if n > best_n:
            best_n = n
            best = p
    return best

def next_run_id(adapters_dir: str) -> int:
    pattern = re.compile(r"^qlora_run(\d+)$")
    nums = []
    if os.path.isdir(adapters_dir):
        for name in os.listdir(adapters_dir):
            m = pattern.match(name)
            if m:
                nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1

def first_txt_file(folder: str) -> str:
    files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    if not files:
        raise SystemExit(f"No .txt found in {folder}")
    return files[0]

def read_sample_lines(path: str, n: int):
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if RE_SKIP_HEADER.search(s):
                continue
            if len(s) < MIN_LEN:
                continue
            out.append(s)
            if len(out) >= n:
                break
    return out

def load_model_with_adapter(adapter_dir: str | None):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map="auto",
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tok, model

def avg_loss_per_token(tok, model, texts):
    losses = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        ).to(model.device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        # out.loss is mean loss over tokens in batch
        losses.append(float(out.loss.detach().cpu()))
    return sum(losses) / max(1, len(losses))

def score_string(tok, model, prompt: str, target: str) -> float:
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1]
    tid = tok.encode(target, add_special_tokens=False)
    if not tid:
        return float("-inf")
    return float(logits[tid[0]].detach().cpu())

def choose_by_loss(tok, model):
    base_dir = CLEAN_DATA_VAL_DIR or CLEAN_DATA_DIR
    losses = {}
    for k in ("A", "B", "C"):
        p = first_txt_file(os.path.join(base_dir, k))
        texts = read_sample_lines(p, SAMPLE_LINES)
        if not texts:
            losses[k] = float("inf")
            continue
        losses[k] = avg_loss_per_token(tok, model, texts)
    choice = min(losses, key=losses.get)
    return choice, losses

def choose_by_score(tok, model):
    scores = {k: score_string(tok, model, SCORE_PROMPT, k) for k in ("A", "B", "C")}
    choice = max(scores, key=scores.get)
    return choice, scores

def log_jsonl(obj: dict):
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, "choice_log.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    adapter = latest_adapter_dir(ADAPTERS_DIR)
    cycle_next = next_run_id(ADAPTERS_DIR)

    tok, model = load_model_with_adapter(adapter)

    print("MODE:", CHOOSE_MODE)
    print("ADAPTER:", adapter)
    if CHOOSE_MODE == "score":
        choice, scores = choose_by_score(tok, model)
        print("SCORES:", scores)
        payload = {"mode": "score", "adapter_used": adapter, "cycle_next": cycle_next, "choice": choice, "scores": scores}
    elif CHOOSE_MODE == "random":
        # compute losses for logging/comparability, but choose randomly
        _unused_choice, losses = choose_by_loss(tok, model)
        print("LOSSES:", losses)
        if RANDOM_SEED_BASE:
            seed = int(RANDOM_SEED_BASE) + int(cycle_next)
            rng = random.Random(seed)
        else:
            seed = None
            rng = random.Random()
        choice = rng.choice(["A", "B", "C"])
        payload = {"mode": "random", "adapter_used": adapter, "cycle_next": cycle_next, "choice": choice, "losses": losses, "random_seed": seed}
    else:
        choice, losses = choose_by_loss(tok, model)
        print("LOSSES:", losses)
        # detailed payload for loss-mode
        payload = {
            "mode": "loss",
            "adapter_used": adapter,
            "cycle_next": cycle_next,
            "choice": choice,
            "losses": losses,
            "eval_dir": (CLEAN_DATA_VAL_DIR or CLEAN_DATA_DIR),
            "min_len": MIN_LEN,
            "sample_lines": SAMPLE_LINES,
            "max_len": MAX_LEN,
        }

    # optional: force choice for baseline runs
    if FORCE_CHOICE in ("A", "B", "C"):
        original = choice
        choice = FORCE_CHOICE
        payload["forced_choice"] = FORCE_CHOICE
        payload["choice_unforced"] = original
        payload["choice"] = choice
        print("FORCED_CHOICE:", FORCE_CHOICE, "(was", original, ")")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(choice + "\n")
    print("CHOICE:", choice)
    print("WROTE:", OUT_FILE)

    log_jsonl(payload)

if __name__ == "__main__":
    main()
