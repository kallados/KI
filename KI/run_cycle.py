import os, re, glob, json, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
ADAPTERS_DIR = os.environ.get("ADAPTERS_DIR", "/workspace/adapters")
RUNS_DIR = os.environ.get("RUNS_DIR", "/workspace/runs")
CHOICE_FILE = os.environ.get("CHOICE_FILE", "/workspace/choice.txt")
CLEAN_DATA_DIR = os.environ.get("CLEAN_DATA_DIR", "/workspace/clean_data")

# Optional: orchestrator pins these
CYCLE_ID_ENV = os.environ.get("CYCLE_ID")
PREV_ADAPTER = os.environ.get("PREV_ADAPTER")
EXCLUDE_JSON = os.environ.get("CLEAN_DATA_EXCLUDE_JSON")

# Training/eval settings
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "800"))
MAX_LINES = int(os.environ.get("MAX_LINES", "0"))  # 0 = all non-empty
MAX_STEPS = int(os.environ.get("MAX_STEPS", "200"))
use_bf16 = os.environ.get("BF16", "").lower() in ("1","true","yes","on")

QUESTIONS = [
    "What is the main limitation of a purely data-driven learning system?",
    "How should a system decide what it needs to learn next?",
    "Does continuous learning necessarily lead to understanding? Why or why not?",
]

def next_run_id():
    pattern = re.compile(r"^qlora_run(\d+)$")
    nums = []
    if os.path.isdir(ADAPTERS_DIR):
        for name in os.listdir(ADAPTERS_DIR):
            m = pattern.match(name)
            if m:
                nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1

def pick_data_file(choice):
    folder = os.path.join(CLEAN_DATA_DIR, choice)
    files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    if not files:
        raise SystemExit(f"no .txt in {folder}")
    return files[0]

def load_bnb_model():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto")
    return model

def load_exclude_map():
    if not EXCLUDE_JSON:
        return {}
    p = EXCLUDE_JSON
    if not os.path.isfile(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j.get("exclude", {})

def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(ADAPTERS_DIR, exist_ok=True)

    choice = open(CHOICE_FILE, "r", encoding="utf-8").read().strip()[:1].upper()
    if choice not in ("A","B","C"):
        raise SystemExit(f"bad choice in {CHOICE_FILE}: {choice!r}")

    cycle_id = int(CYCLE_ID_ENV) if (CYCLE_ID_ENV and CYCLE_ID_ENV.isdigit()) else next_run_id()
    out_dir = os.path.join(ADAPTERS_DIR, f"qlora_run{cycle_id}")

    prev_dir = PREV_ADAPTER
    if prev_dir and not os.path.isfile(os.path.join(prev_dir, "adapter_config.json")):
        raise SystemExit(f"PREV_ADAPTER invalid (no adapter_config.json): {prev_dir}")
    if not prev_dir and cycle_id > 1:
        cand = os.path.join(ADAPTERS_DIR, f"qlora_run{cycle_id-1}")
        if os.path.isfile(os.path.join(cand, "adapter_config.json")):
            prev_dir = cand

    data_path = pick_data_file(choice)
    exclude_map = load_exclude_map()
    rel_key = f"{choice}/{os.path.basename(data_path)}"
    exclude_set = set(exclude_map.get(rel_key, []))

    # Build dataset lines (skip excluded absolute line numbers)
    lines = []
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i in exclude_set:
                continue
            s = line.strip()
            if not s:
                continue
            lines.append(s)
            if MAX_LINES and len(lines) >= MAX_LINES:
                break

    if not lines:
        raise SystemExit(f"no usable lines after exclusions in {data_path}")

    ds = Dataset.from_dict({"text": lines})

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = load_bnb_model()

    if prev_dir and os.path.isdir(prev_dir):
        model = PeftModel.from_pretrained(base, prev_dir, is_trainable=True)
    else:
        lora = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj"]
        )
        model = get_peft_model(base, lora)

    def tok_fn(ex):
        return tok(ex["text"], truncation=True, max_length=512)

    ds2 = ds.map(tok_fn, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=os.path.join(RUNS_DIR, f"tmp_trainer_cycle{cycle_id}"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_steps=MAX_STEPS,
        logging_steps=20,
        save_steps=200,
        bf16=use_bf16,
        fp16=(not use_bf16),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds2,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )

    print(f"TRAIN: cycle{cycle_id}  choice={choice}  data={data_path}  prev={prev_dir}  exclude={len(exclude_set)}")
    trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print("saved:", out_dir)

    # eval cycleN
    base2 = load_bnb_model()
    model2 = PeftModel.from_pretrained(base2, out_dir)
    model2.eval()

    for i, q in enumerate(QUESTIONS, start=1):
        prompt = tok.apply_chat_template([{"role":"user","content":q}], tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model2.device)
        with torch.no_grad():
            out = model2.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        ans = tok.decode(gen, skip_special_tokens=True).strip()
        out_path = os.path.join(RUNS_DIR, f"cycle{cycle_id}_q{i}.txt")
        open(out_path, "w", encoding="utf-8").write(f"Q{i}: {q}\n\nA{i}: {ans}\n")
        print(f"saved: {out_path}")

    os.system(f"python3 /workspace/compare_cycle.py {cycle_id}")

if __name__ == "__main__":
    main()
