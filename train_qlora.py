import os, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DATA_PATH = "/workspace/data/train.txt"
OUT_DIR = "/workspace/adapters/qlora_run1"

text = open(DATA_PATH, "r", encoding="utf-8").read()
lines = [t for t in text.split("\n") if t.strip()]
ds = Dataset.from_dict({"text": lines[:200000]})  # Limit für ersten Test

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"]
)
model = get_peft_model(model, lora)

def tok(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512)
ds = ds.map(tok, batched=True, remove_columns=["text"])

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_steps=400,           # kurzer Testlauf
    logging_steps=20,
    save_steps=200,
    bf16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("saved:", OUT_DIR)
