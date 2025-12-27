import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "/workspace/adapters/qlora_run1"
OUT_DIR = "/workspace/runs"
MAX_NEW_TOKENS = 800

QUESTIONS = [
    "What is the main limitation of a purely data-driven learning system?",
    "How should a system decide what it needs to learn next?",
    "Does continuous learning necessarily lead to understanding? Why or why not?",
]

def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()

    for i, q in enumerate(QUESTIONS, start=1):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        gen = out[0][inputs["input_ids"].shape[1]:]
        ans = tok.decode(gen, skip_special_tokens=True).strip()

        out_path = os.path.join(OUT_DIR, f"cycle1_q{i}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Q{i}: {q}\n\nA{i}: {ans}\n")

        print(f"saved: {out_path}")

if __name__ == "__main__":
    run()
