import sys, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "/workspace/adapters/qlora_run1"

def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: python3 ask.py <out_path> <question...>")

    out_path = sys.argv[1]
    question = " ".join(sys.argv[2:]).strip()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    gen = out[0][inputs["input_ids"].shape[1]:]
    ans = tok.decode(gen, skip_special_tokens=True).strip()

    print(ans)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Q: " + question + "\n\nA: " + ans + "\n")

if __name__ == "__main__":
    main()
