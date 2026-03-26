import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datetime import datetime

# =========================
# Config
# =========================
MODEL_ID = "qwen2.5-7b"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 2048

STRATEGY = "baseline"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = f"{STRATEGY}_{MODEL_ID}_{TIMESTAMP}"

# =========================
# Load Model (runs once)
# =========================
print("Loading Qwen model (4-bit quantized)...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto"
)

print("Model loaded.")
print(f"Device map: {model.hf_device_map}")
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB")
print(f"GPU memory total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# =========================
# Generate Code
# =========================
def generate_code(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful AI that writes correct Python code."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.2,
            do_sample=True
        )

    # Only decode the NEW tokens (skip the echoed prompt)
    new_tokens = outputs[0][input_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return extract_code(decoded)


# =========================
# Code Extraction
# =========================
def extract_code(text):
    text = text.strip()

    # Extract from markdown code fences
    if "```" in text:
        parts = text.split("```")
        # Pick the first fenced block (parts[1])
        if len(parts) >= 2:
            text = parts[1]

    # Remove leading language identifier
    lines = text.splitlines()
    if lines and lines[0].strip().lower() in ("python", "py", "python3"):
        lines = lines[1:]

    cleaned = "\n".join(lines).strip()

    # Anchor to first import statement
    if "import" in cleaned:
        cleaned = cleaned[cleaned.index("import"):]

    return cleaned


# =========================
# Load Prompt
# =========================
def load_prompt(task_folder):
    prompt_path = os.path.join("data", task_folder, "prompt.json")

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"prompt.json not found in {task_folder}")

    with open(prompt_path, "r") as f:
        data = json.load(f)

    return data["prompt"]


# =========================
# Save Output (DataSciBench format)
# =========================
def save_output(task_folder, code):
    output_path = os.path.join(
        "data",
        task_folder,
        f"{RUN_ID}_outputs.jsonl"
    )

    entry = {
        "plan": [
            {"code": code}
        ],
        "output_dir": task_folder,
        "time_cost": 0,
        "error_list": [],
        "cost": 0
    }

    with open(output_path, "w") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"  Saved → {output_path}")


# =========================
# Main
# =========================
if __name__ == "__main__":

    all_tasks = sorted([
        folder for folder in os.listdir("data")
        if folder.startswith("bcb") and os.path.isdir(os.path.join("data", folder))
    ])

    print(f"\nFound {len(all_tasks)} tasks\n")

    for task_folder in all_tasks:

        print(f"\n{'='*40}")
        print(f"Task: {task_folder}")
        print(f"{'='*40}")

        try:
            task_prompt = load_prompt(task_folder)
            generated_code = generate_code(task_prompt)
            save_output(task_folder, generated_code)

        except Exception as e:
            print(f"  FAILED: {e}")