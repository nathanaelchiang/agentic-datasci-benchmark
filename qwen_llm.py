from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. You are NOT using GPU.")

print("Loading Qwen model on GPU...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda"
)

print(f"Model is on device: {next(model.parameters()).device}")

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

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean like your original code
    if "```" in decoded:
        parts = decoded.split("```")
        if len(parts) >= 2:
            decoded = parts[1]

    lines = decoded.splitlines()
    if lines and lines[0].strip().lower() == "python":
        lines = lines[1:]

    return "\n".join(lines).strip()