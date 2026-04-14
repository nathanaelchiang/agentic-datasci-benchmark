import requests
import json
import os

# Config
OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL_ID = "mistral"     
# OLLAMA_MODEL = "qwen2.5:7b"   # for API
# MODEL_ID = "qwen2.5_7b"       # for filenames

OLLAMA_MODEL = "deepseek-coder:6.7b"
MODEL_ID = "deepseek_coder_6.7b"   # for filenames

TASK_FOLDER = "bcb9"       


# Generate code from Ollama
def generate_code(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()

    if "response" not in data:
        raise Exception(f"Ollama error: {data}")

    text = data["response"].strip()

    # Remove markdown blocks if present
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]

    # Remove leading "python" if present
    lines = text.splitlines()
    if lines and lines[0].strip().lower() == "python":
        lines = lines[1:]

    cleaned = "\n".join(lines).strip()

    # Ensure we start at first import
    if "import" in cleaned:
        cleaned = cleaned[cleaned.index("import"):]

    return cleaned


# Load prompt from DataSciBench folder
def load_prompt(task_folder):
    prompt_path = os.path.join("data", task_folder, "prompt.json")

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"prompt.json not found in {task_folder}")

    with open(prompt_path, "r") as f:
        data = json.load(f)

    # DataSciBench prompt format
    return data["prompt"]


# Save to DataSciBench format
def save_output(task_folder, model_id, code):
    output_path = os.path.join(
        "data",
        task_folder,
        f"{model_id}_outputs.jsonl"
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

    print(f"\nSaved output to: {output_path}")


# Main
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL, help="Ollama model ID (e.g. qwen2.5:7b)")
    parser.add_argument("--selected", action="store_true", help="Only run tasks in selected_tasks.json")
    parser.add_argument("--all", action="store_true", help="Run all bcb tasks (default when no selected_tasks.json)")
    parser.add_argument("tasks", nargs="*", help="Specific task folders to run")
    args = parser.parse_args()

    OLLAMA_MODEL = args.model
    MODEL_ID = args.model.replace(":", "_").replace("/", "_")

    if args.tasks:
        all_tasks = args.tasks
    elif args.all:
        all_tasks = sorted([
            folder for folder in os.listdir("data")
            if folder.startswith("bcb") and os.path.isdir(os.path.join("data", folder))
        ])
    else:
        with open("selected_tasks.json") as f:
            all_tasks = sorted(json.load(f))

    print(f"Found {len(all_tasks)} tasks")

    for task_folder in sorted(all_tasks):

        output_path = os.path.join(
            "data",
            task_folder,
            f"{MODEL_ID}_outputs.jsonl"
        )

        if os.path.exists(output_path):
            print(f"Skipping {task_folder} (already exists)")
            continue
    
        print(f"\n==============================")
        print(f"Running task: {task_folder}")
        print(f"==============================")

        try:
            task_prompt = load_prompt(task_folder)
            generated_code = generate_code(task_prompt)
            save_output(task_folder, MODEL_ID, generated_code)
        except Exception as e:
            print(f"Failed on {task_folder}: {e}")