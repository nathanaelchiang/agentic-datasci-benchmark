import requests
import json
import os

# Config
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_ID = "mistral"      
TASK_FOLDER = "bcb9"       


# Generate code from Ollama
def generate_code(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_ID,
            "prompt": prompt,
            "stream": False
        }
    )

    text = response.json()["response"].strip()

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
    if "import pandas" in cleaned:
        cleaned = cleaned[cleaned.index("import pandas"):]

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

    # Automatically detect all bcb task folders
    all_tasks = [
        folder for folder in os.listdir("data")
        if folder.startswith("bcb") and os.path.isdir(os.path.join("data", folder))
    ]

    print(f"Found {len(all_tasks)} tasks")

    for task_folder in sorted(all_tasks):

        print(f"\n==============================")
        print(f"Running task: {task_folder}")
        print(f"==============================")

        try:
            task_prompt = load_prompt(task_folder)
            generated_code = generate_code(task_prompt)
            save_output(task_folder, MODEL_ID, generated_code)
        except Exception as e:
            print(f"Failed on {task_folder}: {e}")