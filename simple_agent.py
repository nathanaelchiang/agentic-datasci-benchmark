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

    task_prompt = """
        You are solving a DataSciBench task.

        Return ONLY valid Python code.
        Do NOT include markdown.
        Do NOT include explanation.
        Do NOT execute the function.

        The code MUST:
        - Import pandas as pd
        - Import matplotlib.pyplot as plt
        - Import seaborn as sns
        - Define: def task_func(list_of_pairs):
        - Create a DataFrame with columns ['Category', 'Value']
        - Create a seaborn barplot of Category vs Value
        - Set the title EXACTLY to 'Category vs Value'
        - Return (df, ax)
        """

    print("Generating code from Ollama...\n")
    generated_code = generate_code(task_prompt)

    print("Generated Code:\n")
    print(generated_code)

    save_output(TASK_FOLDER, MODEL_ID, generated_code)

    print("\nNow run evaluation with:")
    print(f"python -m experiments.evaluate_tmc --model_id {MODEL_ID} --task_id {TASK_FOLDER}")