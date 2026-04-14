import os
import subprocess
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen2.5-coder:7b-instruct", help="Ollama model ID")
parser.add_argument("--selected", action="store_true", help="Only run tasks in selected_tasks.json")
args = parser.parse_args()

RAW_MODEL_ID = args.model
SAFE_MODEL_ID = RAW_MODEL_ID.replace(":", "_").replace("/", "_")
VERSION_TAG = "retry_v3"
OUTPUT_SUFFIX = f"{VERSION_TAG}_outputs"
MAX_EVALS = 200

if args.selected:
    with open("selected_tasks.json") as f:
        selected_tasks = sorted(json.load(f))
else:
    selected_tasks = sorted([
        folder for folder in os.listdir("data")
        if folder.startswith("bcb") and os.path.isdir(os.path.join("data", folder))
    ])
    
evaluated_count = 0

for task in selected_tasks:
    task_path = os.path.join("data", task)
    output_file = os.path.join(task_path, f"{SAFE_MODEL_ID}_{OUTPUT_SUFFIX}.jsonl")
    result_file = os.path.join(task_path, f"{SAFE_MODEL_ID}_{VERSION_TAG}_tmc_results.jsonl")

    if not os.path.exists(output_file):
        print(f"Skipping {task} (no retry output found)")
        continue

    if os.path.exists(result_file):
        print(f"Skipping {task} (TMC results already exist)")
        continue

    if evaluated_count >= MAX_EVALS:
        print(f"\nReached limit of {MAX_EVALS} evaluations. Stopping.")
        break

    print(f"\n==============================")
    print(f"Evaluating {task}")
    print(f"==============================\n")

    subprocess.run([
        "python",
        "-m",
        "experiments.evaluate_tmc",
        "--model_id", f"{SAFE_MODEL_ID}_{VERSION_TAG}",
        "--task_id", task
    ])

    evaluated_count += 1

print(f"\nFinished {evaluated_count} new evaluations.")