import os
import subprocess
import json

MODEL_ID = "mistral"
MAX_EVALS = 25

# Load curated task subset
with open("selected_tasks.json", "r") as f:
    selected_tasks = sorted(json.load(f))

evaluated_count = 0

for task in selected_tasks:

    task_path = os.path.join("data", task)
    result_file = os.path.join(task_path, f"{MODEL_ID}_tmc_results.jsonl")

    # Skip if metrics already exist
    if os.path.exists(result_file):
        print(f"Skipping {task} (already evaluated)")
        continue

    # Stop after MAX_EVALS
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
        "--model_id", MODEL_ID,
        "--task_id", task
    ])

    evaluated_count += 1

print(f"\nFinished {evaluated_count} new evaluations.")