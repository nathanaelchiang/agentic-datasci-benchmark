import os
import subprocess
import json
from datetime import datetime

import sys

MODEL_ID = sys.argv[1]
# MODEL_ID = "qwen2.5-7b"
MAX_EVALS = 100

STRATEGY = "baseline"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = f"{STRATEGY}_{MODEL_ID}_{TIMESTAMP}"
# MODEL_ID = "mistral"

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