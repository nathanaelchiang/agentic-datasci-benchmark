import os
import subprocess
import json

MODEL_ID = "qwen2.5_7b"          # matches qwen2.5_7b_retry_outputs.jsonl
# MODEL_ID = "qwen2.5_3b"
OUTPUT_SUFFIX = "retry_outputs"
MAX_EVALS = 200

# # Load curated task subset
# with open("selected_tasks.json", "r") as f:
#     selected_tasks = sorted(json.load(f))

# Use all bcb folders in data/
selected_tasks = sorted([
    folder for folder in os.listdir("data")
    if folder.startswith("bcb") and os.path.isdir(os.path.join("data", folder))
])

evaluated_count = 0

for task in selected_tasks:

    task_path = os.path.join("data", task)
    output_file = os.path.join(task_path, f"{MODEL_ID}_{OUTPUT_SUFFIX}.jsonl")
    result_file = os.path.join(task_path, f"{MODEL_ID}_retry_tmc_results.jsonl")

    # Skip if no retry output exists
    if not os.path.exists(output_file):
        print(f"Skipping {task} (no retry output found)")
        continue

    # Skip if already evaluated
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
        "--model_id", f"{MODEL_ID}_retry",
        "--task_id", task
    ])

    evaluated_count += 1

print(f"\nFinished {evaluated_count} new evaluations.")