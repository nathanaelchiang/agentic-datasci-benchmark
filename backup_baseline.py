import os
import subprocess

MODEL_ID = "mistral"

all_tasks = [
    folder for folder in os.listdir("data")
    if folder.startswith("bcb") and os.path.isdir(os.path.join("data", folder))
]

for task in sorted(all_tasks):
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