import os
import yaml
import json

DATA_DIR = "data"
METRIC_DIR = "metric"

TARGET_TYPES = {
    "Data cleaning and preprocessing",
    "Data exploration and statistics",
    "Predictive modeling"
}

selected_tasks = []

for task in os.listdir(DATA_DIR):
    if not task.startswith("bcb"):
        continue

    metric_path = os.path.join(METRIC_DIR, task, "metric.yaml")

    if not os.path.exists(metric_path):
        continue

    with open(metric_path, "r") as f:
        metric_data = yaml.safe_load(f)

    tmc_list = metric_data.get("TMC-list", [])

    for entry in tmc_list:
        if entry.get("task_name") in TARGET_TYPES:
            selected_tasks.append(task)
            break

# Sort for reproducibility
selected_tasks = sorted(selected_tasks)

# Save to file
with open("selected_tasks.json", "w") as f:
    json.dump(selected_tasks, f, indent=2)

print(f"Saved {len(selected_tasks)} selected tasks to selected_tasks.json")