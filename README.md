# Agentic DataSciBench

Benchmarking LLM agents on data science tasks using [DataSciBench](https://github.com/THUDM/DataSciBench). We compare single-pass code generation against agentic retry loops to measure the value of self-evaluation and iterative refinement. Agents generate Python code via local [Ollama](https://ollama.com) models, which is then executed and scored using DataSciBench's Task-Function-Code (TFC) evaluation framework.

> **Course Project** — CS6180: Generative AI, Spring 2026

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and running
- DataSciBench task data in `data/` and metric definitions in `metric/` (see [Setup](#setup))

## Setup

### 1. Install Ollama and pull a model

```bash
ollama pull qwen2.5-coder:7b-instruct
ollama serve
```

Other models used in this project include `mistral`, `qwen2.5-coder:3b`, and `deepseek-coder:6.7b`. Pull any model you want to test:

```bash
ollama pull <model_name>
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

Key dependencies include: `requests`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `scipy`, `pyyaml`, `langchain-ollama`, and `langgraph`.

### 3. Prepare benchmark data

Clone the DataSciBench repository and copy the required folders into this project:

```bash
git clone https://github.com/THUDM/DataSciBench.git
cp -r DataSciBench/data/ ./data/
cp -r DataSciBench/metric/ ./metric/
```

Each task folder (`data/bcbXX/`) contains a `prompt.json` with the task description. Each metric folder (`metric/bcbXX/`) contains a `metric.yaml` with ground truth, input data, and TFC evaluation definitions.

## Task Subset

`selected_tasks.json` defines a curated subset of 119 tasks used for our experiments. This subset was chosen to keep evaluation tractable while covering a representative range of task types (data cleaning, statistical analysis, visualization, predictive modeling). All agents can be run against either this subset or the full set of tasks in `data/`.

## Agents

### Baseline (single-pass)

A single LLM call per task with no feedback or retry. Serves as the control for measuring the impact of agentic loops.

```bash
python baseline_agent.py
```

- **Model config:** Edit `MODEL_ID` in the script (default: `mistral`)
- **Ollama endpoint:** `/api/generate`
- **Output:** `data/<task_id>/<model_id>_outputs.jsonl`

### Retry Agent (iterative self-correction)

Generates code, validates it against the task's metric, classifies errors, builds a targeted reflection prompt, and retries up to 3 times.

```bash
python retry_agent.py
```

- **Model config:** `--model <model_id>` flag (default: `qwen2.5-coder:7b-instruct`)
- **Ollama endpoint:** `/v1/chat/completions` (OpenAI-compatible)
- **Output:** `data/<task_id>/<model_id>_retry_v3_outputs.jsonl`

Run only selected tasks or specific tasks:

```bash
python retry_agent.py --model qwen2.5-coder:7b-instruct  # selected_tasks.json
python retry_agent.py --all                                # all tasks
python retry_agent.py bcb9 bcb20                           # specific tasks
```

### Reflection Agent (LangGraph)

Uses a LangGraph state machine with generate → reflect → revise nodes. The reflector reviews code for correctness and the reviser fixes issues, looping up to 3 times.

```bash
python reflection_agent.py
```

- **Model config:** Edit `model` in the script (default: `mistral`)
- **Ollama endpoint:** LangChain's `OllamaLLM` wrapper (port 11434)
- **Output:** `data/<task_id>/mistral0331_outputs.jsonl`

### Planner-Executor-Reflector Agent

A multi-phase pipeline: the Planner decomposes the task into structured JSON steps, the Executor writes code for each step, and the Reflector provides targeted debugging feedback on failures. Includes both inner retries (executor fixes) and outer retries (re-planning).

```bash
python plan_exec_agent.py
```

- **Model config:** Edit `MODEL_PLANNER`, `MODEL_EXECUTOR`, `MODEL_REFLECTOR` in the script (default: `qwen2.5-coder:3b`)
- **Ollama endpoint:** `/api/chat`
- **Output:** `data/<task_id>/PEL-<model_id>_outputs.jsonl`

> **Note on Ollama endpoints:** Different agents use different Ollama API formats. Make sure `ollama serve` is running on port 11434 before launching any agent.

## Evaluation

### Run TFC evaluation on a single task

```bash
python -m experiments.evaluate_tmc --model_id <model_id> --task_id bcb9
```

This compares the agent's generated code against the ground truth using the metric definitions in `metric/bcbXX/metric.yaml`. Results are saved to `data/<task_id>/<model_id>_tmc_results.jsonl`.

### Run evaluation across selected tasks

```bash
python run_all_evals.py
```

Edit `MODEL_ID` in the script to match the agent output you want to evaluate. Skips tasks that already have results.

### View aggregate results

```bash
python experiments/show_bcb_eval_results.py --model_id <model_id>
```

### Example output

A successful evaluation produces a JSONL file with metrics like:

```json
{
  "output_dir": "bcb9",
  "cr": 1,
  "completion": "import pandas as pd\ndef task_func(...):\n    ...",
  "TMC_results": [
    {"function": "data_accuracy", "task_name": "calculate_mse", "result": 1.0}
  ]
}
```

- `cr`: Completion Rate (1 = success, 0 = failure)
- `TMC_results`: Per-metric scores from the TFC evaluation functions

## Project Structure

```
baseline_agent.py       # Single-pass baseline agent
retry_agent.py    # Iterative retry agent with error classification
reflection_agent.py     # LangGraph reflection agent
plan_exec_agent.py      # Planner + Executor + Reflector pipeline
run_all_evals.py        # Batch evaluation runner
selected_tasks.json     # Curated 117-task subset

data/                   # Task folders (data/bcbXX/prompt.json) — not tracked in git
metric/                 # TFC metric definitions (metric/bcbXX/metric.yaml) — not tracked in git
experiments/            # Evaluation scripts (evaluate_tmc.py, show_bcb_eval_results.py)
utils/                  # Shared utilities
results/                # Aggregate result outputs
```

## References

Zhang, D., Zhoubian, S., Cai, M., Li, F., Yang, L., Wang, W., Dong, T., Hu, Z., Tang, J., & Yue, Y. (2025). *DataSciBench: An LLM Agent Benchmark for Data Science.* arXiv:2502.13897. [https://datascibench.github.io/](https://datascibench.github.io/)
