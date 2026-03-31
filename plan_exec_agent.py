"""
Multi-agent Planner+Executor+Reflector pipeline
Author: Jicheng Li
"""

import json
import os
import re
import subprocess
import sys
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PLANNER   = "qwen2.5-coder:3b"
MODEL_EXECUTOR  = "qwen2.5-coder:3b"
MODEL_REFLECTOR = "qwen2.5-coder:3b"
OLLAMA_URL      = "http://localhost:11434/api/chat"
DATASCIBENCH_DIR = "."

INNER_LIMIT = 3   # executor retries per plan
OUTER_LIMIT = 2   # re-plan attempts on total inner failure

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def ollama_call(system: str, user: str, model: str = "", json_mode: bool = False) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    if json_mode:
        payload["format"] = "json"   # only set when needed — None breaks Ollama

    resp = requests.post(OLLAMA_URL, json=payload)
    print("=" * 60)
    print("Response")
    print(resp.json())
    return resp.json()["message"]["content"].strip()


def extract_json(text: str) -> str:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        start = text.index("{")
        end   = text.rindex("}") + 1
        return text[start:end]
    except ValueError as e:
        raise ValueError(f"No JSON object found:\n{text}") from e


def extract_starter_code(prompt: str):
    m = re.search(r"```\s*(.*?)```", prompt, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_return_spec(prompt: str):
    m = re.search(r"The function should output with:\s*(.*?)(?:\n\n|\Z)", prompt, re.DOTALL)
    return m.group(1).strip() if m else None


def clean_code(raw: str) -> str:
    """Strip markdown fences and fix mangled import lines."""
    if "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]

    lines = raw.splitlines()
    if lines and lines[0].strip().lower() == "python":
        lines = lines[1:]

    fixed = []
    for line in lines:
        if line.strip().startswith("import") and " from " in line:
            parts = line.split(" from ")
            fixed.append(parts[0].strip())
            fixed.append("from " + parts[1].strip())
        else:
            fixed.append(line)

    cleaned = "\n".join(fixed).strip()
    if "import" in cleaned:
        cleaned = cleaned[cleaned.index("import"):]
    return cleaned


# ---------------------------------------------------------------------------
# Plan schema
# ---------------------------------------------------------------------------

FUNCTION_TYPES = (
    "data_cleaning, calculate_mse, calculate_f1, calculate_accuracy, "
    "data_visualization, data_mining, statistical_analysis"
)

# Deliberately omits "outputs" and "notes" — small models corrupt those fields
# with Python type literals (pd.DataFrame, np.ndarray) that break JSON parsing
PLAN_SCHEMA = """{
  "steps": [
    {
      "step_id": 1,
      "goal": "short plain-English description",
      "function_type": "<one of the types above>",
      "inputs": "plain text description of inputs — no type annotations"
    }
  ]
}"""

# ---------------------------------------------------------------------------
# Agent: Planner
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = (
    "You are an expert data science task planner. "
    "You produce structured JSON plans and nothing else."
)

def run_planner(task_prompt: str, failure_history: list[str] | None = None) -> dict:
    history_block = ""
    if failure_history:
        joined = "\n---\n".join(failure_history)
        history_block = (
            f"\n\nPrevious plans all failed. Do NOT repeat the same approach. "
            f"Here is the accumulated failure history:\n{joined}"
        )

    user_msg = f"""Produce a JSON plan ONLY. No prose, no code, no explanation.
Stop immediately after the closing brace.

function_type must be one of: {FUNCTION_TYPES}

IMPORTANT rules:
- inputs must be plain text strings only — NO Python type annotations, NO pd.DataFrame, NO np.ndarray
- Never reference column names by string — always use positional access (df.iloc[:, N])
- If ML is involved, note train/test split strategy in the goal

Output schema (follow exactly, no extra fields):
{PLAN_SCHEMA}

Task: {task_prompt}{history_block}"""

    raw = ollama_call(PLANNER_SYSTEM, user_msg, model=MODEL_PLANNER, json_mode=True)
    return json.loads(extract_json(raw))


# ---------------------------------------------------------------------------
# Agent: Executor
# ---------------------------------------------------------------------------

EXECUTOR_SYSTEM = (
    "You are an expert Python data science engineer. "
    "You write clean, correct, self-contained Python functions."
)

def run_executor(
    plan: dict,
    starter_code: str | None = None,
    prev_code: str | None = None,
    reflection: str | None = None,
    return_spec: str | None = None,
) -> str:
    plan_str = json.dumps(plan, indent=2)
    starter  = f"\nStart from this skeleton — complete the body only:\n{starter_code}" if starter_code else ""
    ret_spec = f"\nThe function MUST return exactly: {return_spec}" if return_spec else ""

    if prev_code and reflection:
        user_msg = f"""Fix the Python function below based on the feedback.
ONLY fix what the feedback describes. Do not restructure the rest.
Respond with Python code only — no markdown, no explanation.{starter}{ret_spec}

Plan:
{plan_str}

Previous code:
{prev_code}

Feedback (fix this and only this):
{reflection}"""
    else:
        user_msg = f"""Write a self-contained Python function fulfilling the plan below.
Rules:
- Use pandas, numpy, sklearn, matplotlib as needed
- NEVER hardcode column names — always use df.iloc[:, N] for positional access
- ALWAYS include a return statement matching the required output exactly
- Do NOT add helpers or extra code beyond what the plan requires
- Respond with Python code only — no markdown, no explanation{starter}{ret_spec}

Plan:
{plan_str}"""

    raw = ollama_call(EXECUTOR_SYSTEM, user_msg, MODEL_EXECUTOR)
    return clean_code(raw)


# ---------------------------------------------------------------------------
# Agent: Reflector
# ---------------------------------------------------------------------------

REFLECTOR_SYSTEM = (
    "You are a concise Python debugging assistant. "
    "You give targeted, actionable fix instructions in 3 sentences or fewer."
)

def run_reflector(code: str, error: str, error_history: list[str]) -> str:
    history_block = ""
    if error_history:
        history_block = (
            "\n\nPrevious errors that were NOT fixed by prior attempts "
            "(do not suggest the same fix again):\n"
            + "\n---\n".join(error_history)
        )

    user_msg = f"""A Python data science function failed with this error:
{error}
{history_block}

Rules:
- Identify the exact line causing the error and what to change
- If hardcoded column names (df['col']) are used, say to replace with df.iloc[:, N]
- If the failing code is unnecessary, say to remove it entirely
- Do NOT rewrite the code
- Do NOT suggest changes unrelated to this error

Code:
{code}"""

    return ollama_call(REFLECTOR_SYSTEM, user_msg, model=MODEL_REFLECTOR)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def tfc_eval(code: str, task_folder: str, model_id: str = "", retry: bool = True) -> tuple[bool, str]:
    path = os.path.join("data", task_folder, f"PEL-{model_id}_outputs.jsonl")
    record = {
        "plan": [{"code": code}],
        "output_dir": task_folder,
        "time_cost": 0,
        "error_list": [],
        "cost": 0,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record))

    if not retry:
        return True, ""

    res = subprocess.run(
        ["python", "-m", "experiments.evaluate_tmc",
         "--task_id", task_folder, "--model_id", f"PE-{model_id}"],
        cwd=DATASCIBENCH_DIR,
        capture_output=True,
        text=True,
    )

    output = res.stdout + res.stderr
    print(f"[TFC]\n{output}")

    # Scope to this task's section only — TFC runs multiple prompt IDs per call
    section = _extract_task_section(output, task_folder)
    if section is None:
        return False, f"Prompt ID {task_folder} not found in TFC output"

    if re.search(r"error occurred", section, re.IGNORECASE):
        m = re.search(r"<Error type>\s*(.*?)\s*-{10,}", section, re.DOTALL)
        error_msg = m.group(1).strip() if m else section.strip()
        return False, error_msg

    if re.search(r"<metric_output>\s*1\.0", section):
        return True, ""

    return False, f"No passing metric found for {task_folder}"


def _extract_task_section(output: str, task_folder: str) -> str | None:
    """Isolate the TFC output block belonging to task_folder only."""
    pattern = (
        rf"Evaluating Prompt ID: --{re.escape(task_folder)}--"
        rf"(.*?)"
        rf"(?=Evaluating Prompt ID:|$)"
    )
    m = re.search(pattern, output, re.DOTALL)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(task_prompt: str, task_folder: str, retry: bool = True) -> dict:
    starter     = extract_starter_code(task_prompt)
    return_spec = extract_return_spec(task_prompt)

    failure_history: list[str] = []
    error_history:   list[str] = []
    code = None
    plan = None

    for outer in range(OUTER_LIMIT if retry else 1):
        print(f"\n{'='*60}")
        print(f"OUTER LOOP {outer + 1}/{OUTER_LIMIT}  —  Planning")
        print(f"{'='*60}")

        plan = run_planner(task_prompt, failure_history or None)
        print(json.dumps(plan, indent=2))

        prev_code  = None
        reflection = None
        inner_failure_summary = []

        for inner in range(INNER_LIMIT if retry else 1):
            print(f"\n--- Executor attempt {inner + 1}/{INNER_LIMIT} ---")

            code = run_executor(
                plan,
                starter_code=starter,
                prev_code=prev_code,
                reflection=reflection,
                return_spec=return_spec,
            )
            print(f"[Code]\n{code}\n")

            passed, error = tfc_eval(code, task_folder, model_id=MODEL_EXECUTOR, retry=retry)

            if passed:
                print(f"✓ Passed on outer={outer+1}, inner={inner+1}")
                return {"plan": plan, "code": code}

            print(f"✗ Failed: {error}")
            error_history.append(error)
            inner_failure_summary.append(error)

            if retry:
                reflection = run_reflector(code, error, error_history[:-1])
                print(f"[Reflection] {reflection}")
                prev_code = code

        failure_history.append(
            f"Plan attempt {outer + 1} failed after {INNER_LIMIT} executor retries.\n"
            f"Errors encountered:\n" + "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(inner_failure_summary))
        )

    print("✗ All attempts exhausted.")
    return {"plan": plan, "code": code}


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_prompt(task_folder: str) -> str:
    path = os.path.join("data", task_folder, "prompt.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"prompt.json not found for task {task_folder}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["prompt"].replace("\\n", "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    RETRY_AGENT = len(sys.argv) == 1
    with open("./selected_tasks.json", "r", encoding="utf-8") as tf:
        task_list = json.load(tf)

    for i, task in enumerate(task_list[:16]):
        print(f"\n{'#'*60}")
        print(f"TASK {i+1}: {task}")
        print(f"{'#'*60}")
        prompt = load_prompt(task)
        result = run_pipeline(prompt, task, retry=RETRY_AGENT)
