import requests
import json
import subprocess
import tempfile
import os
import re
from typing import Any, Dict, List

# Config
BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"
# MODEL_ID = "deepseek-coder:6.7b"
# MODEL_ID = "qwen2.5:1.5b"
# MODEL_ID = "qwen2.5-coder:7b"
MODEL_ID = "qwen2.5-coder:7b-instruct"
MAX_RETRIES = 3
TIMEOUT = 15


# YAML loader
def load_yaml_simple(path: str) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML contents.

    Raises:
        RuntimeError: If PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# LLM Call
def generate_code(prompt: str) -> str:
    """Send a prompt to the LLM and return the raw response text.

    Args:
        prompt: The user prompt to send.

    Returns:
        The LLM's response content, or an empty string on failure.
    """
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_ID,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert Python data scientist. "
                            "Return ONLY valid Python code.\n\n"
                            "Requirements:\n"
                            "- Implement exactly the requested task_func signature.\n"
                            "- Include ALL required imports at the top of your code.\n"
                            "- Do not assume placeholder column names like column1/column2/feature1/feature2.\n"
                            "- Infer schema from the provided data when needed.\n"
                            "- Avoid external downloads and internet access.\n"
                            "- Avoid NLTK tokenizers requiring downloaded resources.\n"
                            "- For ML tasks, separate X and y correctly: use X = df.drop(columns=[target_column]).\n"
                            "- Return exactly the required output objects.\n"
                            "- Do NOT add column existence validation guards.\n"
                            "- If the function takes a 'groups' parameter, those values are row filter values "
                            "in a categorical column, NOT column names.\n"
                            "- The input df may be a dict — convert with pd.DataFrame(df) if needed.\n"
                            "- If using numpy, write: import numpy as np\n"
                            "- If using pandas, write: import pandas as pd\n"
                            "- If using matplotlib, write: import matplotlib.pyplot as plt\n"
                            "- If using seaborn, write: import seaborn as sns\n"
                        ),
                    },
                    {"role": "user", "content": str(prompt)},
                ],
                "stream": False,
                "max_tokens": 1000,
            },
            timeout=60,
        )
        data = response.json()
        if "choices" not in data:
            print("LLM ERROR RESPONSE:", data)
            return ""
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print("LLM call exception:", e)
        return ""


# Code Extraction (infrastructure only)
def extract_code(text: str) -> str:
    """Extract raw Python code from an LLM response.

    Strips markdown fences, leading language identifiers, and any text
    before the first import/from/def statement.

    Args:
        text: Raw LLM response text.

    Returns:
        Cleaned Python source code string.
    """
    if not text:
        return ""
    text = text.strip()

    # Remove markdown fences
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]

    # Remove leading language identifier
    lines = text.splitlines()
    if lines and lines[0].strip().lower() in ["python", "py"]:
        lines = lines[1:]

    cleaned = "\n".join(lines).strip()

    # Start at first meaningful Python statement
    for key in ["import", "from", "def"]:
        idx = cleaned.find(key)
        if idx != -1:
            cleaned = cleaned[idx:]
            break

    return cleaned


# Minimal infrastructure patches only
def apply_infrastructure_patches(code: str) -> str:
    """Apply environment-level patches to generated code.

    Only handles infrastructure concerns (e.g. matplotlib backend) rather
    than substantive logic fixes, which are left to LLM reflection.

    Args:
        code: Generated Python source code.

    Returns:
        Patched source code.
    """

    # Ensure matplotlib uses non-interactive backend (environment concern)
    if "matplotlib" in code and "matplotlib.use" not in code:
        code = "import matplotlib\nmatplotlib.use('Agg')\n" + code

    return code


# Error classification
def classify_error(stderr: str) -> str:
    """Classify a stderr string into a known error category.

    Args:
        stderr: The stderr output from a failed validation run.

    Returns:
        A string error category (e.g. ``"syntax"``, ``"missing_import"``,
        ``"key_error"``). Returns ``"unknown"`` if no category matches.
    """
    if "FixedLocator" in stderr or "does not match the number of labels" in stderr:
        return "matplotlib_tick_mismatch"
    if "Could not resolve task_func arguments" in stderr:
        return "input_mapping"
    if "numeric_only" in stderr or "could not convert" in stderr.lower() or "non-numeric" in stderr.lower():
        return "mixed_types"
    if "must be a DataFrame" in stderr or "has no attribute 'empty'" in stderr:
        return "bad_isinstance_guard"
    if "'dict' object has no attribute" in stderr or "'list' object has no attribute" in stderr:
        return "dict_input"
    if re.search(r'File "<string>"[^\n]*\n(ValueError|AssertionError):', stderr):
        return "user_raised_valueerror"
    if "NameError" in stderr:
        return "missing_import"
    if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
        return "missing_import"
    if "SyntaxError" in stderr:
        return "syntax"
    if "TypeError" in stderr:
        return "type_error"
    if "KeyError" in stderr:
        return "key_error"
    if "IndexError" in stderr:
        return "index_error"
    if "AttributeError" in stderr:
        return "attribute_error"
    if "Timeout" in stderr:
        return "timeout"
    return "unknown"


# Validation script
def _build_validation_script(code: str, task_folder: str) -> str:
    """Build a self-contained Python script that validates generated code.

    The script loads the task metric, executes the generated code, resolves
    ``task_func`` arguments from the metric's input globals, calls the
    function, and prints a JSON result dict to stdout.

    Args:
        code: Generated Python source code defining ``task_func``.
        task_folder: Task subfolder name under ``data/`` and ``metric/``.

    Returns:
        Source code of the validation script as a string.
    """
    metric_path = os.path.abspath(os.path.join("metric", task_folder, "metric.yaml"))
    data_dir = os.path.abspath(os.path.join("data", task_folder))

    return f"""
import os
import re
import json
import traceback
import inspect
import matplotlib
matplotlib.use("Agg")

os.chdir({repr(data_dir)})

try:
    import yaml
except ImportError:
    print(json.dumps({{"success": False, "stderr": "PyYAML is required for validation"}}))
    raise SystemExit(0)

metric_path = {repr(metric_path)}
with open(metric_path, "r", encoding="utf-8") as f:
    metric_config = yaml.safe_load(f)

inputs_code = metric_config["data"]
try:
    import pandas as _pd
    import numpy as _np
    import matplotlib.pyplot as _plt
    globals_dict = {{'pd': _pd, 'np': _np, 'plt': _plt}}
except ImportError:
    globals_dict = {{}}
result = {{"success": False, "stderr": "", "stdout": "", "static_issues": []}}
generated_code = {repr(code)}

def record_failure(msg):
    result["success"] = False
    result["stderr"] = msg
    print(json.dumps(result))
    raise SystemExit(0)

# ---- execute metric inputs ----
try:
    exec(inputs_code, globals_dict)
except Exception:
    record_failure("Failed to execute metric inputs:\\n" + traceback.format_exc())

# ---- execute generated code ----
try:
    exec(generated_code, globals_dict)
except Exception:
    record_failure("Failed to define generated code:\\n" + traceback.format_exc())

task_func = globals_dict.get("task_func")
if task_func is None:
    record_failure("Generated code does not define task_func.")

# ---- resolve arguments ----
try:
    sig = inspect.signature(task_func)
    params = list(sig.parameters.keys())
    param_defaults = {{
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }}
except Exception:
    record_failure("Could not inspect task_func signature:\\n" + traceback.format_exc())

alias_map = {{
    "csv_file": ["csv_file", "csv_file_path", "file_path", "filename", "f_1"],
    "csv_file_path": ["csv_file_path", "csv_file", "file_path", "filename", "f_1"],
    "data": ["data", "df", "dataframe", "data_matrix", "f_1", "args"],
    "target_column": ["target_column", "label_col", "target", "y_col"],
    "data_matrix": ["data_matrix", "data", "df", "args"],
    "text": ["text", "input_text", "sentence"],
    "list_of_pairs": ["list_of_pairs", "pairs", "data", "input_data", "args"],
    "df": ["df", "data", "dataframe"],
    "result": ["result", "args", "data", "input_data"],
}}

_skip_keys = {{"__builtins__", "__name__", "__doc__", "__package__",
               "__loader__", "__spec__", "__annotations__",
               "task_func"}}

args_list = []
missing = []

# Fast path: if 'args' is a tuple/list in globals, unpack positionally.
_args_val = globals_dict.get('args')
if isinstance(_args_val, (tuple, list)):
    args_list = list(_args_val[:len(params)])
    for _p in params[len(_args_val):]:
        if _p in param_defaults:
            args_list.append(param_defaults[_p])
        else:
            missing.append(_p)
else:
    for p in params:
        resolved = False

        # 1. Exact match
        if p in globals_dict and p not in _skip_keys:
            args_list.append(globals_dict[p])
            resolved = True

        # 2. Alias map
        if not resolved:
            for alias in alias_map.get(p, []):
                if alias in globals_dict and alias not in _skip_keys:
                    args_list.append(globals_dict[alias])
                    resolved = True
                    break

        # 3. Fuzzy match
        if not resolved:
            candidates = [
                k for k in globals_dict
                if k not in _skip_keys
                and not k.startswith("_")
                and not callable(globals_dict[k])
                and not isinstance(globals_dict[k], type)
                and (p in k or k in p)
            ]
            if len(candidates) == 1:
                args_list.append(globals_dict[candidates[0]])
                resolved = True
            elif len(candidates) > 1:
                best = min(candidates, key=len)
                args_list.append(globals_dict[best])
                resolved = True

        # 4. Default value
        if not resolved and p in param_defaults:
            args_list.append(param_defaults[p])
            resolved = True

        if not resolved:
            missing.append(p)

# Last resort: 1 unresolved param + 1 data variable
if missing:
    import types as _types
    data_vars = [
        k for k in globals_dict
        if k not in _skip_keys
        and not k.startswith("_")
        and not callable(globals_dict[k])
        and not isinstance(globals_dict[k], type)
        and not isinstance(globals_dict[k], _types.ModuleType)
    ]
    if len(missing) == 1 and len(data_vars) == 1:
        args_list.append(globals_dict[data_vars[0]])
        missing = []

    if missing:
        available = [k for k in globals_dict if k not in _skip_keys and not k.startswith("_")]
        record_failure(
            f"Could not resolve task_func arguments: {{missing}}\\n"
            f"Available globals after inputs_code: {{available}}"
        )

# ---- call task_func ----
try:
    output = task_func(*args_list)
except Exception:
    arg_types = ", ".join(params[i] + "=" + type(v).__name__ for i, v in enumerate(args_list))
    record_failure("task_func raised an exception (arg types: " + arg_types + "):\\n" + traceback.format_exc())

result["success"] = True
result["output_type"] = str(type(output))
try:
    result["output_repr"] = repr(output)[:300]
except Exception:
    result["output_repr"] = "<unavailable>"

print(json.dumps(result))
"""


def validate_code(code: str, task_folder: str) -> Dict[str, Any]:
    """Validate generated code by running it in an isolated subprocess.

    Performs a static syntax check first, then writes and executes a
    validation script. Parses the JSON result from stdout.

    Args:
        code: Generated Python source code to validate.
        task_folder: Task subfolder name used to locate metric and data files.

    Returns:
        A dict with keys: ``success`` (bool), ``stdout``, ``stderr``,
        ``timeout`` (bool), ``static_issues`` (list), and optionally
        ``output_type`` and ``output_repr``.
    """
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as e:
        return {"success": False, "stdout": "", "stderr": f"SyntaxError: {e}",
                "timeout": False, "static_issues": []}

    script = _build_validation_script(code, task_folder)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        temp_file = f.name
        f.write(script)

    try:
        result = subprocess.run(
            ["python", temp_file], capture_output=True, text=True, timeout=TIMEOUT,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        parsed = None
        if stdout:
            for line in reversed(stdout.splitlines()):
                try:
                    parsed = json.loads(line)
                    break
                except Exception:
                    continue
        if parsed is not None:
            return {
                "success": bool(parsed.get("success", False)),
                "stdout": stdout,
                "stderr": parsed.get("stderr", stderr),
                "timeout": False,
                "static_issues": parsed.get("static_issues", []),
                "output_type": parsed.get("output_type"),
                "output_repr": parsed.get("output_repr"),
            }
        return {"success": result.returncode == 0, "stdout": stdout,
                "stderr": stderr or "Validation did not return structured output.",
                "timeout": False, "static_issues": []}
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "Timeout: validation exceeded limit",
                "timeout": True, "static_issues": []}
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


# Reflection prompt
def build_reflection_prompt(original_prompt: str, history: List[Dict[str, Any]]) -> str:
    """Build a reflection prompt asking the LLM to fix its previous code.

    Includes the original task, all prior attempts with their errors, a
    fixed rule set, and a targeted hint derived from the most recent error.
    Appends a hard-rewrite warning if the same error repeated twice.

    Args:
        original_prompt: The original task prompt given to the LLM.
        history: List of attempt dicts, each with ``code`` and ``result`` keys.

    Returns:
        A fully formatted reflection prompt string.
    """
    prompt = f"""You are an expert Python data scientist. You must fix your previous code.

Original task:
{original_prompt}

Previous failed attempts:
"""
    for i, h in enumerate(history):
        error_msg = h["result"]["stderr"] or "Unknown error"
        if len(error_msg) > 800:
            error_msg = error_msg[:800] + "\n... (truncated)"
        prompt += f"""
--- Attempt {i+1} ---
Code:
{h['code']}

Error:
{error_msg}
"""

    prompt += """
Fix all issues and return ONLY valid Python code.

IMPORTANT RULES:
1. Include ALL required imports at the top (numpy, pandas, matplotlib, sklearn, scipy, etc.).
2. Implement exactly the requested task_func signature.
3. Do NOT assume placeholder columns like column1, column2, feature1, feature2.
   Infer column names from the actual data dynamically.
4. Avoid external downloads and internet access.
5. Do NOT use nltk.word_tokenize or other tokenizers requiring downloads.
   Use re.findall() as an alternative.
6. For ML tasks with a target_column parameter:
   - y = df[target_column]
   - X = df.drop(columns=[target_column])
   Do NOT use X = df.copy() — this leaks the target.
7. Return exactly the requested output objects.
8. Do NOT add column existence validation guards or isinstance checks.
   The evaluation harness always provides valid input — trust it.
9. If a 'groups' parameter is present, those are ROW VALUES to filter by
   (e.g. df[df['category'] == group]), NOT column names.
10. The input 'df' may be a dict or list — convert at the top of your function:
    if not isinstance(df, pd.DataFrame): df = pd.DataFrame(df)
11. If the data may contain non-numeric columns and you need numeric operations,
    use df.select_dtypes(include=[np.number]).
12. For matplotlib tick operations, do NOT use FixedLocator manually.
    Use plt.xticks() or ax.set_xticklabels() after plotting instead.
"""

    # Add targeted hint based on most recent error type
    if history:
        err = history[-1]["result"]["stderr"]
        error_type = classify_error(err)

        if error_type == "input_mapping":
            match = re.search(r"Available globals after inputs_code: (\[.*?\])", err, re.DOTALL)
            if match:
                try:
                    available = eval(match.group(1))
                    _libs = {"pd", "np", "plt", "sns", "os", "re", "json", "math",
                             "random", "datetime", "collections"}
                    data_vars = [v for v in available if v not in _libs]
                    if data_vars:
                        prompt += (
                            f"\nHINT: The metric exposes these input variables: {data_vars}. "
                            f"Your task_func parameter names must match one of these.\n"
                        )
                except Exception:
                    pass

        elif error_type == "dict_input":
            prompt += (
                "\nHINT: The input is a dict, not a DataFrame. "
                "Add: if not isinstance(df, pd.DataFrame): df = pd.DataFrame(df)\n"
            )

        elif error_type == "missing_import":
            # Extract the missing name from the error
            name_match = re.search(r"name '(\w+)' is not defined", err)
            if name_match:
                prompt += f"\nHINT: '{name_match.group(1)}' is not defined. Add the missing import.\n"
            else:
                prompt += "\nHINT: You have a missing import. Check all names are imported.\n"

        elif error_type == "key_error":
            key_match = re.search(r"KeyError: ['\"](.+?)['\"]", err)
            if key_match:
                prompt += (
                    f"\nHINT: KeyError for '{key_match.group(1)}'. "
                    f"This column/key doesn't exist in the data. "
                    f"Infer actual column names dynamically from the DataFrame.\n"
                )

        elif error_type == "attribute_error":
            prompt += (
                "\nHINT: AttributeError — you may be calling a method on the wrong type. "
                "Check whether your variable is a DataFrame, Series, dict, list, or numpy array.\n"
            )

        elif error_type == "type_error":
            prompt += "\nHINT: TypeError — check argument types and function signatures.\n"

        elif error_type == "mixed_types":
            prompt += (
                "\nHINT: The DataFrame contains non-numeric columns. "
                "Use df.select_dtypes(include=[np.number]) before numeric operations.\n"
            )

        elif error_type == "bad_isinstance_guard":
            prompt += (
                "\nHINT: Do NOT add isinstance checks. Trust the input from the evaluation harness.\n"
            )

        elif error_type == "user_raised_valueerror":
            prompt += (
                "\nHINT: Your code is raising a ValueError that rejects valid input. "
                "Remove strict validation guards. Trust the input.\n"
            )

        elif error_type == "matplotlib_tick_mismatch":
            prompt += (
                "\nHINT: Do NOT use FixedLocator. Use plt.xticks() after plotting.\n"
            )

        elif error_type == "syntax":
            prompt += "\nHINT: Fix the syntax error before anything else.\n"

        elif error_type == "timeout":
            prompt += (
                "\nHINT: Your code timed out — possible infinite loop or very slow algorithm. "
                "Simplify your approach.\n"
            )

    # Force rewrite if repeated identical failure
    if len(history) >= 2:
        if history[-1]["result"]["stderr"] == history[-2]["result"]["stderr"]:
            prompt += (
                "\n\nWARNING: The same error occurred twice in a row. "
                "Your previous approach is fundamentally wrong. "
                "Start over with a completely different solution strategy.\n"
            )

    return prompt


# Solve One Task
def solve_task(task_folder: str) -> bool:
    """Attempt to solve a single benchmark task with iterative LLM retries.

    Generates code, validates it, and on failure builds a reflection prompt
    and retries up to ``MAX_RETRIES`` times. Writes the final attempt to a
    JSONL output file.

    Args:
        task_folder: Task subfolder name under ``data/``.

    Returns:
        ``True`` if any attempt passed validation, ``False`` otherwise.
    """
    safe_model_id = MODEL_ID.replace(":", "_").replace("/", "_")
    output_path = os.path.join("data", task_folder, f"{safe_model_id}_retry_v3_outputs.jsonl")
    prompt_path = os.path.join("data", task_folder, "prompt.json")

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_data = json.load(f)

    original_prompt = prompt_data["prompt"]
    history: List[Dict[str, Any]] = []
    raw = generate_code(original_prompt)

    if not raw.strip():
        print("Initial generation failed.")
        return False

    code = extract_code(raw)
    final_success = False

    for attempt in range(MAX_RETRIES):
        print(f"\n[{task_folder}] Attempt {attempt+1}/{MAX_RETRIES}")

        # Only infrastructure-level patches
        code = apply_infrastructure_patches(code)

        print("\n--- CODE ---")
        print(code[:800])

        result = validate_code(code, task_folder)

        print(f"  Success: {result['success']}")
        if result.get("output_type"):
            print(f"  Output type: {result['output_type']}")
        if result["stderr"]:
            print(f"  Error: {result['stderr'][:400]}")
        if result.get("static_issues"):
            print("  Static issues:")
            for issue in result["static_issues"]:
                print(f"   - {issue}")

        history.append({"code": code, "result": result})

        if result["success"]:
            final_success = True
            print(f"  Task {task_folder} passed validation on attempt {attempt+1}")
            break

        # Stop if model repeated identical code
        if len(history) > 1 and code == history[-2]["code"]:
            print("  Model repeated same code, stopping.")
            break

        # Build reflection prompt and retry
        reflection_prompt = build_reflection_prompt(original_prompt, history)
        raw = generate_code(reflection_prompt)
        if not raw.strip():
            print("  Retry generation failed.")
            break
        code = extract_code(raw)

    # Save output
    entry = {
        "plan": [{"code": code}],
        "output_dir": task_folder,
        "time_cost": 0,
        "error_list": [h["result"]["stderr"] for h in history if not h["result"]["success"]],
        "cost": 0,
        "attempts": len(history),
        "final_success": final_success,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"  Saved -> {output_path}")
    return final_success


# Run All Tasks
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_ID, help="Ollama model ID (e.g. qwen2.5:7b)")
    parser.add_argument("--all", action="store_true", help="Run all bcb tasks instead of selected_tasks.json")
    parser.add_argument("tasks", nargs="*", help="Specific task folders to run")
    args = parser.parse_args()

    MODEL_ID = args.model

    if args.tasks:
        all_tasks = args.tasks
    elif args.all:
        all_tasks = sorted([
            folder for folder in os.listdir("data")
            if folder.startswith("bcb") and os.path.isdir(os.path.join("data", folder))
        ])
    else:
        with open("selected_tasks.json") as f:
            all_tasks = sorted(json.load(f))

    print(f"Found {len(all_tasks)} tasks\n")
    success_count = 0
    fail_count = 0

    for task in all_tasks:
        safe_mid = MODEL_ID.replace(":", "_").replace("/", "_")
        output_path = os.path.join("data", task, f"{safe_mid}_retry_v3_outputs.jsonl")
        if os.path.exists(output_path):
            print(f"Skipping {task} (output already exists)")
            continue
        print(f"\n{'='*40}")
        print(f"Task: {task}")
        print(f"{'='*40}")
        try:
            passed = solve_task(task)
            if passed:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  FATAL error on {task}: {e}")
            fail_count += 1

    total = success_count + fail_count
    print(f"\n{'='*40}")
    print("SUMMARY")
    print(f"{'='*40}")
    print(f"Total:   {total}")
    if total > 0:
        print(f"Passed:  {success_count} ({100 * success_count / total:.1f}%)")
    print(f"Failed:  {fail_count}")