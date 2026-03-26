import requests
import json
import subprocess
import tempfile
import os

# =========================
# Config
# =========================
BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"
MODEL_ID = "qwen2.5:7b"
MAX_RETRIES = 3
TIMEOUT = 10


# =========================
# LLM Call (OpenAI-compatible)
# =========================
def generate_code(prompt):
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": "You are an expert Python data scientist. Return ONLY valid Python code with no explanations or markdown."},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
    )
    return response.json()["choices"][0]["message"]["content"]


# =========================
# Robust Code Extraction
# =========================
def extract_code(text):
    text = text.strip()

    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]

    lines = text.splitlines()

    # remove language tag
    if lines and lines[0].strip().lower() in ["python", "py"]:
        lines = lines[1:]

    cleaned = "\n".join(lines).strip()

    # anchor to first import or def
    for key in ["import", "def"]:
        if key in cleaned:
            cleaned = cleaned[cleaned.index(key):]
            break

    return cleaned


# =========================
# Syntax + Execution Check
# =========================
def validate_code(code, task_folder):
    """
    Since we don't have dedicated test code, we:
    1. Check syntax via compile()
    2. Run the code in a subprocess with the task's data dir
       and check for runtime errors
    """
    # Step 1: syntax check
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"SyntaxError: {e}",
            "timeout": False
        }

    # Step 2: execution check
    data_dir = os.path.abspath(os.path.join("data", task_folder))

    # Wrap code so it runs from the task's data directory
    wrapped_code = f"""
import os
os.chdir({repr(data_dir)})

{code}
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        temp_file = f.name
        f.write(wrapped_code)

    try:
        result = subprocess.run(
            ["python", temp_file],
            capture_output=True,
            text=True,
            timeout=TIMEOUT
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timeout": False
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Timeout: execution exceeded limit",
            "timeout": True
        }

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


# =========================
# Reflection Prompt
# =========================
def build_reflection_prompt(original_prompt, history):
    prompt = f"""You are an expert Python data scientist.

Original task:
{original_prompt}

Previous failed attempts:
"""

    for i, h in enumerate(history):
        error_msg = h["result"]["stderr"]
        # Truncate long errors to keep prompt manageable
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "\n... (truncated)"

        prompt += f"""
--- Attempt {i+1} ---
Code:
{h['code']}

Error:
{error_msg}
"""

    prompt += """
Fix all issues and return ONLY valid, complete Python code.
Do not include explanations or markdown formatting.
"""

    return prompt


# =========================
# Solve One Task
# =========================
def solve_task(task_folder):
    prompt_path = os.path.join("data", task_folder, "prompt.json")

    with open(prompt_path, "r") as f:
        prompt_data = json.load(f)

    original_prompt = prompt_data["prompt"]

    history = []
    code = extract_code(generate_code(original_prompt))
    final_success = False

    for attempt in range(MAX_RETRIES):
        print(f"\n[{task_folder}] Attempt {attempt+1}/{MAX_RETRIES}")

        result = validate_code(code, task_folder)

        print(f"  Success: {result['success']}")
        if result["stderr"]:
            print(f"  Error: {result['stderr'][:200]}")
        if result["stdout"]:
            print(f"  Output: {result['stdout'][:200]}")

        history.append({"code": code, "result": result})

        if result["success"]:
            final_success = True
            print(f"  Task {task_folder} passed on attempt {attempt+1}")
            break

        # Retry with reflection
        reflection_prompt = build_reflection_prompt(original_prompt, history)
        code = extract_code(generate_code(reflection_prompt))

    # Save output in DataSciBench format
    safe_model_id = MODEL_ID.replace(":", "_").replace("/", "_")
    output_path = os.path.join("data", task_folder, f"{safe_model_id}_retry_outputs.jsonl")

    entry = {
        "plan": [{"code": code}],
        "output_dir": task_folder,
        "time_cost": 0,
        "error_list": [h["result"]["stderr"] for h in history if not h["result"]["success"]],
        "cost": 0,
        "attempts": len(history),
        "final_success": final_success
    }

    with open(output_path, "w") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"  Saved → {output_path}")
    return final_success


# =========================
# Run All Tasks
# =========================
if __name__ == "__main__":
    import sys

    # Optional: pass a specific task as argument
    # Usage: python retry_agent.py bcb9
    if len(sys.argv) > 1:
        all_tasks = [sys.argv[1]]
    else:
        all_tasks = sorted([
            folder for folder in os.listdir("data")
            if folder.startswith("bcb") and os.path.isdir(os.path.join("data", folder))
        ])

    print(f"Found {len(all_tasks)} tasks\n")

    success_count = 0
    fail_count = 0

    for task in all_tasks:
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
    print(f"SUMMARY")
    print(f"{'='*40}")
    print(f"Total:   {total}")
    print(f"Passed:  {success_count} ({100*success_count/total:.1f}%)" if total > 0 else "")
    print(f"Failed:  {fail_count}")