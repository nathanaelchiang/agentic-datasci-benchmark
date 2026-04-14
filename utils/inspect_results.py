import os
import json
import sys

MODEL_ID = "qwen2.5_7b_retry"
# MODEL_ID = "qwen2.5_3b_retry"

def inspect_task(task_id):
    """Inspect a single task's prompt, generated code, and eval results."""
    task_dir = os.path.join("data", task_id)

    # 1. Show the prompt
    prompt_path = os.path.join(task_dir, "prompt.json")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as f:
            prompt_data = json.load(f)
        print("=" * 60)
        print("PROMPT")
        print("=" * 60)
        print(prompt_data["prompt"][:500])
        if len(prompt_data["prompt"]) > 500:
            print("... (truncated)")
    else:
        print(f"No prompt.json found for {task_id}")
        return

    # 2. Show the generated code + attempt history
    # output_path = os.path.join(task_dir, "qwen2.5_7b_retry_outputs.jsonl")
    output_path = os.path.join(task_dir, f"{MODEL_ID}_outputs.jsonl")
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            output_data = json.loads(f.readline())

        print("\n" + "=" * 60)
        print(f"AGENT OUTPUT  |  Attempts: {output_data.get('attempts', '?')}  |  Exec Success: {output_data.get('final_success', '?')}")
        print("=" * 60)

        # Show history if available
        if "history" in output_data:
            for h in output_data["history"]:
                status = "PASS" if h["success"] else "FAIL"
                print(f"\n--- Attempt {h['attempt']} [{status}] ---")
                if not h["success"]:
                    print(f"Error: {h['stderr'][:300]}")
                print(f"Code:\n{h['code'][:400]}")
                if len(h["code"]) > 400:
                    print("... (truncated)")
        else:
            # No history, just show final code
            code = output_data["plan"][0]["code"]
            print(f"\nFinal Code:\n{code[:600]}")
            if len(code) > 600:
                print("... (truncated)")
    else:
        print(f"\nNo retry output found for {task_id}")

    # 3. Show eval results
    result_path = os.path.join(task_dir, f"{MODEL_ID}_tmc_results.jsonl")
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            result_data = json.loads(f.readline())

        print("\n" + "=" * 60)
        print(f"EVAL RESULTS  |  CR: {result_data.get('cr', 'N/A')}")
        print("=" * 60)

        if "TMC_results" in result_data:
            for tmc in result_data["TMC_results"]:
                name = tmc.get("function") or tmc.get("task_name", "unknown")
                result = tmc.get("result", "N/A")
                status = "PASS" if result == 1.0 else "FAIL" if result is not None else "ERROR"
                print(f"  [{status}] {name}: {result}")
    else:
        print(f"\nNo eval results found for {task_id}")


def summary():
    """Show pass/fail summary for all evaluated tasks."""
    data_dir = "data"
    passed = []
    failed = []

    for folder in sorted(os.listdir(data_dir)):
        result_path = os.path.join(data_dir, folder, f"{MODEL_ID}_tmc_results.jsonl")
        if not os.path.exists(result_path):
            continue

        with open(result_path, "r") as f:
            line = f.readline().strip()
            if not line:
                continue
            result_data = json.loads(line)

        cr = result_data.get("cr", 0)
        tmc_results = result_data.get("TMC_results", [])
        all_pass = all(t.get("result") == 1.0 for t in tmc_results if t.get("result") is not None)

        if cr >= 1.0 and all_pass:
            passed.append(folder)
        else:
            failed.append((folder, cr, tmc_results))

    print("=" * 60)
    print(f"SUMMARY: {len(passed)} passed, {len(failed)} failed")
    print("=" * 60)

    print(f"\nPASSED ({len(passed)}):")
    for t in passed:
        print(f"  {t}")

    print(f"\nFAILED ({len(failed)}):")
    for t, cr, tmcs in failed:
        tmc_str = ", ".join(
            f"{tmc.get('function', '?')}={tmc.get('result', 'N/A')}"
            for tmc in tmcs
        )
        print(f"  {t}  CR={cr}  [{tmc_str}]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python inspect_results.py summary        # Show all pass/fail")
        print("  python inspect_results.py bcb9           # Inspect one task")
        sys.exit(1)

    if sys.argv[1] == "summary":
        summary()
    else:
        inspect_task(sys.argv[1])