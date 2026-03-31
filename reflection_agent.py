import json
import os
from typing import TypedDict
from pathlib import Path
from langchain_ollama import OllamaLLM  
from langgraph.graph import StateGraph, END

llm = OllamaLLM(
        # model="mistral",
    model="mistral",
    base_url="http://localhost:11434",
    # temperature=0.1
)

class AgentState(TypedDict):
    task: str          
    code: str
    reflection: str
    retry_count: int

def generate_code(state: AgentState):
    print(f"Generating code for task: {state['task']}")
    prompt = f"""
{state['task']}
"""
    code = llm.invoke(prompt)
    print(f"Generated Code (Retry {state['retry_count']}):\n{code}\n{'-'*50}")
    return {
        "code": code.strip()
    }

def reflect_code(state: AgentState):
    print(f"Reflecting on code. Retry count: {state['retry_count']}")
    prompt = f"""
You are a code reviewer.
If there is no error in code, start your response with "no error". If there are errors, start with "has error" and list all issues in detail. Don't write any code, only provide feedback on what needs to be fixed. 
NO NEED to check for code style or efficiency, focus only on correctness and requirement compliance.

TASK REQUIREMENT:
{state['task']}

CODE:
{state['code']}

"""
    reflection = llm.invoke(prompt)
    print(f"Reflection (Retry {state['retry_count']}):\n{reflection}\n{'-'*50}")
    return {
        "reflection": reflection.strip(),
        "retry_count": state.get("retry_count", 0) + 1  
    }

def revise_code(state: AgentState):
    print(f"Revising code based on reflection. Retry count: {state['retry_count']}")
    prompt = f"""
Fix the code based on the review and requirement.
Output ONLY the corrected code.

Requirement: {state['task']}
Original code: {state['code']}
Review: {state['reflection']}

Fixed code:
"""
    new_code = llm.invoke(prompt)
    print(f"Revised Code (Retry {state['retry_count']}):\n{new_code}\n{'-'*50}")
    return {"code": new_code.strip()}

def should_continue(state: AgentState):
    ref = state["reflection"].lower()
    retry = state["retry_count"]

    # check if reflection indicates code is correct
    if "no error" in ref or retry >= 3:  
        return "valid"
    else:  
        return "need_fix"

workflow = StateGraph(AgentState)

workflow.add_node("generate", generate_code)
workflow.add_node("reflect", reflect_code)
workflow.add_node("revise", revise_code)

workflow.set_entry_point("generate")

workflow.add_edge("generate", "reflect")
workflow.add_conditional_edges(
    "reflect",
    should_continue,
    {
        "valid": END,
        "need_fix": "revise"
    }
)
workflow.add_edge("revise", "reflect")

app = workflow.compile()

if __name__ == "__main__":
    with open("selected_tasks.json", 'r', encoding='utf-8') as f:
        selected_tasks = json.load(f)
    
    data_dir = Path("data")
    total = len(selected_tasks)
    
    from datetime import datetime
    log_filename = f"reflection_agent_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_filename, 'w', encoding='utf-8')
    
    log_file.write(f"{'=' * 80}\n")
    log_file.write(f"Reflection Agent Batch Processing Log\n")
    log_file.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Total Tasks: {total}\n")
    log_file.write(f"{'=' * 80}\n\n")
    log_file.flush()
    

    for idx, task_folder in enumerate(selected_tasks, 1):
        task_dir = data_dir / task_folder
        prompt_file = task_dir / "prompt.json"
        output_file = task_dir / "mistral0331_outputs.jsonl"
        

        if not prompt_file.exists():
            msg = f"[{idx}/{total}]   SKIPPED: {task_folder} - prompt.json not found"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()
            continue
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
            
            task = prompt_data.get("prompt", "")
            if not task:
                msg = f"[{idx}/{total}] ⚠️  SKIPPED: {task_folder} - prompt is empty"
                print(msg)
                log_file.write(msg + "\n")
                log_file.flush()
                continue
            
            print(f"[{idx}/{total}]  Processing: {task_folder}...")
            
            result = app.invoke({
                "task": task,
                "code": "",
                "reflection": "",
                "retry_count": 0
            })
            
            output_data = {
                "plan": [{"code": result["code"]}],
                "output_dir": task_folder,
                "time_cost": 0,
                "error_list": [],
                "cost": 0
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False)
            
            log_file.write(f"\n{'=' * 80}\n")
            log_file.write(f"[{idx}/{total}] Task: {task_folder}\n")
            log_file.write(f"{'=' * 80}\n")
            log_file.write(f"\n GENERATED CODE:\n")
            log_file.write(f"{'-' * 80}\n")
            log_file.write(result["code"] + "\n")
            log_file.write(f"{'-' * 80}\n")
            
            log_file.flush()
            
            print(f"[{idx}/{total}]  COMPLETED: {task_folder}")
            print(f"           Output saved to: {output_file}\n")
            
        except Exception as e:
            error_msg = f"[{idx}/{total}]  ERROR in {task_folder}: {str(e)}"
            print(error_msg)
            log_file.write(f"\n{error_msg}\n\n")
            log_file.flush()
            continue
    
    log_file.write(f"\n{'=' * 80}\n")
    log_file.write(f"Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"ALL TASKS COMPLETED!\n")
    log_file.write(f"{'=' * 80}\n")
    log_file.close()
    
    print("=" * 60)
    print("ALL TASKS COMPLETED!")
    print(f"Log file saved: {log_filename}")
    print("=" * 60)