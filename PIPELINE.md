# DataSciBench Agentic Data Analysis Pipeline

## Overview

This project evaluates a language model (Mistral via Ollama) on
DataSciBench tasks.

Each task folder follows this structure:

    data/
      bcbXX/
        prompt.json
        mistral_outputs.jsonl
        mistral_tmc_results.jsonl

These three files represent:

1.  The task specification
2.  The model's generated solution
3.  The benchmark evaluation results


------------------------------------------------------------------------

# 1. prompt.json - The Task Specification

**Created by:** DataSciBench

### What it contains:

-   The natural language data science prompt
-   Required input datasets
-   Ground-truth expectations
-   Evaluation metadata

### Conceptually:

    Prompt → What the model must solve

Example (simplified):

``` json
{
  "prompt": "Load the CSV file, clean missing values, compute mean of column A..."
}
```

This file defines the problem.

------------------------------------------------------------------------

# 2. mistral_outputs.jsonl - Model-Generated Code

**Created by:** Agent Script

**This is the model's answer.**

### What it contains:

``` json
{
  "plan": [
    {
      "code": "import pandas as pd\n..."
    }
  ],
  "output_dir": "bcb9",
  "time_cost": 0,
  "error_list": [],
  "cost": 0
}
```

### Conceptually:

    LLM(prompt) → Python code → saved here

This file contains: - The generated `task_func` - The full solution
code - What will be executed by the evaluator

It represents the independent variable in your experiment (model
behavior).

This file is what the evaluation runs on. 

------------------------------------------------------------------------

# 3. mistral_tmc_results.jsonl - Evaluation Metrics

**Created by:** `experiments.evaluate_tmc`\
**Produced AFTER evaluation runs.**

### What it contains:

-   Completion Rate (CR)
-   Success Rate (SR)
-   TMC score
-   Subtask metric breakdowns

Example (simplified):

``` json
{
  "completion_rate": 1.0,
  "success_rate": 0.8,
  "tmc_score": 0.85
}
```

### Conceptually:

    Execute mistral_outputs.jsonl
    ↓
    Run ground-truth validation
    ↓
    Compute CR / SR / TMC
    ↓
    Save results here

This file represents the dependent variable in your experiment
(performance metrics).

------------------------------------------------------------------------

# Full Pipeline

    prompt.json
        ↓
    LLM generates code
        ↓
    mistral_outputs.jsonl
        ↓
    Evaluator executes code
        ↓
    mistral_tmc_results.jsonl

------------------------------------------------------------------------

# Metric Definitions

### Completion Rate (CR)

Did the code execute without crashing? - Syntax/runtime errors → CR =
0 - Wrong logic but no crash → CR = 1

### Success Rate (SR)

Did the output pass ground-truth checks? - Correct format - Correct
columns - Correct values - Correct saved files

### TMC Score

Weighted aggregate of: - CR - SR - Subtask evaluation functions

TMC is the final benchmark score.

------------------------------------------------------------------------

# Experimental Setup

We run two conditions:

## Baseline

Single-pass code generation

Files: - `mistral_outputs.jsonl` - `mistral_tmc_results.jsonl`

## Retry Agent 

Iterative correction with self-evaluation

Files: - `mistral_retry_outputs.jsonl` -
`mistral_retry_tmc_results.jsonl`
              
