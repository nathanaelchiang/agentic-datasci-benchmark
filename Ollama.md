# Using Ollama (Local LLM Backend)

This project uses Ollama to run a local language model for code generation. No external API keys are required.

---

## 1. Install Ollama

Download and install Ollama from [https://ollama.com](https://ollama.com).

After installation, verify:

```bash
ollama --version
```

## 2. Pull a Model

We recommend using the Mistral model:

```bash
ollama pull mistral
```

Verify installed models:

```bash
ollama list
```

## 3. Start Ollama

On Windows and macOS, Ollama usually runs automatically in the background. To manually start:

```bash
ollama serve
```

The default API endpoint is `http://localhost:11434`.

## 4. Run the Agent

Generate a solution for a DataSciBench task:

```bash
python simple_agent.py
```

This will:
- Call the local LLM
- Generate a `task_func`
- Save results to `data/<task_id>/mistral_outputs.jsonl`

## 5. Run Official Evaluation

```bash
python -m experiments.evaluate_tmc --model_id mistral --task_id bcb9
```

This will:
- Execute your generated function
- Run official unit tests
- Compute CR and TMC metrics
- Save results to `data/<task_id>/mistral_tmc_results.jsonl`

## ⚠️ Important: Headless Plotting

Evaluation requires matplotlib to use a non-GUI backend. `evaluate_tmc.py` has been patched with:

```python
import matplotlib
matplotlib.use("Agg")
```

This prevents blocking from `plt.show()` during automated evaluation.

---

## Requirements

Install dependencies:

```bash
pip install pandas matplotlib seaborn requests pyyaml
```