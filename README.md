# PPIDeclarePilotSandbox

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20338703.svg)](https://doi.org/10.5281/zenodo.20338703)

Pipeline to:
- mine DECLARE constraints from event logs,
- generate KPI proposals with an LLM,
- generate PPINot code,
- execute PPIs on the log,
- produce reports and experiment artifacts.

## Project Structure

- `src/declareppipilot/`: core library (`pipeline.py`, `models.py`, `llm.py`, prompts/schemas)
- `tests/`: runnable integration scripts for the pipeline
- `experiments/`: RQ experiment scripts (`rq1` ... `rq5`)
- `data/logs/`: input logs (e.g. `DomesticDeclarations.xes`)
- `outputs/`: generated artifacts
- `requirements.txt`: Python dependencies

## Requirements

```bash
pip install -r requirements.txt
```

Depending on provider, configure credentials:
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- LM Studio: `LMSTUDIO_BASE_URL` and `LMSTUDIO_MODEL` (optional key env)

## Quick Start

Run from project root.

### 1. Run full pipeline test

```bash
python tests/test_full_pipeline.py
```

The script auto-detects available providers and writes outputs into `outputs/`.

### 2. Run step-by-step pipeline test

```bash
python tests/test_declareppipilot.py
```

### 3. Run web dashboard

```bash
uvicorn webapp.main:app --reload
```

Then open:
- `http://127.0.0.1:8000/`

Dashboard features:
- upload `.xes` log
- set provider/model (and optional API key)
- optional time grouping via quantity + unit (internally mapped to `xWE`, `xME`, `xYE`)
- optional `min_support`
- KPI dashboard grouped by business goal with AI-generated readable PPI text and technical details on demand

## Using the Library

If running scripts outside this repo root, set `PYTHONPATH`:

```bash
export PYTHONPATH=src
```

Main API entry points:
- `run_full_pipeline(...)`
- `mine_declare_model_and_constraints_df(...)`
- `generate_kpis_from_declare_table(...)`
- `generate_ppinot_from_kpi_set(...)`
- `execute_ppinot_kpis(...)`
- `build_kpi_execution_report(...)`

## Experiments

RQ scripts and usage are documented in:
- [`experiments/README.md`](experiments/README.md)

They support multiple providers (`openai`, `anthropic`, `lmstudio`) and save timestamped outputs under `outputs/experiments/...`.

## Notes

- `run_full_pipeline` now includes automatic generation of human-readable PPI definitions from `kpi_metric_str`.
- Generated code execution includes safety checks before `exec`.
