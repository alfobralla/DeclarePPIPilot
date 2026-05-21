# Experiments

Each script runs one RQ independently and writes quantitative outputs under `outputs/experiments/<rq>/`.
For KPI comparison, scripts use the metric assigned to `kpi_metric` in `ppinot4py_definition_code`, resolving dependent intermediate measures before generating signatures and types.
Each RQ also saves raw pipeline artifacts per run under `outputs/experiments/<rq>/raw/`.

## Common

All scripts require explicit provider/model configuration and accept:
- `--xes-path` (default `data/logs/DomesticDeclarations.xes`)
- `--provider` (`openai|anthropic|lmstudio`) (required)
- `--model` (required)
- `--base-url` (required for `lmstudio`; example `http://localhost:1234/v1`)
- `--api-key-env` (optional env var name for API key lookup)
- `--temperature` (optional)
- `--max-output-tokens` (optional)
- `--json-retries` (default `2`)
- `--use-attributes`
- `--n` (runs, configurable per RQ)

Credential defaults by provider:
- `openai`: `OPENAI_API_KEY`
- `anthropic`: `ANTHROPIC_API_KEY`
- `lmstudio`: no key required unless `--api-key-env` is set

## RQ1: KPI Stability (PPINot definition based)

```bash
python experiments/rq1_stability.py --provider openai --model gpt-5.2 --n 20
```

Main outputs:
- `rq1_runs.csv`: per-run KPI/PPI counts and KPI type counts.
- `rq1_kpi_analysis.csv`: KPI-level metric signature, final metric type, and composition profile.
- `rq1_pairwise_jaccard.csv`: pairwise comparison metrics across runs, including exact-match count/Jaccard, fuzzy string similarity, and heuristic exact-match (`same KPI value` + `metric text similarity > 0.8`).
- `rq1_signature_frequency.csv`: signature frequency and core-signature flag.
- `rq1_composition_frequency.csv`: frequency of composition profiles (e.g. `AggregatedMeasure->DerivedMeasure->CountMeasure`).
- `rq1_summary.json`: aggregate stability metrics.

## RQ2: Retry usefulness

```bash
python experiments/rq2_retry.py --provider anthropic --model claude-sonnet-4-5 --n 20 --max-retries 2
```

Main outputs:
- `rq2_runs.csv`: baseline error rate (`max_retries=0`), recovered count, final failures after retry.
- `rq2_kpi_level.csv`: KPI-level baseline vs retry status and errors.
- `rq2_summary.json`: average recovery and residual-failure metrics.

## RQ3: CV-threshold filtering impact

```bash
python experiments/rq3_cv_filter.py --provider lmstudio --model local-model --base-url http://localhost:1234/v1 --n 20 --time-grouper-freq 1ME --cv-threshold 0.2
```

Main outputs:
- `rq3_runs.csv`: per-run counts of `ok/skipped/error` variability analysis and filtered KPIs.
- `rq3_variability_by_kpi.csv`: KPI-level CV and filtering flags with metric type/profile.
- `rq3_signature_filtering.csv`: filtering frequency by normalized `kpi_metric` signature.
- `rq3_summary.json`: aggregate filtering rates.

## RQ4: min_support sensitivity

```bash
python experiments/rq4_min_support.py --provider openai --model gpt-5.2 --n 10 --supports 0.8,0.6,0.4,0.2 --max-constraints 600
```

Main outputs:
- `rq4_runs.csv`: per-run KPI/PPI counts, success rates, and KPI type counts by support.
- `rq4_support_summary.csv`: aggregated metrics per support value.
- `rq4_signatures.csv`: raw metric signature occurrences per support/run, including metric type/profile.
- `rq4_composition_summary.csv`: composition profile counts per support value.
- `rq4_support_signature_overlap.csv`: overlap between support values including exact-match count and Jaccard.
- `rq4_summary.json`: run configuration metadata.

## RQ5: KPI/PPI Alignment

```bash
python experiments/rq5_alignment.py --provider openai --model gpt-5.2 --n 10
```

Optional separate evaluator model:

```bash
python experiments/rq5_alignment.py \
  --provider openai --model gpt-5.2 \
  --eval-provider anthropic --eval-model claude-sonnet-4-5
```

Main outputs:
- `rq5_runs.csv`: per-run alignment aggregates.
- `rq5_alignment_by_kpi.csv`: KPI-level alignment score/rationale, plus approximation text.
- `rq5_summary.json`: overall alignment summary and model configs.
