from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common import (
    build_llm_config_from_args,
    configure_logging,
    ensure_llm_credentials,
    load_event_log,
    parse_common_args,
    save_json,
    save_pipeline_run_artifacts,
)
from declareppipilot import build_kpi_execution_report, run_multi_execution_pipeline
from declareppipilot.models import LLMConfig


def _build_selector_llm_config(args, default_llm_config: LLMConfig) -> LLMConfig:
    if not args.selector_provider:
        return default_llm_config
    if not args.selector_model:
        raise ValueError("--selector-model is required when --selector-provider is set.")
    return LLMConfig(
        provider=args.selector_provider,
        model=args.selector_model,
        base_url=args.selector_base_url,
        api_key_env=args.selector_api_key_env,
        temperature=args.selector_temperature,
        max_output_tokens=args.selector_max_output_tokens,
        json_retries=args.selector_json_retries,
    )


def main() -> None:
    parser = parse_common_args(
        "Run the KPI/PPI pipeline multiple times and select the best final PPI set."
    )
    parser.add_argument("--n-runs", type=int, default=3, help="Number of pipeline executions.")
    parser.add_argument("--max-selected-ppis", type=int, default=15, help="Maximum number of final selected PPIs.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multi_run_pipeline"))
    parser.add_argument("--run-tag", type=str, default=None, help="Optional run tag. Defaults to current timestamp.")
    parser.add_argument("--min-support", type=float, default=0.8)
    parser.add_argument("--itemsets-support", type=float, default=0.9)
    parser.add_argument("--max-declare-cardinality", type=int, default=3)
    parser.add_argument("--max-constraints", type=int, default=300)
    parser.add_argument("--max-activities", type=int, default=50)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--time-grouper-freq", type=str, default=None)
    parser.add_argument("--cv-threshold", type=float, default=None)
    parser.add_argument("--selector-provider", type=str, choices=("openai", "anthropic", "lmstudio"), default=None)
    parser.add_argument("--selector-model", type=str, default=None)
    parser.add_argument("--selector-base-url", type=str, default=None)
    parser.add_argument("--selector-api-key-env", type=str, default=None)
    parser.add_argument("--selector-temperature", type=float, default=None)
    parser.add_argument("--selector-max-output-tokens", type=int, default=None)
    parser.add_argument("--selector-json-retries", type=int, default=2)
    args = parser.parse_args()

    configure_logging()
    llm_config = build_llm_config_from_args(args)
    ensure_llm_credentials(llm_config)
    selector_llm_config = _build_selector_llm_config(args, llm_config)
    if selector_llm_config is not llm_config:
        ensure_llm_credentials(selector_llm_config)

    event_log = load_event_log(args.xes_path)
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    result, selected_report_df = run_multi_execution_pipeline(
        event_log=event_log,
        llm_config=llm_config,
        selector_llm_config=selector_llm_config,
        n_runs=args.n_runs,
        max_selected_ppis=args.max_selected_ppis,
        min_support=args.min_support,
        itemsets_support=args.itemsets_support,
        max_declare_cardinality=args.max_declare_cardinality,
        max_constraints_to_include=args.max_constraints,
        max_activities_to_include=args.max_activities,
        use_attributes=args.use_attributes,
        max_retries=args.max_retries,
        time_grouper_freq=args.time_grouper_freq,
        cv_threshold=args.cv_threshold,
    )

    for run_id, run_result in enumerate(result.runs):
        selected_kpi_ids = {
            item.kpi_id
            for item in result.selected_ppis.items
            if item.run_id == run_id
        }
        run_selected_report_df = selected_report_df[selected_report_df["run_id"] == run_id].copy()
        run_report_df = build_kpi_execution_report(
            kpi_set=run_result.kpi_set,
            ppinot_set=run_result.ppinot_set,
            executions=run_result.executions,
            categories_output=run_result.kpi_categories,
            variability_results=run_result.variability_results,
            readable_ppi_definitions=run_result.readable_ppi_definitions,
        )
        save_pipeline_run_artifacts(
            raw_root_dir=raw_dir,
            run_label=f"run_{run_id:03d}",
            kpi_set=run_result.kpi_set,
            kpi_categories=run_result.kpi_categories,
            ppinot_set=run_result.ppinot_set,
            executions=run_result.executions,
            variability_results=run_result.variability_results,
            readable_ppi_definitions=run_result.readable_ppi_definitions,
            selected_ppis={
                "items": [
                    item.model_dump(mode="json")
                    for item in result.selected_ppis.items
                    if item.run_id == run_id
                ]
            },
            final_report_df=run_report_df,
            selected_report_df=run_selected_report_df if selected_kpi_ids else pd.DataFrame(),
            metadata={
                "run_id": run_id,
                "selected_kpi_ids": sorted(selected_kpi_ids),
            },
        )

    save_json(output_dir / "selection_candidates.json", result.selection_candidates.model_dump(mode="json"))
    save_json(output_dir / "selected_ppis.json", result.selected_ppis.model_dump(mode="json"))
    selected_report_df.to_csv(output_dir / "selected_report.csv", index=False)

    summary = {
        "n_runs": args.n_runs,
        "max_selected_ppis": args.max_selected_ppis,
        "candidate_count": len(result.selection_candidates.items),
        "selected_count": len(result.selected_ppis.items),
        "llm_config": llm_config.model_dump(mode="json"),
        "selector_llm_config": selector_llm_config.model_dump(mode="json"),
    }
    save_json(output_dir / "summary.json", summary)

    print(f"Completed multi-run pipeline. Output dir: {output_dir.resolve()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
