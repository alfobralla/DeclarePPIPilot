import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from common import (
    build_llm_config_from_args,
    configure_logging,
    ensure_llm_credentials,
    load_event_log,
    parse_common_args,
    save_json,
    save_pipeline_run_artifacts,
)


def _build_eval_llm_config(args, default_llm_config):
    eval_provider = args.eval_provider
    if not eval_provider:
        return default_llm_config
    if not args.eval_model:
        raise ValueError("--eval-model is required when --eval-provider is set.")

    from declareppipilot.models import LLMConfig

    return LLMConfig(
        provider=eval_provider,
        model=args.eval_model,
        base_url=args.eval_base_url,
        api_key_env=args.eval_api_key_env,
        temperature=args.eval_temperature,
        max_output_tokens=args.eval_max_output_tokens,
        json_retries=args.eval_json_retries,
    )


def main() -> None:
    parser = parse_common_args("RQ5: LLM-based alignment between KPI definition and computed kpi_metric string.")
    parser.add_argument("--n", type=int, default=10, help="Number of executions.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/rq5"))
    parser.add_argument("--min-support", type=float, default=0.8)
    parser.add_argument("--itemsets-support", type=float, default=0.9)
    parser.add_argument("--max-declare-cardinality", type=int, default=3)
    parser.add_argument("--max-constraints", type=int, default=300)
    parser.add_argument("--max-activities", type=int, default=50)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--run-tag", type=str, default=None, help="Optional run tag. Defaults to current timestamp.")

    parser.add_argument("--eval-provider", type=str, choices=("openai", "anthropic", "lmstudio"), default=None)
    parser.add_argument("--eval-model", type=str, default=None)
    parser.add_argument("--eval-base-url", type=str, default=None)
    parser.add_argument("--eval-api-key-env", type=str, default=None)
    parser.add_argument("--eval-temperature", type=float, default=None)
    parser.add_argument("--eval-max-output-tokens", type=int, default=None)
    parser.add_argument("--eval-json-retries", type=int, default=2)

    args = parser.parse_args()

    configure_logging()
    llm_config = build_llm_config_from_args(args)
    ensure_llm_credentials(llm_config)

    eval_llm_config = _build_eval_llm_config(args, llm_config)
    if eval_llm_config is not llm_config:
        ensure_llm_credentials(eval_llm_config)

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    event_log = load_event_log(args.xes_path)

    from declareppipilot import evaluate_kpi_metric_alignment, run_full_pipeline

    run_rows: list[dict] = []
    alignment_rows: list[dict] = []

    for run_idx in range(args.n):
        run_result, final_report_df = run_full_pipeline(
            event_log=event_log,
            llm_config=llm_config,
            min_support=args.min_support,
            itemsets_support=args.itemsets_support,
            max_declare_cardinality=args.max_declare_cardinality,
            max_constraints_to_include=args.max_constraints,
            max_activities_to_include=args.max_activities,
            use_attributes=args.use_attributes,
            max_retries=args.max_retries,
            time_grouper_freq=None,
            cv_threshold=None,
        )
        run_dir = save_pipeline_run_artifacts(
            raw_root_dir=raw_dir,
            run_label=f"run_{run_idx:03d}",
            kpi_set=run_result.kpi_set,
            kpi_categories=run_result.kpi_categories,
            ppinot_set=run_result.ppinot_set,
            executions=run_result.executions,
            variability_results=run_result.variability_results,
            readable_ppi_definitions=run_result.readable_ppi_definitions,
            final_report_df=final_report_df,
            metadata={
                "rq": "rq5",
                "run_id": run_idx,
                "llm_config": llm_config.model_dump(mode="json"),
                "eval_llm_config": eval_llm_config.model_dump(mode="json"),
                "use_attributes": bool(args.use_attributes),
                "min_support": args.min_support,
                "itemsets_support": args.itemsets_support,
                "max_declare_cardinality": args.max_declare_cardinality,
                "max_constraints": args.max_constraints,
                "max_activities": args.max_activities,
                "max_retries": args.max_retries,
            },
        )

        alignment_set = evaluate_kpi_metric_alignment(
            kpi_set=run_result.kpi_set,
            ppinot_set=run_result.ppinot_set,
            executions=run_result.executions,
            llm_config=eval_llm_config,
        )
        save_json(run_dir / "alignment.json", alignment_set.model_dump(mode="json"))

        kpi_by_id = {k.id: k for k in run_result.kpi_set.kpis}
        ppi_by_id = {p.source_kpi_id: p for p in run_result.ppinot_set.ppis}
        exe_by_id = {e.kpi_id: e for e in run_result.executions}
        readable_by_id = {}
        if run_result.readable_ppi_definitions is not None:
            readable_by_id = {item.kpi_id: item for item in run_result.readable_ppi_definitions.items}

        scores = []
        for item in alignment_set.items:
            scores.append(item.alignment_score)
            kpi = kpi_by_id.get(item.kpi_id)
            ppi = ppi_by_id.get(item.kpi_id)
            exe = exe_by_id.get(item.kpi_id)
            readable = readable_by_id.get(item.kpi_id)
            alignment_rows.append(
                {
                    "run_id": run_idx,
                    "kpi_id": item.kpi_id,
                    "kpi_name": None if kpi is None else kpi.name,
                    "kpi_definition": None if kpi is None else kpi.definition,
                    "kpi_metric_str": None if exe is None else exe.kpi_metric_str,
                    "human_readable_ppi_definition": None if readable is None else readable.human_readable_definition,
                    "approximation": None if ppi is None else ppi.approximation,
                    "alignment_score": item.alignment_score,
                    "alignment_rationale": item.rationale
                }
            )

        run_rows.append(
            {
                "run_id": run_idx,
                "kpi_count": len(run_result.kpi_set.kpis),
                "alignment_items": len(alignment_set.items),
                "avg_alignment_score": float(sum(scores) / len(scores)) if scores else 0.0,
                "min_alignment_score": float(min(scores)) if scores else 0.0,
                "max_alignment_score": float(max(scores)) if scores else 0.0,
                "count_below_0_5": sum(1 for s in scores if s < 0.5),
                "count_below_0_7": sum(1 for s in scores if s < 0.7),
            }
        )

    run_df = pd.DataFrame(run_rows)
    alignment_df = pd.DataFrame(alignment_rows)

    summary = {
        "n_runs": args.n,
        "llm_config": llm_config.model_dump(mode="json"),
        "eval_llm_config": eval_llm_config.model_dump(mode="json"),
        "avg_run_alignment_score": float(run_df["avg_alignment_score"].mean()) if not run_df.empty else 0.0,
        "std_run_alignment_score": float(run_df["avg_alignment_score"].std(ddof=0)) if not run_df.empty else 0.0,
    }

    run_df.to_csv(output_dir / "rq5_runs.csv", index=False)
    alignment_df.to_csv(output_dir / "rq5_alignment_by_kpi.csv", index=False)
    save_json(output_dir / "rq5_summary.json", summary)

    print(f"RQ5 completed. Output dir: {output_dir.resolve()}")
    print(summary)


if __name__ == "__main__":
    main()
