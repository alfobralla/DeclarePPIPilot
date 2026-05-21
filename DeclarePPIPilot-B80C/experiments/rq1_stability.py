import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from common import (
    analyze_kpi_metric_definition,
    best_match_similarity_metrics,
    build_llm_config_from_args,
    configure_logging,
    exact_set_overlap_metrics,
    ensure_llm_credentials,
    heuristic_exact_match_metrics,
    load_event_log,
    normalize_text_for_similarity,
    normalize_value_for_matching,
    parse_common_args,
    save_pipeline_run_artifacts,
    save_json,
)


def main() -> None:
    parser = parse_common_args("RQ1: Stability of generated KPIs based on PPINot definition code.")
    parser.add_argument("--n", type=int, default=20, help="Number of executions.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/rq1"))
    parser.add_argument("--min-support", type=float, default=0.8)
    parser.add_argument("--itemsets-support", type=float, default=0.9)
    parser.add_argument("--max-declare-cardinality", type=int, default=3)
    parser.add_argument("--max-constraints", type=int, default=300)
    parser.add_argument("--max-activities", type=int, default=50)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--heuristic-sim-threshold", type=float, default=0.8)
    parser.add_argument("--run-tag", type=str, default=None, help="Optional run tag. Defaults to current timestamp.")
    args = parser.parse_args()

    configure_logging()
    llm_config = build_llm_config_from_args(args)
    ensure_llm_credentials(llm_config)
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    event_log = load_event_log(args.xes_path)
    from declareppipilot import run_full_pipeline

    run_rows = []
    kpi_rows = []
    run_signature_sets: list[set[str]] = []
    run_metric_text_sets: list[set[str]] = []
    run_text_value_items: list[list[tuple[str, str]]] = []
    condition_rows: list[dict] = []
    metric_text_counter = Counter()
    metric_text_type: dict[str, str] = {}
    composition_counter = Counter()
    unresolved_total = 0

    for run_idx in range(args.n):
        run_result, _ = run_full_pipeline(
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
        save_pipeline_run_artifacts(
            raw_root_dir=raw_dir,
            run_label=f"run_{run_idx:03d}",
            kpi_set=run_result.kpi_set,
            kpi_categories=run_result.kpi_categories,
            ppinot_set=run_result.ppinot_set,
            executions=run_result.executions,
            variability_results=run_result.variability_results,
            readable_ppi_definitions=run_result.readable_ppi_definitions,
            final_report_df=None,
            metadata={
                "rq": "rq1",
                "run_id": run_idx,
                "llm_config": llm_config.model_dump(mode="json"),
                "use_attributes": bool(args.use_attributes),
                "min_support": args.min_support,
                "itemsets_support": args.itemsets_support,
                "max_declare_cardinality": args.max_declare_cardinality,
                "max_constraints": args.max_constraints,
                "max_activities": args.max_activities,
                "max_retries": args.max_retries,
            },
        )
        signatures = []
        metric_texts = []
        text_value_items = []
        type_counter = Counter()
        contains_counter = Counter()
        aggregated_feature_counter = Counter()
        aggregated_count_counter = Counter()
        run_unresolved = 0
        time_conditions_counter = Counter()
        count_conditions_counter = Counter()
        data_conditions_counter = Counter()
        exec_by_kpi_id = {item.kpi_id: item for item in run_result.executions}
        for ppi in run_result.ppinot_set.ppis:
            analysis = analyze_kpi_metric_definition(ppi.ppinot4py_definition_code)
            sig = analysis["metric_signature"]
            execution = exec_by_kpi_id.get(ppi.source_kpi_id)
            metric_text = normalize_text_for_similarity(
                execution.kpi_metric_str if execution is not None else None
            )
            metric_value = normalize_value_for_matching(
                execution.kpi_metric_computation_value if execution is not None else None
            )
            signatures.append(sig)
            if metric_text:
                metric_texts.append(metric_text)
                text_value_items.append((metric_text, metric_value))
                metric_text_counter[metric_text] += 1
                metric_text_type[metric_text] = analysis["final_metric_type"]

            for metric_type in analysis["type_labels"]:
                type_counter[metric_type] += 1

            contains_counter["time"] += int(analysis["contains_time_measure"])
            contains_counter["data"] += int(analysis["contains_data_measure"])
            contains_counter["frequency"] += int(analysis["contains_frequency_measure"])
            contains_counter["aggregated"] += int(analysis["contains_aggregated_measure"])
            contains_counter["derived"] += int(analysis["contains_derived_measure"])
            aggregated_feature_counter["with_filter"] += int(analysis["aggregated_has_filter"])
            aggregated_feature_counter["with_grouper"] += int(analysis["aggregated_has_grouper"])
            aggregated_count_counter["filter_count"] += int(analysis["aggregated_filter_count"])
            aggregated_count_counter["grouper_count"] += int(analysis["aggregated_grouper_count"])

            composition_counter[analysis["composition_profile"]] += 1

            if analysis["analysis_status"] != "ok":
                unresolved_total += 1
                run_unresolved += 1
            for condition in analysis["time_conditions"]:
                time_conditions_counter[condition] += 1
            for condition in analysis["count_conditions"]:
                count_conditions_counter[condition] += 1
            for condition in analysis["data_conditions"]:
                data_conditions_counter[condition] += 1

            kpi_rows.append(
                {
                    "run_id": run_idx,
                    "kpi_id": ppi.source_kpi_id,
                    "metric_signature": sig,
                    "kpi_metric_text": metric_text,
                    "kpi_metric_computation_value": metric_value,
                    "analysis_status": analysis["analysis_status"],
                    "final_metric_type": analysis["final_metric_type"],
                    "base_metric_types": "|".join(analysis["base_metric_types"]),
                    "type_labels": "|".join(analysis["type_labels"]),
                    "composition_profile": analysis["composition_profile"],
                    "contains_time_measure": analysis["contains_time_measure"],
                    "contains_data_measure": analysis["contains_data_measure"],
                    "contains_frequency_measure": analysis["contains_frequency_measure"],
                    "contains_aggregated_measure": analysis["contains_aggregated_measure"],
                    "contains_derived_measure": analysis["contains_derived_measure"],
                    "aggregated_has_filter": analysis["aggregated_has_filter"],
                    "aggregated_has_grouper": analysis["aggregated_has_grouper"],
                    "aggregated_filter_count": analysis["aggregated_filter_count"],
                    "aggregated_grouper_count": analysis["aggregated_grouper_count"],
                    "time_conditions": " | ".join(analysis["time_conditions"]),
                    "count_conditions": " | ".join(analysis["count_conditions"]),
                    "data_conditions": " | ".join(analysis["data_conditions"]),
                }
            )

        signature_set = set(signatures)
        metric_text_set = set(metric_texts)
        run_signature_sets.append(signature_set)
        run_metric_text_sets.append(metric_text_set)
        run_text_value_items.append(text_value_items)
        run_rows.append(
            {
                "run_id": run_idx,
                "kpi_count": len(run_result.kpi_set.kpis),
                "ppi_count": len(run_result.ppinot_set.ppis),
                "unique_signatures": len(signature_set),
                "unique_kpi_metric_strings": len(metric_text_set),
                "type_time": type_counter["time"],
                "type_data": type_counter["data"],
                "type_frequency": type_counter["frequency"],
                "type_aggregated": type_counter["aggregated"],
                "type_derived": type_counter["derived"],
                "type_other": type_counter["other"],
                "analysis_unresolved": run_unresolved,
                "sum_contains_time_measure": contains_counter["time"],
                "sum_contains_data_measure": contains_counter["data"],
                "sum_contains_frequency_measure": contains_counter["frequency"],
                "sum_contains_aggregated_measure": contains_counter["aggregated"],
                "sum_contains_derived_measure": contains_counter["derived"],
                "sum_aggregated_with_filter": aggregated_feature_counter["with_filter"],
                "sum_aggregated_with_grouper": aggregated_feature_counter["with_grouper"],
                "sum_aggregated_filter_count": aggregated_count_counter["filter_count"],
                "sum_aggregated_grouper_count": aggregated_count_counter["grouper_count"],
                "time_conditions_top": json.dumps(time_conditions_counter.most_common(5)),
                "count_conditions_top": json.dumps(count_conditions_counter.most_common(5)),
                "data_conditions_top": json.dumps(data_conditions_counter.most_common(5)),
            }
        )
        for condition, count in time_conditions_counter.items():
            condition_rows.append(
                {"run_id": run_idx, "measure_type": "time", "condition": condition, "count": count}
            )
        for condition, count in count_conditions_counter.items():
            condition_rows.append(
                {"run_id": run_idx, "measure_type": "count", "condition": condition, "count": count}
            )
        for condition, count in data_conditions_counter.items():
            condition_rows.append(
                {"run_id": run_idx, "measure_type": "data", "condition": condition, "count": count}
            )

    pairwise_rows = []
    for i in range(args.n):
        for j in range(i + 1, args.n):
            exact_metrics = exact_set_overlap_metrics(run_metric_text_sets[i], run_metric_text_sets[j])
            heuristic_metrics = heuristic_exact_match_metrics(
                run_text_value_items[i],
                run_text_value_items[j],
                similarity_threshold=args.heuristic_sim_threshold,
            )
            pairwise_rows.append(
                {
                    "run_i": i,
                    "run_j": j,
                    "exact_match_count": int(exact_metrics["exact_match_count"]),
                    "exact_match_jaccard": exact_metrics["exact_match_jaccard"],
                    "heuristic_exact_match_count": int(heuristic_metrics["heuristic_exact_match_count"]),
                    "heuristic_exact_match_rate": heuristic_metrics["heuristic_exact_match_rate"],
                    "value_exact_match_count": int(heuristic_metrics["value_exact_match_count"]),
                    "value_exact_match_rate": heuristic_metrics["value_exact_match_rate"]
                }
            )
    pairwise_df = pd.DataFrame(pairwise_rows)
    run_df = pd.DataFrame(run_rows)
    kpi_df = pd.DataFrame(kpi_rows)
    conditions_by_run_df = pd.DataFrame(condition_rows)
    if not conditions_by_run_df.empty:
        conditions_by_run_df = conditions_by_run_df.sort_values(
            ["run_id", "measure_type", "count", "condition"],
            ascending=[True, True, False, True],
        )

    core_threshold = max(1, int(0.8 * args.n))
    metric_freq_df = pd.DataFrame(
        [
            {
                "metric_text": t,
                "final_metric_type": metric_text_type.get(t, "other"),
                "count": c,
                "frequency": c / args.n,
                "is_core": c >= core_threshold,
            }
            for t, c in metric_text_counter.items()
        ]
    ).sort_values(["count", "metric_text"], ascending=[False, True])
    composition_df = pd.DataFrame(
        [
            {"composition_profile": profile, "count": count, "frequency": count / max(1, len(kpi_rows))}
            for profile, count in composition_counter.items()
        ]
    ).sort_values(["count", "composition_profile"], ascending=[False, True])

    summary = {
        "n_runs": args.n,
        "avg_exact_match_jaccard": float(pairwise_df["exact_match_jaccard"].mean()) if not pairwise_df.empty else 1.0,
        "std_exact_match_jaccard": float(pairwise_df["exact_match_jaccard"].std(ddof=0)) if not pairwise_df.empty else 0.0,
        "avg_exact_match_count": float(pairwise_df["exact_match_count"].mean()) if not pairwise_df.empty else 0.0,
        "avg_heuristic_exact_match_count": float(pairwise_df["heuristic_exact_match_count"].mean()) if not pairwise_df.empty else 0.0,
        "avg_heuristic_exact_match_rate": float(pairwise_df["heuristic_exact_match_rate"].mean()) if not pairwise_df.empty else 0.0,
        "avg_value_exact_match_count": float(pairwise_df["value_exact_match_count"].mean()) if not pairwise_df.empty else 0.0,
        "avg_value_exact_match_rate": float(pairwise_df["value_exact_match_rate"].mean()) if not pairwise_df.empty else 0.0,
        "avg_kpis_per_run": float(run_df["kpi_count"].mean()),
        "std_kpis_per_run": float(run_df["kpi_count"].std(ddof=0)),
        "core_metric_threshold_count": core_threshold,
        "n_core_metrics": int(metric_freq_df["is_core"].sum()) if not metric_freq_df.empty else 0,
        "n_unique_metrics": int(metric_freq_df.shape[0]),
        "analysis_unresolved_count": unresolved_total,
        "analysis_unresolved_rate": unresolved_total / max(1, len(kpi_rows)),
    }

    run_df.to_csv(output_dir / "rq1_runs.csv", index=False)
    kpi_df.to_csv(output_dir / "rq1_kpi_analysis.csv", index=False)
    pairwise_df.to_csv(output_dir / "rq1_pairwise_jaccard.csv", index=False)
    metric_freq_df.to_csv(output_dir / "rq1_metric_frequency.csv", index=False)
    composition_df.to_csv(output_dir / "rq1_composition_frequency.csv", index=False)
    conditions_by_run_df.to_csv(output_dir / "rq1_conditions_by_run.csv", index=False)
    save_json(output_dir / "rq1_summary.json", summary)

    print(f"RQ1 completed. Output dir: {output_dir.resolve()}")
    print(summary)


if __name__ == "__main__":
    main()
