import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from common import (
    analyze_kpi_metric_definition,
    build_llm_config_from_args,
    configure_logging,
    ensure_llm_credentials,
    exact_set_overlap_metrics,
    heuristic_exact_match_metrics,
    load_event_log,
    normalize_text_for_similarity,
    normalize_value_for_matching,
    parse_common_args,
    save_pipeline_run_artifacts,
    save_json,
)


def _parse_support_values(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("No min_support values provided.")
    return values


def main() -> None:
    parser = parse_common_args("RQ4: Effect of min_support on generated KPI sets.")
    parser.add_argument("--n", type=int, default=10, help="Executions per min_support value.")
    parser.add_argument("--supports", type=str, default="0.8,0.6,0.4,0.2")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/rq4"))
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
    supports = _parse_support_values(args.supports)
    from declareppipilot import run_full_pipeline

    run_rows: list[dict] = []
    signature_rows: list[dict] = []
    condition_rows: list[dict] = []
    signatures_by_support: dict[float, set[str]] = defaultdict(set)
    metric_texts_by_support: dict[float, set[str]] = defaultdict(set)
    text_value_items_by_support: dict[float, list[tuple[str, str]]] = defaultdict(list)
    composition_rows: list[dict] = []

    for min_support in supports:
        for run_idx in range(args.n):
            run_result, final_report_df = run_full_pipeline(
                event_log=event_log,
                llm_config=llm_config,
                min_support=min_support,
                itemsets_support=args.itemsets_support,
                max_declare_cardinality=args.max_declare_cardinality,
                max_constraints_to_include=args.max_constraints,
                max_activities_to_include=args.max_activities,
                use_attributes=args.use_attributes,
                max_retries=args.max_retries,
                time_grouper_freq=None,
                cv_threshold=None,
            )
            support_label = str(min_support).replace(".", "_")
            save_pipeline_run_artifacts(
                raw_root_dir=raw_dir,
                run_label=f"support_{support_label}_run_{run_idx:03d}",
                kpi_set=run_result.kpi_set,
                kpi_categories=run_result.kpi_categories,
                ppinot_set=run_result.ppinot_set,
                executions=run_result.executions,
                variability_results=run_result.variability_results,
                readable_ppi_definitions=run_result.readable_ppi_definitions,
                final_report_df=final_report_df,
                metadata={
                    "rq": "rq4",
                    "run_id": run_idx,
                    "min_support": min_support,
                    "llm_config": llm_config.model_dump(mode="json"),
                    "use_attributes": bool(args.use_attributes),
                    "itemsets_support": args.itemsets_support,
                    "max_declare_cardinality": args.max_declare_cardinality,
                    "max_constraints": args.max_constraints,
                    "max_activities": args.max_activities,
                    "max_retries": args.max_retries,
                },
            )

            type_counter = Counter()
            contains_counter = Counter()
            aggregated_feature_counter = Counter()
            aggregated_count_counter = Counter()
            time_conditions_counter = Counter()
            count_conditions_counter = Counter()
            data_conditions_counter = Counter()
            signatures = []
            metric_texts = []
            text_value_items = []
            unresolved_count = 0
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
                signatures_by_support[min_support].add(sig)
                if metric_text:
                    metric_texts.append(metric_text)
                    text_value_items.append((metric_text, metric_value))

                for metric_type in analysis["type_labels"]:
                    type_counter[metric_type] += 1
                contains_counter["time"] += int(analysis["contains_time_measure"])
                contains_counter["data"] += int(analysis["contains_data_measure"])
                contains_counter["frequency"] += int(analysis["contains_frequency_measure"])
                contains_counter["aggregated"] += int(analysis["contains_aggregated_measure"])
                contains_counter["derived"] += int(analysis["contains_derived_measure"])
                aggregated_feature_counter["with_filter"] += int(analysis.get("aggregated_has_filter", False))
                aggregated_feature_counter["with_grouper"] += int(analysis.get("aggregated_has_grouper", False))
                aggregated_count_counter["filter_count"] += int(analysis.get("aggregated_filter_count", 0))
                aggregated_count_counter["grouper_count"] += int(analysis.get("aggregated_grouper_count", 0))
                for condition in analysis.get("time_conditions", []):
                    time_conditions_counter[condition] += 1
                for condition in analysis.get("count_conditions", []):
                    count_conditions_counter[condition] += 1
                for condition in analysis.get("data_conditions", []):
                    data_conditions_counter[condition] += 1
                if analysis["analysis_status"] != "ok":
                    unresolved_count += 1
                signature_rows.append(
                    {
                        "min_support": min_support,
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
                        "aggregated_has_filter": analysis.get("aggregated_has_filter", False),
                        "aggregated_has_grouper": analysis.get("aggregated_has_grouper", False),
                        "aggregated_filter_count": analysis.get("aggregated_filter_count", 0),
                        "aggregated_grouper_count": analysis.get("aggregated_grouper_count", 0),
                        "time_conditions": " | ".join(analysis.get("time_conditions", [])),
                        "count_conditions": " | ".join(analysis.get("count_conditions", [])),
                        "data_conditions": " | ".join(analysis.get("data_conditions", [])),
                        "data_preconditions": " | ".join(analysis.get("data_preconditions", [])),
                        "data_content_selection_values": " | ".join(analysis.get("data_content_selection_values", [])),
                        "derived_variables_total": analysis.get("derived_variables_total", 0),
                        "derived_variables_avg": analysis.get("derived_variables_avg", 0.0),
                    }
                )
                composition_rows.append(
                    {
                        "min_support": min_support,
                        "run_id": run_idx,
                        "composition_profile": analysis["composition_profile"],
                    }
                )
            metric_texts_by_support[min_support].update(metric_texts)
            text_value_items_by_support[min_support].extend(text_value_items)

            for condition, count in time_conditions_counter.items():
                condition_rows.append(
                    {
                        "min_support": min_support,
                        "run_id": run_idx,
                        "measure_type": "time",
                        "condition": condition,
                        "count": count,
                    }
                )
            for condition, count in count_conditions_counter.items():
                condition_rows.append(
                    {
                        "min_support": min_support,
                        "run_id": run_idx,
                        "measure_type": "count",
                        "condition": condition,
                        "count": count,
                    }
                )
            for condition, count in data_conditions_counter.items():
                condition_rows.append(
                    {
                        "min_support": min_support,
                        "run_id": run_idx,
                        "measure_type": "data",
                        "condition": condition,
                        "count": count,
                    }
                )

            ppi_count = len(run_result.ppinot_set.ppis)
            success_count = sum(1 for e in run_result.executions if e.final_status == "success")
            run_rows.append(
                {
                    "min_support": min_support,
                    "run_id": run_idx,
                    "kpi_count": len(run_result.kpi_set.kpis),
                    "ppi_count": ppi_count,
                    "execution_success_count": success_count,
                    "execution_success_rate": success_count / ppi_count if ppi_count else 0.0,
                    "unique_signatures": len(set(signatures)),
                    "unique_kpi_metric_strings": len(set(metric_texts)),
                    "type_time": type_counter["time"],
                    "type_data": type_counter["data"],
                    "type_frequency": type_counter["frequency"],
                    "type_aggregated": type_counter["aggregated"],
                    "type_derived": type_counter["derived"],
                    "type_other": type_counter["other"],
                    "analysis_unresolved": unresolved_count,
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

    run_df = pd.DataFrame(run_rows)
    signature_df = pd.DataFrame(signature_rows)
    conditions_by_run_df = pd.DataFrame(condition_rows)
    if not conditions_by_run_df.empty:
        conditions_by_run_df = conditions_by_run_df.sort_values(
            ["min_support", "run_id", "measure_type", "count", "condition"],
            ascending=[True, True, True, False, True],
        )

    support_summary_df = (
        run_df.groupby("min_support", as_index=False)
        .agg(
            runs=("run_id", "count"),
            avg_kpi_count=("kpi_count", "mean"),
            std_kpi_count=("kpi_count", "std"),
            avg_ppi_count=("ppi_count", "mean"),
            avg_success_rate=("execution_success_rate", "mean"),
            avg_unique_signatures=("unique_signatures", "mean"),
            avg_type_time=("type_time", "mean"),
            avg_type_data=("type_data", "mean"),
            avg_type_frequency=("type_frequency", "mean"),
            avg_type_aggregated=("type_aggregated", "mean"),
            avg_type_derived=("type_derived", "mean"),
            avg_type_other=("type_other", "mean"),
            avg_analysis_unresolved=("analysis_unresolved", "mean"),
        )
        .sort_values("min_support", ascending=False)
    )
    composition_df = pd.DataFrame(composition_rows)
    if not composition_df.empty:
        composition_summary_df = (
            composition_df.groupby(["min_support", "composition_profile"], as_index=False)
            .agg(count=("composition_profile", "count"))
            .sort_values(["min_support", "count"], ascending=[False, False])
        )
    else:
        composition_summary_df = pd.DataFrame(columns=["min_support", "composition_profile", "count"])

    overlap_rows = []
    for i, support_a in enumerate(supports):
        for support_b in supports[i + 1 :]:
            set_a = signatures_by_support[support_a]
            set_b = signatures_by_support[support_b]
            exact_metrics = exact_set_overlap_metrics(set_a, set_b)
            heuristic_metrics = heuristic_exact_match_metrics(
                text_value_items_by_support[support_a],
                text_value_items_by_support[support_b],
                similarity_threshold=args.heuristic_sim_threshold,
            )
            overlap_rows.append(
                {
                    "support_a": support_a,
                    "support_b": support_b,
                    "exact_match_count": int(exact_metrics["exact_match_count"]),
                    "exact_match_jaccard": exact_metrics["exact_match_jaccard"],
                    "metric_text_exact_match_count": len(
                        metric_texts_by_support[support_a].intersection(metric_texts_by_support[support_b])
                    ),
                    "metric_text_exact_match_jaccard": exact_set_overlap_metrics(
                        metric_texts_by_support[support_a], metric_texts_by_support[support_b]
                    )["exact_match_jaccard"],
                    "heuristic_exact_match_count": int(heuristic_metrics["heuristic_exact_match_count"]),
                    "heuristic_exact_match_rate": heuristic_metrics["heuristic_exact_match_rate"],
                    "value_exact_match_count": int(heuristic_metrics["value_exact_match_count"]),
                    "value_exact_match_rate": heuristic_metrics["value_exact_match_rate"],
                    "unique_signatures_a": len(set_a),
                    "unique_signatures_b": len(set_b),
                }
            )
    overlap_df = pd.DataFrame(overlap_rows)

    summary = {
        "supports": supports,
        "runs_per_support": args.n,
        "max_constraints": args.max_constraints,
        "max_activities": args.max_activities,
    }

    run_df.to_csv(output_dir / "rq4_runs.csv", index=False)
    support_summary_df.to_csv(output_dir / "rq4_support_summary.csv", index=False)
    signature_df.to_csv(output_dir / "rq4_signatures.csv", index=False)
    composition_summary_df.to_csv(output_dir / "rq4_composition_summary.csv", index=False)
    conditions_by_run_df.to_csv(output_dir / "rq4_conditions_by_run.csv", index=False)
    overlap_df.to_csv(output_dir / "rq4_support_signature_overlap.csv", index=False)
    save_json(output_dir / "rq4_summary.json", summary)

    print(f"RQ4 completed. Output dir: {output_dir.resolve()}")
    print(summary)


if __name__ == "__main__":
    main()
