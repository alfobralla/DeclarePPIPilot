import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from common import (
    analyze_kpi_metric_definition,
    build_llm_config_from_args,
    configure_logging,
    ensure_llm_credentials,
    load_event_log,
    parse_common_args,
    save_pipeline_run_artifacts,
    save_json,
)


def main() -> None:
    parser = parse_common_args("RQ3: CV-threshold filtering impact with time-grouped KPI computation.")
    parser.add_argument("--n", type=int, default=20, help="Number of executions.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/rq3"))
    parser.add_argument("--time-grouper-freq", type=str, default="1ME")
    parser.add_argument("--cv-threshold", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--min-support", type=float, default=0.8)
    parser.add_argument("--itemsets-support", type=float, default=0.9)
    parser.add_argument("--max-declare-cardinality", type=int, default=3)
    parser.add_argument("--max-constraints", type=int, default=300)
    parser.add_argument("--max-activities", type=int, default=50)
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

    run_rows: list[dict] = []
    variability_rows: list[dict] = []
    signature_total = Counter()
    signature_filtered = Counter()

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
            time_grouper_freq=args.time_grouper_freq,
            cv_threshold=args.cv_threshold,
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
            final_report_df=final_report_df,
            metadata={
                "rq": "rq3",
                "run_id": run_idx,
                "llm_config": llm_config.model_dump(mode="json"),
                "use_attributes": bool(args.use_attributes),
                "min_support": args.min_support,
                "itemsets_support": args.itemsets_support,
                "max_declare_cardinality": args.max_declare_cardinality,
                "max_constraints": args.max_constraints,
                "max_activities": args.max_activities,
                "max_retries": args.max_retries,
                "time_grouper_freq": args.time_grouper_freq,
                "cv_threshold": args.cv_threshold,
            },
        )

        variability = run_result.variability_results or []
        variability_by_id = {item.kpi_id: item for item in variability}
        signature_by_id = {}
        analysis_by_id = {}
        for ppi in run_result.ppinot_set.ppis:
            analysis = analyze_kpi_metric_definition(ppi.ppinot4py_definition_code)
            signature_by_id[ppi.source_kpi_id] = analysis["metric_signature"]
            analysis_by_id[ppi.source_kpi_id] = analysis

        ok_count = 0
        filtered_count = 0
        skipped_count = 0
        error_count = 0
        unresolved_count = 0

        for kpi_id, signature in signature_by_id.items():
            analysis = variability_by_id.get(kpi_id)
            metric_analysis = analysis_by_id[kpi_id]
            if analysis is None:
                status = "missing"
                is_filtered = False
            else:
                status = analysis.status
                is_filtered = bool(analysis.cv_flag_below_threshold) if analysis.cv_flag_below_threshold is not None else False
            if metric_analysis["analysis_status"] != "ok":
                unresolved_count += 1

            signature_total[signature] += 1
            if is_filtered:
                signature_filtered[signature] += 1

            if status == "ok":
                ok_count += 1
                filtered_count += int(is_filtered)
            elif status == "skipped":
                skipped_count += 1
            elif status == "error":
                error_count += 1

            variability_rows.append(
                {
                    "run_id": run_idx,
                    "kpi_id": kpi_id,
                    "metric_signature": signature,
                    "final_metric_type": metric_analysis["final_metric_type"],
                    "composition_profile": metric_analysis["composition_profile"],
                    "analysis_status": metric_analysis["analysis_status"],
                    "status": status,
                    "coefficient_of_variation": None if analysis is None else analysis.coefficient_of_variation,
                    "cv_flag_below_threshold": None if analysis is None else analysis.cv_flag_below_threshold,
                }
            )

        total_ppis = len(signature_by_id)
        run_rows.append(
            {
                "run_id": run_idx,
                "ppi_count": total_ppis,
                "variability_ok_count": ok_count,
                "variability_skipped_count": skipped_count,
                "variability_error_count": error_count,
                "filtered_count": filtered_count,
                "filtered_rate_on_ok": filtered_count / ok_count if ok_count else 0.0,
                "filtered_rate_on_all_ppis": filtered_count / total_ppis if total_ppis else 0.0,
                "analysis_unresolved_count": unresolved_count,
            }
        )

    run_df = pd.DataFrame(run_rows)
    variability_df = pd.DataFrame(variability_rows)
    signature_df = pd.DataFrame(
        [
            {
                "metric_signature": signature,
                "occurrences": total,
                "filtered_occurrences": signature_filtered.get(signature, 0),
                "filtered_rate": signature_filtered.get(signature, 0) / total if total else 0.0,
            }
            for signature, total in signature_total.items()
        ]
    ).sort_values(["filtered_rate", "occurrences"], ascending=[False, False])

    summary = {
        "n_runs": args.n,
        "time_grouper_freq": args.time_grouper_freq,
        "cv_threshold": args.cv_threshold,
        "avg_ppi_count": float(run_df["ppi_count"].mean()) if not run_df.empty else 0.0,
        "avg_variability_ok_count": float(run_df["variability_ok_count"].mean()) if not run_df.empty else 0.0,
        "avg_filtered_count": float(run_df["filtered_count"].mean()) if not run_df.empty else 0.0,
        "avg_filtered_rate_on_ok": float(run_df["filtered_rate_on_ok"].mean()) if not run_df.empty else 0.0,
        "avg_filtered_rate_on_all_ppis": float(run_df["filtered_rate_on_all_ppis"].mean()) if not run_df.empty else 0.0,
    }

    run_df.to_csv(output_dir / "rq3_runs.csv", index=False)
    variability_df.to_csv(output_dir / "rq3_variability_by_kpi.csv", index=False)
    signature_df.to_csv(output_dir / "rq3_signature_filtering.csv", index=False)
    save_json(output_dir / "rq3_summary.json", summary)

    print(f"RQ3 completed. Output dir: {output_dir.resolve()}")
    print(summary)


if __name__ == "__main__":
    main()
