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
    save_pipeline_run_artifacts,
    save_json,
)


def main() -> None:
    parser = parse_common_args("RQ2: Retry usefulness for PPINot execution failures.")
    parser.add_argument("--n", type=int, default=20, help="Number of executions.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/rq2"))
    parser.add_argument("--max-retries", type=int, default=2, help="Retries for retry condition.")
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
    from declareppipilot import (
        compute_activity_case_coverage,
        execute_ppinot_kpis,
        generate_kpis_from_declare_table,
        generate_ppinot_from_kpi_set,
        mine_declare_model_and_constraints_df,
    )

    run_rows: list[dict] = []
    kpi_rows: list[dict] = []

    for run_idx in range(args.n):
        declare_table = mine_declare_model_and_constraints_df(
            event_log=event_log,
            min_support=args.min_support,
            itemsets_support=args.itemsets_support,
            max_declare_cardinality=args.max_declare_cardinality,
        )
        activities_table = compute_activity_case_coverage(event_log)
        kpi_set = generate_kpis_from_declare_table(
            declare_table=declare_table,
            activities_table=activities_table,
            event_log=event_log,
            use_attributes=args.use_attributes,
            llm_config=llm_config,
            max_constraints_to_include=args.max_constraints,
            max_activities_to_include=args.max_activities,
        )
        ppinot_set = generate_ppinot_from_kpi_set(kpi_set=kpi_set, llm_config=llm_config)

        no_retry = execute_ppinot_kpis(
            ppinot_set=ppinot_set,
            kpi_set=kpi_set,
            event_log=event_log,
            declare_table=declare_table,
            max_retries=0,
            llm_config=llm_config,
        )
        with_retry = execute_ppinot_kpis(
            ppinot_set=ppinot_set,
            kpi_set=kpi_set,
            event_log=event_log,
            declare_table=declare_table,
            max_retries=args.max_retries,
            llm_config=llm_config,
        )
        save_pipeline_run_artifacts(
            raw_root_dir=raw_dir,
            run_label=f"run_{run_idx:03d}",
            kpi_set=kpi_set,
            ppinot_set=ppinot_set,
            executions_no_retry=no_retry,
            executions_with_retry=with_retry,
            declare_table=declare_table,
            activities_table=activities_table,
            metadata={
                "rq": "rq2",
                "run_id": run_idx,
                "llm_config": llm_config.model_dump(mode="json"),
                "use_attributes": bool(args.use_attributes),
                "min_support": args.min_support,
                "itemsets_support": args.itemsets_support,
                "max_declare_cardinality": args.max_declare_cardinality,
                "max_constraints": args.max_constraints,
                "max_activities": args.max_activities,
                "retry_limit": args.max_retries,
            },
        )

        no_retry_by_id = {e.kpi_id: e for e in no_retry}
        with_retry_by_id = {e.kpi_id: e for e in with_retry}

        baseline_failed = 0
        recovered = 0
        still_failed = 0
        total_retry_attempts = 0

        for kpi_id, e0 in no_retry_by_id.items():
            er = with_retry_by_id.get(kpi_id)
            if er is None:
                continue

            baseline_failed_kpi = e0.final_status == "failed"
            recovered_kpi = baseline_failed_kpi and er.final_status == "success"
            still_failed_kpi = baseline_failed_kpi and er.final_status == "failed"

            baseline_failed += int(baseline_failed_kpi)
            recovered += int(recovered_kpi)
            still_failed += int(still_failed_kpi)
            total_retry_attempts += er.retry_attempts

            kpi_rows.append(
                {
                    "run_id": run_idx,
                    "kpi_id": kpi_id,
                    "baseline_status": e0.final_status,
                    "status_with_retry": er.final_status,
                    "retry_attempts": er.retry_attempts,
                    "recovered": recovered_kpi,
                    "baseline_definition_error": e0.definition_execution_error,
                    "baseline_metric_error": e0.metric_computation_error,
                    "final_definition_error": er.definition_execution_error,
                    "final_metric_error": er.metric_computation_error,
                }
            )

        kpi_count = len(kpi_set.kpis)
        run_rows.append(
            {
                "run_id": run_idx,
                "kpi_count": kpi_count,
                "baseline_failed": baseline_failed,
                "baseline_failed_rate": baseline_failed / kpi_count if kpi_count else 0.0,
                "recovered_with_retry": recovered,
                "recovery_rate_on_failed": recovered / baseline_failed if baseline_failed else 0.0,
                "still_failed_after_retry": still_failed,
                "final_failed_rate": still_failed / kpi_count if kpi_count else 0.0,
                "total_retry_attempts": total_retry_attempts,
                "avg_retry_attempts_per_kpi": total_retry_attempts / kpi_count if kpi_count else 0.0,
            }
        )

    runs_df = pd.DataFrame(run_rows)
    kpi_df = pd.DataFrame(kpi_rows)

    summary = {
        "n_runs": args.n,
        "retry_limit": args.max_retries,
        "avg_baseline_failed": float(runs_df["baseline_failed"].mean()) if not runs_df.empty else 0.0,
        "avg_baseline_failed_rate": float(runs_df["baseline_failed_rate"].mean()) if not runs_df.empty else 0.0,
        "avg_recovered_with_retry": float(runs_df["recovered_with_retry"].mean()) if not runs_df.empty else 0.0,
        "avg_recovery_rate_on_failed": float(runs_df["recovery_rate_on_failed"].mean()) if not runs_df.empty else 0.0,
        "avg_still_failed_after_retry": float(runs_df["still_failed_after_retry"].mean()) if not runs_df.empty else 0.0,
        "avg_final_failed_rate": float(runs_df["final_failed_rate"].mean()) if not runs_df.empty else 0.0,
    }

    runs_df.to_csv(output_dir / "rq2_runs.csv", index=False)
    kpi_df.to_csv(output_dir / "rq2_kpi_level.csv", index=False)
    save_json(output_dir / "rq2_summary.json", summary)

    print(f"RQ2 completed. Output dir: {output_dir.resolve()}")
    print(summary)


if __name__ == "__main__":
    main()
