import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from Declare4Py.D4PyEventLog import D4PyEventLog

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from declareppipilot import (
    analyze_ppi_variability_with_time_grouper,
    build_kpi_execution_report,
    categorize_kpis_by_business_goal,
    compute_activity_case_coverage,
    execute_ppinot_kpis,
    generate_kpis_from_declare_table,
    generate_ppinot_from_kpi_set,
    mine_declare_model_and_constraints_df,
)
from declareppipilot.models import LLMConfig


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    xes_path = PROJECT_ROOT / "data" / "logs" / "DomesticDeclarations.xes"
    if not xes_path.exists():
        raise FileNotFoundError(f"XES file not found: {xes_path.resolve()}")

    event_log = D4PyEventLog(case_name="case:concept:name")
    event_log.parse_xes_log(str(xes_path))

    declare_table = mine_declare_model_and_constraints_df(
        event_log=event_log,
        consider_vacuity=False,
        min_support=0.80,
        itemsets_support=0.90,
        max_declare_cardinality=3,
        conformance_consider_vacuity=False,
    )
    activities_table = compute_activity_case_coverage(event_log)

    print("declare_table shape:", declare_table.shape)
    print("\nactivities_table shape:", activities_table.shape)

    if not os.getenv("OPENAI_API_KEY"):
        print("\nSkipping OpenAI KPI generation test: OPENAI_API_KEY is not set.")
        return
    llm_config = LLMConfig(provider="openai", model="gpt-5.2")

    kpi_payload = generate_kpis_from_declare_table(
        declare_table=declare_table,
        activities_table=activities_table,
        event_log=event_log,
        llm_config=llm_config,
    )
    kpis = kpi_payload.kpis
    output_path = PROJECT_ROOT / "outputs" / f"kpi_output_{run_ts}.json"
    output_path.write_text(
        json.dumps(kpi_payload.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    print("\nGenerated KPIs:", len(kpis))
    print("Saved KPI JSON to:", output_path.resolve())
    if kpis:
        first = kpis[0]
        print("First KPI preview:")
        print("-", first.id)
        print("-", first.name)

    categories_payload = categorize_kpis_by_business_goal(kpi_payload, llm_config=llm_config)
    categories_output_path = PROJECT_ROOT / "outputs" / f"kpi_categories_{run_ts}.json"
    categories_output_path.write_text(
        json.dumps(categories_payload.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    print("\nGenerated KPI categories:", len(categories_payload.kpi_categories))
    print("Saved KPI categories JSON to:", categories_output_path.resolve())

    ppinot_payload = generate_ppinot_from_kpi_set(kpi_payload, llm_config=llm_config)
    ppis = ppinot_payload.ppis
    ppinot_output_path = PROJECT_ROOT / "outputs" / f"ppinot_output_{run_ts}.json"
    ppinot_output_path.write_text(
        json.dumps(ppinot_payload.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    print("\nGenerated PPIs:", len(ppis))
    print("Saved PPINot JSON to:", ppinot_output_path.resolve())

    executions = execute_ppinot_kpis(
        ppinot_set=ppinot_payload,
        kpi_set=kpi_payload,
        event_log=event_log,
        declare_table=declare_table,
        max_retries=2,
        llm_config=llm_config,
    )
    execution_df = pd.DataFrame([item.model_dump(mode="json") for item in executions])
    execution_output_path = PROJECT_ROOT / "outputs" / f"ppi_execution_raw_{run_ts}.csv"
    execution_df.to_csv(execution_output_path, index=False)

    print("\nPPI execution rows:", len(execution_df))
    print("Saved raw PPI execution output to:", execution_output_path.resolve())

    successful_kpi_ids = {e.kpi_id for e in executions if e.final_status == "success"}
    successful_ppis = [ppi for ppi in ppinot_payload.ppis if ppi.source_kpi_id in successful_kpi_ids]
    variability_results = analyze_ppi_variability_with_time_grouper(
        ppinot_set=type(ppinot_payload)(ppis=successful_ppis),
        event_log=event_log,
        time_grouper_freq="1ME",
        cv_threshold=0.2,
    )
    variability_df = pd.DataFrame([item.model_dump(mode="json") for item in variability_results])
    variability_output_path = PROJECT_ROOT / "outputs" / f"ppi_variability_raw_{run_ts}.csv"
    variability_df.to_csv(variability_output_path, index=False)
    print("Saved raw PPI variability output to:", variability_output_path.resolve())

    final_report_df = build_kpi_execution_report(
        kpi_set=kpi_payload,
        ppinot_set=ppinot_payload,
        executions=executions,
        categories_output=categories_payload,
        variability_results=variability_results,
    )
    final_report_output_path = PROJECT_ROOT / "outputs" / f"ppi_execution_{run_ts}.csv"
    final_report_df.to_csv(final_report_output_path, index=False)
    print("\nFinal report rows:", len(final_report_df))
    print("Saved final PPI execution report to:", final_report_output_path.resolve())


if __name__ == "__main__":
    main()
