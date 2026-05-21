import ast
import json
import logging
import re
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer
from Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner

from .llm import generate_structured_model
from .models import (
    KPIAlignmentItem,
    KPIAlignmentSet,
    LLMKPIAlignmentSet,
    LLMSelectedPPISet,
    KPICategorySet,
    LLMConfig,
    KPIItem,
    KPISet,
    MultiRunPipelineResult,
    PipelineRunResult,
    PPIExecution,
    PPIItem,
    PPISelectionCandidateItem,
    PPISelectionCandidateSet,
    PPIReadableDefinitionSet,
    PPISet,
    PPIVariabilityAnalysis,
    SelectedPPIItem,
    SelectedPPISet,
)

logger = logging.getLogger(__name__)


_KPI_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "kpi_prompt_template.txt"
_KPI_PROMPT_WITH_ATTRIBUTES_TEMPLATE_PATH = (
    Path(__file__).parent / "prompts" / "kpi_prompt_template_with_attributes.txt"
)
_KPI_CATEGORIES_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "kpi_categories_prompt_template.txt"
_KPI_ALIGNMENT_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "kpi_alignment_prompt_template.txt"
_PPI_READABLE_DEFINITION_PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent / "prompts" / "ppi_readable_definition_prompt_template.txt"
)
_PPI_SELECTION_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "ppi_selection_prompt_template.txt"
_PPINOT_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "ppinot_prompt_template.txt"
_PPINOT_RETRY_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "ppinot_retry_prompt_template.txt"
_PPINOT_README_PATH = Path(__file__).parent / "prompts" / "ppinot4py-README.md"
_REQUIRED_DECLARE_TABLE_COLUMNS = {"constraint", "support_pct", "support_traces", "total_traces"}
_REQUIRED_ACTIVITY_TABLE_COLUMNS = {"activity", "cases_with_activity", "exists_pct"}
_CORE_PROCESS_CONTROL_COLUMNS = {"time:timestamp", "concept:name", "case:concept:name"}
_BLOCKED_MODULES = {
    "os",
    "pathlib",
    "shutil",
    "subprocess",
    "glob",
    "importlib",
    "builtins",
    "socket",
    "requests",
    "urllib",
}
_BLOCKED_CALL_NAMES = {"open", "exec", "eval", "compile", "__import__"}
_BLOCKED_CALL_ATTRS = {
    "unlink",
    "rmdir",
    "remove",
    "rename",
    "replace",
    "rmtree",
    "system",
    "popen",
}


def _parse_constraint(constraint: str) -> dict[str, str | None]:
    match = re.match(r"^(?P<template>[^\[]+)\[(?P<inside>[^\]]+)\]", constraint.strip())
    if not match:
        return {"raw": constraint, "template": None, "a": None, "b": None}

    template = match.group("template").strip()
    parts = [part.strip() for part in match.group("inside").split(",")]
    activity_a = parts[0] if len(parts) > 0 else None
    activity_b = parts[1] if len(parts) > 1 else None
    return {"raw": constraint, "template": template, "a": activity_a, "b": activity_b}


def compute_activity_case_coverage(
    event_log,
    case_col: str = "case:concept:name",
    activity_col: str = "concept:name",
) -> pd.DataFrame:
    logger.info("Computing activity case coverage (case_col=%s, activity_col=%s).", case_col, activity_col)
    df = _event_log_to_dataframe(event_log)
    if case_col not in df.columns:
        raise ValueError(f"Missing case column '{case_col}' in event log dataframe.")
    if activity_col not in df.columns:
        raise ValueError(f"Missing activity column '{activity_col}' in event log dataframe.")

    n_cases = df[case_col].nunique()
    if n_cases == 0:
        return pd.DataFrame(columns=["activity", "cases_with_activity", "exists_pct"])

    case_cov = (
        df.groupby(activity_col)[case_col]
        .nunique()
        .sort_values(ascending=False)
        .rename("cases_with_activity")
        .to_frame()
    )
    case_cov["exists_pct"] = case_cov["cases_with_activity"] / n_cases * 100.0
    result = case_cov.reset_index().rename(columns={activity_col: "activity"})
    logger.info("Computed activity coverage for %d activities across %d cases.", len(result), n_cases)
    return result


def _event_log_to_dataframe(event_log) -> pd.DataFrame:
    if isinstance(event_log, pd.DataFrame):
        logger.debug("Event log already provided as DataFrame.")
        return event_log.copy()

    log_obj = event_log
    if hasattr(event_log, "get_log"):
        log_obj = event_log.get_log()
    elif hasattr(event_log, "log"):
        log_obj = event_log.log

    if isinstance(log_obj, pd.DataFrame):
        logger.debug("Resolved event log object as DataFrame.")
        return log_obj.copy()

    try:
        from pm4py.objects.conversion.log import converter as log_converter
    except ImportError as exc:
        raise ImportError(
            "pm4py is required to convert a non-dataframe event log into a dataframe."
        ) from exc

    logger.debug("Converting event log object to DataFrame with pm4py log converter.")
    return log_converter.apply(log_obj, variant=log_converter.Variants.TO_DATA_FRAME)


def mine_declare_model_and_constraints_df(
    event_log,
    consider_vacuity: bool = False,
    min_support: float = 0.80,
    itemsets_support: float = 0.90,
    max_declare_cardinality: int = 3,
    conformance_consider_vacuity: bool = False,
):
    logger.info(
        "Starting Declare mining and conformance (min_support=%s, itemsets_support=%s, max_declare_cardinality=%s).",
        min_support,
        itemsets_support,
        max_declare_cardinality,
    )
    miner = DeclareMiner(
        log=event_log,
        consider_vacuity=consider_vacuity,
        min_support=min_support,
        itemsets_support=itemsets_support,
        max_declare_cardinality=max_declare_cardinality,
    )
    discovered_model = miner.run()

    checker = MPDeclareAnalyzer(
        log=event_log,
        declare_model=discovered_model,
        consider_vacuity=conformance_consider_vacuity,
    )
    conf_res = checker.run()
    state_df = conf_res.get_metric(metric="state")
    n_traces = len(state_df)

    support_pct = state_df.mean(axis=0) * 100.0
    support_cnt = state_df.sum(axis=0).astype(int)

    declare_table = (
        pd.DataFrame(
            {
                "constraint": support_pct.index,
                "support_pct": support_pct.values,
                "support_traces": support_cnt.values,
                "total_traces": n_traces,
            }
        )
        .sort_values("support_pct", ascending=False)
        .reset_index(drop=True)
    )
    logger.info("Built DECLARE table with %d constraints and %d traces.", len(declare_table), n_traces)
    return declare_table

def generate_kpis_from_declare_table(
    declare_table: pd.DataFrame,
    llm_config: LLMConfig,
    activities_table: pd.DataFrame | None = None,
    event_log=None,
    df: pd.DataFrame | None = None,
    use_attributes: bool = False,
    max_constraints_to_include: int = 300,
    max_activities_to_include: int = 50,
) -> KPISet:
    logger.info("Generating KPI set from DECLARE table.")
    _validate_declare_table(declare_table)
    if activities_table is not None:
        _validate_activities_table(activities_table)
    activities_output = "Activities\n\nNot provided in this input."
    if activities_table is not None:
        activities_output = serialize_activities_table_as_text(
            activities_table=activities_table,
            max_activities_to_include=max_activities_to_include,
        )
    constraints_output = serialize_declare_table_as_text(
        declare_table=declare_table,
        max_constraints_to_include=max_constraints_to_include,
    )
    if use_attributes:
        profile_df = df
        if profile_df is None:
            if event_log is None:
                raise ValueError("use_attributes=True requires either df or event_log.")
            profile_df = _event_log_to_dataframe(event_log)
        attributes_profile = build_attribute_profile_text(profile_df)
        prompt = _build_kpi_prompt_with_attributes(
            activities_output=activities_output,
            constraints_output=constraints_output,
            attributes_profile=attributes_profile,
        )
    else:
        prompt = _build_kpi_prompt(
            activities_output=activities_output,
            constraints_output=constraints_output,
        )

    logger.debug(
        "Calling LLM provider for KPI generation (provider=%s, model=%s).",
        llm_config.provider,
        llm_config.model,
    )
    kpi_set = generate_structured_model(
        llm_config=llm_config,
        prompt=prompt,
        schema_name="kpi_set",
        model_type=KPISet,
    )
    logger.info("Received KPI set with %d KPIs.", len(kpi_set.kpis))
    return kpi_set


def _validate_declare_table(declare_table: pd.DataFrame) -> None:
    missing_cols = _REQUIRED_DECLARE_TABLE_COLUMNS.difference(set(declare_table.columns))
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"declare_table is missing required columns: {missing}")


def _validate_activities_table(activities_table: pd.DataFrame) -> None:
    missing_cols = _REQUIRED_ACTIVITY_TABLE_COLUMNS.difference(set(activities_table.columns))
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"activities_table is missing required columns: {missing}")


def serialize_activities_table_as_text(
    activities_table: pd.DataFrame, max_activities_to_include: int = 50
) -> str:
    _validate_activities_table(activities_table)
    if max_activities_to_include <= 0:
        raise ValueError("max_activities_to_include must be > 0")

    acts_lines = ["Activities", ""]
    for i, row in activities_table.head(max_activities_to_include).iterrows():
        acts_lines.append(
            f"{i + 1}) {row['activity']} : Exists in {row['exists_pct']:.2f}% of traces in the log"
        )
    output = "\n".join(acts_lines)
    logger.debug("Serialized activities summary with %d entries.", min(len(activities_table), max_activities_to_include))
    return output


def serialize_declare_table_as_text(
    declare_table: pd.DataFrame,
    max_constraints_to_include: int = 300,
) -> str:
    _validate_declare_table(declare_table)
    if max_constraints_to_include <= 0:
        raise ValueError("max_constraints_to_include must be > 0")

    constraints_lines = ["Constraints", ""]
    for i, row in declare_table.head(max_constraints_to_include).iterrows():
        constraints_lines.append(
            f"{i + 1}) In {row['support_pct']:.2f}% of cases in the log, {row['constraint']}"
        )
    constraints_output = "\n".join(constraints_lines)
    logger.debug("Serialized DECLARE constraints summary with %d entries.", min(len(declare_table), max_constraints_to_include))
    return constraints_output


def _build_kpi_prompt(activities_output: str, constraints_output: str) -> str:
    template = _KPI_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.format(
        activities_output=activities_output,
        constraints_output=constraints_output,
    ).strip()


def _build_kpi_prompt_with_attributes(
    activities_output: str,
    constraints_output: str,
    attributes_profile: str,
) -> str:
    template = _KPI_PROMPT_WITH_ATTRIBUTES_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.format(
        activities_output=activities_output,
        constraints_output=constraints_output,
        attributes_profile=attributes_profile,
    ).strip()


def build_attribute_profile_text(df: pd.DataFrame, max_categorical_values: int = 20) -> str:
    if max_categorical_values <= 0:
        raise ValueError("max_categorical_values must be > 0.")

    lines = ["Attributes Profile", ""]
    attribute_cols = [c for c in df.columns if c not in _CORE_PROCESS_CONTROL_COLUMNS]
    if not attribute_cols:
        lines.append("No non-core attributes available.")
        return "\n".join(lines)

    for col in attribute_cols:
        series = df[col]
        non_null = series.dropna()
        lines.append(f"- {col}")
        lines.append(f"  - non_null_count: {int(non_null.shape[0])}")
        lines.append(f"  - missing_count: {int(series.isna().sum())}")

        if pd.api.types.is_bool_dtype(series):
            true_count = int((series == True).sum())  # noqa: E712
            false_count = int((series == False).sum())  # noqa: E712
            lines.append("  - type: boolean")
            lines.append(f"  - true_count: {true_count}")
            lines.append(f"  - false_count: {false_count}")
            continue

        if pd.api.types.is_numeric_dtype(series):
            desc = non_null.astype("float64")
            if desc.empty:
                lines.append("  - type: numeric")
                lines.append("  - no numeric values available")
                continue
            lines.append("  - type: numeric")
            lines.append(f"  - min: {desc.min()}")
            lines.append(f"  - max: {desc.max()}")
            lines.append(f"  - mean: {desc.mean()}")
            lines.append(f"  - std: {desc.std(ddof=0)}")
            lines.append(f"  - median: {desc.median()}")
            lines.append(f"  - q25: {desc.quantile(0.25)}")
            lines.append(f"  - q75: {desc.quantile(0.75)}")
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            lines.append("  - type: datetime")
            if non_null.empty:
                lines.append("  - no datetime values available")
            else:
                dt = pd.to_datetime(non_null, errors="coerce").dropna()
                if dt.empty:
                    lines.append("  - no parseable datetime values available")
                else:
                    lines.append(f"  - min: {dt.min()}")
                    lines.append(f"  - max: {dt.max()}")
            continue

        # Treat remaining values as categorical/text-like.
        unique_values = non_null.astype(str).drop_duplicates()
        total_unique = int(unique_values.shape[0])
        sample = unique_values.head(max_categorical_values).tolist()
        lines.append("  - type: categorical")
        lines.append(f"  - distinct_values_count: {total_unique}")
        if total_unique > max_categorical_values:
            lines.append(
                f"  - sample_values(first {max_categorical_values}): {sample}"
            )
        else:
            lines.append(f"  - values: {sample}")

    return "\n".join(lines)


def _build_kpi_categories_prompt(kpi_json_str: str) -> str:
    template = _KPI_CATEGORIES_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.format(kpi_json_str=kpi_json_str).strip()


def _build_kpi_alignment_prompt(items_json_str: str) -> str:
    template = _KPI_ALIGNMENT_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.format(items_json_str=items_json_str).strip()


def _build_ppi_readable_definition_prompt(items_json_str: str) -> str:
    template = _PPI_READABLE_DEFINITION_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.format(items_json_str=items_json_str).strip()


def _build_ppi_selection_prompt(items_json_str: str, max_selected_ppis: int) -> str:
    template = _PPI_SELECTION_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.format(
        items_json_str=items_json_str,
        max_selected_ppis=max_selected_ppis,
    ).strip()


def categorize_kpis_by_business_goal(
    kpi_set: KPISet,
    llm_config: LLMConfig,
) -> KPICategorySet:
    logger.info("Categorizing KPIs into high-level business goals.")
    prompt = _build_kpi_categories_prompt(
        kpi_json_str=kpi_set.model_dump_json(indent=2)
    )

    categories = generate_structured_model(
        llm_config=llm_config,
        prompt=prompt,
        schema_name="kpi_categories",
        model_type=KPICategorySet,
    )
    logger.info("Received KPI categories for %d KPI entries.", len(categories.kpi_categories))
    return categories


def evaluate_kpi_metric_alignment(
    kpi_set: KPISet,
    ppinot_set: PPISet,
    executions: list[PPIExecution],
    llm_config: LLMConfig,
) -> KPIAlignmentSet:
    logger.info("Evaluating KPI/PPI alignment with LLM.")

    ppi_by_kpi_id = {ppi.source_kpi_id: ppi for ppi in ppinot_set.ppis}
    exec_by_kpi_id = {exe.kpi_id: exe for exe in executions}

    items = []
    for kpi in kpi_set.kpis:
        ppi = ppi_by_kpi_id.get(kpi.id)
        exe = exec_by_kpi_id.get(kpi.id)
        items.append(
            {
                "kpi_id": kpi.id,
                "kpi_name": kpi.name,
                "kpi_objective": kpi.objective,
                "kpi_definition": kpi.definition,
                "kpi_metric_str": exe.kpi_metric_str if exe is not None else None,
            }
        )
    prompt = _build_kpi_alignment_prompt(items_json_str=json.dumps(items, indent=2))
    llm_alignment = generate_structured_model(
        llm_config=llm_config,
        prompt=prompt,
        schema_name="kpi_alignment_set",
        model_type=LLMKPIAlignmentSet,
    )
    logger.info("Received KPI alignment for %d items.", len(llm_alignment.items))
    alignment = KPIAlignmentSet(items=[KPIAlignmentItem(
        kpi_id=item.kpi_id,
        alignment_score=item.alignment_score,
        rationale=item.rationale,
        approximation=ppi_by_kpi_id[item.kpi_id].approximation if exec_by_kpi_id.get(item.kpi_id) is not None else None,
    ) for item in llm_alignment.items])    

    return alignment


def generate_human_readable_ppi_definitions(
    executions: list[PPIExecution],
    llm_config: LLMConfig,
) -> PPIReadableDefinitionSet:
    logger.info("Generating human-readable PPI definitions from execution metric strings.")
    items = [
        {"kpi_id": exe.kpi_id, "kpi_metric_str": exe.kpi_metric_str}
        for exe in executions
    ]
    prompt = _build_ppi_readable_definition_prompt(items_json_str=json.dumps(items, indent=2))
    readable_set = generate_structured_model(
        llm_config=llm_config,
        prompt=prompt,
        schema_name="ppi_readable_definition_set",
        model_type=PPIReadableDefinitionSet,
    )
    logger.info("Received human-readable definitions for %d KPIs.", len(readable_set.items))
    return readable_set


def build_ppi_selection_candidates(
    run_results: list[PipelineRunResult],
) -> PPISelectionCandidateSet:
    logger.info("Building PPI selection candidate pool from %d runs.", len(run_results))
    items: list[PPISelectionCandidateItem] = []

    for run_id, run_result in enumerate(run_results):
        category_by_kpi_id = {item.kpi_id: item for item in run_result.kpi_categories.kpi_categories}
        execution_by_kpi_id = {item.kpi_id: item for item in run_result.executions}
        readable_by_kpi_id = {}
        if run_result.readable_ppi_definitions is not None:
            readable_by_kpi_id = {item.kpi_id: item for item in run_result.readable_ppi_definitions.items}

        for ppi in run_result.ppinot_set.ppis:
            kpi_id = ppi.source_kpi_id
            execution = execution_by_kpi_id.get(kpi_id)
            category = category_by_kpi_id.get(kpi_id)
            readable = readable_by_kpi_id.get(kpi_id)
            if execution is None or execution.final_status != "success":
                continue
            if category is None or readable is None:
                continue
            if not category.goal_name.strip() or not category.goal_description.strip():
                continue
            if not readable.human_readable_definition.strip():
                continue

            items.append(
                PPISelectionCandidateItem(
                    candidate_id=f"run_{run_id:03d}:{kpi_id}",
                    run_id=run_id,
                    kpi_id=kpi_id,
                    business_goal=category.goal_name,
                    business_goal_description=category.goal_description,
                    human_readable_ppi_definition=readable.human_readable_definition,
                )
            )

    logger.info("Built candidate pool with %d successful PPIs.", len(items))
    return PPISelectionCandidateSet(items=items)


def select_best_ppis_across_runs(
    candidates: PPISelectionCandidateSet,
    llm_config: LLMConfig,
    max_selected_ppis: int = 15,
) -> SelectedPPISet:
    if max_selected_ppis <= 0:
        raise ValueError("max_selected_ppis must be > 0.")
    if not candidates.items:
        logger.info("No PPI selection candidates available.")
        return SelectedPPISet(items=[])

    prompt_items = [
        {
            "candidate_id": item.candidate_id,
            "business_goal": item.business_goal,
            "business_goal_description": item.business_goal_description,
            "human_readable_ppi_definition": item.human_readable_ppi_definition,
        }
        for item in candidates.items
    ]
    prompt = _build_ppi_selection_prompt(
        items_json_str=json.dumps(prompt_items, indent=2),
        max_selected_ppis=max_selected_ppis,
    )
    llm_selection = generate_structured_model(
        llm_config=llm_config,
        prompt=prompt,
        schema_name="selected_ppi_set",
        model_type=LLMSelectedPPISet,
    )

    candidate_by_id = {item.candidate_id: item for item in candidates.items}
    selected_items: list[SelectedPPIItem] = []
    seen_candidate_ids: set[str] = set()
    for item in llm_selection.items:
        if item.candidate_id in seen_candidate_ids:
            continue
        candidate = candidate_by_id.get(item.candidate_id)
        if candidate is None:
            logger.warning("Skipping unknown selected candidate_id=%s.", item.candidate_id)
            continue
        selected_items.append(
            SelectedPPIItem(
                candidate_id=candidate.candidate_id,
                run_id=candidate.run_id,
                kpi_id=candidate.kpi_id,
                business_goal=candidate.business_goal,
                business_goal_description=candidate.business_goal_description,
                human_readable_ppi_definition=candidate.human_readable_ppi_definition,
                rationale=item.rationale,
            )
        )
        seen_candidate_ids.add(item.candidate_id)
        if len(selected_items) >= max_selected_ppis:
            break

    logger.info("Selected %d final PPIs across runs.", len(selected_items))
    return SelectedPPISet(items=selected_items)


def generate_ppinot_from_kpi_set(
    kpi_set: KPISet,
    llm_config: LLMConfig,
) -> PPISet:
    logger.info("Generating PPINot implementation set from KPI set.")
    kpi_json_str = serialize_kpi_set_for_prompt(kpi_set)
    prompt = _build_ppinot_prompt(kpi_json_str=kpi_json_str)

    logger.debug(
        "Calling LLM provider for PPINot generation (provider=%s, model=%s).",
        llm_config.provider,
        llm_config.model,
    )
    ppinot_set = generate_structured_model(
        llm_config=llm_config,
        prompt=prompt,
        schema_name="ppinot_kpi_set",
        model_type=PPISet,
    )
    logger.info("Received PPINot set with %d PPIs.", len(ppinot_set.ppis))
    return ppinot_set


def serialize_kpi_set_for_prompt(kpi_set: KPISet) -> str:
    return kpi_set.model_dump_json(indent=2)


def _build_ppinot_prompt(kpi_json_str: str) -> str:
    template = _PPINOT_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    ppinot4py_readme = _PPINOT_README_PATH.read_text(encoding="utf-8")
    return template.format(
        kpi_json_str=kpi_json_str,
        ppinot4py_readme=ppinot4py_readme,
    ).strip()


def _build_ppinot_retry_prompt(
    kpi_json_str: str,
    failed_kpi_json_str: str,
    failed_ppi_json_str: str,
    executed_definition_code: str,
    definition_execution_error: str,
    metric_computation_error: str,
) -> str:
    template = _PPINOT_RETRY_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    ppinot4py_readme = _PPINOT_README_PATH.read_text(encoding="utf-8")
    return template.format(
        kpi_json_str=kpi_json_str,
        failed_kpi_json_str=failed_kpi_json_str,
        failed_ppi_json_str=failed_ppi_json_str,
        executed_definition_code=executed_definition_code,
        definition_execution_error=definition_execution_error,
        metric_computation_error=metric_computation_error,
        ppinot4py_readme=ppinot4py_readme,
    ).strip()


def execute_ppinot_kpis(
    ppinot_set: PPISet,
    kpi_set: KPISet,
    llm_config: LLMConfig,
    event_log=None,
    df: pd.DataFrame | None = None,
    declare_table: pd.DataFrame | None = None,
    max_retries: int = 0,
) -> list[PPIExecution]:
    logger.info("Executing PPINot KPI code for %d PPIs.", len(ppinot_set.ppis))
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0.")

    if df is None:
        if event_log is None:
            raise ValueError("Provide either df or event_log to execute PPINot compute code.")
        df = _event_log_to_dataframe(event_log)

    kpi_by_id = {kpi.id: kpi for kpi in kpi_set.kpis}
    results: list[PPIExecution] = []

    for ppi in ppinot_set.ppis:
        current_ppi = ppi
        source_kpi_id = current_ppi.source_kpi_id
        logger.debug("Executing PPI for source_kpi_id=%s.", source_kpi_id)
        kpi = kpi_by_id.get(source_kpi_id)
        retry_attempts = 0

        execution_result = _execute_single_ppi(
            ppi=current_ppi,
            df=df,
            declare_table=declare_table,
        )
        while execution_result["kpi_metric_error"] is not None and retry_attempts < max_retries:
            retry_attempts += 1
            logger.info(
                "Retrying PPI generation for source_kpi_id=%s (attempt %d/%d).",
                source_kpi_id,
                retry_attempts,
                max_retries,
            )
            current_ppi = regenerate_failed_ppi(
                kpi_set=kpi_set,
                failed_kpi=kpi,
                failed_ppi=current_ppi,
                definition_execution_error=execution_result["definition_execution_error"] or "None",
                metric_computation_error=execution_result["metric_computation_error"] or "None",
                executed_definition_code=current_ppi.ppinot4py_definition_code,
                llm_config=llm_config,
            )
            execution_result = _execute_single_ppi(
                ppi=current_ppi,
                df=df,
                declare_table=declare_table,
            )

        final_status = "success"
        last_error = execution_result["kpi_metric_error"]
        if execution_result["definition_execution_error"] or execution_result["kpi_metric_error"]:
            final_status = "failed"

        results.append(
            PPIExecution(
                kpi_id=source_kpi_id,
                kpi_metric_str=execution_result["kpi_metric_str"],
                kpi_metric_computation_value=_normalize_computation_value(
                    execution_result["kpi_metric_computation_value"]
                ),
                definition_execution_error=execution_result["definition_execution_error"],
                metric_computation_error=execution_result["metric_computation_error"],
                retry_attempts=retry_attempts,
                final_status=final_status,
                last_error=last_error,
            )
        )

    logger.info("Execution completed for %d PPIs.", len(results))
    return results


def build_kpi_execution_report(
    kpi_set: KPISet,
    ppinot_set: PPISet,
    executions: list[PPIExecution],
    categories_output: KPICategorySet | None = None,
    variability_results: list[PPIVariabilityAnalysis] | None = None,
    readable_ppi_definitions: PPIReadableDefinitionSet | None = None,
) -> pd.DataFrame:
    def _ensure_object_id_column(df: pd.DataFrame) -> pd.DataFrame:
        if "id" not in df.columns:
            df["id"] = pd.Series(dtype="object")
        else:
            df["id"] = df["id"].astype("object")
        return df

    execution_df = pd.DataFrame([e.model_dump(mode="json") for e in executions]).rename(
        columns={
            "kpi_id": "id",
            "kpi_metric_str": "string conversion of kpi_metric",
            "kpi_metric_computation_value": "output of measure_computer with kpi_metric",
            "definition_execution_error": "definition execution error",
            "metric_computation_error": "metric computation error",
            "retry_attempts": "retry attempts",
            "final_status": "final status",
            "last_error": "last error",
        }
    )
    execution_columns = [
        "id",
        "string conversion of kpi_metric",
        "output of measure_computer with kpi_metric",
        "definition execution error",
        "metric computation error",
        "retry attempts",
        "final status",
        "last error",
    ]
    execution_df = _ensure_object_id_column(execution_df.reindex(columns=execution_columns))

    kpi_rows = [
        {
            "id": kpi.id,
            "name": kpi.name,
            "objective": kpi.objective,
            "definition": kpi.definition,
            "threshold": kpi.threshold.model_dump_json(),
            "rationale": kpi.rationale,
        }
        for kpi in kpi_set.kpis
    ]
    base_df = _ensure_object_id_column(pd.DataFrame(
        kpi_rows,
        columns=["id", "name", "objective", "definition", "threshold", "rationale"],
    ))

    ppi_rows = []
    for ppi in ppinot_set.ppis:
        approximation = ppi.approximation
        ppi_rows.append(
            {
                "id": ppi.source_kpi_id,
                "explanation of the approximation in the computation (if it applies)": approximation,
            }
        )
    ppi_df = _ensure_object_id_column(pd.DataFrame(
        ppi_rows,
        columns=["id", "explanation of the approximation in the computation (if it applies)"],
    ))

    merged = base_df.merge(execution_df, on="id", how="left").merge(ppi_df, on="id", how="left")

    if categories_output is not None:
        categories_rows = [
            {
                "id": item.kpi_id,
                "business goal": item.goal_name,
                "business goal description": item.goal_description,
                "business goal rationale": item.rationale,
            }
            for item in categories_output.kpi_categories
        ]
        categories_df = _ensure_object_id_column(pd.DataFrame(
            categories_rows,
            columns=[
                "id",
                "business goal",
                "business goal description",
                "business goal rationale",
            ],
        ))
        merged = merged.merge(categories_df, on="id", how="left")
    else:
        merged["business goal"] = None
        merged["business goal description"] = None
        merged["business goal rationale"] = None

    if variability_results is not None:
        variability_df = pd.DataFrame([v.model_dump(mode="json") for v in variability_results]).rename(
            columns={
                "kpi_id": "id",
                "time_grouper_freq": "time grouper frequency",
                "grouped_values": "time grouper computed values",
                "coefficient_of_variation": "coefficient of variation",
                "cv_flag_below_threshold": "cv flag below threshold",
                "status": "time grouper analysis status",
            }
        )
        variability_columns = [
            "id",
            "time grouper frequency",
            "time grouper computed values",
            "coefficient of variation",
            "cv flag below threshold",
            "time grouper analysis status",
        ]
        variability_df = _ensure_object_id_column(variability_df.reindex(columns=variability_columns))
        merged = merged.merge(variability_df, on="id", how="left")
    else:
        merged["time grouper frequency"] = None
        merged["time grouper computed values"] = None
        merged["coefficient of variation"] = None
        merged["cv flag below threshold"] = None
        merged["time grouper analysis status"] = None

    if readable_ppi_definitions is not None:
        readable_df = pd.DataFrame(
            [
                {
                    "id": item.kpi_id,
                    "human readable ppi definition": item.human_readable_definition,
                    "human readable ppi rationale": item.rationale,
                }
                for item in readable_ppi_definitions.items
            ]
        )
        readable_df = _ensure_object_id_column(
            readable_df.reindex(
                columns=["id", "human readable ppi definition", "human readable ppi rationale"]
            )
        )
        merged = merged.merge(readable_df, on="id", how="left")
    else:
        merged["human readable ppi definition"] = None
        merged["human readable ppi rationale"] = None

    priority_cols = ["id", "name", "objective", "definition", "threshold", "rationale",
                     "business goal", "business goal description", "business goal rationale"]
    existing_priority_cols = [c for c in priority_cols if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in existing_priority_cols]
    final_df = merged[existing_priority_cols + other_cols]
    required_report_cols = set(priority_cols)
    missing_report_cols = required_report_cols.difference(final_df.columns)
    if missing_report_cols:
        missing = ", ".join(sorted(missing_report_cols))
        raise ValueError(f"Final report is missing required KPI/category columns: {missing}")
    logger.info("Built final KPI execution report with %d rows.", len(final_df))
    return final_df


def analyze_ppi_variability_with_time_grouper(
    ppinot_set: PPISet,
    event_log=None,
    df: pd.DataFrame | None = None,
    time_grouper_freq: str = "1M",
    cv_threshold: float = 0.2,
) -> list[PPIVariabilityAnalysis]:
    logger.info(
        "Analyzing PPI variability with time grouper freq=%s for %d PPIs.",
        time_grouper_freq,
        len(ppinot_set.ppis),
    )
    if df is None:
        if event_log is None:
            raise ValueError("Provide either df or event_log for time-grouper analysis.")
        df = _event_log_to_dataframe(event_log)

    results: list[PPIVariabilityAnalysis] = []
    for ppi in ppinot_set.ppis:
        kpi_id = ppi.source_kpi_id
        grouped_values = None
        cv = None
        cv_flag = None
        status = "ok"

        try:
            validate_generated_python_code(ppi.ppinot4py_definition_code, "definition")
            ns: dict[str, Any] = {"df": df, "pd": pd}
            exec(ppi.ppinot4py_definition_code, ns, ns)
            if "kpi_metric" not in ns:
                status = "skipped"
            else:
                from ppinot4py.computers import measure_computer

                grouped_output = measure_computer(
                    ns["kpi_metric"],
                    df,
                    time_grouper=pd.Grouper(freq=time_grouper_freq),
                )
                grouped_values, numeric_series = _normalize_grouped_output_for_cv(grouped_output)
                if numeric_series is None or len(numeric_series) < 2:
                    status = "skipped"
                else:
                    mean_val = float(numeric_series.mean())
                    if abs(mean_val) < 1e-12:
                        status = "skipped"
                    else:
                        cv = float(numeric_series.std(ddof=0) / abs(mean_val))
                        cv_flag = cv < cv_threshold
        except Exception as exc:
            logger.warning("Variability analysis failed for kpi_id=%s: %s", kpi_id, exc)
            status = "error"

        results.append(
            PPIVariabilityAnalysis(
                kpi_id=kpi_id,
                time_grouper_freq=time_grouper_freq,
                grouped_values=grouped_values,
                coefficient_of_variation=cv,
                cv_flag_below_threshold=cv_flag,
                status=status,
            )
        )
    return results


def run_full_pipeline(
    event_log,
    llm_config: LLMConfig,
    min_support: float = 0.80,
    itemsets_support: float = 0.90,
    max_declare_cardinality: int = 3,
    max_constraints_to_include: int = 300,
    max_activities_to_include: int = 50,
    use_attributes: bool = False,
    max_retries: int = 2,
    time_grouper_freq: str | None = None,
    cv_threshold: float | None = None,
    **kwargs,
) -> tuple[PipelineRunResult, pd.DataFrame]:
    logger.info("Running full KPI/PPI pipeline.")
    consider_vacuity = kwargs.get("consider_vacuity", False)
    conformance_consider_vacuity = kwargs.get("conformance_consider_vacuity", False)

    declare_table = mine_declare_model_and_constraints_df(
        event_log=event_log,
        consider_vacuity=consider_vacuity,
        min_support=min_support,
        itemsets_support=itemsets_support,
        max_declare_cardinality=max_declare_cardinality,
        conformance_consider_vacuity=conformance_consider_vacuity,
    )
    activities_table = compute_activity_case_coverage(event_log)
    kpi_set = generate_kpis_from_declare_table(
        declare_table=declare_table,
        activities_table=activities_table,
        event_log=event_log,
        use_attributes=use_attributes,
        llm_config=llm_config,
        max_constraints_to_include=max_constraints_to_include,
        max_activities_to_include=max_activities_to_include,
    )
    kpi_categories = categorize_kpis_by_business_goal(
        kpi_set=kpi_set,
        llm_config=llm_config,
    )
    ppinot_set = generate_ppinot_from_kpi_set(
        kpi_set=kpi_set,
        llm_config=llm_config,
    )
    executions = execute_ppinot_kpis(
        ppinot_set=ppinot_set,
        kpi_set=kpi_set,
        event_log=event_log,
        declare_table=declare_table,
        max_retries=max_retries,
        llm_config=llm_config,
    )
    readable_ppi_definitions = generate_human_readable_ppi_definitions(
        executions=executions,
        llm_config=llm_config,
    )

    variability_results: list[PPIVariabilityAnalysis] | None = None
    if time_grouper_freq is not None and cv_threshold is not None:
        successful_kpi_ids = {e.kpi_id for e in executions if e.final_status == "success"}
        successful_ppis = [ppi for ppi in ppinot_set.ppis if ppi.source_kpi_id in successful_kpi_ids]
        if successful_ppis:
            variability_results = analyze_ppi_variability_with_time_grouper(
                ppinot_set=PPISet(ppis=successful_ppis),
                event_log=event_log,
                time_grouper_freq=time_grouper_freq,
                cv_threshold=cv_threshold,
            )
        else:
            variability_results = []

    final_report_df = build_kpi_execution_report(
        kpi_set=kpi_set,
        ppinot_set=ppinot_set,
        executions=executions,
        categories_output=kpi_categories,
        variability_results=variability_results,
        readable_ppi_definitions=readable_ppi_definitions,
    )

    result = PipelineRunResult(
        kpi_set=kpi_set,
        kpi_categories=kpi_categories,
        ppinot_set=ppinot_set,
        executions=executions,
        variability_results=variability_results,
        readable_ppi_definitions=readable_ppi_definitions,
    )
    return result, final_report_df


def run_multi_execution_pipeline(
    event_log,
    llm_config: LLMConfig,
    n_runs: int = 3,
    max_selected_ppis: int = 15,
    selector_llm_config: LLMConfig | None = None,
    **kwargs,
) -> tuple[MultiRunPipelineResult, pd.DataFrame]:
    if n_runs <= 0:
        raise ValueError("n_runs must be > 0.")

    logger.info("Running multi-execution KPI/PPI pipeline for %d runs.", n_runs)
    run_results: list[PipelineRunResult] = []
    final_reports: list[pd.DataFrame] = []
    for run_idx in range(n_runs):
        logger.info("Starting pipeline run %d/%d.", run_idx + 1, n_runs)
        run_result, final_report_df = run_full_pipeline(
            event_log=event_log,
            llm_config=llm_config,
            **kwargs,
        )
        run_results.append(run_result)
        final_report_with_run = final_report_df.copy()
        final_report_with_run.insert(0, "run_id", run_idx)
        final_reports.append(final_report_with_run)

    candidates = build_ppi_selection_candidates(run_results)
    selected_ppis = select_best_ppis_across_runs(
        candidates=candidates,
        llm_config=selector_llm_config or llm_config,
        max_selected_ppis=max_selected_ppis,
    )

    selected_rows = []
    selected_lookup = {(item.run_id, item.kpi_id): item for item in selected_ppis.items}
    for row in final_reports:
        if row.empty:
            continue
        filtered = row[
            row.apply(
                lambda current: (int(current["run_id"]), str(current["id"])) in selected_lookup,
                axis=1,
            )
        ].copy()
        if filtered.empty:
            continue
        filtered["selection candidate id"] = filtered.apply(
            lambda current: selected_lookup[(int(current["run_id"]), str(current["id"]))].candidate_id,
            axis=1,
        )
        filtered["selection rationale"] = filtered.apply(
            lambda current: selected_lookup[(int(current["run_id"]), str(current["id"]))].rationale,
            axis=1,
        )
        selected_rows.append(filtered)

    if selected_rows:
        selected_report_df = pd.concat(selected_rows, ignore_index=True)
    else:
        selected_report_df = pd.DataFrame(
            columns=[
                "run_id",
                "id",
                "business goal",
                "business goal description",
                "human readable ppi definition",
                "selection candidate id",
                "selection rationale",
            ]
        )

    result = MultiRunPipelineResult(
        runs=run_results,
        selection_candidates=candidates,
        selected_ppis=selected_ppis,
    )
    return result, selected_report_df



def regenerate_failed_ppi(
    kpi_set: KPISet,
    failed_kpi: KPIItem | None,
    failed_ppi: PPIItem,
    definition_execution_error: str,
    metric_computation_error: str,
    executed_definition_code: str,
    llm_config: LLMConfig,
) -> PPIItem:
    failed_kpi_json = "null"
    if failed_kpi is not None:
        failed_kpi_json = failed_kpi.model_dump_json(indent=2)

    prompt = _build_ppinot_retry_prompt(
        kpi_json_str=kpi_set.model_dump_json(indent=2),
        failed_kpi_json_str=failed_kpi_json,
        failed_ppi_json_str=failed_ppi.model_dump_json(indent=2),
        executed_definition_code=executed_definition_code,
        definition_execution_error=definition_execution_error,
        metric_computation_error=metric_computation_error,
    )
    logger.debug(
        "Calling LLM provider for PPINot retry (provider=%s, model=%s).",
        llm_config.provider,
        llm_config.model,
    )
    regenerated = generate_structured_model(
        llm_config=llm_config,
        prompt=prompt,
        schema_name="ppinot_kpi_set_retry",
        model_type=PPISet,
    )
    if not regenerated.ppis:
        raise ValueError("Retry generation returned no PPIs.")
    return regenerated.ppis[0]


def _execute_single_ppi(
    ppi: PPIItem,
    df: pd.DataFrame,
    declare_table: pd.DataFrame | None,
) -> dict[str, Any]:
    source_kpi_id = ppi.source_kpi_id
    kpi_metric_str = None
    kpi_metric_computation_value = None
    definition_execution_error = None
    metric_computation_error = None
    kpi_metric_error = None
    ns: dict[str, Any] = {"df": df, "declare_table": declare_table, "pd": pd}

    try:
        validate_generated_python_code(ppi.ppinot4py_definition_code, "definition")
        with StringIO() as buf, redirect_stdout(buf):
            exec(ppi.ppinot4py_definition_code, ns, ns)
            _ = buf.getvalue().strip()
        kpi_metric_str = _stringify_kpi_metric(ns)
        if "kpi_metric" not in ns:
            metric_computation_error = "Missing variable 'kpi_metric' after executing definition code."
            kpi_metric_error = metric_computation_error
        else:
            try:
                kpi_metric_computation_value = _compute_kpi_metric_value(ns, df)
            except Exception as exc:
                metric_computation_error = f"{exc.__class__.__name__}: {exc}"
                kpi_metric_error = metric_computation_error
    except Exception as exc:
        definition_execution_error = f"{exc.__class__.__name__}: {exc}"
        logger.warning(
            "PPI definition execution failed for source_kpi_id=%s: %s",
            source_kpi_id,
            definition_execution_error,
        )
        kpi_metric_computation_value = f"ERROR: {definition_execution_error}"
        kpi_metric_error = definition_execution_error

    if kpi_metric_error is not None:
        logger.warning("kpi_metric computation failed for source_kpi_id=%s: %s", source_kpi_id, kpi_metric_error)

    return {
        "kpi_metric_str": kpi_metric_str,
        "kpi_metric_computation_value": kpi_metric_computation_value,
        "definition_execution_error": definition_execution_error,
        "metric_computation_error": metric_computation_error,
        "kpi_metric_error": kpi_metric_error,
    }


def _normalize_computation_value(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_string(max_rows=10, max_cols=10)
    if isinstance(value, pd.Series):
        if len(value) == 1:
            return _normalize_computation_value(value.iloc[0])
        return value.to_string(max_rows=10)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if hasattr(value, "item"):
        try:
            python_scalar = value.item()
        except (TypeError, ValueError):
            pass
        else:
            if python_scalar is not value:
                return _normalize_computation_value(python_scalar)
    return value


def _stringify_kpi_metric(namespace: dict[str, Any]) -> str | None:
    if "kpi_metric" not in namespace:
        return None
    return str(namespace["kpi_metric"])


def _compute_kpi_metric_value(namespace: dict[str, Any], df: pd.DataFrame) -> Any:
    if "kpi_metric" not in namespace:
        return None
    try:
        from ppinot4py.computers import measure_computer
    except ImportError as exc:
        raise ImportError("ppinot4py is required to compute 'kpi_metric'.") from exc
    return measure_computer(namespace["kpi_metric"], df)


def _normalize_grouped_output_for_cv(grouped_output: Any) -> tuple[list[dict[str, Any]], pd.Series | None]:
    series: pd.Series | None = None
    if isinstance(grouped_output, pd.Series):
        series = grouped_output
    elif isinstance(grouped_output, pd.DataFrame):
        if grouped_output.shape[1] == 1:
            series = grouped_output.iloc[:, 0]
        else:
            return [], None
    else:
        return [], None

    grouped_values = [
        {"bucket": str(idx), "value": str(val)}
        for idx, val in series.items()
    ]

    numeric_series: pd.Series | None
    if pd.api.types.is_timedelta64_dtype(series):
        numeric_series = series.dt.total_seconds()
    else:
        numeric_series = pd.to_numeric(series, errors="coerce")

    numeric_series = numeric_series.dropna()
    if numeric_series.empty:
        return grouped_values, None
    return grouped_values, numeric_series


def validate_generated_python_code(code: str, code_label: str) -> None:
    if not code.strip():
        return

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Unsafe {code_label} code: syntax error ({exc.msg}).") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BLOCKED_MODULES:
                    raise ValueError(f"Unsafe {code_label} code: import of '{root}' is blocked.")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _BLOCKED_MODULES:
                    raise ValueError(f"Unsafe {code_label} code: import from '{root}' is blocked.")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALL_NAMES:
                raise ValueError(
                    f"Unsafe {code_label} code: call to '{node.func.id}' is blocked."
                )
            if isinstance(node.func, ast.Attribute) and node.func.attr in _BLOCKED_CALL_ATTRS:
                raise ValueError(
                    f"Unsafe {code_label} code: call to '.{node.func.attr}(...)' is blocked."
                )
