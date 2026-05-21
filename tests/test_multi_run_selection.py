from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

declare4py = types.ModuleType("Declare4Py")
process_mining_tasks = types.ModuleType("Declare4Py.ProcessMiningTasks")
conformance_checking = types.ModuleType("Declare4Py.ProcessMiningTasks.ConformanceChecking")
discovery = types.ModuleType("Declare4Py.ProcessMiningTasks.Discovery")
mpdeclare_module = types.ModuleType(
    "Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer"
)
declare_miner_module = types.ModuleType(
    "Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner"
)
mpdeclare_module.MPDeclareAnalyzer = object
declare_miner_module.DeclareMiner = object

sys.modules.setdefault("Declare4Py", declare4py)
sys.modules.setdefault("Declare4Py.ProcessMiningTasks", process_mining_tasks)
sys.modules.setdefault(
    "Declare4Py.ProcessMiningTasks.ConformanceChecking",
    conformance_checking,
)
sys.modules.setdefault("Declare4Py.ProcessMiningTasks.Discovery", discovery)
sys.modules.setdefault(
    "Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer",
    mpdeclare_module,
)
sys.modules.setdefault(
    "Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner",
    declare_miner_module,
)

from declareppipilot.models import (
    KPICategoryItem,
    KPICategorySet,
    KPISet,
    LLMConfig,
    LLMSelectedPPIItem,
    LLMSelectedPPISet,
    MultiRunPipelineResult,
    PipelineRunResult,
    PPIExecution,
    PPIItem,
    PPIReadableDefinitionItem,
    PPIReadableDefinitionSet,
    PPISet,
)
from declareppipilot.pipeline import (
    _normalize_computation_value,
    build_kpi_execution_report,
    build_ppi_selection_candidates,
    run_multi_execution_pipeline,
    select_best_ppis_across_runs,
)
from experiments.common import _model_to_json_data


def _build_run_result(
    *,
    kpi_id: str,
    goal_name: str,
    goal_description: str,
    human_readable_definition: str,
    final_status: str = "success",
) -> PipelineRunResult:
    return PipelineRunResult(
        kpi_set=KPISet(kpis=[]),
        kpi_categories=KPICategorySet(
            kpi_categories=[
                KPICategoryItem(
                    kpi_id=kpi_id,
                    goal_name=goal_name,
                    goal_description=goal_description,
                    rationale="grouped by goal",
                )
            ]
        ),
        ppinot_set=PPISet(
            ppis=[
                PPIItem(
                    source_kpi_id=kpi_id,
                    ppinot4py_definition_code="kpi_metric = None",
                    approximation=None,
                )
            ]
        ),
        executions=[
            PPIExecution(
                kpi_id=kpi_id,
                kpi_metric_str="metric",
                kpi_metric_computation_value=1,
                final_status=final_status,
            )
        ],
        readable_ppi_definitions=PPIReadableDefinitionSet(
            items=[
                PPIReadableDefinitionItem(
                    kpi_id=kpi_id,
                    human_readable_definition=human_readable_definition,
                    rationale="readable",
                )
            ]
        ),
    )


def test_build_ppi_selection_candidates_filters_unsuccessful_runs() -> None:
    run_results = [
        _build_run_result(
            kpi_id="KPI_001",
            goal_name="Reduce delays",
            goal_description="Improve process timeliness.",
            human_readable_definition="Average completion time per case.",
        ),
        _build_run_result(
            kpi_id="KPI_002",
            goal_name="Improve compliance",
            goal_description="Ensure rules are respected.",
            human_readable_definition="Percentage of compliant cases.",
            final_status="failed",
        ),
    ]

    candidates = build_ppi_selection_candidates(run_results)

    assert len(candidates.items) == 1
    assert candidates.items[0].candidate_id == "run_000:KPI_001"
    assert candidates.items[0].business_goal == "Reduce delays"
    assert candidates.items[0].human_readable_ppi_definition == "Average completion time per case."


def test_select_best_ppis_across_runs_enforces_maximum(monkeypatch) -> None:
    run_results = [
        _build_run_result(
            kpi_id=f"KPI_{idx:03d}",
            goal_name=f"Goal {idx}",
            goal_description=f"Description {idx}",
            human_readable_definition=f"Definition {idx}",
        )
        for idx in range(20)
    ]
    candidates = build_ppi_selection_candidates(run_results)

    def _fake_generate_structured_model(*, llm_config, prompt, schema_name, model_type):
        return LLMSelectedPPISet(
            items=[
                LLMSelectedPPIItem(candidate_id=item.candidate_id, rationale="selected")
                for item in candidates.items
            ]
        )

    monkeypatch.setattr(
        "declareppipilot.pipeline.generate_structured_model",
        _fake_generate_structured_model,
    )

    selected = select_best_ppis_across_runs(
        candidates=candidates,
        llm_config=LLMConfig(provider="openai", model="test"),
        max_selected_ppis=15,
    )

    assert len(selected.items) == 15
    assert selected.items[0].candidate_id == "run_000:KPI_000"


def test_run_multi_execution_pipeline_returns_selected_report(monkeypatch) -> None:
    call_count = {"value": 0}

    def _fake_run_full_pipeline(*, event_log, llm_config, **kwargs):
        run_id = call_count["value"]
        call_count["value"] += 1
        run_result = _build_run_result(
            kpi_id=f"KPI_{run_id:03d}",
            goal_name=f"Goal {run_id}",
            goal_description=f"Description {run_id}",
            human_readable_definition=f"Definition {run_id}",
        )
        final_report_df = pd.DataFrame(
            [
                {
                    "id": f"KPI_{run_id:03d}",
                    "business goal": f"Goal {run_id}",
                    "business goal description": f"Description {run_id}",
                    "human readable ppi definition": f"Definition {run_id}",
                }
            ]
        )
        return run_result, final_report_df

    def _fake_generate_structured_model(*, llm_config, prompt, schema_name, model_type):
        return LLMSelectedPPISet(
            items=[LLMSelectedPPIItem(candidate_id="run_000:KPI_000", rationale="best coverage")]
        )

    monkeypatch.setattr("declareppipilot.pipeline.run_full_pipeline", _fake_run_full_pipeline)
    monkeypatch.setattr(
        "declareppipilot.pipeline.generate_structured_model",
        _fake_generate_structured_model,
    )

    result, selected_report_df = run_multi_execution_pipeline(
        event_log=object(),
        llm_config=LLMConfig(provider="openai", model="test"),
        n_runs=3,
        max_selected_ppis=1,
    )

    assert isinstance(result, MultiRunPipelineResult)
    assert len(result.runs) == 3
    assert len(result.selection_candidates.items) == 3
    assert len(result.selected_ppis.items) == 1
    assert list(selected_report_df["selection candidate id"]) == ["run_000:KPI_000"]
    assert list(selected_report_df["selection rationale"]) == ["best coverage"]


def test_numpy_scalar_execution_value_is_json_serializable() -> None:
    execution = PPIExecution(
        kpi_id="KPI_001",
        kpi_metric_str="metric",
        kpi_metric_computation_value=_normalize_computation_value(pd.Series([np.int64(7)])),
        final_status="success",
    )

    payload = _model_to_json_data(execution)

    assert payload["kpi_metric_computation_value"] == 7
    json.dumps(payload)


def test_build_kpi_execution_report_handles_empty_inputs() -> None:
    report_df = build_kpi_execution_report(
        kpi_set=KPISet(kpis=[]),
        ppinot_set=PPISet(ppis=[]),
        executions=[],
        categories_output=KPICategorySet(kpi_categories=[]),
        variability_results=[],
        readable_ppi_definitions=PPIReadableDefinitionSet(items=[]),
    )

    assert report_df.empty
    assert list(report_df.columns[:9]) == [
        "id",
        "name",
        "objective",
        "definition",
        "threshold",
        "rationale",
        "business goal",
        "business goal description",
        "business goal rationale",
    ]
