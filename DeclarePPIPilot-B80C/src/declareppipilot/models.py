from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KPIThresholdLevel(_StrictModel):
    level: Literal["info", "warning", "critical"]
    condition: str


class KPIThreshold(_StrictModel):
    levels: list[KPIThresholdLevel]
    rationale: str


class KPIItem(_StrictModel):
    id: str = Field(description="Short stable identifier, e.g., KPI_001")
    name: str
    objective: str
    definition: str
    threshold: KPIThreshold
    rationale: str


class KPISet(_StrictModel):
    kpis: list[KPIItem]


class PPIItem(_StrictModel):
    source_kpi_id: str
    ppinot4py_definition_code: str
    approximation: str | None


class PPISet(_StrictModel):
    ppis: list[PPIItem]


class KPICategoryItem(_StrictModel):
    kpi_id: str
    goal_name: str
    goal_description: str
    rationale: str


class KPICategorySet(_StrictModel):
    kpi_categories: list[KPICategoryItem]


class PPIExecution(_StrictModel):
    kpi_id: str
    kpi_metric_str: str | None = None
    kpi_metric_computation_value: Any = None
    definition_execution_error: str | None = None
    metric_computation_error: str | None = None
    retry_attempts: int = 0
    final_status: Literal["success", "failed"]
    last_error: str | None = None


class PPIVariabilityAnalysis(_StrictModel):
    kpi_id: str
    time_grouper_freq: str
    grouped_values: list[dict[str, Any]] | None = None
    coefficient_of_variation: float | None = None
    cv_flag_below_threshold: bool | None = None
    status: Literal["ok", "skipped", "error"]


class PPIReadableDefinitionItem(_StrictModel):
    kpi_id: str
    human_readable_definition: str
    rationale: str


class PPIReadableDefinitionSet(_StrictModel):
    items: list[PPIReadableDefinitionItem]


class PipelineRunResult(_StrictModel):
    kpi_set: KPISet
    kpi_categories: KPICategorySet
    ppinot_set: PPISet
    executions: list[PPIExecution]
    variability_results: list[PPIVariabilityAnalysis] | None = None
    readable_ppi_definitions: PPIReadableDefinitionSet | None = None


class PPISelectionCandidateItem(_StrictModel):
    candidate_id: str
    run_id: int
    kpi_id: str
    business_goal: str
    business_goal_description: str
    human_readable_ppi_definition: str


class PPISelectionCandidateSet(_StrictModel):
    items: list[PPISelectionCandidateItem]


class LLMSelectedPPIItem(_StrictModel):
    candidate_id: str
    rationale: str


class LLMSelectedPPISet(_StrictModel):
    items: list[LLMSelectedPPIItem]


class SelectedPPIItem(_StrictModel):
    candidate_id: str
    run_id: int
    kpi_id: str
    business_goal: str
    business_goal_description: str
    human_readable_ppi_definition: str
    rationale: str


class SelectedPPISet(_StrictModel):
    items: list[SelectedPPIItem]


class MultiRunPipelineResult(_StrictModel):
    runs: list[PipelineRunResult]
    selection_candidates: PPISelectionCandidateSet
    selected_ppis: SelectedPPISet


class LLMKPIAlignmentItem(_StrictModel):
    kpi_id: str
    alignment_score: float = Field(ge=0.0, le=1.0)
    rationale: str


class LLMKPIAlignmentSet(_StrictModel):
    items: list[LLMKPIAlignmentItem]

class KPIAlignmentItem(_StrictModel):
    kpi_id: str
    alignment_score: float = Field(ge=0.0, le=1.0)
    rationale: str
    approximation: str | None = None


class KPIAlignmentSet(_StrictModel):
    items: list[KPIAlignmentItem]

class LLMConfig(_StrictModel):
    provider: Literal["openai", "anthropic", "lmstudio"]
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    json_retries: int = 2
