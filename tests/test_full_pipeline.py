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

from declareppipilot import run_full_pipeline
from declareppipilot.models import LLMConfig


def _build_provider_configs() -> list[LLMConfig]:
    configs: list[LLMConfig] = []

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        configs.append(
            LLMConfig(
                provider="openai",
                model=os.getenv("OPENAI_MODEL", "gpt-5.2"),
            )
        )

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    if anthropic_key and anthropic_model:
        configs.append(
            LLMConfig(
                provider="anthropic",
                model=anthropic_model,
            )
        )
    elif anthropic_key and not anthropic_model:
        print("Skipping anthropic provider: ANTHROPIC_MODEL is not set.")

    lmstudio_base_url = os.getenv("LMSTUDIO_BASE_URL")
    lmstudio_model = os.getenv("LMSTUDIO_MODEL")
    if lmstudio_base_url and lmstudio_model:
        configs.append(
            LLMConfig(
                provider="lmstudio",
                model=lmstudio_model,
                base_url=lmstudio_base_url,
                api_key_env=os.getenv("LMSTUDIO_API_KEY_ENV"),
            )
        )

    return configs


def _run_provider_test(
    *,
    llm_config: LLMConfig,
    event_log: D4PyEventLog,
    run_ts: str,
) -> None:
    provider_tag = llm_config.provider
    if llm_config.provider == "lmstudio":
        provider_tag = f"{provider_tag}_{llm_config.model.replace('/', '_').replace(':', '_')}"
    run_result, final_report_df = run_full_pipeline(
        event_log=event_log,
        llm_config=llm_config,
        max_retries=2,
        time_grouper_freq="1ME",
        cv_threshold=0.2,
        use_attributes=True
    )

    assert len(run_result.kpi_set.kpis) > 0, "No KPIs generated."
    assert len(run_result.ppinot_set.ppis) > 0, "No PPIs generated."
    assert len(final_report_df) > 0, "Final report is empty."

    expected_cols = {
        "id",
        "name",
        "objective",
        "definition",
        "threshold",
        "rationale",
        "business goal",
        "business goal description",
        "business goal rationale",
    }
    missing_cols = expected_cols.difference(set(final_report_df.columns))
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise AssertionError(f"Missing expected columns in final report: {missing}")

    kpi_output_path = PROJECT_ROOT / "outputs" / f"kpi_output_{provider_tag}_{run_ts}.json"
    kpi_output_path.write_text(
        json.dumps(run_result.kpi_set.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    categories_output_path = PROJECT_ROOT / "outputs" / f"kpi_categories_{provider_tag}_{run_ts}.json"
    categories_output_path.write_text(
        json.dumps(run_result.kpi_categories.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    ppinot_output_path = PROJECT_ROOT / "outputs" / f"ppinot_output_{provider_tag}_{run_ts}.json"
    ppinot_output_path.write_text(
        json.dumps(run_result.ppinot_set.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    if run_result.readable_ppi_definitions is not None:
        readable_output_path = PROJECT_ROOT / "outputs" / f"ppi_readable_definitions_{provider_tag}_{run_ts}.json"
        readable_output_path.write_text(
            json.dumps(run_result.readable_ppi_definitions.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

    execution_df = pd.DataFrame([item.model_dump(mode="json") for item in run_result.executions])
    execution_output_path = PROJECT_ROOT / "outputs" / f"ppi_execution_raw_{provider_tag}_{run_ts}.csv"
    execution_df.to_csv(execution_output_path, index=False)

    if run_result.variability_results is not None:
        variability_df = pd.DataFrame(
            [item.model_dump(mode="json") for item in run_result.variability_results]
        )
        variability_output_path = PROJECT_ROOT / "outputs" / f"ppi_variability_raw_{provider_tag}_{run_ts}.csv"
        variability_df.to_csv(variability_output_path, index=False)

    final_report_output_path = PROJECT_ROOT / "outputs" / f"ppi_execution_{provider_tag}_{run_ts}.csv"
    final_report_df.to_csv(final_report_output_path, index=False)

    print(f"Full pipeline test passed for provider={llm_config.provider}.")
    print("KPIs:", len(run_result.kpi_set.kpis))
    print("PPIs:", len(run_result.ppinot_set.ppis))
    print("Final report rows:", len(final_report_df))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    xes_path = PROJECT_ROOT / "data" / "logs" / "DomesticDeclarations.xes"
    if not xes_path.exists():
        raise FileNotFoundError(f"XES file not found: {xes_path.resolve()}")

    provider_configs = _build_provider_configs()
    if not provider_configs:
        print(
            "Skipping full pipeline test: no provider credentials/config found "
            "(OPENAI_API_KEY, ANTHROPIC_API_KEY+ANTHROPIC_MODEL, or LMSTUDIO_BASE_URL+LMSTUDIO_MODEL)."
        )
        return

    event_log = D4PyEventLog(case_name="case:concept:name")
    event_log.parse_xes_log(str(xes_path))

    for llm_config in provider_configs:
        _run_provider_test(
            llm_config=llm_config,
            event_log=event_log,
            run_ts=run_ts,
        )


if __name__ == "__main__":
    main()
