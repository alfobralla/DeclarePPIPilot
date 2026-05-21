import argparse
import ast
import difflib
import json
import logging
import os
import re
import sys
from statistics import median
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from declareppipilot import validate_generated_python_code
from declareppipilot.models import LLMConfig

def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def build_llm_config_from_args(args: argparse.Namespace) -> LLMConfig:
    return LLMConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        json_retries=args.json_retries,
    )


def ensure_llm_credentials(llm_config: LLMConfig) -> None:
    if llm_config.provider == "openai":
        key_env = llm_config.api_key_env or "OPENAI_API_KEY"
        if not os.getenv(key_env):
            raise RuntimeError(f"{key_env} is not set.")
        return

    if llm_config.provider == "anthropic":
        key_env = llm_config.api_key_env or "ANTHROPIC_API_KEY"
        if not os.getenv(key_env):
            raise RuntimeError(f"{key_env} is not set.")
        return

    if llm_config.provider == "lmstudio":
        if not llm_config.base_url:
            raise RuntimeError("LM Studio requires --base-url.")
        if llm_config.api_key_env and not os.getenv(llm_config.api_key_env):
            raise RuntimeError(f"{llm_config.api_key_env} is not set.")
        return

    raise RuntimeError(f"Unsupported provider: {llm_config.provider}")


def load_event_log(xes_path: Path):
    if not xes_path.exists():
        raise FileNotFoundError(f"XES file not found: {xes_path.resolve()}")
    from Declare4Py.D4PyEventLog import D4PyEventLog

    event_log = D4PyEventLog(case_name="case:concept:name")
    event_log.parse_xes_log(str(xes_path))
    return event_log


def normalize_definition_code(code: str) -> str:
    code = code.strip()
    if not code:
        return ""
    try:
        tree = ast.parse(code)
        return ast.dump(tree, annotate_fields=False, include_attributes=False)
    except SyntaxError:
        return re.sub(r"\s+", " ", code)


_MEASURE_CLASS_TO_BUCKET = {
    "TimeMeasure": "time",
    "DataMeasure": "data",
    "CountMeasure": "frequency",
    "AggregatedMeasure": "aggregated",
    "DerivedMeasure": "derived",
}
_SUPPORTED_BUCKETS = ("time", "data", "frequency", "aggregated", "derived", "other")
_AGGREGATED_GROUPER_KEYS = {"grouper", "group_by", "time_grouper"}
_AGGREGATED_FILTER_KEYS = {"filter_to_apply", "filter", "filters", "condition"}
_CONDITION_KEYS = {
    "condition",
    "when",
    "predicate",
    "filter",
    "filters",
    "filter_to_apply",
    "time_condition",
    "data_condition",
    "event_condition",
}


def _extract_call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_measure_call_name(name: str | None) -> bool:
    if not name:
        return False
    return name.endswith("Measure")


def _collect_assignments(module: ast.Module) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}
    for stmt in module.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            assignments[stmt.targets[0].id] = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            assignments[stmt.target.id] = stmt.value
    return assignments


def _resolve_name_expr(
    expr: ast.AST,
    assignments: dict[str, ast.AST],
    active_names: set[str] | None = None,
) -> ast.AST:
    if not isinstance(expr, ast.Name):
        return expr
    if expr.id not in assignments:
        return expr
    active = active_names or set()
    if expr.id in active:
        return expr
    return _resolve_name_expr(assignments[expr.id], assignments, active | {expr.id})


def _canonicalize_expr(
    expr: ast.AST,
    assignments: dict[str, ast.AST],
    name_map: dict[str, str],
    active_names: set[str] | None = None,
) -> str:
    active = active_names or set()
    node = _resolve_name_expr(expr, assignments, active)

    if isinstance(node, ast.Name):
        placeholder = name_map.setdefault(node.id, f"v{len(name_map) + 1}")
        return f"Name({placeholder})"

    if isinstance(node, ast.Constant):
        return f"Const({repr(node.value)})"

    if isinstance(node, ast.Call):
        call_name = _extract_call_name(node.func) or "UnknownCall"
        args = ",".join(_canonicalize_expr(arg, assignments, name_map, active) for arg in node.args)
        kw_items = []
        for kw in sorted(node.keywords, key=lambda k: k.arg or ""):
            key = kw.arg or "**"
            kw_items.append(f"{key}={_canonicalize_expr(kw.value, assignments, name_map, active)}")
        kws = ",".join(kw_items)
        return f"Call({call_name};args=[{args}];kws=[{kws}])"

    if isinstance(node, ast.Attribute):
        base = _canonicalize_expr(node.value, assignments, name_map, active)
        return f"Attr({base}.{node.attr})"

    if isinstance(node, ast.BinOp):
        left = _canonicalize_expr(node.left, assignments, name_map, active)
        right = _canonicalize_expr(node.right, assignments, name_map, active)
        return f"BinOp({node.op.__class__.__name__};{left};{right})"

    if isinstance(node, ast.UnaryOp):
        operand = _canonicalize_expr(node.operand, assignments, name_map, active)
        return f"UnaryOp({node.op.__class__.__name__};{operand})"

    if isinstance(node, ast.BoolOp):
        values = ",".join(_canonicalize_expr(v, assignments, name_map, active) for v in node.values)
        return f"BoolOp({node.op.__class__.__name__};[{values}])"

    if isinstance(node, ast.Compare):
        left = _canonicalize_expr(node.left, assignments, name_map, active)
        comps = ",".join(_canonicalize_expr(c, assignments, name_map, active) for c in node.comparators)
        ops = ",".join(op.__class__.__name__ for op in node.ops)
        return f"Compare({ops};{left};[{comps}])"

    if isinstance(node, ast.Tuple):
        return "Tuple([" + ",".join(_canonicalize_expr(e, assignments, name_map, active) for e in node.elts) + "])"
    if isinstance(node, ast.List):
        return "List([" + ",".join(_canonicalize_expr(e, assignments, name_map, active) for e in node.elts) + "])"
    if isinstance(node, ast.Dict):
        pairs = []
        for key, val in zip(node.keys, node.values):
            key_str = _canonicalize_expr(key, assignments, name_map, active) if key is not None else "None"
            val_str = _canonicalize_expr(val, assignments, name_map, active)
            pairs.append(f"{key_str}:{val_str}")
        return "Dict({" + ",".join(pairs) + "})"

    if isinstance(node, ast.Subscript):
        value = _canonicalize_expr(node.value, assignments, name_map, active)
        sl = _canonicalize_expr(node.slice, assignments, name_map, active)
        return f"Subscript({value};{sl})"

    return ast.dump(node, annotate_fields=False, include_attributes=False)


def _collect_measure_paths(
    expr: ast.AST,
    assignments: dict[str, ast.AST],
    active_names: set[str] | None = None,
) -> list[list[str]]:
    active = active_names or set()
    node = expr
    if isinstance(node, ast.Name) and node.id in assignments and node.id not in active:
        return _collect_measure_paths(assignments[node.id], assignments, active | {node.id})

    if isinstance(node, ast.Call):
        measure_name = _extract_call_name(node.func)
        child_paths: list[list[str]] = []
        for arg in node.args:
            child_paths.extend(_collect_measure_paths(arg, assignments, active))
        for kw in node.keywords:
            child_paths.extend(_collect_measure_paths(kw.value, assignments, active))

        if _is_measure_call_name(measure_name):
            if not child_paths:
                return [[measure_name]]
            return [[measure_name, *path] for path in child_paths]
        return child_paths

    paths: list[list[str]] = []
    for child in ast.iter_child_nodes(node):
        paths.extend(_collect_measure_paths(child, assignments, active))
    return paths


def _bucket_from_measure_name(name: str | None) -> str:
    if not name:
        return "other"
    return _MEASURE_CLASS_TO_BUCKET.get(name, "other")


def _normalize_ast_text(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        text = ast.unparse(node)
    except Exception:
        text = ast.dump(node, annotate_fields=False, include_attributes=False)
    return re.sub(r"\s+", " ", text).strip()


def _is_effectively_empty_node(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        if node.value is None:
            return True
        if isinstance(node.value, (str, bytes)) and node.value == "":
            return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and len(node.elts) == 0:
        return True
    if isinstance(node, ast.Dict) and len(node.keys) == 0:
        return True
    return False


def _collect_measure_calls(
    expr: ast.AST,
    assignments: dict[str, ast.AST],
    active_names: set[str] | None = None,
) -> list[ast.Call]:
    active = active_names or set()
    node = expr
    if isinstance(node, ast.Name) and node.id in assignments and node.id not in active:
        return _collect_measure_calls(assignments[node.id], assignments, active | {node.id})

    calls: list[ast.Call] = []
    if isinstance(node, ast.Call):
        measure_name = _extract_call_name(node.func)
        if _is_measure_call_name(measure_name):
            calls.append(node)
        for arg in node.args:
            calls.extend(_collect_measure_calls(arg, assignments, active))
        for kw in node.keywords:
            calls.extend(_collect_measure_calls(kw.value, assignments, active))
        return calls

    for child in ast.iter_child_nodes(node):
        calls.extend(_collect_measure_calls(child, assignments, active))
    return calls


def _extract_conditions_by_measure_type(
    measure_calls: list[ast.Call],
) -> tuple[list[str], list[str], list[str]]:
    time_conditions: list[str] = []
    count_conditions: list[str] = []
    data_conditions: list[str] = []

    def add_unique(target: list[str], value: str) -> None:
        if value and value not in target:
            target.append(value)

    for call in measure_calls:
        measure_name = _extract_call_name(call.func)
        if measure_name not in {"TimeMeasure", "CountMeasure", "DataMeasure"}:
            continue
        condition_chunks: list[str] = []
        for kw in call.keywords:
            key = kw.arg or ""
            if key.lower() in _CONDITION_KEYS or "condition" in key.lower() or "filter" in key.lower():
                value_text = _normalize_ast_text(kw.value)
                if value_text:
                    condition_chunks.append(f"{key}={value_text}")
        if not condition_chunks:
            continue
        packed = " ; ".join(condition_chunks)
        if measure_name == "TimeMeasure":
            add_unique(time_conditions, packed)
        elif measure_name == "CountMeasure":
            add_unique(count_conditions, packed)
        elif measure_name == "DataMeasure":
            add_unique(data_conditions, packed)
    return time_conditions, count_conditions, data_conditions


def analyze_kpi_metric_definition(definition_code: str) -> dict[str, Any]:
    fallback = normalize_definition_code(definition_code)
    unresolved_payload = {
        "analysis_status": "unresolved",
        "metric_signature": fallback,
        "kpi_metric_text": fallback,
        "final_metric_type": "other",
        "composition_profile": "none",
        "base_metric_types": ["other"],
        "type_labels": ["other"],
        "contains_time_measure": False,
        "contains_data_measure": False,
        "contains_frequency_measure": False,
        "contains_aggregated_measure": False,
        "contains_derived_measure": False,
        "aggregated_has_filter": False,
        "aggregated_has_grouper": False,
        "aggregated_filter_count": 0,
        "aggregated_grouper_count": 0,
        "time_conditions": [],
        "count_conditions": [],
        "data_conditions": [],
        "data_preconditions": [],
        "data_content_selection_values": [],
        "derived_variables_total": 0,
        "derived_variables_avg": 0.0,
    }
    if not definition_code.strip():
        return unresolved_payload

    try:
        module = ast.parse(definition_code)
    except SyntaxError:
        return unresolved_payload

    assignments = _collect_assignments(module)
    kpi_expr = assignments.get("kpi_metric")
    if kpi_expr is None:
        return unresolved_payload

    name_map: dict[str, str] = {}
    metric_signature = _canonicalize_expr(kpi_expr, assignments, name_map)
    resolved_kpi_expr = _resolve_name_expr(kpi_expr, assignments)
    try:
        kpi_metric_text = ast.unparse(resolved_kpi_expr)
    except Exception:
        kpi_metric_text = metric_signature

    final_measure_name = None
    if isinstance(resolved_kpi_expr, ast.Call):
        call_name = _extract_call_name(resolved_kpi_expr.func)
        if _is_measure_call_name(call_name):
            final_measure_name = call_name

    paths = _collect_measure_paths(kpi_expr, assignments)
    unique_profiles = sorted({"->".join(path) for path in paths if path})
    measure_names = {name for path in paths for name in path}
    bucket_set = {_bucket_from_measure_name(name) for name in measure_names}
    base_measure_names = {path[-1] for path in paths if path}
    base_bucket_candidates = {
        _bucket_from_measure_name(name)
        for name in base_measure_names
    }
    base_metric_types = sorted(t for t in base_bucket_candidates if t in {"time", "data", "frequency"})
    if not base_metric_types:
        base_metric_types = ["other"]
    type_labels = sorted(({"aggregated"} if "aggregated" in bucket_set else set()) |
                         ({"derived"} if "derived" in bucket_set else set()) |
                         set(base_metric_types))
    runtime_features = extract_measure_features_from_definition_code(definition_code)
    runtime_ok = runtime_features.get("analysis_status") == "ok"
    contains_time = runtime_features.get("n_time_measures", 0) > 0 if runtime_ok else ("time" in bucket_set)
    contains_data = runtime_features.get("n_data_measures", 0) > 0 if runtime_ok else ("data" in bucket_set)
    contains_frequency = runtime_features.get("n_count_measures", 0) > 0 if runtime_ok else ("frequency" in bucket_set)
    contains_aggregated = runtime_features.get("n_aggregated_measures", 0) > 0 if runtime_ok else ("aggregated" in bucket_set)
    contains_derived = runtime_features.get("n_derived_measures", 0) > 0 if runtime_ok else ("derived" in bucket_set)
    aggregated_has_filter = runtime_features.get("n_aggregated_with_filter", 0) > 0 if runtime_ok else False
    aggregated_has_grouper = runtime_features.get("n_aggregated_with_grouper", 0) > 0 if runtime_ok else False
    aggregated_filter_count = int(runtime_features.get("n_aggregated_with_filter", 0)) if runtime_ok else 0
    aggregated_grouper_count = int(runtime_features.get("n_aggregated_with_grouper", 0)) if runtime_ok else 0
    time_conditions = runtime_features.get("time_conditions", []) if runtime_ok else []
    count_conditions = runtime_features.get("count_conditions", []) if runtime_ok else []
    data_preconditions = runtime_features.get("data_preconditions", []) if runtime_ok else []
    data_content_selection_values = runtime_features.get("data_content_selection_values", []) if runtime_ok else []
    # Keep legacy key used by current RQ1 output.
    data_conditions = data_preconditions

    return {
        "analysis_status": "ok",
        "metric_signature": metric_signature,
        "kpi_metric_text": kpi_metric_text,
        "final_metric_type": _bucket_from_measure_name(final_measure_name),
        "composition_profile": " | ".join(unique_profiles) if unique_profiles else "none",
        "base_metric_types": base_metric_types,
        "type_labels": type_labels,
        "contains_time_measure": contains_time,
        "contains_data_measure": contains_data,
        "contains_frequency_measure": contains_frequency,
        "contains_aggregated_measure": contains_aggregated,
        "contains_derived_measure": contains_derived,
        "aggregated_has_filter": aggregated_has_filter,
        "aggregated_has_grouper": aggregated_has_grouper,
        "aggregated_filter_count": aggregated_filter_count,
        "aggregated_grouper_count": aggregated_grouper_count,
        "time_conditions": time_conditions,
        "count_conditions": count_conditions,
        "data_conditions": data_conditions,
        "data_preconditions": data_preconditions,
        "data_content_selection_values": data_content_selection_values,
        "derived_variables_total": int(runtime_features.get("derived_variables_total", 0)) if runtime_ok else 0,
        "derived_variables_avg": float(runtime_features.get("derived_variables_avg", 0.0)) if runtime_ok else 0.0,
    }


def classify_kpi_type(definition_code: str) -> str:
    return analyze_kpi_metric_definition(definition_code)["final_metric_type"]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a.union(b)
    if not union:
        return 0.0
    return len(a.intersection(b)) / len(union)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe_value(payload), indent=2), encoding="utf-8")


def model_type_breakdown(definition_codes: list[str]) -> dict[str, int]:
    result = {bucket: 0 for bucket in _SUPPORTED_BUCKETS}
    for code in definition_codes:
        result[classify_kpi_type(code)] += 1
    return result


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def exact_set_overlap_metrics(a: set[str], b: set[str]) -> dict[str, float]:
    exact_match_count = len(a.intersection(b))
    return {
        "exact_match_count": float(exact_match_count),
        "exact_match_jaccard": jaccard(a, b),
    }


def best_match_similarity_metrics(a: set[str], b: set[str]) -> dict[str, float]:
    if not a and not b:
        return {"best_match_similarity_mean": 1.0, "best_match_similarity_median": 1.0}
    if not a or not b:
        return {"best_match_similarity_mean": 0.0, "best_match_similarity_median": 0.0}

    left = list(a)
    right = list(b)
    if len(left) > len(right):
        left, right = right, left

    remaining = set(range(len(right)))
    scores: list[float] = []
    for s in left:
        best_idx = None
        best_score = -1.0
        for idx in remaining:
            score = difflib.SequenceMatcher(None, s, right[idx]).ratio()
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None:
            remaining.remove(best_idx)
            scores.append(best_score)

    if not scores:
        return {"best_match_similarity_mean": 0.0, "best_match_similarity_median": 0.0}
    return {
        "best_match_similarity_mean": float(sum(scores) / len(scores)),
        "best_match_similarity_median": float(median(scores)),
    }


def normalize_text_for_similarity(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_value_for_matching(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, (int, float, bool, str)):
        return repr(value)
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return repr(value)


def heuristic_exact_match_metrics(
    a_items: list[tuple[str, str]],
    b_items: list[tuple[str, str]],
    similarity_threshold: float = 0.8,
) -> dict[str, float]:
    if not a_items and not b_items:
        return {"heuristic_exact_match_count": 0.0, "heuristic_exact_match_rate": 1.0}
    if not a_items or not b_items:
        return {"heuristic_exact_match_count": 0.0, "heuristic_exact_match_rate": 0.0}

    left = list(a_items)
    right = list(b_items)
    if len(left) > len(right):
        left, right = right, left

    remaining = set(range(len(right)))
    exact_count = 0
    value_count = 0
    for l in left:
        for idx in remaining:
            r = right[idx]
            value_match, heuristic_match = value_heuristic_match(l, r, similarity_threshold)
            if value_match:
                value_count += 1
            
            if heuristic_match:
                exact_count += 1

    denom = min(len(a_items), len(b_items))
    heuristic_rate = exact_count / denom if denom else 0.0
    value_rate = value_count / denom if denom else 0.0
    return {
        "heuristic_exact_match_count": float(exact_count),
        "heuristic_exact_match_rate": float(heuristic_rate),
        "value_exact_match_count": float(value_count),
        "value_exact_match_rate": float(value_rate)
    }

def value_heuristic_match(left: tuple[str,str], right: tuple[str, str], similarity_threshold: float = 0.8):
    left_text, left_value = left
    right_text, right_value = right

    if left_value != right_value:
        return False, False
    
    score = difflib.SequenceMatcher(None, left_text, right_text).ratio()
    if score > similarity_threshold:
        return True, True
    else:
        return True, False


def _model_to_json_data(model_obj: Any) -> Any:
    if model_obj is None:
        return None
    if hasattr(model_obj, "model_dump"):
        try:
            return model_obj.model_dump(mode="json")
        except Exception:
            return _json_safe_value(model_obj.model_dump(mode="python"))
    return _json_safe_value(model_obj)


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
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
                return _json_safe_value(python_scalar)
    return value


def _runtime_is_measure_instance(obj: Any) -> bool:
    return obj is not None and obj.__class__.__name__.endswith("Measure")


def _runtime_value_to_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        return re.sub(r"\s+", " ", str(value)).strip()
    except Exception:
        return repr(value)



def _get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(obj, attr)
    except Exception:
        return default




def extract_measure_instance_features(metric_instance: Any) -> dict[str, Any]:
    if metric_instance is None:
        return {
            "analysis_status": "error",
            "error": "metric_instance is None.",
            "n_time_measures": 0,
            "n_count_measures": 0,
            "n_data_measures": 0,
            "n_aggregated_measures": 0,
            "n_derived_measures": 0,
            "n_aggregated_with_filter": 0,
            "n_aggregated_with_grouper": 0,
            "time_conditions": [],
            "count_conditions": [],
            "data_preconditions": [],
            "data_content_selection_values": [],
            "derived_variables_total": 0,
            "derived_variables_avg": 0.0,
        }

    counts = {
        "n_time_measures": 0,
        "n_count_measures": 0,
        "n_data_measures": 0,
        "n_aggregated_measures": 0,
        "n_derived_measures": 0,
        "n_aggregated_with_filter": 0,
        "n_aggregated_with_grouper": 0,
    }
    time_conditions: list[str] = []
    count_conditions: list[str] = []
    data_preconditions: list[str] = []
    data_content_selection_values: list[str] = []
    derived_variable_counts: list[int] = []

    visited: set[int] = set()
    stack = [metric_instance]

    def _append_unique(target: list[str], text: str) -> None:
        if text and text not in target:
            target.append(text)

    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)

        class_name = node.__class__.__name__
        if class_name == "TimeMeasure":
            counts["n_time_measures"] += 1
            _append_unique(time_conditions, _runtime_value_to_text(_get_attr(node, "from_condition")))
            _append_unique(time_conditions, _runtime_value_to_text(_get_attr(node, "to_condition")))
        elif class_name == "CountMeasure":
            counts["n_count_measures"] += 1
            _append_unique(count_conditions, _runtime_value_to_text(_get_attr(node, "when")))
        elif class_name == "DataMeasure":
            counts["n_data_measures"] += 1
            _append_unique(data_preconditions, _runtime_value_to_text(_get_attr(node, "precondition")))
            _append_unique(
                data_content_selection_values,
                _runtime_value_to_text(_get_attr(node, "data_content_selection")),
            )
        elif class_name == "AggregatedMeasure":
            counts["n_aggregated_measures"] += 1

            if node.filter_to_apply is not None:            
                counts["n_aggregated_with_filter"] += 1
                stack.append(node.filter_to_apply)

            if node.grouper is not None:
                if isinstance(node.grouper, list):
                    if len(node.grouper) > 0:
                        counts["n_aggregated_with_grouper"] += 1
                        for g in node.grouper:
                            if not isinstance(g, str):
                                stack.append(g)
                else:
                    counts["n_aggregated_with_grouper"] += 1
                    if not isinstance(node.grouper, str):
                        stack.append(node.grouper)

        elif class_name == "DerivedMeasure":
            counts["n_derived_measures"] += 1
            value = _get_attr(node, "measure_map")
            if value is None:
                continue
            if isinstance(value, dict):
                derived_variable_counts.append(len(value))
                for _,m in value.items():
                    stack.append(m)

    derived_variables_total = int(sum(derived_variable_counts))
    derived_variables_avg = (
        float(derived_variables_total / len(derived_variable_counts))
        if derived_variable_counts
        else 0.0
    )

    return {
        "analysis_status": "ok",
        "error": None,
        **counts,
        "time_conditions": time_conditions,
        "count_conditions": count_conditions,
        "data_preconditions": data_preconditions,
        "data_content_selection_values": data_content_selection_values,
        "derived_variables_total": derived_variables_total,
        "derived_variables_avg": derived_variables_avg,
    }


def extract_measure_features_from_definition_code(
    definition_code: str,
    namespace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_generated_python_code(definition_code, "definition")
    ns: dict[str, Any] = {"pd": pd}
    if namespace:
        ns.update(namespace)
    try:
        exec(definition_code, ns, ns)
    except Exception as exc:
        return {
            "analysis_status": "error",
            "error": f"{exc.__class__.__name__}: {exc}",
            "n_time_measures": 0,
            "n_count_measures": 0,
            "n_data_measures": 0,
            "n_aggregated_measures": 0,
            "n_derived_measures": 0,
            "n_aggregated_with_filter": 0,
            "n_aggregated_with_grouper": 0,
            "time_conditions": [],
            "count_conditions": [],
            "data_preconditions": [],
            "data_content_selection_values": [],
            "derived_variables_total": 0,
            "derived_variables_avg": 0.0,
        }
    metric = ns.get("kpi_metric")
    if metric is None:
        return {
            "analysis_status": "error",
            "error": "Missing 'kpi_metric' after executing definition code.",
            "n_time_measures": 0,
            "n_count_measures": 0,
            "n_data_measures": 0,
            "n_aggregated_measures": 0,
            "n_derived_measures": 0,
            "n_aggregated_with_filter": 0,
            "n_aggregated_with_grouper": 0,
            "time_conditions": [],
            "count_conditions": [],
            "data_preconditions": [],
            "data_content_selection_values": [],
            "derived_variables_total": 0,
            "derived_variables_avg": 0.0,
        }
    return extract_measure_instance_features(metric)


def save_pipeline_run_artifacts(
    raw_root_dir: Path,
    run_label: str,
    *,
    kpi_set: Any | None = None,
    kpi_categories: Any | None = None,
    ppinot_set: Any | None = None,
    executions: list[Any] | None = None,
    executions_no_retry: list[Any] | None = None,
    executions_with_retry: list[Any] | None = None,
    variability_results: list[Any] | None = None,
    readable_ppi_definitions: Any | None = None,
    selection_candidates: Any | None = None,
    selected_ppis: Any | None = None,
    final_report_df: pd.DataFrame | None = None,
    selected_report_df: pd.DataFrame | None = None,
    declare_table: pd.DataFrame | None = None,
    activities_table: pd.DataFrame | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    run_dir = raw_root_dir / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    if metadata is not None:
        save_json(run_dir / "metadata.json", metadata)

    if kpi_set is not None:
        save_json(run_dir / "kpi_set.json", _model_to_json_data(kpi_set))
    if kpi_categories is not None:
        save_json(run_dir / "kpi_categories.json", _model_to_json_data(kpi_categories))
    if ppinot_set is not None:
        save_json(run_dir / "ppinot_set.json", _model_to_json_data(ppinot_set))

    if executions is not None:
        payload = [_model_to_json_data(item) for item in executions]
        save_json(run_dir / "executions.json", payload)
        pd.DataFrame(payload).to_csv(run_dir / "executions.csv", index=False)
    if executions_no_retry is not None:
        payload = [_model_to_json_data(item) for item in executions_no_retry]
        save_json(run_dir / "executions_no_retry.json", payload)
        pd.DataFrame(payload).to_csv(run_dir / "executions_no_retry.csv", index=False)
    if executions_with_retry is not None:
        payload = [_model_to_json_data(item) for item in executions_with_retry]
        save_json(run_dir / "executions_with_retry.json", payload)
        pd.DataFrame(payload).to_csv(run_dir / "executions_with_retry.csv", index=False)
    if variability_results is not None:
        payload = [_model_to_json_data(item) for item in variability_results]
        save_json(run_dir / "variability_results.json", payload)
        pd.DataFrame(payload).to_csv(run_dir / "variability_results.csv", index=False)
    if readable_ppi_definitions is not None:
        save_json(run_dir / "readable_ppi_definitions.json", _model_to_json_data(readable_ppi_definitions))
    if selection_candidates is not None:
        save_json(run_dir / "selection_candidates.json", _model_to_json_data(selection_candidates))
    if selected_ppis is not None:
        save_json(run_dir / "selected_ppis.json", _model_to_json_data(selected_ppis))

    if final_report_df is not None:
        final_report_df.to_csv(run_dir / "final_report.csv", index=False)
    if selected_report_df is not None:
        selected_report_df.to_csv(run_dir / "selected_report.csv", index=False)
    if declare_table is not None:
        declare_table.to_csv(run_dir / "declare_table.csv", index=False)
    if activities_table is not None:
        activities_table.to_csv(run_dir / "activities_table.csv", index=False)

    return run_dir


def parse_common_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--xes-path",
        type=Path,
        default=Path("data/logs/DomesticDeclarations.xes"),
        help="Path to XES log.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=("openai", "anthropic", "lmstudio"),
        help="LLM provider.",
    )
    parser.add_argument("--model", type=str, required=True, help="Model identifier.")
    parser.add_argument("--base-url", type=str, default=None, help="Provider base URL.")
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=None,
        help="Optional environment variable name to read provider API key.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--json-retries", type=int, default=2)
    parser.add_argument("--use-attributes", action="store_true", help="Enable attribute-aware KPI generation.")
    return parser
