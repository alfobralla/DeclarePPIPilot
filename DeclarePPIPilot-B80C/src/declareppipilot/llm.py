import json
import logging
import os
from typing import Any

from pydantic import BaseModel, ValidationError

from .models import LLMConfig

logger = logging.getLogger(__name__)


def _extract_first_json_object(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("LLM response is empty.")
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start == -1:
        raise ValueError("LLM response does not contain a JSON object.")

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("Could not find a complete JSON object in LLM response.")


def _build_json_only_prompt(prompt: str, schema: dict[str, Any]) -> str:
    schema_text = json.dumps(schema, indent=2)
    return (
        f"{prompt}\n\n"
        "Return only a valid JSON object and nothing else.\n"
        "The JSON must follow this JSON Schema exactly:\n"
        f"{schema_text}\n"
    )


def _resolve_api_key(llm_config: LLMConfig) -> str | None:
    if llm_config.provider == "openai":
        key_env = llm_config.api_key_env or "OPENAI_API_KEY"
        key = os.getenv(key_env)
        if not key:
            raise RuntimeError(f"{key_env} is not set.")
        return key

    if llm_config.provider == "anthropic":
        key_env = llm_config.api_key_env or "ANTHROPIC_API_KEY"
        key = os.getenv(key_env)
        if not key:
            raise RuntimeError(f"{key_env} is not set.")
        return key

    if llm_config.provider == "lmstudio":
        key_env = llm_config.api_key_env
        return os.getenv(key_env) if key_env else None

    raise ValueError(f"Unsupported provider '{llm_config.provider}'.")


def _call_openai_compatible(
    *,
    llm_config: LLMConfig,
    prompt: str,
    schema: dict[str, Any],
    schema_name: str,
) -> str:
    from openai import OpenAI

    api_key = _resolve_api_key(llm_config)
    client_kwargs: dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if llm_config.base_url:
        client_kwargs["base_url"] = llm_config.base_url
    client = OpenAI(**client_kwargs)

    response_kwargs: dict[str, Any] = {
        "model": llm_config.model,
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
        "truncation": "auto",
    }
    if llm_config.temperature is not None:
        response_kwargs["temperature"] = llm_config.temperature
    if llm_config.max_output_tokens is not None:
        response_kwargs["max_output_tokens"] = llm_config.max_output_tokens

    try:
        resp = client.responses.create(**response_kwargs)
        return resp.output_text
    except Exception as exc:
        if llm_config.provider != "lmstudio":
            raise
        logger.info(
            "LM Studio strict JSON schema call failed (%s). Falling back to JSON-only prompt.",
            exc.__class__.__name__,
        )

    chat_kwargs: dict[str, Any] = {
        "model": llm_config.model,
        "messages": [{"role": "user", "content": _build_json_only_prompt(prompt, schema)}],
    }
    if llm_config.temperature is not None:
        chat_kwargs["temperature"] = llm_config.temperature
    if llm_config.max_output_tokens is not None:
        chat_kwargs["max_tokens"] = llm_config.max_output_tokens
    chat = client.chat.completions.create(**chat_kwargs)
    content = chat.choices[0].message.content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return content or ""


def _call_anthropic(
    *,
    llm_config: LLMConfig,
    prompt: str,
    schema: dict[str, Any],
) -> str:
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise ImportError("anthropic package is required for provider='anthropic'.") from exc

    api_key = _resolve_api_key(llm_config)
    client = Anthropic(api_key=api_key)
    msg_kwargs: dict[str, Any] = {
        "model": llm_config.model,
        "max_tokens": llm_config.max_output_tokens or 4096,
        "messages": [{"role": "user", "content": prompt}],#_build_json_only_prompt(prompt, schema)}],
        "output_config": {
            "format": {"type": "json_schema", "schema": schema},
        },
    }
    if llm_config.temperature is not None:
        msg_kwargs["temperature"] = llm_config.temperature

    resp = client.messages.create(**msg_kwargs)
    parts: list[str] = []
    for block in resp.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def _call_provider(
    *,
    llm_config: LLMConfig,
    prompt: str,
    schema: dict[str, Any],
    schema_name: str,
) -> str:
    if llm_config.provider in {"openai", "lmstudio"}:
        return _call_openai_compatible(
            llm_config=llm_config,
            prompt=prompt,
            schema=schema,
            schema_name=schema_name,
        )
    if llm_config.provider == "anthropic":
        return _call_anthropic(
            llm_config=llm_config,
            prompt=prompt,
            schema=schema,
        )
    raise ValueError(f"Unsupported provider '{llm_config.provider}'.")


def generate_structured_model(
    *,
    llm_config: LLMConfig,
    prompt: str,
    schema_name: str,
    model_type: type[BaseModel],
) -> BaseModel:
    retries = max(0, llm_config.json_retries)
    schema = model_type.model_json_schema()
    current_prompt = prompt
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        raw_text = _call_provider(
            llm_config=llm_config,
            prompt=current_prompt,
            schema=schema,
            schema_name=schema_name,
        )
        try:
            json_text = _extract_first_json_object(raw_text)
            return model_type.model_validate_json(json_text)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            current_prompt = (
                f"{prompt}\n\n"
                "The previous response was invalid for the schema. "
                "Return only valid JSON matching the schema exactly."
            )
            logger.warning(
                "Invalid structured response from provider=%s model=%s (attempt %d/%d): %s",
                llm_config.provider,
                llm_config.model,
                attempt + 1,
                retries + 1,
                exc,
            )

    raise ValueError(
        f"Failed to parse valid structured response after {retries + 1} attempts: {last_error}"
    )
