import json
import os
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI


MODEL_NAME = "gemini-3.1-flash-lite-preview"
_LLM: ChatGoogleGenerativeAI | None = None
_quota_exhausted: bool = False
_quota_error_message: str = ""


def is_quota_exhausted() -> bool:
    return _quota_exhausted


def get_quota_error_message() -> str:
    return _quota_error_message


def reset_quota_flag() -> None:
    global _quota_exhausted, _quota_error_message
    _quota_exhausted = False
    _quota_error_message = ""


def _get_llm() -> ChatGoogleGenerativeAI:
    global _LLM
    if _LLM is not None:
        return _LLM

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your environment or .env file.")

    _LLM = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0.2,
    )
    return _LLM


def _is_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "quota exceeded" in message
        or "resource_exhausted" in message
        or "429" in message
        or "rate limit" in message
        or "too many requests" in message
    )


def call_llm(prompt: str) -> str:
    global _quota_exhausted, _quota_error_message

    try:
        response = _get_llm().invoke(prompt)
    except Exception as exc:
        if _is_quota_error(exc):
            _quota_exhausted = True
            _quota_error_message = f"Gemini API quota exhausted. Details: {exc}"
        raise

    text = getattr(response, "content", "")
    if isinstance(text, list):
        text = "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in text
        ).strip()
    if not text:
        raise ValueError("LLM returned an empty response")
    return str(text)


def safe_parse_json(text: str) -> dict[str, Any] | list[Any] | None:
    clean = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(clean[start : end + 1])
            except json.JSONDecodeError:
                return None
        start = clean.find("[")
        end = clean.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(clean[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _repair_json_response(text: str) -> dict[str, Any] | list[Any] | None:
    repair_prompt = (
        "Convert the following model output into strict valid JSON.\n"
        "Return JSON only, with no markdown fences or commentary.\n"
        "Preserve the original structure and values as closely as possible.\n\n"
        f"Model output:\n{text}"
    )
    repaired = call_llm(repair_prompt)
    return safe_parse_json(repaired)


def call_llm_json(prompt: str) -> dict[str, Any]:
    text = call_llm(prompt)
    parsed = safe_parse_json(text)
    if parsed is None:
        parsed = _repair_json_response(text)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"items": parsed}
    raise ValueError("LLM response could not be parsed as JSON")


def _value_is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _validate_payload(
    payload: dict[str, Any],
    required_fields: list[str] | None = None,
    nonempty_fields: list[str] | None = None,
    enum_fields: dict[str, set[str]] | None = None,
) -> list[str]:
    problems: list[str] = []
    required_fields = required_fields or []
    nonempty_fields = nonempty_fields or []
    enum_fields = enum_fields or {}

    for field in required_fields:
        if field not in payload:
            problems.append(f"Missing required field: {field}")
    for field in nonempty_fields:
        if _value_is_missing(payload.get(field)):
            problems.append(f"Empty required field: {field}")
    for field, allowed in enum_fields.items():
        value = payload.get(field)
        if _value_is_missing(value):
            problems.append(f"Empty enum field: {field}")
            continue
        if str(value).strip() not in allowed:
            problems.append(f"Invalid enum value for {field}: {value!r}")
    return problems


def _repair_structured_payload(
    original_prompt: str,
    invalid_payload: dict[str, Any],
    problems: list[str],
) -> dict[str, Any]:
    repair_prompt = (
        "The previous JSON response did not satisfy the required schema.\n"
        "Return one corrected JSON object only, with no markdown and no commentary.\n"
        "Preserve the intended meaning of the previous response while fixing the schema issues.\n\n"
        f"Original prompt:\n{original_prompt}\n\n"
        f"Previous JSON:\n{json.dumps(invalid_payload, indent=2)}\n\n"
        f"Schema issues:\n{json.dumps(problems, indent=2)}"
    )
    return call_llm_json(repair_prompt)


def call_llm_json_checked(
    prompt: str,
    required_fields: list[str] | None = None,
    nonempty_fields: list[str] | None = None,
    enum_fields: dict[str, set[str]] | None = None,
    max_repairs: int = 2,
) -> dict[str, Any]:
    payload = call_llm_json(prompt)
    problems = _validate_payload(payload, required_fields, nonempty_fields, enum_fields)
    repairs = 0
    while problems and repairs < max_repairs:
        payload = _repair_structured_payload(prompt, payload, problems)
        problems = _validate_payload(payload, required_fields, nonempty_fields, enum_fields)
        repairs += 1
    if problems:
        joined = "; ".join(problems)
        raise ValueError(f"LLM JSON failed schema validation: {joined}")
    return payload
