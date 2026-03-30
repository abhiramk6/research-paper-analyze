from __future__ import annotations

import json
import os
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI


# MODEL_NAME = "gemini-2.5-flash-lite"
MODEL_NAME = "gemini-3.1-flash-lite-preview"
_LLM: ChatGoogleGenerativeAI | None = None

# Module-level quota exhaustion flag — set when any call hits a 429 / RESOURCE_EXHAUSTED.
# Readable by the Streamlit app to surface a visible warning.
_quota_exhausted: bool = False
_quota_error_message: str = ""


def is_quota_exhausted() -> bool:
    return _quota_exhausted


def get_quota_error_message() -> str:
    return _quota_error_message


def reset_quota_flag() -> None:
    """Call this at the start of a new pipeline run to clear stale flags."""
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
    msg = str(exc).lower()
    return (
        "quota exceeded" in msg
        or "resource_exhausted" in msg
        or "429" in msg
        or "rate limit" in msg
        or "too many requests" in msg
    )


def call_llm(prompt: str) -> str:
    global _quota_exhausted, _quota_error_message

    try:
        response = _get_llm().invoke(prompt)
    except Exception as exc:
        if _is_quota_error(exc):
            _quota_exhausted = True
            _quota_error_message = (
                "Gemini API quota exhausted. "
                "Free-tier limit reached — results are falling back to heuristics. "
                f"Details: {exc}"
            )
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


def _summarize_fallback_reason(reason: str) -> str:
    lower = reason.lower()
    if "quota exceeded" in lower or "resource_exhausted" in lower or "429" in lower or "rate limit" in lower:
        return "⚠️ Gemini free-tier quota is currently exhausted — heuristic fallback used."
    if "empty response" in lower:
        return "The LLM returned an empty response."
    if "json" in lower:
        return "The LLM returned output that was not valid JSON."
    return "The LLM could not provide a usable structured response."


def fallback_payload(fallback: dict[str, Any], reason: str) -> dict[str, Any]:
    payload = dict(fallback)
    summary = _summarize_fallback_reason(reason)
    if "reasoning" in payload:
        payload["reasoning"] = f"{payload['reasoning']} {summary}".strip()
    if "summary" in payload:
        payload["summary"] = f"{payload['summary']} {summary}".strip()
    return payload


def call_llm_json(prompt: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        text = call_llm(prompt)
    except Exception as exc:
        if fallback is not None:
            return fallback_payload(fallback, f"Fallback used because LLM request failed: {exc}")
        raise

    parsed = safe_parse_json(text)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"items": parsed}
    if fallback is not None:
        return fallback_payload(fallback, "Fallback used because the LLM returned JSON that could not be parsed cleanly.")
    raise ValueError("LLM response could not be parsed as JSON")
