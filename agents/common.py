from __future__ import annotations

import json

from agents.llm_client import call_llm_json
from chunker.token_chunker import chunk_text
from prompt_loader import load_prompt


def _local_summary(chunks: list[str], per_chunk_chars: int = 1000) -> str:
    trimmed = [chunk[:per_chunk_chars].strip() for chunk in chunks[:3] if chunk.strip()]
    return "\n\n".join(trimmed)


def summarize_chunks(label: str, text: str) -> str:
    chunks = chunk_text(text)
    if len(chunks) == 1:
        return text

    stitched_text = "\n\n".join(
        f"[Chunk {index}/{len(chunks)}]\n{chunk[:2500].strip()}"
        for index, chunk in enumerate(chunks[:4], start=1)
    )
    synthesis_prompt = load_prompt(
        "section_summary.txt",
        schema_json='{"summary": "string"}',
        label=label,
        chunk_excerpts=json.dumps(stitched_text),
    )
    payload = call_llm_json(synthesis_prompt, fallback={"summary": _local_summary(chunks)})
    return str(payload.get("summary", "")).strip()
