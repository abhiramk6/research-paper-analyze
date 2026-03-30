from chunker.token_chunker import rolling_window_text, truncate_to_token_limit


def summarize_chunks(label: str, text: str, max_tokens: int = 1_600) -> str:
    del label
    windows = rolling_window_text(text, max_tokens=max_tokens, overlap_tokens=150)
    if not windows:
        return ""
    combined = "\n\n".join(windows[:2]).strip()
    return truncate_to_token_limit(combined, max_tokens)
