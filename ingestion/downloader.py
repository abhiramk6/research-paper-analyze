from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import requests


ARXIV_ID_PATTERN = re.compile(r"(?P<paper_id>\d{4}\.\d{4,5})(v\d+)?$")


def is_probable_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return bool(parsed.scheme and parsed.netloc)


def is_local_pdf(value: str) -> bool:
    path = Path(value).expanduser()
    return path.exists() and path.is_file() and path.suffix.lower() == ".pdf"


def extract_paper_id(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.netloc not in {"arxiv.org", "www.arxiv.org"}:
        raise ValueError("Expected an arXiv URL like https://arxiv.org/abs/1706.03762")

    path = parsed.path.strip("/")
    if path.startswith("abs/"):
        candidate = path.removeprefix("abs/")
    elif path.startswith("pdf/"):
        candidate = path.removeprefix("pdf/").removesuffix(".pdf")
    else:
        raise ValueError("arXiv URL must use /abs/ or /pdf/ format")

    match = ARXIV_ID_PATTERN.search(candidate)
    if not match:
        raise ValueError("Only standard modern arXiv IDs are supported in this build")
    return match.group("paper_id")


def build_pdf_url(url: str) -> tuple[str, str]:
    paper_id = extract_paper_id(url)
    return f"https://arxiv.org/pdf/{paper_id}.pdf", paper_id


def download_pdf(url: str, output_dir: str | Path = "/tmp") -> Path:
    pdf_url, paper_id = build_pdf_url(url)
    response = requests.get(pdf_url, timeout=60)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
        raise ValueError("Downloaded file was not a PDF")

    output_path = Path(output_dir) / f"{paper_id}.pdf"
    output_path.write_bytes(response.content)
    return output_path


def resolve_pdf_input(source: str, output_dir: str | Path = "/tmp") -> tuple[Path, str]:
    if is_local_pdf(source):
        path = Path(source).expanduser().resolve()
        return path, path.as_posix()
    if is_probable_url(source):
        path = download_pdf(source, output_dir=output_dir)
        return path, source.strip()
    raise ValueError("Input must be an arXiv URL or a local PDF path.")
