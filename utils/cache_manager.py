import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def clear_runtime_cache() -> list[str]:
    removed: list[str] = []

    for pycache_dir in ROOT.rglob("__pycache__"):
        if pycache_dir.is_dir():
            shutil.rmtree(pycache_dir, ignore_errors=True)
            removed.append(str(pycache_dir.relative_to(ROOT)))

    reports_dir = ROOT / "reports"
    if reports_dir.exists():
        for report_dir in reports_dir.iterdir():
            if report_dir.is_dir() and report_dir.name != "test-case":
                shutil.rmtree(report_dir, ignore_errors=True)
                removed.append(str(report_dir.relative_to(ROOT)))

    for pdf_path in Path("/tmp").glob("*.pdf"):
        try:
            pdf_path.unlink()
            removed.append(str(pdf_path))
        except OSError:
            continue

    return removed
