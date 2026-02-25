"""
Generate the Headless Evaluation document (Markdown) from a benchmark report CSV.
This fulfills the task: document which sites work / fail in headless mode.
"""
import csv
from pathlib import Path
from typing import List, Dict, Optional

from .config import REPORT_DIR


def find_latest_report(report_dir: Path) -> Optional[Path]:
    """Find the most recent headless_report_*.csv in report_dir."""
    reports = list(report_dir.glob("headless_report_*.csv"))
    if not reports:
        return None
    return max(reports, key=lambda p: p.stat().st_mtime)


def read_report(csv_path: Path) -> List[Dict[str, str]]:
    """Read report CSV and return list of row dicts."""
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def generate_markdown(rows: List[Dict[str, str]], report_name: str) -> str:
    """Turn report rows into a Markdown document."""
    passed = sum(1 for r in rows if (r.get("works_in_headless") or "").strip().lower() == "yes")
    failed = len(rows) - passed

    lines = [
        "# Headless Mode Evaluation",
        "",
        "This document summarizes which websites work or fail when the agent runs in **headless** mode (no visible browser window).",
        "",
        f"**Source report:** `{report_name}`",
        f"**Sites tested:** {len(rows)} | **Passed:** {passed} | **Failed:** {failed}",
        "",
        "---",
        "",
        "## Results Table",
        "",
        "| URL | Works in headless? | Error / Notes |",
        "|-----|--------------------|---------------|",
    ]

    for r in rows:
        url = (r.get("url") or "").strip()
        works = (r.get("works_in_headless") or "").strip()
        err = (r.get("error") or "").strip()
        # Keep error short in table; escape pipe for markdown
        err_cell = err.replace("|", "\\|")[:120] + ("..." if len(err) > 120 else "")
        lines.append(f"| {url} | {works} | {err_cell} |")

    lines.extend([
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- **Passed:** {passed} sites completed (browse + summarize) without errors.",
        f"- **Failed:** {failed} sites raised an error (e.g. timeout, blocked, HTTP/2 error).",
        "",
        "## Categorization / Findings (edit manually)",
        "",
        "You can add notes here after reviewing the table, for example:",
        "",
        "- Sites that **always work**: e.g. simple static pages, documentation.",
        "- Sites that **often fail**: e.g. e‑commerce (bot blocking), heavy JavaScript, login walls.",
        "- Common error types: `ERR_HTTP2_PROTOCOL_ERROR`, timeout, redirect loops.",
        "",
    ])

    return "\n".join(lines)


def generate_headless_doc(
    report_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> str:
    """
    Generate HEADLESS_EVALUATION.md from a report CSV.
    If report_path is None, use the latest report in REPORT_DIR.
    If out_path is None, use docs/HEADLESS_EVALUATION.md (relative to cwd).
    Returns the path to the written file.
    """
    report_path = report_path or find_latest_report(REPORT_DIR)
    if not report_path or not report_path.exists():
        raise FileNotFoundError(
            "No report CSV found. Run: python main.py benchmark --csv data/sites_to_test.csv"
        )

    rows = read_report(report_path)
    if not rows:
        raise ValueError(f"Report is empty: {report_path}")

    out_path = out_path or Path("docs") / "HEADLESS_EVALUATION.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md_content = generate_markdown(rows, report_path.name)
    out_path.write_text(md_content, encoding="utf-8")
    return str(out_path)
