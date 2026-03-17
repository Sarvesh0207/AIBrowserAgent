"""
Batch headless evaluation for the browser-based SerpAPI agent.

Given a text file with one URL per line (or a simple CSV with a `url`
column), this script will:

- open each URL with Playwright (respecting HEADLESS from config)
- collect title / description length / screenshot path
- record whether the run succeeded in headless mode
- write a CSV report under outputs/reports
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from browser_agent import build_graph
from config import REPORT_DIR, ensure_dirs


def _load_urls(path: Path) -> List[str]:
    urls: List[str] = []
    if not path.exists():
        return urls
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "url" not in (reader.fieldnames or []):
                return urls
            for row in reader:
                u = (row.get("url") or "").strip()
                if not u:
                    continue
                if not u.startswith("http"):
                    u = "https://" + u
                urls.append(u)
        return urls
    # Fallback: plain text, one URL per line
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("http"):
            line = "https://" + line
        urls.append(line)
    return urls


def run_headless_evaluation(path: str) -> str:
    ensure_dirs()
    in_path = Path(path)
    urls = _load_urls(in_path)
    if not urls:
        raise FileNotFoundError(f"No URLs found in {in_path}")

    app = build_graph()
    rows: List[Dict[str, str]] = []

    for url in urls:
        try:
            final = app.invoke({"url": url})  # type: ignore[arg-type]
            title = (final.get("title") or "").strip()
            desc = (final.get("description") or "").strip()
            rows.append(
                {
                    "url": url,
                    "final_url": (final.get("final_url") or "").strip(),
                    "title": title,
                    "description_len": str(len(desc)),
                    "works_in_headless": "Yes",
                    "error": "",
                    "screenshot_path": final.get("screenshot_path") or "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "url": url,
                    "final_url": "",
                    "title": "",
                    "description_len": "0",
                    "works_in_headless": "No",
                    "error": repr(exc),
                    "screenshot_path": "",
                }
            )

    ensure_dirs()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORT_DIR / "headless_report.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "url",
                "final_url",
                "title",
                "description_len",
                "works_in_headless",
                "error",
                "screenshot_path",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return str(out_csv)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python headless_eval.py urls.txt")
        raise SystemExit(1)

    report_path = run_headless_evaluation(sys.argv[1])
    print(f"\nHeadless report written to {report_path}")

