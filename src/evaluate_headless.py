import csv
from pathlib import Path
from typing import List, Dict

from .agent_graph import build_graph
from .config import REPORT_DIR, ensure_dirs
from .logger import jsonl_path, append_jsonl, utc_ts


async def run_headless_evaluation(csv_path: str) -> str:
    """
    Input CSV: must have a column named 'url'
    Output: report csv path
    """
    ensure_dirs()
    app = build_graph()

    in_path = Path(csv_path)
    if not in_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # read urls
    urls: List[str] = []
    with in_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "url" not in reader.fieldnames:
            raise ValueError("CSV must include a column named 'url'")
        for row in reader:
            u = (row.get("url") or "").strip()
            if u:
                urls.append(u)

    bench_log = jsonl_path("benchmark")
    rows: List[Dict] = []

    for url in urls:
        try:
            out = await app.ainvoke({"url": url})
            rows.append({
                "url": url,
                "final_url": out.get("final_url", ""),
                "title": out.get("title", ""),
                "description_len": len(out.get("description", "") or ""),
                "works_in_headless": "Yes",
                "error": "",
                "screenshot_path": out.get("screenshot_path", ""),
            })
            append_jsonl(bench_log, {"ts": utc_ts(), "url": url, "status": "ok"})
        except Exception as e:
            rows.append({
                "url": url,
                "final_url": "",
                "title": "",
                "description_len": 0,
                "works_in_headless": "No",
                "error": repr(e),
                "screenshot_path": "",
            })
            append_jsonl(bench_log, {"ts": utc_ts(), "url": url, "status": "fail", "error": repr(e)})

    out_csv = REPORT_DIR / f"headless_report_{utc_ts()}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["url"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return str(out_csv)
