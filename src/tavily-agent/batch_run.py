import csv
import sys
from pathlib import Path
from typing import List

from nl_agent import search_with_instruction


PROJECT_ROOT = Path(__file__).parent
URLS_FILE = PROJECT_ROOT / "urls.txt"
OUTPUT_CSV = PROJECT_ROOT / "results.csv"


def load_urls(path: Path) -> List[str]:
    urls: List[str] = []
    if not path.exists():
        return urls

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def main():
    urls = load_urls(URLS_FILE)
    if not urls:
        print(f"No URLs found in {URLS_FILE}.")
        print("Add one instruction/query per line.")
        sys.exit(1)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "instruction",
                "query",
                "search_depth",
                "max_results",
                "start_date",
                "end_date",
                "response_time_s",
                "result_count",
            ]
        )

        for instruction in urls:
            print(f"\n=== Running Tavily search for: {instruction} ===")
            payload = search_with_instruction(instruction)
            parsed = payload["parsed"]
            writer.writerow(
                [
                    payload["instruction"],
                    parsed["query"],
                    parsed["search_depth"],
                    parsed["max_results"],
                    parsed["start_date"] or "",
                    parsed["end_date"] or "",
                    payload["response_time_s"],
                    payload["result_count"],
                ]
            )

    print(f"\nSaved CSV results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


