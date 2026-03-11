import csv
import sys
from pathlib import Path
from typing import List

from agent import build_agent, AgentState, print_metrics


PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
URLS_FILE = PROJECT_ROOT / "urls.txt"
OUTPUT_CSV = RESULTS_DIR / "results.csv"


def load_urls(path: Path) -> List[str]:
    urls: List[str] = []
    if not path.exists():
        return urls

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("http"):
            line = "https://" + line
        urls.append(line)
    return urls


def run_once(agent, url: str) -> AgentState:
    initial: AgentState = {
        "url": url,
        "title": "",
        "description": "",
        "title_fetched": False,
        "desc_fetched": False,
        "response_time": None,
        "rate_limit_hit": False,
        "raw_result": {},
        "error": "",
    }
    final = agent.invoke(initial)
    print_metrics(final)
    return final


def main():
    urls = load_urls(URLS_FILE)
    if not urls:
        print(f"No URLs found in {URLS_FILE}.")
        print("Add one URL per line (you can omit 'https://').")
        sys.exit(1)

    agent = build_agent()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "url", "title", "description",
            "title_fetched", "description_fetched", "response_time_s",
        ])

        for url in urls:
            print(f"\n=== Running SerpAPI agent for: {url} ===")
            state = run_once(agent, url)
            writer.writerow([
                state["url"],
                state["title"],
                state["description"],
                "Yes" if state["title_fetched"] else "No",
                "Yes" if state["desc_fetched"] else "No",
                state["response_time"] or "",
            ])

    print(f"\nSaved CSV results to {OUTPUT_CSV}")
    print(f"Text reports for each URL are in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

