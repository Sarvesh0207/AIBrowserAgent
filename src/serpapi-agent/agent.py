"""
SerpAPI Search API — Metadata Fetch Agent (SerpAPI + LangGraph)
==============================================================
Given a website URL, queries SerpAPI (Google engine), fetches the first
organic result, and displays:

  • Title Fetched         (Yes/No)
  • Description Fetched  (Yes/No)
  • Response Time (s)
  (Scoring is left to the user.)

Usage:
    python agent.py https://www.example.com
    python agent.py                          # prompts for URL
"""

import os
import re
import sys
import time
from pathlib import Path
from typing import TypedDict, Optional, Dict, Any

import requests
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END


load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
SERPAPI_SEARCH_URL = "https://serpapi.com/search"


class AgentState(TypedDict):
    url: str
    title: str
    description: str
    title_fetched: bool
    desc_fetched: bool
    response_time: Optional[float]
    rate_limit_hit: bool
    raw_result: Dict[str, Any]        # top SerpAPI organic result
    error: str


def serpapi_search_node(state: AgentState) -> AgentState:
    print("\nQuerying SerpAPI (Google engine)...")

    if not SERPAPI_API_KEY:
        return {**state, "error": "SERPAPI_API_KEY not set in .env"}

    params = {
        "engine": "google",
        "q": state["url"],
        "num": 1,
        "api_key": SERPAPI_API_KEY,
    }

    try:
        t_start = time.perf_counter()
        response = requests.get(SERPAPI_SEARCH_URL, params=params, timeout=10)
        t_end = time.perf_counter()
        response_time = round(t_end - t_start, 3)

        if response.status_code == 429:
            return {
                **state,
                "rate_limit_hit": True,
                "response_time": response_time,
                "error": "Rate limit hit (HTTP 429)",
            }

        response.raise_for_status()
        data = response.json()
        results = data.get("organic_results", [])
        top = results[0] if results else {}

        title = top.get("title", "").strip()
        # SerpAPI Google engine uses "snippet" for description
        description = top.get("snippet", "").strip()

        return {
            **state,
            "title": title,
            "description": description,
            "title_fetched": bool(title),
            "desc_fetched": bool(description),
            "response_time": response_time,
            "raw_result": top,
        }

    except requests.exceptions.Timeout:
        return {**state, "error": "Request timed out after 10s"}
    except Exception as e:
        return {**state, "error": str(e)}


def build_agent():
    g = StateGraph(AgentState)
    g.add_node("serpapi", serpapi_search_node)
    g.set_entry_point("serpapi")
    g.add_edge("serpapi", END)
    return g.compile()


RESULTS_DIR = Path(__file__).parent / "results"


def _sanitize_url_for_filename(url: str, max_len: int = 40) -> str:
    """Turn URL into a safe filename string."""
    s = url.replace("https://", "").replace("http://", "").strip().rstrip("/")
    s = re.sub(r"[^\w\-.]", "_", s)[:max_len].strip("_") or "url"
    return s


def format_metrics(state: AgentState) -> str:
    """Build the same report text as the terminal (for saving to file); no scoring."""
    yes = "✅  Yes"
    no = "❌  No"

    lines = []
    lines.append("\n" + "═" * 62)
    lines.append("  📊  SERPAPI SEARCH API — METRICS")
    lines.append("═" * 62)
    lines.append(f"  URL               : {state['url']}")
    lines.append("─" * 62)
    lines.append(f"  Title             : {state['title'] or '(not found)'}")
    desc = state["description"]
    preview = desc[:200] + "…" if len(desc) > 200 else (desc or "(not found)")
    lines.append(f"  Description       : {preview}")
    lines.append("─" * 62)
    lines.append(f"  Title Fetched     : {yes if state['title_fetched'] else no}")
    lines.append(f"  Desc Fetched      : {yes if state['desc_fetched'] else no}")
    rt = state["response_time"]
    lines.append(f"  Response Time     : {rt}s" if rt else "  Response Time     : N/A")
    lines.append(f"  Rate Limit Hit    : {yes if state['rate_limit_hit'] else no}")
    if state.get("error"):
        lines.append(f"  ⚠  Error          : {state['error']}")
    lines.append("═" * 62 + "\n")
    return "\n".join(lines)


def print_metrics(state: AgentState, save_to_results: bool = True) -> Optional[Path]:
    """Print report to terminal and optionally save the same content under results/."""
    text = format_metrics(state)
    print(text)

    if not save_to_results:
        return None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_url = _sanitize_url_for_filename(state["url"])
    filename = f"{safe_url}.txt"
    out_path = RESULTS_DIR / filename
    out_path.write_text(text, encoding="utf-8")
    print(f"  💾  Saved to {out_path}")
    return out_path


def run(url: str):
    agent = build_agent()
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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_url = sys.argv[1]
    else:
        target_url = input("\nEnter website URL: ").strip()

    if not target_url.startswith("http"):
        target_url = "https://" + target_url

    run(target_url)

