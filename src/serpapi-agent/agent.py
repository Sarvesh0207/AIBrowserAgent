"""
SerpAPI Search API — Metadata Fetch Agent (SerpAPI + LangGraph)
===============================================================

Given a website URL, this agent:

- calls SerpAPI (Google engine) using that URL as the query
- reads the first organic result
- reports whether a title / description (snippet) was found
- measures response time and basic rate‑limit information

The file is organized into three parts to make it easier to explain:

1) Configuration & shared types        (API keys, AgentState)
2) Core SerpAPI call + LangGraph node (one search step)
3) Output / CLI helpers               (printing, saving, __main__)

Public entrypoints:

- build_agent()     → compiled LangGraph app
- print_metrics()   → pretty terminal + file output for one AgentState
- run(url: str)     → one‑shot wrapper used by CLI and batch_run.py
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph


# ────────────────────────────────────────────────────────────────────
# 1. Configuration & shared types
# ────────────────────────────────────────────────────────────────────

load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
SERPAPI_SEARCH_URL = "https://serpapi.com/search"


class AgentState(TypedDict):
    """
    Shared state that flows through the LangGraph graph.

    For this simple agent there is only a single node, but we still use a
    typed dict so the structure is explicit.
    """

    # Input
    url: str

    # Output from SerpAPI
    title: str
    description: str
    title_fetched: bool
    desc_fetched: bool
    response_time: Optional[float]
    rate_limit_hit: bool
    raw_result: Dict[str, Any]

    # Error message, empty string when everything is OK.
    error: str


RESULTS_DIR = Path(__file__).parent / "results"


def _require_api_key() -> None:
    """Raise a clear error if SERPAPI_API_KEY is missing."""
    if not SERPAPI_API_KEY:
        raise RuntimeError(
            "Missing SERPAPI_API_KEY. Put it in .env or environment variables."
        )


def _sanitize_url_for_filename(url: str, max_len: int = 40) -> str:
    """
    Turn a URL into a safe filename stem.

    Example:
        https://www.example.com/path  →  www_example_com_path
    """
    s = url.replace("https://", "").replace("http://", "").strip().rstrip("/")
    s = re.sub(r"[^\w\-.]", "_", s)[:max_len].strip("_") or "url"
    return s


# ────────────────────────────────────────────────────────────────────
# 2. Core SerpAPI call + LangGraph node
# ────────────────────────────────────────────────────────────────────

def _call_serpapi(url: str) -> tuple[Dict[str, Any], float, bool, str]:
    """
    Low‑level SerpAPI HTTP call.

    Returns:
        (top_result, response_time_seconds, rate_limit_hit, error_message)

    When error_message is a non‑empty string the caller should treat the run
    as failed, even if a partial result is present.
    """
    _require_api_key()

    params = {
        "engine": "google",
        "q": url,
        "num": 1,
        "api_key": SERPAPI_API_KEY,
    }

    try:
        t_start = time.perf_counter()
        response = requests.get(SERPAPI_SEARCH_URL, params=params, timeout=10)
        t_end = time.perf_counter()
        response_time = round(t_end - t_start, 3)

        if response.status_code == 429:
            return {}, response_time, True, "Rate limit hit (HTTP 429)"

        response.raise_for_status()
        data = response.json()
        results = data.get("organic_results", []) or []
        top: Dict[str, Any] = results[0] if results else {}
        return top, response_time, False, ""

    except requests.exceptions.Timeout:
        return {}, 10.0, False, "Request timed out after 10s"
    except Exception as exc:  # noqa: BLE001 – convert to string for the state
        return {}, 0.0, False, str(exc)


def serpapi_search_node(state: AgentState) -> AgentState:
    """
    LangGraph node: read `url` from the state, call SerpAPI once, and
    populate the rest of the fields.
    """
    print("\n[1/1] Querying SerpAPI (Google engine)...")

    top, response_time, rate_limit_hit, error = _call_serpapi(state["url"])

    if error:
        return {
            **state,
            "title": "",
            "description": "",
            "title_fetched": False,
            "desc_fetched": False,
            "response_time": response_time or None,
            "rate_limit_hit": rate_limit_hit,
            "raw_result": top,
            "error": error,
        }

    title = (top.get("title") or "").strip()
    # SerpAPI Google engine uses "snippet" for the description field.
    description = (top.get("snippet") or "").strip()

    return {
        **state,
        "title": title,
        "description": description,
        "title_fetched": bool(title),
        "desc_fetched": bool(description),
        "response_time": response_time,
        "rate_limit_hit": rate_limit_hit,
        "raw_result": top,
        "error": "",
    }


def build_agent():
    """
    Build and compile the LangGraph agent:

        AgentState ──serpapi_search_node──► END
    """
    g = StateGraph(AgentState)
    g.add_node("serpapi", serpapi_search_node)
    g.set_entry_point("serpapi")
    g.add_edge("serpapi", END)
    return g.compile()


# ────────────────────────────────────────────────────────────────────
# 3. Output / CLI helpers
# ────────────────────────────────────────────────────────────────────

def format_metrics(state: AgentState) -> str:
    """
    Convert one AgentState into a human‑readable text report.

    The same text is printed to the terminal and written to disk.
    """
    yes = "✅  Yes"
    no = "❌  No"

    lines: list[str] = []
    lines.append("\n" + "═" * 62)
    lines.append("  📊  SERPAPI SEARCH API — METRICS")
    lines.append("═" * 62)
    lines.append(f"  URL               : {state['url']}")
    lines.append("─" * 62)

    title = state.get("title") or ""
    lines.append(f"  Title             : {title or '(not found)'}")

    desc = state.get("description") or ""
    preview = desc[:200] + "…" if len(desc) > 200 else (desc or "(not found)")
    lines.append(f"  Description       : {preview}")

    lines.append("─" * 62)
    lines.append(f"  Title Fetched     : {yes if state['title_fetched'] else no}")
    lines.append(f"  Desc Fetched      : {yes if state['desc_fetched'] else no}")

    rt = state.get("response_time")
    lines.append(f"  Response Time     : {rt}s" if rt else "  Response Time     : N/A")
    lines.append(f"  Rate Limit Hit    : {yes if state['rate_limit_hit'] else no}")

    if state.get("error"):
        lines.append(f"  ⚠  Error          : {state['error']}")

    lines.append("═" * 62 + "\n")
    return "\n".join(lines)


def print_metrics(state: AgentState, save_to_results: bool = True) -> Optional[Path]:
    """
    Print metrics for one run and optionally save them under
    `results/<sanitized_url>.txt`.
    """
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


def run(url: str) -> AgentState:
    """
    One‑shot helper used by the CLI and batch_run.py.

    Builds the agent, runs a single invocation, prints and saves metrics,
    and returns the final AgentState for further processing.
    """
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
    return final


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_url = sys.argv[1].strip()
    else:
        target_url = input("\nEnter website URL: ").strip()

    if not target_url:
        print("No URL given, exiting.")
        sys.exit(1)

    if not target_url.startswith("http"):
        target_url = "https://" + target_url

    run(target_url)


