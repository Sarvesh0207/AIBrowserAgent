"""
SerpAPI Search API — Natural Language Agent (SerpAPI + LangGraph + LLM)
======================================================================

This script mirrors the Brave Search natural‑language agent, but uses
SerpAPI (Google engine) as the backend instead of Brave Search.

High‑level behavior:

- **Input**: a natural‑language instruction
    e.g. "latest AI research papers from last month"
         "Python tutorials published in 2024 top 10"
         "news about LangGraph between Jan and Mar 2025"

- **Step 1 (LLM parse)**:
    Use a small Claude model to extract:
        - search_query   (clean query string)
        - date_from/to   (YYYY‑MM‑DD) or ""
        - freshness      ("pd"/"pw"/"pm"/"py") or ""
        - result_count   (1–20)

- **Step 2 (SerpAPI search)**:
    Build a SerpAPI/Google query from those fields, including a
    time‑range constraint when provided, and fetch multiple results.

- **Output**:
    Pretty printed summary with:
        - original instruction
        - final query that was sent
        - any date / freshness constraint
        - response time and basic rate‑limit info
        - a small list of top results (title + URL)

Environment variables expected in .env:

    SERPAPI_API_KEY=...
    ANTHROPIC_API_KEY=...
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agent import SERPAPI_SEARCH_URL, _require_api_key


load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Re‑use the same model as brave-search-agent for consistency.
LLM_MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
llm = ChatAnthropic(model=LLM_MODEL_NAME, temperature=0)


# ────────────────────────────────────────────────────────────────────
# State definition
# ────────────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    # Input
    user_instruction: str

    # Parsed by LLM (node 1)
    search_query: str
    date_from: str        # YYYY-MM-DD or ""
    date_to: str          # YYYY-MM-DD or ""
    freshness: str        # "pd" | "pw" | "pm" | "py" | ""
    result_count: int     # 1–20
    parse_reason: str

    # SerpAPI results (node 2)
    results: List[dict]
    response_time: Optional[float]
    rate_limit_hit: bool
    error: str


# ────────────────────────────────────────────────────────────────────
# Node 1 — LLM parses the natural language instruction
# ────────────────────────────────────────────────────────────────────


def parse_instruction_node(state: AgentState) -> AgentState:
    print("\n[1/2] Parsing instruction with LLM...")

    today = datetime.today().strftime("%Y-%m-%d")

    system = f"""You are a search query parser for Google (via SerpAPI).
Today's date is {today}.

The user gives a natural language search instruction. Extract:
1. search_query    — a clean, keyword-focused search string for a web search API
2. date_from       — start date as YYYY-MM-DD if the user specified one, else ""
3. date_to         — end date as YYYY-MM-DD if the user specified one, else ""
4. freshness       — shorthand if user says a relative time window:
                     "pd" = past day, "pw" = past week, "pm" = past month, "py" = past year
                     Leave "" if the user gave explicit dates or no time filter at all.
                     NOTE: use freshness OR date_from/date_to, never both.
5. result_count    — how many results to fetch. Default 5. Max 20.
6. parse_reason    — one sentence explaining what you interpreted

Rules:
- "last week" / "past week"         → freshness = "pw", dates blank
- "last month" / "past month"       → freshness = "pm", dates blank
- "last year" / "past year"         → freshness = "py", dates blank
- "today" / "past 24 hours"         → freshness = "pd", dates blank
- "between Jan and Mar 2025"        → date_from = "2025-01-01", date_to = "2025-03-31", freshness = ""
- "in 2024"                         → date_from = "2024-01-01", date_to = "2024-12-31", freshness = ""
- "top 10" or "give me 10"          → result_count = 10

Respond ONLY with a valid JSON object, no markdown, no extra text:
{{
  "search_query": "...",
  "date_from": "...",
  "date_to": "...",
  "freshness": "...",
  "result_count": 5,
  "parse_reason": "..."
}}"""

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=state["user_instruction"]),
            ]
        )
        raw = response.content.strip()
        # Be forgiving if the model wrapped JSON in ``` fences.
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
                if raw.lstrip().startswith("json"):
                    raw = raw.lstrip()[4:]
        parsed = json.loads(raw.strip())

        print(f"   Query     : {parsed.get('search_query')}")
        print(f"   Date from : {parsed.get('date_from') or '(none)'}")
        print(f"   Date to   : {parsed.get('date_to') or '(none)'}")
        print(f"   Freshness : {parsed.get('freshness') or '(none)'}")
        print(f"   Count     : {parsed.get('result_count')}")
        print(f"   Reason    : {parsed.get('parse_reason')}")

        return {
            **state,
            "search_query": parsed.get("search_query", state["user_instruction"]),
            "date_from": parsed.get("date_from", ""),
            "date_to": parsed.get("date_to", ""),
            "freshness": parsed.get("freshness", ""),
            "result_count": min(int(parsed.get("result_count", 5)), 20),
            "parse_reason": parsed.get("parse_reason", ""),
        }

    except Exception as exc:  # noqa: BLE001 – fall back gracefully
        print(f"   ⚠ LLM parse failed ({exc}), using raw instruction as query")
        return {
            **state,
            "search_query": state["user_instruction"],
            "date_from": "",
            "date_to": "",
            "freshness": "",
            "result_count": 5,
            "parse_reason": f"Fallback — parse error: {exc}",
        }


# ────────────────────────────────────────────────────────────────────
# Node 2 — SerpAPI search
# ────────────────────────────────────────────────────────────────────


def serpapi_search_node(state: AgentState) -> AgentState:
    print("\n[2/2] Querying SerpAPI (Google engine)...")

    _require_api_key()
    if not SERPAPI_API_KEY:
        return {**state, "error": "SERPAPI_API_KEY not set in .env"}

    # Base query from the parsed instruction (without time filters).
    base_query = state["search_query"]

    def _build_params(with_time_filters: bool) -> dict:
        query = base_query
        if with_time_filters:
            # Encode explicit date range directly into the Google query string.
            # Google understands things like:
            #   after:2024-01-01 before:2024-03-31
            if state["date_from"]:
                query += f" after:{state['date_from']}"
            if state["date_to"]:
                query += f" before:{state['date_to']}"

        params: dict = {
            "engine": "google",
            "q": query,
            "num": state["result_count"],
            "api_key": SERPAPI_API_KEY,
        }

        # For relative freshness, we can optionally map to Google's "qdr"
        # shortcut via the tbs parameter. This is best-effort.
        freshness = state.get("freshness")
        if with_time_filters and freshness and not (state["date_from"] or state["date_to"]):
            mapping = {"pd": "d", "pw": "w", "pm": "m", "py": "y"}
            suffix = mapping.get(freshness)
            if suffix:
                params["tbs"] = f"qdr:{suffix}"
        return params

    freshness = state.get("freshness")
    params = _build_params(with_time_filters=True)

    print(f"   Final query : {params['q']}")
    if freshness:
        labels = {"pd": "Past day", "pw": "Past week", "pm": "Past month", "py": "Past year"}
        print(f"   Freshness   : {labels.get(freshness, freshness)}")

    try:
        # First attempt: with time filters
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
        items = data.get("organic_results", []) or []

        # Fallback: if time-filtered search returns no organic_results,
        # retry once without any time constraint so the demo still shows results.
        if not items:
            print("   No organic results with time filter; retrying without time constraint...")
            fallback_params = _build_params(with_time_filters=False)
            t_start2 = time.perf_counter()
            response2 = requests.get(SERPAPI_SEARCH_URL, params=fallback_params, timeout=10)
            t_end2 = time.perf_counter()
            response2.raise_for_status()
            data = response2.json()
            items = data.get("organic_results", []) or []
            response_time = round(t_end2 - t_start2, 3)

        results: List[dict] = []
        for item in items:
            results.append(
                {
                    "title": (item.get("title") or "").strip(),
                    "url": (item.get("link") or item.get("formattedUrl") or "").strip(),
                    "description": (item.get("snippet") or "").strip(),
                }
            )

        print(f"   Got {len(results)} result(s) in {response_time}s")

        return {
            **state,
            "results": results,
            "response_time": response_time,
            "rate_limit_hit": False,
        }

    except requests.exceptions.Timeout:
        return {**state, "error": "Request timed out after 10s"}
    except Exception as exc:  # noqa: BLE001
        return {**state, "error": str(exc)}


# ────────────────────────────────────────────────────────────────────
# Build graph (parse → search → END)
# ────────────────────────────────────────────────────────────────────


def build_agent():
    g = StateGraph(AgentState)
    g.add_node("parse", parse_instruction_node)
    g.add_node("search", serpapi_search_node)
    g.set_entry_point("parse")
    g.add_edge("parse", "search")
    g.add_edge("search", END)
    return g.compile()


# ────────────────────────────────────────────────────────────────────
# Pretty printing
# ────────────────────────────────────────────────────────────────────


def print_results(state: AgentState) -> None:
    width = 70
    yes = "✅ Yes"
    no = "❌ No"

    print("\n" + "═" * width)
    print("  📊  SERPAPI (GOOGLE) — RESULTS")
    print("═" * width)
    print(f"  Instruction  : {state['user_instruction']}")
    print(f"  Query sent   : {state['search_query']}")

    if state["date_from"] or state["date_to"]:
        df = state["date_from"] or "—"
        dt = state["date_to"] or "—"
        print(f"  Date range   : {df}  →  {dt}")
    elif state["freshness"]:
        labels = {"pd": "Past day", "pw": "Past week", "pm": "Past month", "py": "Past year"}
        print(f"  Freshness    : {labels.get(state['freshness'], state['freshness'])}")
    else:
        print("  Date filter  : None")

    rt = state["response_time"]
    print(f"  Response time: {rt}s" if rt else "  Response time: N/A")
    print(f"  Rate limit   : {yes if state['rate_limit_hit'] else no}")

    if state.get("error"):
        print(f"\n  ⚠  Error: {state['error']}")
        print("═" * width + "\n")
        return

    results = state.get("results", [])
    print(f"  Results      : {len(results)} found")
    print("─" * width)

    for i, r in enumerate(results, 1):
        title = r.get("title") or "(no title)"
        url = r.get("url", "")
        desc = r.get("description", "") or "(no description)"

        print(f"\n  [{i}] {title}")
        if url:
            print(f"       URL  : {url}")
        print(f"       Desc : {desc[:100] + '…' if len(desc) > 100 else desc}")

    print("\n" + "═" * width + "\n")


def search_top_url(instruction: str) -> Optional[str]:
    """
    Helper for other agents: run the same graph as `run`, but return
    the first result URL (if any) instead of printing everything.
    """
    app = build_agent()
    initial: AgentState = {
        "user_instruction": instruction,
        "search_query": "",
        "date_from": "",
        "date_to": "",
        "freshness": "",
        "result_count": 5,
        "parse_reason": "",
        "results": [],
        "response_time": None,
        "rate_limit_hit": False,
        "error": "",
    }
    final = app.invoke(initial)
    results = final.get("results", [])
    if not results:
        return None
    url = (results[0].get("url") or "").strip()
    return url or None


# ────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ────────────────────────────────────────────────────────────────────


def run(instruction: str) -> None:
    app = build_agent()
    initial: AgentState = {
        "user_instruction": instruction,
        "search_query": "",
        "date_from": "",
        "date_to": "",
        "freshness": "",
        "result_count": 5,
        "parse_reason": "",
        "results": [],
        "response_time": None,
        "rate_limit_hit": False,
        "error": "",
    }
    final = app.invoke(initial)
    print_results(final)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        instruction = " ".join(sys.argv[1:]).strip()
    else:
        print("\nExamples:")
        print('  "latest AI research papers from last month"')
        print('  "Python tutorials published in 2024"')
        print('  "LangGraph news between Jan and Mar 2025"')
        print('  "top 10 machine learning blogs from past week"\n')
        instruction = input("Enter your instruction: ").strip()

    if not instruction:
        print("No instruction given, exiting.")
        sys.exit(1)

    run(instruction)

