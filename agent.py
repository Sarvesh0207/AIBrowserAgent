"""
Brave Search API — Natural Language Agent (Brave + LangGraph + LLM)
=====================================================================
Accepts natural language instructions and performs web searches.
Supports date range filtering and multiple results.

Examples:
    python agent.py "latest AI research papers from last month"
    python agent.py "Python tutorials published in 2024"
    python agent.py "news about LangGraph between Jan and Mar 2025"
    python agent.py   ← prompts you to type an instruction

Requirements in .env:
    BRAVE_API_KEY=...
    ANTHROPIC_API_KEY=...
"""

import os
import sys
import time
import json
import requests
from datetime import datetime
from typing import TypedDict, Optional, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

load_dotenv()

BRAVE_API_KEY    = os.getenv("BRAVE_API_KEY")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# ── LLM ───────────────────────────────────────────────────────────
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)


# ─────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────
    user_instruction: str

    # ── Parsed by LLM (Node 1) ─────────────────────────────────────
    search_query:     str          # clean search query for Brave
    date_from:        str          # YYYY-MM-DD or ""
    date_to:          str          # YYYY-MM-DD or ""
    freshness:        str          # "pd" | "pw" | "pm" | "py" | ""
    result_count:     int          # how many results to fetch (1–20)
    parse_reason:     str          # LLM explanation of its parsing

    # ── Brave results (Node 2) ─────────────────────────────────────
    results:          List[dict]
    response_time:    Optional[float]
    rate_limit_hit:   bool
    error:            str


# ─────────────────────────────────────────────────────────────────
# NODE 1 — LLM parses the natural language instruction
# ─────────────────────────────────────────────────────────────────
def parse_instruction_node(state: AgentState) -> AgentState:
    print("\n[1/2] Parsing instruction with LLM...")

    today = datetime.today().strftime("%Y-%m-%d")

    system = f"""You are a search query parser. Today's date is {today}.

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
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=state["user_instruction"]),
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
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
            "date_from":    parsed.get("date_from", ""),
            "date_to":      parsed.get("date_to", ""),
            "freshness":    parsed.get("freshness", ""),
            "result_count": min(int(parsed.get("result_count", 5)), 20),
            "parse_reason": parsed.get("parse_reason", ""),
        }

    except Exception as e:
        print(f"   ⚠ LLM parse failed ({e}), using raw instruction as query")
        return {
            **state,
            "search_query": state["user_instruction"],
            "date_from": "", "date_to": "", "freshness": "",
            "result_count": 5,
            "parse_reason": f"Fallback — parse error: {e}",
        }


# ─────────────────────────────────────────────────────────────────
# NODE 2 — Brave Search API
# ─────────────────────────────────────────────────────────────────
def brave_search_node(state: AgentState) -> AgentState:
    print("\n[2/2] Querying Brave Search API...")

    if not BRAVE_API_KEY:
        return {**state, "error": "BRAVE_API_KEY not set in .env"}

    headers = {
        "Accept":               "application/json",
        "Accept-Encoding":      "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }

    params = {
        "q":     state["search_query"],
        "count": state["result_count"],
    }

    # Append explicit date range to query string
    if state["date_from"]:
        params["q"] += f" after:{state['date_from']}"
    if state["date_to"]:
        params["q"] += f" before:{state['date_to']}"

    # Use Brave's freshness param for relative ranges
    if state["freshness"] and not (state["date_from"] or state["date_to"]):
        params["freshness"] = state["freshness"]

    print(f"   Final query : {params['q']}")
    if params.get("freshness"):
        labels = {"pd": "Past day", "pw": "Past week", "pm": "Past month", "py": "Past year"}
        print(f"   Freshness   : {labels.get(params['freshness'], params['freshness'])}")

    try:
        t_start  = time.perf_counter()
        response = requests.get(BRAVE_SEARCH_URL, headers=headers, params=params, timeout=10)
        t_end    = time.perf_counter()
        response_time = round(t_end - t_start, 3)

        if response.status_code == 429:
            return {**state, "rate_limit_hit": True, "response_time": response_time,
                    "error": "Rate limit hit (HTTP 429)"}

        response.raise_for_status()
        data  = response.json()
        items = data.get("web", {}).get("results", [])

        results = []
        for item in items:
            results.append({
                "title":       item.get("title", "").strip(),
                "url":         item.get("url", "").strip(),
                "description": item.get("description", "").strip(),
                "age":         item.get("age", ""),
                "language":    item.get("language", ""),
            })

        print(f"   Got {len(results)} result(s) in {response_time}s")

        return {
            **state,
            "results":        results,
            "response_time":  response_time,
            "rate_limit_hit": False,
        }

    except requests.exceptions.Timeout:
        return {**state, "error": "Request timed out after 10s"}
    except Exception as e:
        return {**state, "error": str(e)}


# ─────────────────────────────────────────────────────────────────
# BUILD GRAPH  (parse → search → END)
# ─────────────────────────────────────────────────────────────────
def build_agent():
    g = StateGraph(AgentState)
    g.add_node("parse",  parse_instruction_node)
    g.add_node("search", brave_search_node)
    g.set_entry_point("parse")
    g.add_edge("parse",  "search")
    g.add_edge("search", END)
    return g.compile()


# ─────────────────────────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────────────────────────
def print_results(state: AgentState):
    W   = 70
    yes = "✅ Yes"; no = "❌ No"

    print("\n" + "═" * W)
    print("  📊  BRAVE SEARCH — RESULTS")
    print("═" * W)
    print(f"  Instruction  : {state['user_instruction']}")
    print(f"  Query sent   : {state['search_query']}")

    if state["date_from"] or state["date_to"]:
        df = state["date_from"] or "—"
        dt = state["date_to"]   or "—"
        print(f"  Date range   : {df}  →  {dt}")
    elif state["freshness"]:
        labels = {"pd": "Past day", "pw": "Past week", "pm": "Past month", "py": "Past year"}
        print(f"  Freshness    : {labels.get(state['freshness'], state['freshness'])}")
    else:
        print(f"  Date filter  : None")

    rt = state["response_time"]
    print(f"  Response time: {rt}s" if rt else "  Response time: N/A")
    print(f"  Rate limit   : {yes if state['rate_limit_hit'] else no}")

    if state.get("error"):
        print(f"\n  ⚠  Error: {state['error']}")
        print("═" * W + "\n")
        return

    results = state.get("results", [])
    print(f"  Results      : {len(results)} found")
    print("─" * W)

    for i, r in enumerate(results, 1):
        title = r.get("title") or "(no title)"
        url   = r.get("url", "")
        desc  = r.get("description", "") or "(no description)"
        age   = r.get("age", "")

        print(f"\n  [{i}] {title}")
        print(f"       URL  : {url}")
        print(f"       Desc : {desc[:100] + '…' if len(desc) > 100 else desc}")
        if age:
            print(f"       Date : {age}")

    print("\n" + "═" * W + "\n")


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
def run(instruction: str):
    agent = build_agent()

    initial: AgentState = {
        "user_instruction": instruction,
        "search_query":     "",
        "date_from":        "",
        "date_to":          "",
        "freshness":        "",
        "result_count":     5,
        "parse_reason":     "",
        "results":          [],
        "response_time":    None,
        "rate_limit_hit":   False,
        "error":            "",
    }

    final = agent.invoke(initial)
    print_results(final)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        instruction = " ".join(sys.argv[1:])
    else:
        print("\nExamples:")
        print('  "latest AI research papers from last month"')
        print('  "Python tutorials published in 2024"')
        print('  "LangGraph news between Jan and Mar 2025"')
        print('  "top 10 machine learning blogs from past week"\n')
        instruction = input("Enter your instruction: ").strip()

    run(instruction)
