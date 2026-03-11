"""
Tavily Search API — Metadata Fetch Agent (Tavily + LangGraph only)
==================================================================
Give it a website URL → it fetches and auto-scores:
  • Title Fetched         (Yes/No)
  • Description Fetched   (Yes/No)
  • Response Time (s)
  • Result Accuracy       (1–5, auto-scored)
  • Metadata Quality      (1–5, auto-scored)
  • Rate Limit Hit        (Yes/No)

Scoring Guide:
  Result Accuracy   1 = off-topic/wrong  |  3 = partially relevant  |  5 = perfectly on-point
  Metadata Quality  1 = URL only         |  3 = URL+title+snippet   |  5 = full metadata

Run:
    python agent.py https://www.example.com
    python agent.py                          ← prompts you to enter a URL
"""

import os
import sys
import time
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ─────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    url:              str
    title:            str
    description:      str
    title_fetched:    bool
    desc_fetched:     bool
    response_time:    Optional[float]
    rate_limit_hit:   bool
    raw_result:       dict             # full top result from Tavily for scoring
    result_accuracy:  int              # 1–5
    metadata_quality: int              # 1–5
    accuracy_reason:  str
    quality_reason:   str
    error:            str


# ─────────────────────────────────────────────────────────────────
# NODE 1 — Tavily Search API
# ─────────────────────────────────────────────────────────────────
def tavily_search_node(state: AgentState) -> AgentState:
    print("\n[1/2] Querying Tavily Search API...")

    if not TAVILY_API_KEY:
        return {**state, "error": "TAVILY_API_KEY not set in .env"}

    client = TavilyClient(api_key=TAVILY_API_KEY)

    try:
        t_start = time.perf_counter()
        # Use the URL as the query and keep only the top result.
        response = client.search(
            query=state["url"],
            max_results=1,
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
        )
        t_end = time.perf_counter()
        response_time = round(t_end - t_start, 3)

        results = response.get("results", []) if isinstance(response, dict) else []
        top = results[0] if results else {}

        title = top.get("title", "").strip()
        # Tavily uses "content" for the snippet/description.
        description = top.get("content", "").strip()

        return {
            **state,
            "title":         title,
            "description":   description,
            "title_fetched": bool(title),
            "desc_fetched":  bool(description),
            "response_time": response_time,
            "raw_result":    top,
        }

    except Exception as e:
        msg = str(e)
        rate_limited = "rate limit" in msg.lower()
        return {
            **state,
            "error":          msg,
            "rate_limit_hit": rate_limited,
        }


# ─────────────────────────────────────────────────────────────────
# NODE 2 — Auto-scorer
#
# Result Accuracy  — does the returned result actually match the URL?
#   1  No results at all, or result is completely unrelated
#   2  Different domain but some metadata present
#   3  Same domain but different page returned
#   4  Very close URL match (subdomain or path difference)
#   5  Exact URL match
#
# Metadata Quality — how rich is the metadata returned?
#   1  Only a URL returned (no title, no description)
#   2  URL + title only
#   3  URL + title + snippet/description
#   4  Above + extra fields (date, language, thumbnail, etc.)
#   5  Full metadata (date, author, language, structured schema, etc.)
# ─────────────────────────────────────────────────────────────────
def score_node(state: AgentState) -> AgentState:
    print("[2/2] Auto-scoring results...")

    raw        = state.get("raw_result", {})
    input_url  = state["url"].rstrip("/").lower()
    result_url = raw.get("url", "").rstrip("/").lower()

    # ── Result Accuracy ──────────────────────────────────────────
    if not raw:
        accuracy        = 1
        accuracy_reason = "No results returned by Tavily"
    elif result_url == input_url:
        accuracy        = 5
        accuracy_reason = "Exact URL match returned"
    elif input_url in result_url or result_url in input_url:
        accuracy        = 4
        accuracy_reason = "Very close URL match (subdomain or path difference)"
    elif _domain(input_url) == _domain(result_url):
        accuracy        = 3
        accuracy_reason = "Same domain but different page returned"
    elif state["title_fetched"] or state["desc_fetched"]:
        accuracy        = 2
        accuracy_reason = "Different domain returned but some metadata present"
    else:
        accuracy        = 1
        accuracy_reason = "Completely off-topic result"

    # ── Metadata Quality ─────────────────────────────────────────
    has_url         = bool(result_url)
    has_title       = bool(raw.get("title", "").strip())
    # Tavily uses "content" instead of "description".
    has_description = bool(raw.get("content", "").strip())

    # Heuristic extra metadata fields commonly present in Tavily results.
    extra_fields = [
        raw.get("score"),
        raw.get("raw_content"),
        raw.get("published_date"),
        raw.get("author"),
        raw.get("images"),
    ]
    extra_count = sum(1 for f in extra_fields if f)

    if not has_url and not has_title and not has_description:
        quality        = 1
        quality_reason = "Only URL returned (no title or description)"
    elif has_url and has_title and not has_description:
        quality        = 2
        quality_reason = "URL + title returned, no description"
    elif has_url and has_title and has_description and extra_count == 0:
        quality        = 3
        quality_reason = "URL + title + description returned"
    elif has_url and has_title and has_description and 0 < extra_count < 3:
        quality        = 4
        quality_reason = f"URL + title + description + {extra_count} extra field(s) (e.g. date, language, thumbnail)"
    else:
        quality        = 5
        quality_reason = "Full metadata including date, language, structured schema"

    return {
        **state,
        "result_accuracy":  accuracy,
        "metadata_quality": quality,
        "accuracy_reason":  accuracy_reason,
        "quality_reason":   quality_reason,
    }


def _domain(url: str) -> str:
    url = url.replace("https://", "").replace("http://", "").replace("www.", "")
    return url.split("/")[0]


# ─────────────────────────────────────────────────────────────────
# BUILD GRAPH  (2 nodes: tavily → score → END)
# ─────────────────────────────────────────────────────────────────
def build_agent():
    g = StateGraph(AgentState)
    g.add_node("tavily", tavily_search_node)
    g.add_node("score", score_node)
    g.set_entry_point("tavily")
    g.add_edge("tavily", "score")
    g.add_edge("score", END)
    return g.compile()


# ─────────────────────────────────────────────────────────────────
# PRINT METRICS
# ─────────────────────────────────────────────────────────────────
def print_metrics(state: AgentState):
    yes = "✅  Yes"
    no  = "❌  No"
    def stars(n): return "★" * n + "☆" * (5 - n)

    print("\n" + "═" * 62)
    print("  📊  TAVILY SEARCH API — METRICS")
    print("═" * 62)
    print(f"  URL               : {state['url']}")
    print("─" * 62)
    print(f"  Title             : {state['title'] or '(not found)'}")
    desc = state["description"]
    # Show up to ~200 words of description in the console output.
    if desc:
        words = desc.split()
        if len(words) > 200:
            desc_preview = " ".join(words[:200]) + "…"
        else:
            desc_preview = desc
    else:
        desc_preview = "(not found)"
    print(f"  Description       : {desc_preview}")
    print("─" * 62)
    print(f"  Title Fetched     : {yes if state['title_fetched'] else no}")
    print(f"  Desc Fetched      : {yes if state['desc_fetched']  else no}")
    rt = state["response_time"]
    print(f"  Response Time     : {rt}s" if rt else "  Response Time     : N/A")
    print(f"  Rate Limit Hit    : {yes if state['rate_limit_hit'] else no}")
    print("─" * 62)
    acc = state["result_accuracy"]
    qua = state["metadata_quality"]
    print(f"  Result Accuracy   : {acc}/5  {stars(acc)}")
    print(f"    → {state['accuracy_reason']}")
    print(f"  Metadata Quality  : {qua}/5  {stars(qua)}")
    print(f"    → {state['quality_reason']}")
    if state.get("error"):
        print(f"  ⚠  Error          : {state['error']}")
    print("═" * 62)
    print()
    print("  📝  Copy into spreadsheet:")
    print(f"      Title Fetched       → {'Yes' if state['title_fetched'] else 'No'}")
    print(f"      Description Fetched → {'Yes' if state['desc_fetched']  else 'No'}")
    print(f"      Response Time (s)   → {rt or 'N/A'}")
    print(f"      Result Accuracy     → {acc}/5")
    print(f"      Metadata Quality    → {qua}/5")
    print(f"      Rate Limit Hit      → {'Yes' if state['rate_limit_hit'] else 'No'}")
    print("═" * 62 + "\n")


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
def run(url: str):
    agent = build_agent()

    initial: AgentState = {
        "url":              url,
        "title":            "",
        "description":      "",
        "title_fetched":    False,
        "desc_fetched":     False,
        "response_time":    None,
        "rate_limit_hit":   False,
        "raw_result":       {},
        "result_accuracy":  0,
        "metadata_quality": 0,
        "accuracy_reason":  "",
        "quality_reason":   "",
        "error":            "",
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
