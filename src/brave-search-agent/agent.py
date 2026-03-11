"""
Brave Search API — Metadata Fetch Agent (Brave + LangGraph only)
=================================================================
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
import requests
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

load_dotenv()

BRAVE_API_KEY    = os.getenv("BRAVE_API_KEY")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


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
    raw_result:       dict             # full top result from Brave for scoring
    result_accuracy:  int              # 1–5
    metadata_quality: int              # 1–5
    accuracy_reason:  str
    quality_reason:   str
    error:            str


# ─────────────────────────────────────────────────────────────────
# NODE 1 — Brave Search API
# ─────────────────────────────────────────────────────────────────
def brave_search_node(state: AgentState) -> AgentState:
    print("\n[1/2] Querying Brave Search API...")

    if not BRAVE_API_KEY:
        return {**state, "error": "BRAVE_API_KEY not set in .env"}

    headers = {
        "Accept":               "application/json",
        "Accept-Encoding":      "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {"q": state["url"], "count": 1}

    try:
        t_start  = time.perf_counter()
        response = requests.get(BRAVE_SEARCH_URL, headers=headers, params=params, timeout=10)
        t_end    = time.perf_counter()
        response_time = round(t_end - t_start, 3)

        if response.status_code == 429:
            return {**state, "rate_limit_hit": True, "response_time": response_time,
                    "error": "Rate limit hit (HTTP 429)"}

        response.raise_for_status()
        data    = response.json()
        results = data.get("web", {}).get("results", [])
        top     = results[0] if results else {}

        title       = top.get("title", "").strip()
        description = top.get("description", "").strip()

        return {
            **state,
            "title":         title,
            "description":   description,
            "title_fetched": bool(title),
            "desc_fetched":  bool(description),
            "response_time": response_time,
            "raw_result":    top,
        }

    except requests.exceptions.Timeout:
        return {**state, "error": "Request timed out after 10s"}
    except Exception as e:
        return {**state, "error": str(e)}


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
        accuracy_reason = "No results returned by Brave"
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
    has_description = bool(raw.get("description", "").strip())

    extra_fields = [
        raw.get("age"),
        raw.get("language"),
        raw.get("thumbnail"),
        raw.get("meta_url"),
        raw.get("extra_snippets"),
        raw.get("profile"),
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
# BUILD GRAPH  (2 nodes: brave → score → END)
# ─────────────────────────────────────────────────────────────────
def build_agent():
    g = StateGraph(AgentState)
    g.add_node("brave", brave_search_node)
    g.add_node("score", score_node)
    g.set_entry_point("brave")
    g.add_edge("brave", "score")
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
    print("  📊  BRAVE SEARCH API — METRICS")
    print("═" * 62)
    print(f"  URL               : {state['url']}")
    print("─" * 62)
    print(f"  Title             : {state['title'] or '(not found)'}")
    desc = state["description"]
    print(f"  Description       : {desc[:88] + '…' if len(desc) > 88 else desc or '(not found)'}")
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
