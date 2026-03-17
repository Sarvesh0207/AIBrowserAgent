from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from firecrawl import Firecrawl
from langgraph.graph import END, StateGraph

from live_browser import LiveBrowserConfig, LiveBrowserSession, run_live_plan

try:
    from llm_claude import ClaudeJSON
except Exception:  # pragma: no cover
    ClaudeJSON = None  # type: ignore


class TaskState(TypedDict, total=False):
    instruction: str
    query: str
    tbs: Optional[str]
    search_limit: int
    search_results: List[Dict[str, Any]]
    chosen_url: Optional[str]
    chosen_title: Optional[str]
    chosen_description: Optional[str]
    live_plan: List[Dict[str, Any]]
    live_result: Dict[str, Any]
    browser_session: Any
    started_at: float
    finished_at: float
    duration_ms: float
    error: Optional[str]


@dataclass
class InstructionAgentConfig:
    search_limit: int = 5
    # If the instruction doesn't specify dates, you can bias to recent info by enabling this.
    default_sort_by_date: bool = True
    live_browser: LiveBrowserConfig = field(default_factory=LiveBrowserConfig)


SYSTEM_EXTRACT = """You convert a user's natural-language instruction into web-search parameters.

Return ONLY valid JSON with:
{
  "query": "string",
  "tbs": "string|null",
  "search_limit": 5
}

Rules:
- If user requests a date range, set tbs to: "cdr:1,cd_min:MM/DD/YYYY,cd_max:MM/DD/YYYY"
- If user requests "latest/recent" but no explicit dates, set tbs to "sbd:1,qdr:m" (past month, sorted by date)
- If no time intent, set tbs to null.
"""


SYSTEM_PICK = """Pick the single best URL to open for a live demo.
Return ONLY JSON: {"url":"...","title":"...","description":"..."}.
Prefer pages that are likely to load without paywalls or heavy bot protection.
"""


SYSTEM_PLAN = """You create a small, safe browser action plan for Playwright.
Return ONLY JSON: {"plan":[...]}.

Supported action objects:
- {"type":"click","text":"..."}  (button or visible text)
- {"type":"fill","label":"...","value":"..."} (label/placeholder/name)
- {"type":"press","key":"Enter"}
- {"type":"wait","ms":1500}
- {"type":"screenshot","label":"..."}

Goal: demonstrate the task live (usually: search within site, navigate to relevant result, capture screenshots).
Keep it short (max 8 actions).
"""


def _heuristic_extract(instruction: str, config: InstructionAgentConfig) -> Dict[str, Any]:
    ins = instruction.strip()
    # Detect simple "between" date ranges like "between 01/01/2025 and 02/01/2025"
    m = re.search(r"between\s+(\d{1,2}/\d{1,2}/\d{4})\s+and\s+(\d{1,2}/\d{1,2}/\d{4})", ins, re.I)
    tbs = None
    if m:
        tbs = f"cdr:1,cd_min:{m.group(1)},cd_max:{m.group(2)}"
    elif re.search(r"\b(latest|recent|newest|this month|past month)\b", ins, re.I):
        tbs = "sbd:1,qdr:m" if config.default_sort_by_date else None

    # crude query: remove obvious "find/search/please" prefixes
    query = re.sub(r"^(please\s+)?(find|search|look up)\s+", "", ins, flags=re.I).strip()
    return {"query": query or ins, "tbs": tbs, "search_limit": config.search_limit}


def _firecrawl_search(client: Firecrawl, query: str, limit: int, tbs: Optional[str]) -> List[Dict[str, Any]]:
    # firecrawl-py (v4.x) expects `sources=["web"]` and `tbs="..."` as top-level args.
    resp = client.search(
        query=query,
        sources=["web"],
        limit=limit,
        tbs=tbs,
    )

    # Normalize SearchData (pydantic) or dict to list[dict]
    if hasattr(resp, "model_dump"):
        payload = resp.model_dump()  # type: ignore[assignment]
    elif isinstance(resp, dict):
        payload = resp
    else:
        payload = {}

    web = None
    if isinstance(payload, dict):
        # firecrawl-py SearchData uses top-level arrays: {"web":[...], "news":[...], "images":[...]}
        web = payload.get("web")
        if web is None:
            data = payload.get("data")
            if isinstance(data, dict):
                web = data.get("web")
    if not isinstance(web, list):
        return []

    out: List[Dict[str, Any]] = []
    for r in web:
        if isinstance(r, dict) and r.get("url"):
            out.append(r)
    return out


def build_instruction_graph(
    firecrawl_client: Firecrawl,
    config: Optional[InstructionAgentConfig] = None,
):
    if config is None:
        config = InstructionAgentConfig()

    graph = StateGraph(TaskState)

    def extract_node(state: TaskState) -> TaskState:
        instruction = (state.get("instruction") or "").strip()
        if not instruction:
            return {**state, "error": "No instruction provided"}

        started_at = state.get("started_at") or time.perf_counter()
        new_state: TaskState = {**state, "started_at": started_at, "error": None}

        # Prefer Claude if available, else heuristic.
        extracted: Dict[str, Any]
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if ClaudeJSON is not None and api_key:
            try:
                claude = ClaudeJSON(api_key=api_key)
                extracted = claude.complete_json(system=SYSTEM_EXTRACT, user=instruction)
            except Exception:
                extracted = _heuristic_extract(instruction, config)
        else:
            extracted = _heuristic_extract(instruction, config)

        new_state["query"] = str(extracted.get("query") or instruction)
        new_state["tbs"] = extracted.get("tbs")
        new_state["search_limit"] = int(extracted.get("search_limit") or config.search_limit)
        return new_state

    def search_node(state: TaskState) -> TaskState:
        query = state.get("query")
        if not query:
            return {**state, "error": "No query to search"}
        limit = int(state.get("search_limit") or config.search_limit)
        tbs = state.get("tbs")
        results = _firecrawl_search(firecrawl_client, query=query, limit=limit, tbs=tbs)
        return {**state, "search_results": results, "error": None}

    def pick_node(state: TaskState) -> TaskState:
        results = state.get("search_results") or []
        if not results:
            return {**state, "error": "No search results returned"}

        url = None
        title = None
        desc = None
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if ClaudeJSON is not None and api_key:
            try:
                claude = ClaudeJSON(api_key=api_key)
                pick = claude.complete_json(
                    system=SYSTEM_PICK,
                    user=f"Instruction:\n{state.get('instruction')}\n\nResults:\n{results}",
                )
                url = pick.get("url")
                title = pick.get("title")
                desc = pick.get("description")
            except Exception:
                url = None

        if not url:
            top = results[0]
            url = top.get("url")
            title = top.get("title")
            desc = top.get("description")

        return {
            **state,
            "chosen_url": url,
            "chosen_title": title,
            "chosen_description": desc,
            "error": None,
        }

    def plan_node(state: TaskState) -> TaskState:
        url = state.get("chosen_url")
        if not url:
            return {**state, "error": "No URL chosen to open"}

        instruction = state.get("instruction") or ""
        plan: Any = None
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if ClaudeJSON is not None and api_key:
            try:
                claude = ClaudeJSON(api_key=api_key)
                res = claude.complete_json(
                    system=SYSTEM_PLAN,
                    user=f"Website: {url}\nUser goal: {instruction}\nCreate a plan now.",
                )
                plan = res.get("plan")
            except Exception:
                plan = None

        if plan is None:
            # Minimal generic demo: take a few screenshots around a short pause.
            plan = [
                {"type": "wait", "ms": 1200},
                {"type": "screenshot", "label": "landing"},
                {"type": "wait", "ms": 1200},
                {"type": "screenshot", "label": "still"},
            ]

        if not isinstance(plan, list):
            plan = []

        # Safety cap
        plan = plan[:8]
        return {**state, "live_plan": plan, "error": None}

    def live_node(state: TaskState) -> TaskState:
        url = state.get("chosen_url")
        if not url:
            return {**state, "error": "No URL to run live browser on"}
        plan = state.get("live_plan") or []
        session = state.get("browser_session")
        if session is not None and not isinstance(session, LiveBrowserSession):
            session = None
        result = run_live_plan(url=url, plan=plan, config=config.live_browser, session=session)
        return {**state, "live_result": result, "error": result.get("error")}

    def finalize_node(state: TaskState) -> TaskState:
        started_at = float(state.get("started_at") or time.perf_counter())
        finished_at = time.perf_counter()
        return {**state, "finished_at": finished_at, "duration_ms": (finished_at - started_at) * 1000.0}

    graph.add_node("extract", extract_node)
    graph.add_node("search", search_node)
    graph.add_node("pick", pick_node)
    graph.add_node("plan", plan_node)
    graph.add_node("live", live_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "search")
    graph.add_edge("search", "pick")
    graph.add_edge("pick", "plan")
    graph.add_edge("plan", "live")
    graph.add_edge("live", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()

