from typing import TypedDict, Optional, List, Dict, Any

from langgraph.graph import StateGraph, END
from anthropic import Anthropic

from .browser import fetch_page
from .config import ANTHROPIC_MODEL, require_api_key
from .logger import jsonl_path, append_jsonl, utc_ts

RUN_LOG = jsonl_path("run")


class AgentState(TypedDict, total=False):
    url: str
    final_url: str
    title: str
    description: str
    screenshot_path: str
    summary: str
    logs: List[Dict[str, Any]]


async def browse_node(state: AgentState) -> AgentState:
    state.setdefault("logs", [])
    url = state["url"]
    try:
        res = await fetch_page(url)
        state["final_url"] = res.final_url
        state["title"] = res.title
        state["description"] = res.description
        state["screenshot_path"] = res.screenshot_path

        rec = {
            "ts": utc_ts(),
            "node": "browse",
            "url": url,
            "final_url": res.final_url,
            "title_len": len(res.title or ""),
            "desc_len": len(res.description or ""),
            "screenshot_path": res.screenshot_path,
            "status": "ok",
        }
        append_jsonl(RUN_LOG, rec)
        state["logs"].append(rec)
        return state
    except Exception as e:
        rec = {"ts": utc_ts(), "node": "browse", "url": url, "status": "fail", "error": repr(e)}
        append_jsonl(RUN_LOG, rec)
        state["logs"].append(rec)
        raise


def summarize_node(state: AgentState) -> AgentState:
    state.setdefault("logs", [])
    require_api_key()

    title = state.get("title", "") or ""
    desc = state.get("description", "") or ""

    client = Anthropic()
    prompt = (
        "Write a short, factual 1-2 sentence description of the website based on the title and snippet.\n"
        f"Title: {title}\n"
        f"Snippet: {desc}\n"
        "Output:"
    )

    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=120,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )

    parts = []
    for block in msg.content:
        if getattr(block, "type", "") == "text":
            parts.append(block.text)

    summary = (" ".join(parts)).strip()
    state["summary"] = summary

    rec = {"ts": utc_ts(), "node": "summarize", "status": "ok", "summary_len": len(summary)}
    append_jsonl(RUN_LOG, rec)
    state["logs"].append(rec)
    return state


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("browse", browse_node)
    g.add_node("summarize", summarize_node)

    g.set_entry_point("browse")
    g.add_edge("browse", "summarize")
    g.add_edge("summarize", END)

    return g.compile()
