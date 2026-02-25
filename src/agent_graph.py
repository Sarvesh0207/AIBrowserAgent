from typing import TypedDict, Optional, List, Dict, Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from anthropic import Anthropic

from .browser import fetch_page
from .config import ANTHROPIC_MODEL, require_api_key
from .logger import jsonl_path, append_jsonl, utc_ts, truncate_for_log

RUN_LOG = jsonl_path("run")


class AgentState(TypedDict, total=False):
    url: str
    final_url: str
    title: str
    description: str
    screenshot_path: str
    summary: str
    logs: List[Dict[str, Any]]
    followup_question: str
    followup_answer: str
    followup_answers: List[Dict[str, str]]  # list of {"question": ..., "answer": ...}


def ask_url_node(state: AgentState) -> AgentState:
    """
    HITL: Pause the graph and ask for the user's intent (natural language or URL).
    The user's input is resolved to a URL in main.py (if needed) before resume.
    """
    payload = (
        "What would you like me to do? "
        "You can enter a URL (e.g. https://example.com) or describe in natural language (e.g. visit example.com and summarize):"
    )
    url = interrupt(payload)
    url = (url or "").strip()
    if not url:
        raise ValueError("URL cannot be empty")
    state["url"] = url
    return state


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
            "title": res.title or "",
            "description": truncate_for_log(res.description),
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

    rec = {
        "ts": utc_ts(),
        "node": "summarize",
        "status": "ok",
        "summary": truncate_for_log(summary),
        "summary_len": len(summary),
    }
    append_jsonl(RUN_LOG, rec)
    state["logs"].append(rec)
    return state


def ask_followup_node(state: AgentState) -> AgentState:
    """
    HITL: Pause after summary and ask for a follow-up question.
    User can enter a question (we answer it) or press Enter to finish.
    """
    payload = (
        "Enter your follow-up question about this page (or press Enter to finish):"
    )
    user_input = interrupt(payload)
    user_input = (user_input or "").strip()
    state["followup_question"] = user_input
    return state


def answer_followup_node(state: AgentState) -> AgentState:
    """
    Answer the user's follow-up question using page context (title, description, summary).
    """
    state.setdefault("logs", [])
    state.setdefault("followup_answers", [])
    require_api_key()

    question = state.get("followup_question", "") or ""
    if not question:
        return state

    title = state.get("title", "") or ""
    desc = state.get("description", "") or ""
    summary = state.get("summary", "") or ""

    client = Anthropic()
    prompt = (
        "Answer the following question based ONLY on the webpage information below. "
        "Be concise (1-3 sentences). If the information is not in the page, say so.\n\n"
        f"Page title: {title}\n"
        f"Page description/snippet: {desc[:800]}\n"
        f"Summary: {summary}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=200,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )

    parts = []
    for block in msg.content:
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
    answer = (" ".join(parts)).strip()
    state["followup_answer"] = answer
    state["followup_answers"].append({"question": question, "answer": answer})

    rec = {
        "ts": utc_ts(),
        "node": "answer_followup",
        "status": "ok",
        "question": truncate_for_log(question),
        "answer": truncate_for_log(answer),
        "question_len": len(question),
    }
    append_jsonl(RUN_LOG, rec)
    state["logs"].append(rec)
    return state


def _route_after_followup(state: AgentState) -> str:
    """If user entered a question, answer it; else end."""
    q = state.get("followup_question", "") or ""
    return "answer_followup" if q else END


def build_graph():
    """Graph for run --url and benchmark: browse → summarize (no interrupt)."""
    g = StateGraph(AgentState)
    g.add_node("browse", browse_node)
    g.add_node("summarize", summarize_node)

    g.set_entry_point("browse")
    g.add_edge("browse", "summarize")
    g.add_edge("summarize", END)

    return g.compile()


def build_graph_hitl():
    """
    Graph for run-hitl: ask_url → browse → summarize → ask_followup (loop).
    ask_url and ask_followup use interrupt to pause for user input.
    Requires checkpointer for interrupt to work; use same thread_id when resuming.
    """
    g = StateGraph(AgentState)
    g.add_node("ask_url", ask_url_node)
    g.add_node("browse", browse_node)
    g.add_node("summarize", summarize_node)
    g.add_node("ask_followup", ask_followup_node)
    g.add_node("answer_followup", answer_followup_node)

    g.set_entry_point("ask_url")
    g.add_edge("ask_url", "browse")
    g.add_edge("browse", "summarize")
    g.add_edge("summarize", "ask_followup")
    g.add_conditional_edges("ask_followup", _route_after_followup)
    g.add_edge("answer_followup", "ask_followup")

    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)
