from __future__ import annotations

"""
Browser + LangGraph agent for SerpAPI.

This file combines two concerns so it is easier to explain:

1) Playwright browser helpers (fetch_page, PageResult)
2) LangGraph flows (build_graph, build_graph_hitl) that use the browser
   and Claude to summarize pages and support HITL (human-in-the-loop).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict

from anthropic import Anthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from playwright.async_api import async_playwright

from config import (
    ANTHROPIC_API_KEY,
    BROWSER_TIMEOUT_MS,
    HEADLESS,
    SCREENSHOT_DIR,
    ensure_dirs,
    require_api_key,
)


# ────────────────────────────────────────────────────────────────────
# 1. Playwright browser helpers
# ────────────────────────────────────────────────────────────────────


@dataclass
class PageResult:
    url: str
    final_url: str
    title: str
    description: str
    screenshot_path: str


async def _extract_title_and_description(page) -> tuple[str, str]:
    """
    Best-effort extraction of page title and description.

    Prefer the <title> tag and standard description metas.
    """
    title = (await page.title()) or ""
    desc = await page.evaluate(
        """() => {
            const meta = document.querySelector('meta[name="description"]');
            if (meta && meta.content) return meta.content;
            const og = document.querySelector('meta[property="og:description"]');
            if (og && og.content) return og.content;
            return "";
        }"""
    )
    return (title or "").strip(), (desc or "").strip()


async def fetch_page(url: str) -> PageResult:
    """
    Open `url` in Chromium, wait for DOMContentLoaded, grab title/description,
    take a screenshot, and return a PageResult.
    """
    ensure_dirs()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()
        page.set_default_timeout(BROWSER_TIMEOUT_MS)

        try:
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_timeout(800)

            title, desc = await _extract_title_and_description(page)
            final_url = page.url

            safe_name = (
                url.replace("https://", "")
                .replace("http://", "")
                .rstrip("/")
                .replace("/", "_")
            )
            if not safe_name:
                safe_name = "page"
            screenshot_path = str(SCREENSHOT_DIR / f"{safe_name}.png")
            await page.screenshot(path=screenshot_path, full_page=True)

            return PageResult(
                url=url,
                final_url=final_url,
                title=title,
                description=desc,
                screenshot_path=screenshot_path,
            )
        finally:
            await page.close()
            await context.close()
            await browser.close()


# ────────────────────────────────────────────────────────────────────
# 2. LangGraph flows (browse + summarize + HITL)
# ────────────────────────────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    # Input
    url: str

    # From browser
    final_url: str
    title: str
    description: str
    screenshot_path: str

    # LLM summary
    summary: str

    # Follow-up Q&A
    followup_question: str
    followup_answer: str
    followup_answers: List[Dict[str, str]]  # {"question", "answer"}


def browse_node(state: AgentState) -> AgentState:
    """
    Synchronous wrapper around fetch_page for use in LangGraph.

    LangGraph apps can be async, but for this simple CLI-oriented agent we
    run the async function inside a temporary event loop.
    """
    import asyncio

    ensure_dirs()
    url = state["url"]

    async def _run() -> AgentState:
        res = await fetch_page(url)
        return {
            **state,
            "final_url": res.final_url,
            "title": res.title,
            "description": res.description,
            "screenshot_path": res.screenshot_path,
        }

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_run())
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def summarize_node(state: AgentState) -> AgentState:
    """Use Claude to write a short summary of the page."""
    require_api_key("ANTHROPIC")

    client = Anthropic()
    title = state.get("title", "") or ""
    desc = state.get("description", "") or ""

    prompt = (
        "Write a short, factual 1–2 sentence description of this webpage "
        "based on the title and snippet.\n"
        f"Title: {title}\n"
        f"Snippet: {desc}\n"
        "Output:"
    )

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=120,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )

    parts: List[str] = []
    for block in msg.content:
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
    summary = (" ".join(parts)).strip()
    state["summary"] = summary
    return state


def ask_url_node(state: AgentState) -> AgentState:
    """
    HITL: ask the user what to browse.

    For now we expect a URL; if the user omits the scheme we prepend https://.
    """
    prompt = (
        "Enter a website URL to browse (e.g. https://example.com). "
        "For this demo we expect a URL, not a long instruction:"
    )
    user_input = interrupt(prompt)  # type: ignore[assignment]
    text = (user_input or "").strip()
    if not text:
        raise ValueError("URL cannot be empty")
    if not text.startswith("http"):
        text = "https://" + text.lstrip("/")
    state["url"] = text
    return state


def ask_followup_node(state: AgentState) -> AgentState:
    """
    HITL: ask for an optional follow-up question about the page.
    Empty input ends the HITL loop.
    """
    prompt = "Enter a follow-up question about this page (or press Enter to finish):"
    user_input = interrupt(prompt)  # type: ignore[assignment]
    state["followup_question"] = (user_input or "").strip()
    return state


def answer_followup_node(state: AgentState) -> AgentState:
    """Answer the follow-up question using the page context."""
    require_api_key("ANTHROPIC")

    question = state.get("followup_question", "") or ""
    if not question:
        return state

    client = Anthropic()
    title = state.get("title", "") or ""
    desc = state.get("description", "") or ""
    summary = state.get("summary", "") or ""

    prompt = (
        "Answer the user's question based ONLY on the webpage information below. "
        "Be concise (1–3 sentences). If the information is not in the page, say so.\n\n"
        f"Page title: {title}\n"
        f"Snippet: {desc[:800]}\n"
        f"Summary: {summary}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )

    parts: List[str] = []
    for block in msg.content:
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
    answer = (" ".join(parts)).strip()

    state.setdefault("followup_answers", [])
    state["followup_answers"].append({"question": question, "answer": answer})
    state["followup_answer"] = answer
    return state


def _route_after_followup(state: AgentState) -> str:
    """If user entered a question, answer it; else end."""
    if state.get("followup_question"):
        return "answer_followup"
    return END  # type: ignore[return-value]


def build_graph():
    """Graph for one-shot browse + summarize (no HITL)."""
    g = StateGraph(AgentState)
    g.add_node("browse", browse_node)
    g.add_node("summarize", summarize_node)

    g.set_entry_point("browse")
    g.add_edge("browse", "summarize")
    g.add_edge("summarize", END)
    return g.compile()


def build_graph_hitl():
    """
    Human-in-the-loop graph:
        ask_url -> browse -> summarize -> ask_followup -> (answer_followup)* -> END
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

