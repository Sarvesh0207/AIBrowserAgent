"""
Unified SerpAPI agent (LLM parsing + SerpAPI calls).

This version removes all browser automation (Playwright). It supports:
- a plain URL            → SerpAPI URL->metadata (title + snippet)
- a search-style query  → LLM parses instruction, then SerpAPI searches
- a request about a URL → resolves the URL and shows title/snippet

Run:
    python unified_agent.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from agent import AgentState as UrlAgentState
from agent import build_agent as build_url_agent
from agent import print_metrics as print_url_metrics
from nl_agent import run as run_search, search_top_url


IntentKind = Literal["url_browse", "page_summary", "search", "exit", "go_to"]


@dataclass
class Intent:
    kind: IntentKind
    url: str | None = None
    text: str | None = None  # full user message (for search or context)


_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)


def _extract_url(text: str) -> str | None:
    m = _URL_RE.search(text)
    if m:
        return m.group(0)
    return None


def parse_intent(message: str, current_url: Optional[str] = None) -> Intent:
    """
    Very lightweight intent parser:
    - if message is 'exit'/'bye'/'quit'          → exit
    - if contains URL and mentions summary words → page_summary
    - if contains URL                            → url_browse
    - if starts with 'go to'/'open'/'visit'      → go_to
    - otherwise                                  → search
    """
    msg = (message or "").strip()
    lower = msg.lower()

    # Explicit exit intents
    if lower in {"exit", "bye", "quit"}:
        return Intent(kind="exit", url=None, text=msg)

    url = _extract_url(msg)
    wants_summary = any(k in lower for k in ["title", "summary", "overview", "outline"])

    if url and wants_summary:
        return Intent(kind="page_summary", url=url, text=msg)
    if url:
        return Intent(kind="url_browse", url=url, text=msg)

    # If we already have a current page and the user asks for a summary
    # without specifying a URL, treat it as a request about the current page.
    if current_url and wants_summary:
        return Intent(kind="page_summary", url=current_url, text=msg)

    # "go to ..." style navigation without explicit URL
    if lower.startswith(("go to ", "open ", "visit ")):
        return Intent(kind="go_to", url=None, text=msg)

    return Intent(kind="search", url=None, text=msg)


def _url_browse(url: str) -> None:
    """
    URL -> metadata using the SerpAPI URL agent.

    Note: This path does not use an additional LLM step; the snippet/description
    comes directly from SerpAPI's organic result.
    """
    print(f"\n[SerpAPI] Resolving metadata for: {url}")
    app = build_url_agent()
    initial: UrlAgentState = {
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
    final = app.invoke(initial)  # type: ignore[arg-type]
    print_url_metrics(final, save_to_results=True)


def main() -> None:
    print(
        "\nUnified SerpAPI Agent (LLM + SerpAPI)\n"
        "You can type:\n"
        "  - a URL (e.g. https://www.uic.edu/)\n"
        "  - a search query (e.g. Python tutorials published in 2024 top 10)\n"
        "Type 'exit' or 'bye' or press Enter on an empty line to exit.\n"
    )

    current_url: Optional[str] = None

    while True:
        try:
            msg = input("What would you like to do?\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        intent = parse_intent(msg, current_url=current_url)

        if intent.kind == "exit":
            print("Goodbye.")
            break

        if intent.kind == "url_browse" and intent.url:
            current_url = intent.url
            _url_browse(intent.url)
            continue

        if intent.kind == "page_summary" and intent.url:
            current_url = intent.url
            _url_browse(intent.url)
            continue

        if intent.kind == "go_to":
            # Strip leading navigation verbs to get the target site text.
            lower = (intent.text or "").lower()
            query = intent.text or ""
            for prefix in ("go to ", "open ", "visit "):
                if lower.startswith(prefix):
                    query = intent.text[len(prefix) :].strip()  # type: ignore[arg-type]
                    break
            print(f"\n[GoTo] Resolving destination for: {query!r}")
            url = search_top_url(query)
            if not url:
                print("Could not resolve a destination URL from that instruction.")
            else:
                current_url = url
                _url_browse(url)
            continue

        # Default: treat as natural-language search instruction.
        print("\n[Search] Running natural-language SerpAPI search...")
        run_search(intent.text or msg)


if __name__ == "__main__":
    main()

