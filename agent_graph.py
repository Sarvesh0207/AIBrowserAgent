from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

import requests
from firecrawl import Firecrawl
from langgraph.graph import END, StateGraph


class PageState(TypedDict, total=False):
    url: str
    title: Optional[str]
    description: Optional[str]
    summary: Optional[str]
    markdown: Optional[str]
    markdown_preview: Optional[str]
    screenshot_url: Optional[str]
    screenshot_path: Optional[str]
    started_at: float
    finished_at: float
    duration_ms: float
    error: Optional[str]
    raw_metadata: Dict[str, Any]


@dataclass
class AgentConfig:
    preview_chars: int = 800
    screenshots_dir: str = "screenshots"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_meta_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    return str(value)


def _download_screenshot(url: str, screenshots_dir: str) -> Optional[str]:
    if not url:
        return None

    _ensure_dir(screenshots_dir)
    # Use timestamp-based filename to avoid collisions
    ts = int(time.time() * 1000)
    filename = f"screenshot_{ts}.png"
    path = os.path.join(screenshots_dir, filename)

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        # Try to pick an extension from content-type if available
        content_type = resp.headers.get("Content-Type", "")
        if "jpeg" in content_type:
            filename = f"screenshot_{ts}.jpg"
        elif "png" in content_type:
            filename = f"screenshot_{ts}.png"
        elif "webp" in content_type:
            filename = f"screenshot_{ts}.webp"
        path = os.path.join(screenshots_dir, filename)

        with open(path, "wb") as f:
            f.write(resp.content)

        return path
    except Exception:
        # If screenshot download fails, we just skip saving locally
        return None


def _summarize_markdown(markdown: str, max_sentences: int = 4, max_chars: int = 500) -> Optional[str]:
    """
    Very lightweight, heuristic summarizer:
    - Strips markdown headings/lists
    - Splits into sentences
    - Returns the first few sentences capped at max_chars.
    """
    if not markdown:
        return None

    lines: List[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Skip heavy headings and navigation junk
        if line.startswith("#"):
            continue
        # Skip short menu-like bullets
        if line.startswith(("-", "*", "+")) and len(line.split()) <= 3:
            # likely a short nav/menu bullet
            continue
        # Skip standalone images
        if line.startswith("!["):
            continue
        lines.append(line)

    if not lines:
        return None

    text = " ".join(lines)
    # Simple sentence split on punctuation.
    parts = []
    sentence = []
    for ch in text:
        sentence.append(ch)
        if ch in ".!?":
            s = "".join(sentence).strip()
            if s:
                parts.append(s)
            sentence = []
    tail = "".join(sentence).strip()
    if tail:
        parts.append(tail)

    if not parts:
        parts = [text]

    chosen: List[str] = []
    total_len = 0
    for s in parts:
        if len(chosen) >= max_sentences:
            break
        if total_len + len(s) > max_chars and chosen:
            break
        chosen.append(s)
        total_len += len(s)

    summary = " ".join(chosen).strip()
    return summary or None


def _build_business_summary(raw_metadata: Dict[str, Any], markdown: str) -> Optional[str]:
    """
    Try to describe what the site/business is about.
    Priority:
    1) Meta description / OG / Twitter descriptions
    2) Fallback to markdown heuristic summary.
    """

    def _coerce_meta_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, list):
            value = " ".join(str(v) for v in value if v)
        s = str(value).strip()
        return s or None

    pieces: List[str] = []
    for key in ["description", "og:description", "twitter:description"]:
        v = _coerce_meta_str(raw_metadata.get(key))
        if v and v not in pieces:
            pieces.append(v)

    if pieces:
        text = " ".join(pieces)
        if len(text) > 500:
            text = text[:497].rstrip() + "..."
        return text

    if markdown:
        return _summarize_markdown(markdown)

    return None


def build_graph(
    firecrawl_client: Firecrawl, config: Optional[AgentConfig] = None
):
    """
    Build a minimal LangGraph that:
    - Scrapes a URL via Firecrawl (markdown + metadata + screenshot)
    - Extracts title/description
    - Downloads the screenshot locally
    - Measures total response time
    """

    if config is None:
        config = AgentConfig()

    graph = StateGraph(PageState)

    def scrape_node(state: PageState) -> PageState:
        url = state.get("url")
        if not url:
            return {
                **state,
                "error": "No URL provided",
            }

        started_at = time.perf_counter()
        new_state: PageState = {
            **state,
            "started_at": started_at,
        }

        try:
            # Ask Firecrawl for markdown + screenshot + metadata
            doc = firecrawl_client.scrape(
                url,
                formats=["markdown", "screenshot"],
            )

            # `doc` is a mapping-like object
            markdown = doc.get("markdown") if isinstance(doc, dict) else getattr(doc, "markdown", None)
            screenshot_url = doc.get("screenshot") if isinstance(doc, dict) else getattr(doc, "screenshot", None)
            metadata = doc.get("metadata") if isinstance(doc, dict) else getattr(doc, "metadata", None)

            # Handle metadata whether it's a dict or a metadata object
            if isinstance(metadata, dict):
                raw_meta_for_title = metadata.get("title")
                raw_meta_for_description = metadata.get("description")
                raw_metadata_dict: Dict[str, Any] = dict(metadata)
            else:
                raw_meta_for_title = getattr(metadata, "title", None)
                raw_meta_for_description = getattr(metadata, "description", None)
                if hasattr(metadata, "model_dump"):
                    raw_metadata_dict = metadata.model_dump()  # type: ignore[assignment]
                else:
                    raw_metadata_dict = {}

            markdown_text = str(markdown) if markdown is not None else ""
            preview = markdown_text[: config.preview_chars] if markdown_text else None
            title = _normalize_meta_value(raw_meta_for_title)
            description = _normalize_meta_value(raw_meta_for_description)
            summary = _build_business_summary(raw_metadata_dict, markdown_text)

            screenshot_path = None
            if isinstance(screenshot_url, str) and screenshot_url:
                screenshot_path = _download_screenshot(screenshot_url, config.screenshots_dir)

            finished_at = time.perf_counter()
            duration_ms = (finished_at - started_at) * 1000.0

            new_state.update(
                {
                    "title": title,
                    "description": description,
                    "summary": summary,
                    "markdown": markdown_text or None,
                    "markdown_preview": preview,
                    "screenshot_url": screenshot_url,
                    "screenshot_path": screenshot_path,
                    "finished_at": finished_at,
                    "duration_ms": duration_ms,
                    "error": None,
                    "raw_metadata": raw_metadata_dict,
                }
            )
        except Exception as exc:
            finished_at = time.perf_counter()
            duration_ms = (finished_at - started_at) * 1000.0
            new_state.update(
                {
                    "finished_at": finished_at,
                    "duration_ms": duration_ms,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        return new_state

    graph.add_node("scrape_page", scrape_node)
    graph.set_entry_point("scrape_page")
    graph.add_edge("scrape_page", END)

    return graph.compile()

