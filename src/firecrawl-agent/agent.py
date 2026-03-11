from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from typing import Any, Dict

from dotenv import load_dotenv
from firecrawl import Firecrawl

from agent_graph import AgentConfig, build_graph


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "page"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangGraph + Firecrawl web agent (basic)."
    )
    parser.add_argument(
        "--url",
        required=True,
        help="URL to scrape (e.g. https://example.com)",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=60000,
        help="Soft timeout in milliseconds (used for reporting only).",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=800,
        help="Number of characters of markdown to keep in the preview.",
    )
    args = parser.parse_args()

    # Load env vars from .env if present
    load_dotenv()

    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise SystemExit(
            "FIRECRAWL_API_KEY is not set. "
            "Create a .env file with FIRECRAWL_API_KEY=fc-... or set it in your environment."
        )

    client = Firecrawl(api_key=api_key)

    # Build the LangGraph agent
    agent_config = AgentConfig(preview_chars=args.preview_chars)
    graph = build_graph(client, agent_config)

    # Invoke the graph
    initial_state: Dict[str, Any] = {"url": args.url}
    result = graph.invoke(initial_state)

    # Prepare outputs
    outputs_dir = "outputs"
    _ensure_dir(outputs_dir)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    slug = _slugify(args.url)
    basename = f"{ts}_{slug}"

    # Prefer our heuristic summary when showing "description"-like text
    summary = result.get("summary")
    original_description = result.get("description")
    effective_description = summary or original_description

    response_time_ms = float(result.get("duration_ms") or 0.0)
    response_time_s = response_time_ms / 1000.0
    timeout_exceeded = response_time_ms > float(args.timeout_ms)

    output: Dict[str, Any] = {
        "url": args.url,
        "title": result.get("title"),
        # For consumers: description is our best-effort summary.
        "description": effective_description,
        "summary": summary,
        "markdown_preview": result.get("markdown_preview"),
        "response_time_ms": response_time_ms,
        "response_time_s": response_time_s,
        "timeout_ms_soft": args.timeout_ms,
        "timeout_exceeded": timeout_exceeded,
        "screenshot_url": result.get("screenshot_url"),
        "screenshot_path": result.get("screenshot_path"),
        "error": result.get("error"),
        "raw_metadata": result.get("raw_metadata"),
    }

    json_path = os.path.join(outputs_dir, f"{basename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print a concise summary to the console
    print(f"URL: {output['url']}")
    print(f"Title: {output.get('title')}")
    print(f"Description (summary): {output.get('description')}")
    print(f"Response time: {output['response_time_ms']:.2f} ms ({output['response_time_s']:.2f} s)")
    print(f"Timeout soft limit: {output['timeout_ms_soft']} ms (exceeded={output['timeout_exceeded']})")
    print(f"Screenshot (local): {output.get('screenshot_path')}")
    print(f"Screenshot (remote): {output.get('screenshot_url')}")
    print(f"Error: {output.get('error')}")
    print(f"Output JSON: {json_path}")


if __name__ == "__main__":
    main()

