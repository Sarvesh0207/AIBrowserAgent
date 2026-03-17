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
from instruction_agent_graph import InstructionAgentConfig, build_instruction_graph
from live_browser import LiveBrowserSession


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
        description="LangGraph + Firecrawl web agent (scrape + live instruction mode)."
    )
    parser.add_argument(
        "--url",
        required=False,
        help="URL to scrape (e.g. https://example.com)",
    )
    parser.add_argument(
        "--instruction",
        required=False,
        help='Natural language instruction (e.g. "Find papers on X between 01/01/2025 and 02/01/2025 and open the best result")',
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="When used with --instruction, runs a visible Playwright session and captures screenshots.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep the browser open and prompt for more tasks (use with --instruction --live).",
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

    if not args.url and not args.instruction:
        raise SystemExit("Provide either --url (scrape mode) or --instruction (agent mode).")

    # Load env vars from .env if present
    load_dotenv()

    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise SystemExit(
            "FIRECRAWL_API_KEY is not set. "
            "Create a .env file with FIRECRAWL_API_KEY=fc-... or set it in your environment."
        )

    client = Firecrawl(api_key=api_key)

    outputs_dir = "outputs"
    _ensure_dir(outputs_dir)

    if args.instruction:
        # Instruction mode: search the web + (optional) do it live in browser
        instr_config = InstructionAgentConfig()
        if args.live:
            instr_config.live_browser.headless = False
        else:
            # non-live runs headless to be faster
            instr_config.live_browser.headless = True
            instr_config.live_browser.slow_mo_ms = 0

        graph = build_instruction_graph(client, instr_config)

        session: LiveBrowserSession | None = None
        if args.live and args.interactive:
            # One shared browser window for multiple tasks
            session = LiveBrowserSession(config=instr_config.live_browser)

        try:
            instruction = args.instruction
            while True:
                initial_state: Dict[str, Any] = {"instruction": instruction}
                if session is not None:
                    initial_state["browser_session"] = session

                result = graph.invoke(initial_state)

                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                slug = _slugify(instruction[:60])
                basename = f"{ts}_instruction_{slug}"

                output: Dict[str, Any] = {
                    "instruction": instruction,
                    "query": result.get("query"),
                    "tbs": result.get("tbs"),
                    "search_results": result.get("search_results"),
                    "chosen_url": result.get("chosen_url"),
                    "chosen_title": result.get("chosen_title"),
                    "chosen_description": result.get("chosen_description"),
                    "live_plan": result.get("live_plan"),
                    "live_result": result.get("live_result"),
                    "response_time_ms": float(result.get("duration_ms") or 0.0),
                    "response_time_s": float(result.get("duration_ms") or 0.0) / 1000.0,
                    "timeout_ms_soft": args.timeout_ms,
                    "timeout_exceeded": float(result.get("duration_ms") or 0.0) > float(args.timeout_ms),
                    "error": result.get("error"),
                }

                json_path = os.path.join(outputs_dir, f"{basename}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)

                print(f"Instruction: {instruction}")
                print(f"Query: {output.get('query')}")
                print(f"TBS: {output.get('tbs')}")
                print(f"Chosen URL: {output.get('chosen_url')}")
                print(f"Live screenshots: {len((output.get('live_result') or {}).get('screenshots') or [])}")
                print(f"Error: {output.get('error')}")
                print(f"Output JSON: {json_path}")

                if not (args.live and args.interactive):
                    return

                ans = input("\nDo you want to perform another task? (y/n): ").strip().lower()
                if ans not in ("y", "yes"):
                    return
                instruction = input("Enter next task instruction: ").strip()
                if not instruction:
                    return
        finally:
            if session is not None:
                session.close()

    # URL scrape mode (existing)
    agent_config = AgentConfig(preview_chars=args.preview_chars)
    graph = build_graph(client, agent_config)
    initial_state = {"url": args.url}
    result = graph.invoke(initial_state)

    # Prepare outputs
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

