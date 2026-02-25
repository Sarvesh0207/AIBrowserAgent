import argparse
import asyncio
import uuid
from pathlib import Path

from langgraph.types import Command

from src.agent_graph import build_graph, build_graph_hitl
from src.browser import fetch_page_with_action
from src.evaluate_headless import run_headless_evaluation
from src.generate_headless_doc import generate_headless_doc
from src.logger import jsonl_path, append_jsonl, utc_ts
from src.prompt_to_url import parse_prompt_to_url, looks_like_url


def parse_args():
    parser = argparse.ArgumentParser("AIBrowserAgent")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Browse URL and summarize; use --url or --prompt (natural language)")
    run_parser.add_argument("--url", default=None, help="URL to browse (use this or --prompt)")
    run_parser.add_argument("--prompt", default=None, help="Natural language, e.g. 'Visit example.com and summarize' (LLM extracts URL)")

    hitl_parser = subparsers.add_parser("run-hitl", help="HITL mode: agent pauses to ask for URL via interrupt")
    hitl_parser.add_argument("--thread-id", default=None, help="Optional thread_id; auto-generated if omitted")

    eval_parser = subparsers.add_parser("benchmark")
    eval_parser.add_argument("--csv", required=True)

    doc_parser = subparsers.add_parser(
        "benchmark-doc",
        help="Generate Headless Evaluation doc (Markdown) from a benchmark report CSV",
    )
    doc_parser.add_argument(
        "--report",
        default=None,
        help="Path to headless_report_*.csv; if omitted, use latest in outputs/reports",
    )
    doc_parser.add_argument(
        "--out",
        default=None,
        help="Output path for .md file; default docs/HEADLESS_EVALUATION.md",
    )

    action_parser = subparsers.add_parser(
        "run-action",
        help="Headless: load URL, then click or fill one element and confirm (screenshot + log)",
    )
    action_parser.add_argument("--url", required=True, help="Page URL to load")
    action_parser.add_argument("--click", default=None, metavar="SELECTOR", help="CSS selector to click (e.g. 'a' or 'button.submit')")
    action_parser.add_argument("--open-search", default=None, metavar="SELECTOR", help="Click this first to open search bar (e.g. search icon), then use --fill and --fill-value")
    action_parser.add_argument("--fill", default=None, metavar="SELECTOR", help="CSS selector of input to fill (use with --fill-value)")
    action_parser.add_argument("--fill-value", default=None, help="Value to type into the element given by --fill")
    action_parser.add_argument("--submit", action="store_true", help="After --fill, press Enter to submit (e.g. search)")

    return parser.parse_args()

async def run_single(url):
    app = build_graph()
    result = await app.ainvoke({"url": url})

    print("\n=== RESULT ===")
    print("URL:", result.get("url"))
    print("Title:", result.get("title"))
    print("Summary:", result.get("summary"))
    print("================\n")

async def run_hitl(thread_id: str | None):
    """
    HITL mode: graph starts at ask_url, calls interrupt(), pauses.
    main.py prompts the user in terminal, then invokes again with Command(resume=url).
    """
    app = build_graph_hitl()
    tid = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    # Initial invoke: ask_url will call interrupt() and pause
    result = await app.ainvoke({}, config=config)

    while "__interrupt__" in result:
        interrupts = result["__interrupt__"]
        if not interrupts:
            break
        # If we have a follow-up answer from the previous cycle, show it
        last_answer = result.get("followup_answer")
        if last_answer:
            print(f"\n--- Answer ---\n{last_answer}\n")
        payload = interrupts[0].value
        user_input = input(f"{payload}\n> ").strip()
        # If this is the URL/intent question (not follow-up), resolve natural language to URL
        if payload and "follow-up" not in payload.lower() and "follow up" not in payload.lower():
            if not looks_like_url(user_input):
                try:
                    resolved = parse_prompt_to_url(user_input)
                    print("Resolved URL:", resolved)
                    user_input = resolved
                except Exception as e:
                    print("Could not resolve to URL:", e)
        result = await app.ainvoke(Command(resume=user_input), config=config)

    print("\n=== RESULT ===")
    print("URL:", result.get("url"))
    print("Title:", result.get("title"))
    print("Summary:", result.get("summary"))
    followups = result.get("followup_answers") or []
    if followups:
        print("\n--- Follow-up Q&A ---")
        for i, qa in enumerate(followups, 1):
            print(f"Q{i}: {qa.get('question', '')}")
            print(f"A{i}: {qa.get('answer', '')}\n")
    print("================\n")


async def run_benchmark(csv_path):
    await run_headless_evaluation(csv_path)


async def run_demo_action(
    url: str,
    click_selector: str | None,
    fill_selector: str | None,
    fill_value: str | None,
    submit_after_fill: bool = False,
    open_search_selector: str | None = None,
) -> None:
    """
    Headless: load URL, perform one click or one fill, then confirm (screenshot + log).
    If open_search_selector, click it first to open search bar, then fill.
    If submit_after_fill, after filling we press Enter to submit (e.g. search).
    """
    if click_selector and fill_selector:
        raise SystemExit("Use either --click or --fill/--fill-value, not both.")
    if fill_selector and fill_value is None:
        raise SystemExit("--fill requires --fill-value.")
    if not click_selector and not fill_selector:
        raise SystemExit("Provide --click SELECTOR or --fill SELECTOR and --fill-value VALUE.")
    if submit_after_fill and not fill_selector:
        raise SystemExit("--submit can only be used with --fill and --fill-value.")
    if open_search_selector and not fill_selector:
        raise SystemExit("--open-search requires --fill and --fill-value.")

    page_result, confirmation = await fetch_page_with_action(
        url,
        click_selector=click_selector,
        fill_selector=fill_selector,
        fill_value=fill_value,
        submit_after_fill=submit_after_fill,
        open_search_selector=open_search_selector,
    )

    # Log to JSONL (action log)
    action_log = jsonl_path("action")
    rec = {
        "ts": utc_ts(),
        "url": url,
        "final_url": page_result.final_url,
        "action_type": confirmation.action_type if confirmation else None,
        "selector": confirmation.selector if confirmation else None,
        "value": getattr(confirmation, "value", None) if confirmation else None,
        "success": confirmation.success if confirmation else None,
        "message": confirmation.message if confirmation else None,
        "submitted": getattr(confirmation, "submitted", False) if confirmation else False,
        "screenshot_path": page_result.screenshot_path,
    }
    append_jsonl(action_log, rec)

    # Print confirmation
    print("\n=== ACTION RESULT (headless) ===")
    print("URL:", page_result.url)
    print("Final URL:", page_result.final_url)
    print("Title:", page_result.title)
    if confirmation:
        print("Action:", confirmation.action_type)
        print("Selector:", confirmation.selector)
        if confirmation.value is not None:
            print("Value:", confirmation.value)
        print("Submitted (pressed Enter):", getattr(confirmation, "submitted", False))
        print("Success:", confirmation.success)
        print("Message:", confirmation.message)
    print("Screenshot:", page_result.screenshot_path)
    print("Log:", str(action_log))
    print("================================\n")


def run_benchmark_doc(report_path: str | None, out_path: str | None) -> None:
    """Generate docs/HEADLESS_EVALUATION.md from a benchmark report CSV."""
    out = generate_headless_doc(
        report_path=Path(report_path) if report_path else None,
        out_path=Path(out_path) if out_path else None,
    )
    print(f"Wrote: {out}")


def main():
    args = parse_args()

    if args.command == "run":
        url = args.url
        if getattr(args, "prompt", None):
            if url:
                raise SystemExit("Use either --url or --prompt, not both.")
            url = parse_prompt_to_url(args.prompt)
            print("Resolved URL:", url)
        elif not url:
            raise SystemExit("Provide --url or --prompt.")
        asyncio.run(run_single(url))

    elif args.command == "run-hitl":
        asyncio.run(run_hitl(args.thread_id))

    elif args.command == "benchmark":
        asyncio.run(run_benchmark(args.csv))

    elif args.command == "benchmark-doc":
        run_benchmark_doc(
            getattr(args, "report", None),
            getattr(args, "out", None),
        )

    elif args.command == "run-action":
        asyncio.run(run_demo_action(
            args.url,
            getattr(args, "click", None),
            getattr(args, "fill", None),
            getattr(args, "fill_value", None),
            submit_after_fill=getattr(args, "submit", False),
            open_search_selector=getattr(args, "open_search", None),
        ))

if __name__ == "__main__":
    main()
