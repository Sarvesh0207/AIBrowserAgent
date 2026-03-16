"""
main.py
~~~~~~~
Terminal entry point for the WebAgent.

Usage:
    python main.py
    python main.py --url https://example.com
    python main.py --headless
"""

from __future__ import annotations

import argparse
import json
import traceback
import uuid

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.theme import Theme

from agent import _browser, build_graph

load_dotenv()

# ── rich console ──────────────────────────────────────────────────────────────

THEME = Theme({
    "agent":     "bold cyan",
    "human":     "bold green",
    "tool":      "dim yellow",
    "interrupt": "bold magenta",
    "error":     "bold red",
    "info":      "dim white",
    "path":      "underline bright_blue",
})

console = Console(theme=THEME, highlight=False)

BANNER = """
╭──────────────────────────────────────────────────────╮
│                                                      │
│   🌐  WebAgent  —  AI Browser with SoM + HIL        │
│       Powered by LangGraph + Claude + Playwright     │
│                                                      │
│   Commands:  quit / exit / q  →  exit                │
│              help             →  show this banner    │
│                                                      │
│   Screenshots saved to:  ./screenshots/              │
│                                                      │
╰──────────────────────────────────────────────────────╯
"""


# ── display helpers ───────────────────────────────────────────────────────────

def print_banner() -> None:
    console.print(BANNER, style="bold cyan")


def print_agent(text: str) -> None:
    console.print()
    console.print(Panel(
        Markdown(text),
        title="[agent]🤖 Agent[/agent]",
        border_style="cyan",
        padding=(0, 1),
    ))


def print_interrupt_panel(question: str) -> None:
    console.print()
    console.print(Panel(
        f"[interrupt]{question}[/interrupt]",
        title="[interrupt]⏸  Agent needs your input[/interrupt]",
        border_style="magenta",
        padding=(0, 1),
    ))


def print_tool_line(tool_call_id: str, content: str) -> None:
    preview = content[:150].replace("\n", " ")
    console.print(f"  [tool]⚙  [{tool_call_id[:16]}][/tool]  [info]{preview}[/info]")


def highlight_screenshots(content: str) -> None:
    try:
        data = json.loads(content)
        if "screenshot_som" in data:
            console.print(f"  [path]📸 SoM   → {data['screenshot_som']}[/path]")
        if "screenshot_clean" in data:
            console.print(f"  [path]📷 Clean → {data['screenshot_clean']}[/path]")
    except Exception:
        pass


def display_result(messages: list) -> None:
    console.rule(style="dim")
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            print_tool_line(msg.tool_call_id, msg.content)
            highlight_screenshots(msg.content)

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            if isinstance(msg.content, str):
                print_agent(msg.content)
            elif isinstance(msg.content, list):
                parts = [
                    b["text"] for b in msg.content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                if parts:
                    print_agent("\n".join(parts))
            break


# ── interrupt-aware invocation ────────────────────────────────────────────────

def run_with_interrupts(graph, config: dict, initial_input: dict) -> dict:
    """
    Run the graph, handling LangGraph interrupt() calls.

    When the agent calls ask_human(), LangGraph raises GraphInterrupt.
    We catch it, show the question to the user, collect their answer,
    then resume the graph with Command(resume=answer).
    This loops until the graph reaches END normally.
    """
    current: dict | Command = initial_input

    while True:
        try:
            result = graph.invoke(current, config=config)
            return result

        except GraphInterrupt as gi:
            # gi.args[0] is a tuple/list of Interrupt(value=...) objects
            interrupts = gi.args[0] if gi.args else []
            question = "The agent needs your input:"

            if interrupts:
                iv = interrupts[0].value
                if isinstance(iv, dict):
                    question = iv.get("question", str(iv))
                else:
                    question = str(iv)

            print_interrupt_panel(question)
            console.print()
            answer = Prompt.ask("[bold magenta]Your answer[/bold magenta]")
            # Resume the graph from where interrupt() was called
            current = Command(resume=answer)


# ── main REPL ─────────────────────────────────────────────────────────────────

def run(initial_url: str | None = None, headless: bool = False) -> None:
    _browser.headless = headless
    _browser.open()

    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print_banner()
    console.print(
        "[info]Browser launched. Type a URL or a natural language instruction.[/info]\n"
    )

    first_input = initial_url

    while True:
        # ── prompt ────────────────────────────────────────────────────────────
        if first_input is not None:
            user_input = first_input
            first_input = None
            console.print(f"[human]You:[/human] {user_input}")
        else:
            console.print()
            try:
                user_input = Prompt.ask("[bold green]You[/bold green]")
            except (KeyboardInterrupt, EOFError):
                break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "help":
            print_banner()
            continue

        # ── invoke agent ──────────────────────────────────────────────────────
        console.print()
        console.rule("[dim]Agent thinking…[/dim]", style="dim")

        try:
            result = run_with_interrupts(
                graph,
                config,
                {"messages": [HumanMessage(content=user_input)]},
            )
            display_result(result.get("messages", []))

        except KeyboardInterrupt:
            console.print("\n[info]Interrupted.[/info]")
        except Exception as exc:
            console.print(f"\n[error]❌ {exc}[/error]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")

    # ── shutdown ──────────────────────────────────────────────────────────────
    console.print("\n[info]Closing browser… Goodbye![/info]")
    _browser.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="WebAgent — AI browser with SoM + HIL")
    parser.add_argument("--url", "-u", default=None,
                        help="URL to navigate to on startup")
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Run Chromium headless (no browser window)")
    args = parser.parse_args()
    run(initial_url=args.url, headless=args.headless)


if __name__ == "__main__":
    main()
