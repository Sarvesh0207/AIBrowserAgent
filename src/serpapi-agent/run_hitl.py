"""
Terminal entrypoint for the human-in-the-loop browser agent.

Run:
    python run_hitl.py

The agent will:
- ask you for a URL
- open it in the browser (headless or headed based on HEADLESS)
- summarize the page
- let you ask follow-up questions in a loop
"""

from __future__ import annotations

import sys

from browser_agent import AgentState, build_graph_hitl


def main() -> None:
    app = build_graph_hitl()
    thread_id = sys.argv[1] if len(sys.argv) > 1 else "default-thread"

    print(
        "\nStarting HITL browser session.\n"
        "You can resume this session later by running:\n"
        f"  python run_hitl.py {thread_id}\n"
    )

    final: AgentState = app.invoke(  # type: ignore[assignment]
        {"url": ""},  # initial empty state
        config={"configurable": {"thread_id": thread_id}},
    )

    print("\nSession finished.")
    if final.get("summary"):
        print("\nSummary:\n", final["summary"])
    if final.get("followup_answers"):
        print("\nFollow-up Q&A:")
        for qa in final["followup_answers"]:
            print(f"- Q: {qa['question']}\n  A: {qa['answer']}\n")


if __name__ == "__main__":
    main()

