"""
Tavily Search CLI (user-facing, no scoring)
==========================================
This entrypoint is meant to run searches for a user's need (not evaluate Tavily).

Run:
  python agent.py "site:openai.com gpt-4.1 release notes last 30 days top 10"
  python agent.py "Find posts about X from 2025-01-01 to 2025-03-01 top 8 depth:advanced"

This uses LangGraph for orchestration, and `nl_agent.py` for lightweight parsing/search.
"""

import sys
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from nl_agent import _print_summary, search_with_instruction


class AgentState(TypedDict):
    instruction: str
    payload: Optional[dict]
    error: Optional[str]


def search_node(state: AgentState) -> AgentState:
    try:
        payload = search_with_instruction(state["instruction"])
        return {**state, "payload": payload, "error": None}
    except Exception as e:
        return {**state, "payload": None, "error": str(e)}


def print_node(state: AgentState) -> AgentState:
    if state.get("payload"):
        _print_summary(state["payload"])
    if state.get("error"):
        print(f"\nError: {state['error']}\n")
    return state


def build_agent():
    g = StateGraph(AgentState)
    g.add_node("search", search_node)
    g.add_node("print", print_node)
    g.set_entry_point("search")
    g.add_edge("search", "print")
    g.add_edge("print", END)
    return g.compile()


def run(instruction: str) -> dict:
    agent = build_agent()
    final = agent.invoke({"instruction": instruction, "payload": None, "error": None})
    return final.get("payload") or {}


def main() -> None:
    if len(sys.argv) > 1:
        instruction = " ".join(sys.argv[1:]).strip()
    else:
        instruction = input("\nEnter search instruction: ").strip()

    run(instruction)


if __name__ == "__main__":
    main()
