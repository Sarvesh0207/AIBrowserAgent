"""
agent.py
~~~~~~~~
LangGraph web-browsing agent with:
  • SoM (Set-of-Mark) screenshots
  • Human-in-the-Loop via LangGraph interrupt()
  • Multi-turn conversation memory
  • Dropdown clarification
  • Vague instruction detection + clarification/refusal
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from browser_tools import BrowserController, ElementInfo, PageState

load_dotenv()

# ── global browser instance (lives for the whole session) ─────────────────────

_browser = BrowserController(headless=False)   # headed mode as requested

# ── shared state between tools and graph ──────────────────────────────────────

_current_elements: list[ElementInfo] = []
_current_page: PageState | None = None


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def navigate_to_url(url: str) -> str:
    """
    Navigate the browser to a URL.
    Returns the page title, a description of visible content, and SoM screenshot path.
    """
    global _current_elements, _current_page
    state = _browser.navigate(url)
    _current_elements = state.elements
    _current_page = state

    elem_summary = _elements_summary(state.elements)
    return json.dumps({
        "url": state.url,
        "title": state.title,
        "screenshot_clean": str(state.screenshot_clean),
        "screenshot_som": str(state.screenshot_som),
        "page_text_preview": state.page_text[:800],
        "interactive_elements": elem_summary,
        "total_elements": len(state.elements),
    })


@tool
def capture_current_page() -> str:
    """
    Take a fresh screenshot of the current page with SoM bounding boxes.
    Use this after any action to see the updated page state.
    """
    global _current_elements, _current_page
    state = _browser.capture_state()
    _current_elements = state.elements
    _current_page = state

    elem_summary = _elements_summary(state.elements)
    return json.dumps({
        "url": state.url,
        "title": state.title,
        "screenshot_clean": str(state.screenshot_clean),
        "screenshot_som": str(state.screenshot_som),
        "page_text_preview": state.page_text[:800],
        "interactive_elements": elem_summary,
        "total_elements": len(state.elements),
    })


@tool
def click_element(element_index: int) -> str:
    """
    Click an interactive element by its SoM index number.
    Use the SoM screenshot to identify the correct index.
    """
    result = _browser.click_element(element_index, _current_elements)
    return result


@tool
def type_text(element_index: int, text: str) -> str:
    """
    Type text into an input field identified by its SoM index number.
    Clears existing content first.
    """
    return _browser.type_into(element_index, text, _current_elements)


@tool
def press_keyboard_key(key: str) -> str:
    """
    Press a keyboard key. Examples: 'Enter', 'Tab', 'Escape', 'ArrowDown'.
    """
    return _browser.press_key(key)


@tool
def select_dropdown_option(element_index: int, option_text: str) -> str:
    """
    Select an option in a <select> dropdown by its SoM index and the visible option text.
    """
    return _browser.select_option(element_index, option_text, _current_elements)


@tool
def scroll_page(direction: str, amount: int = 400) -> str:
    """
    Scroll the page. direction must be 'up' or 'down'. amount is pixels (default 400).
    """
    return _browser.scroll(direction, amount)


@tool
def go_back() -> str:
    """Navigate the browser back to the previous page."""
    return _browser.go_back()


@tool
def ask_human(question: str) -> str:
    """
    Pause execution and ask the human a question or for confirmation.
    Use this when:
    - A dropdown needs user selection input
    - You need to clarify a vague instruction
    - You need credentials or sensitive info
    - You're about to perform a destructive/irreversible action
    The human's answer will be returned as a string.
    """
    # LangGraph interrupt — pauses the graph and waits for human input
    answer = interrupt({"question": question})
    return str(answer)


ALL_TOOLS = [
    navigate_to_url,
    capture_current_page,
    click_element,
    type_text,
    press_keyboard_key,
    select_dropdown_option,
    scroll_page,
    go_back,
    ask_human,
]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}


# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a highly capable web browsing assistant — think of yourself like Claude, ChatGPT, or Gemini but with the ability to actually visit and interact with websites in real time.

## YOUR CAPABILITIES
- Navigate to any URL
- Click buttons, links, and interactive elements
- Type into forms and search boxes
- Select dropdown options
- Scroll pages
- Take screenshots (clean + with SoM bounding boxes)
- Ask the human for clarification or confirmation

## SET-OF-MARK (SoM) SYSTEM
Every screenshot comes with a SoM version where interactive elements have **numbered bounding boxes**:
- 🔵 Blue = Links
- 🟢 Green = Buttons
- 🟡 Amber = Input fields
- 🟣 Purple = Dropdowns (<select>)
- 🔴 Red = Text areas
When you want to interact with an element, reference it by number (e.g. "clicking element #7").

## CRITICAL BEHAVIOURS

### 1. CLARIFY VAGUE INSTRUCTIONS
If the user gives a vague instruction like "do the thing" or "go somewhere interesting", use `ask_human` to clarify BEFORE acting. Ask specifically what they mean.

### 2. REFUSE IRRELEVANT REQUESTS  
If someone asks you to do something unrelated to web browsing (write code, solve math, etc.) politely explain that you are a web browsing agent and redirect them.

### 3. DROPDOWN HANDLING (MANDATORY)
When you encounter a dropdown (<select> element, purple bounding box) and the user hasn't specified what to choose:
- ALWAYS call `ask_human` with the question and list all available options
- Wait for the user's answer before calling `select_dropdown_option`
- Example: "I see a dropdown #5 with options: [English, French, Spanish, German]. Which would you like?"

### 4. DESTRUCTIVE ACTIONS
Before submitting forms, making purchases, deleting items, or any irreversible action — use `ask_human` to confirm.

### 5. AFTER EVERY ACTION
Always call `capture_current_page` after clicking, typing, or navigating to see the updated state.

### 6. SCREENSHOTS
Always tell the user the screenshot paths so they can view them:
- `screenshot_clean` = raw screenshot
- `screenshot_som` = annotated with numbered bounding boxes

### 7. CONVERSATION MEMORY
You remember the full conversation history. Reference past steps naturally.

## WORKFLOW FOR EACH USER MESSAGE
1. Understand the intent — clarify if vague
2. Navigate / interact using tools
3. Capture state after each action
4. If dropdowns appear and choice not given → ask_human
5. Report findings clearly: title, URL, description, screenshot paths, what you did

Be concise but thorough. Always show the user what you found.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  LANGGRAPH STATE + NODES
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _make_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=4096,
    ).bind_tools(ALL_TOOLS)


llm = _make_llm()


def agent_node(state: AgentState) -> dict:
    """Main reasoning node — calls the LLM with full message history."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """Execute tool calls from the last AI message."""
    last_msg: AIMessage = state["messages"][-1]
    tool_messages: list[ToolMessage] = []

    for tc in last_msg.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_fn = TOOL_MAP.get(tool_name)

        if tool_fn is None:
            result = f"❌ Unknown tool: {tool_name}"
        else:
            try:
                result = tool_fn.invoke(tool_args)
            except Exception as exc:
                result = f"❌ Tool error ({tool_name}): {exc}"

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )

    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return END


# ── graph assembly ─────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    memory = MemorySaver()
    # interrupt() is called inside ask_human tool directly — no interrupt_before needed
    return builder.compile(checkpointer=memory)


# ── helpers ────────────────────────────────────────────────────────────────────

def _elements_summary(elements: list[ElementInfo]) -> list[dict]:
    """Compact summary of elements for LLM context (avoids token bloat)."""
    out = []
    for el in elements[:60]:   # cap at 60 to keep context manageable
        entry: dict[str, Any] = {
            "index": el.index,
            "type": el.element_type,
            "text": el.text[:80],
        }
        if el.placeholder:
            entry["placeholder"] = el.placeholder[:50]
        if el.href:
            entry["href"] = el.href[:100]
        if el.is_dropdown and el.dropdown_options:
            entry["options"] = el.dropdown_options[:20]
        out.append(entry)
    return out
