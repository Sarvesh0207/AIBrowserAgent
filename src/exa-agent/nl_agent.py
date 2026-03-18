import os
import sys
import re
import time
from datetime import datetime, timedelta
from typing import TypedDict, Optional, List

from dotenv import load_dotenv
from exa_py import Exa
from langgraph.graph import StateGraph, END

load_dotenv()

EXA_API_KEY = os.getenv("EXA_API_KEY")

if not EXA_API_KEY:
    raise ValueError("EXA_API_KEY not set in .env")

exa = Exa(EXA_API_KEY)


class AgentState(TypedDict):
    user_instruction: str
    search_query: str
    date_from: str
    date_to: str
    result_count: int
    parse_reason: str
    results: List[dict]
    response_time: Optional[float]
    error: str


def last_day_of_month(year: int, month: int) -> int:
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    return (next_month - timedelta(days=1)).day


def parse_between_months_2025(text: str):
    month_map = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }

    pattern = r"between\s+([a-zA-Z]+)\s+and\s+([a-zA-Z]+)\s+(20\d{2})"
    match = re.search(pattern, text.lower())
    if not match:
        return "", ""

    m1, m2, year_str = match.groups()
    if m1 not in month_map or m2 not in month_map:
        return "", ""

    year = int(year_str)
    start_month = month_map[m1]
    end_month = month_map[m2]

    date_from = f"{year:04d}-{start_month:02d}-01"
    end_day = last_day_of_month(year, end_month)
    date_to = f"{year:04d}-{end_month:02d}-{end_day:02d}"
    return date_from, date_to


def parse_instruction_node(state: AgentState) -> AgentState:
    print("\n[1/2] Parsing instruction (rule-based)...")

    instruction = state["user_instruction"]
    text = instruction.lower()

    search_query = instruction
    date_from = ""
    date_to = ""
    result_count = 5
    parse_reason = "Default parsing"

    today = datetime.today()

    # result count
    match_top = re.search(r"top\s+(\d+)", text)
    match_give = re.search(r"give me\s+(\d+)", text)
    match_results = re.search(r"(\d+)\s+results", text)

    if match_top:
        result_count = min(int(match_top.group(1)), 10)
        parse_reason = "Detected result count from 'top N'"
    elif match_give:
        result_count = min(int(match_give.group(1)), 10)
        parse_reason = "Detected result count from 'give me N'"
    elif match_results:
        result_count = min(int(match_results.group(1)), 10)
        parse_reason = "Detected result count from 'N results'"

    # explicit year
    year_match = re.search(r"\bin\s+(20\d{2})\b", text)
    if year_match:
        year = int(year_match.group(1))
        date_from = f"{year}-01-01"
        date_to = f"{year}-12-31"
        parse_reason = f"Detected explicit year {year}"

    # between Jan and Mar 2025
    between_from, between_to = parse_between_months_2025(text)
    if between_from and between_to:
        date_from = between_from
        date_to = between_to
        parse_reason = "Detected explicit month range"

    # relative ranges
    if "last month" in text or "past month" in text:
        first_this_month = datetime(today.year, today.month, 1)
        last_prev_month = first_this_month - timedelta(days=1)
        first_prev_month = datetime(last_prev_month.year, last_prev_month.month, 1)
        date_from = first_prev_month.strftime("%Y-%m-%d")
        date_to = last_prev_month.strftime("%Y-%m-%d")
        parse_reason = "Detected last month"

    elif "last week" in text or "past week" in text:
        date_from = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        date_to = today.strftime("%Y-%m-%d")
        parse_reason = "Detected last week"

    elif "last year" in text or "past year" in text:
        year = today.year - 1
        date_from = f"{year}-01-01"
        date_to = f"{year}-12-31"
        parse_reason = "Detected last year"

    print(f"   Query     : {search_query}")
    print(f"   Date from : {date_from or '(none)'}")
    print(f"   Date to   : {date_to or '(none)'}")
    print(f"   Count     : {result_count}")
    print(f"   Reason    : {parse_reason}")

    return {
        **state,
        "search_query": search_query,
        "date_from": date_from,
        "date_to": date_to,
        "result_count": result_count,
        "parse_reason": parse_reason,
    }


def exa_search_node(state: AgentState) -> AgentState:
    print("\n[2/2] Querying Exa Search API...")

    try:
        start = time.perf_counter()

        params = {
            "query": state["search_query"],
            "num_results": state["result_count"],
        }

        if state["date_from"]:
            params["start_published_date"] = state["date_from"]
        if state["date_to"]:
            params["end_published_date"] = state["date_to"]

        response = exa.search(**params)

        end = time.perf_counter()
        response_time = round(end - start, 3)

        results = []
        for item in response.results:
            results.append({
                "title": (item.title or "").strip(),
                "url": (item.url or "").strip(),
                "description": (item.text[:300].replace("\n", " ").strip() if item.text else ""),
            })

        return {
            **state,
            "results": results,
            "response_time": response_time,
            "error": "",
        }

    except Exception as e:
        return {
            **state,
            "results": [],
            "response_time": None,
            "error": str(e),
        }


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("parse", parse_instruction_node)
    graph.add_node("search", exa_search_node)
    graph.set_entry_point("parse")
    graph.add_edge("parse", "search")
    graph.add_edge("search", END)
    return graph.compile()


def print_results(state: AgentState):
    print("\n" + "=" * 70)
    print("EXA NATURAL LANGUAGE SEARCH — RESULTS")
    print("=" * 70)
    print(f"Instruction   : {state['user_instruction']}")
    print(f"Query sent    : {state['search_query']}")

    if state["date_from"] or state["date_to"]:
        print(f"Date range    : {state['date_from'] or '—'} -> {state['date_to'] or '—'}")
    else:
        print("Date range    : None")

    print(f"Response time : {state['response_time']}s" if state["response_time"] else "Response time : N/A")

    if state["error"]:
        print(f"Error         : {state['error']}")
        print("=" * 70)
        return

    print(f"Results found : {len(state['results'])}")
    print("-" * 70)

    for i, result in enumerate(state["results"], start=1):
        print(f"\n[{i}] {result.get('title') or '(no title)'}")
        print(f"URL  : {result.get('url') or 'N/A'}")
        desc = result.get("description") or "(no description)"
        print(f"Desc : {desc[:160]}")

    print("\n" + "=" * 70)


def run(instruction: str):
    agent = build_agent()

    initial_state: AgentState = {
        "user_instruction": instruction,
        "search_query": "",
        "date_from": "",
        "date_to": "",
        "result_count": 5,
        "parse_reason": "",
        "results": [],
        "response_time": None,
        "error": "",
    }

    final_state = agent.invoke(initial_state)
    print_results(final_state)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        instruction = " ".join(sys.argv[1:])
    else:
        print("\nExamples:")
        print('  python3 nl_agent.py "latest AI research papers from last month"')
        print('  python3 nl_agent.py "Python tutorials published in 2024"')
        print('  python3 nl_agent.py "news about LangGraph between Jan and Mar 2025"')
        print('  python3 nl_agent.py "top 3 cloud security articles from last week"\n')
        instruction = input("Enter your instruction: ").strip()

    run(instruction)
