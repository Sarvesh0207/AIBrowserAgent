"""
Natural-language Tavily search (lightweight, no LLM)
===================================================
This is a small CLI that accepts a flexible-ish natural language instruction,
extracts a few common constraints (date range, result count, site filters),
runs a Tavily search, and post-filters results by date when possible.

Examples:
  python nl_agent.py "Find articles about Apple DMA enforcement in the EU from 2025-01-01 to 2025-03-01 top 10"
  python nl_agent.py "site:openai.com gpt-4.1 release notes last 30 days"
  python nl_agent.py "Kubernetes CVE-2025 last 2 weeks exclude blog"

Notes:
  - Date filtering is best-effort because some results may not include a published_date.
  - If a date range is requested and a result has no date metadata, we keep it but mark it as undated.
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Optional, Tuple

from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


@dataclass(frozen=True)
class ParsedInstruction:
    original: str
    query: str
    max_results: int
    search_depth: str  # "basic" | "advanced"
    start_date: Optional[date]
    end_date: Optional[date]
    sites: Tuple[str, ...]
    excludes: Tuple[str, ...]


_RE_ISO_DATE = re.compile(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b")
_RE_TOP_N = re.compile(r"\b(top|first|limit|max)\s+(\d{1,3})\b", re.IGNORECASE)
_RE_LAST_N_UNITS = re.compile(
    r"\b(last|past)\s+(\d{1,3})\s*(day|days|week|weeks|month|months)\b",
    re.IGNORECASE,
)
_RE_SINCE = re.compile(r"\b(since|after)\s+(\d{4}[-/]\d{2}[-/]\d{2})\b", re.IGNORECASE)
_RE_UNTIL = re.compile(r"\b(until|before)\s+(\d{4}[-/]\d{2}[-/]\d{2})\b", re.IGNORECASE)
_RE_BETWEEN_AND = re.compile(
    r"\b(between|from)\s+(\d{4}[-/]\d{2}[-/]\d{2})\s+(and|to)\s+(\d{4}[-/]\d{2}[-/]\d{2})\b",
    re.IGNORECASE,
)
_RE_SITE = re.compile(r"\bsite:([^\s]+)", re.IGNORECASE)
_RE_EXCLUDE = re.compile(r"\bexclude\s+\"([^\"]+)\"|\bexclude\s+([^\s]+)", re.IGNORECASE)
_RE_DEPTH = re.compile(r"\bdepth:(basic|advanced)\b", re.IGNORECASE)

_RE_PRICEY = re.compile(r"\b(price|pricing|cost|msrp|starts at|\$)\b", re.IGNORECASE)
_RE_IPHONE = re.compile(r"\biphone\b", re.IGNORECASE)
_RE_JOBS = re.compile(r"\b(job|jobs|career|careers|opening|openings|position|positions|hiring)\b", re.IGNORECASE)
_RE_AT_DOMAIN = re.compile(r"\bat\s+([a-z0-9.-]+\.[a-z]{2,})\b", re.IGNORECASE)
_RE_CCCIS = re.compile(r"\bccc?is\b", re.IGNORECASE)
_RE_EACH = re.compile(r"\b(each|all)\b", re.IGNORECASE)


def _parse_iso_date(s: str) -> Optional[date]:
    m = _RE_ISO_DATE.search(s)
    if not m:
        return None
    y, mo, d = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def _parse_any_iso_date(s: str) -> Optional[date]:
    """Parse a full ISO-ish date string (YYYY-MM-DD or YYYY/MM/DD)."""
    m = _RE_ISO_DATE.fullmatch(s.strip())
    if not m:
        return None
    y, mo, d = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _strip_spans(text: str, spans: Iterable[Tuple[int, int]]) -> str:
    spans = sorted(spans, key=lambda t: t[0])
    out = []
    last = 0
    for a, b in spans:
        if a < last:
            continue
        out.append(text[last:a])
        last = b
    out.append(text[last:])
    cleaned = " ".join("".join(out).split())
    return cleaned.strip()


def parse_instruction(instruction: str) -> ParsedInstruction:
    original = " ".join(instruction.strip().split())
    if not original:
        raise ValueError("Instruction is empty.")

    max_results = 5
    search_depth = "basic"
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    spans_to_strip: list[Tuple[int, int]] = []

    # depth:basic / depth:advanced
    m_depth = _RE_DEPTH.search(original)
    if m_depth:
        search_depth = m_depth.group(1).lower()
        spans_to_strip.append(m_depth.span())

    # top N
    m_top = _RE_TOP_N.search(original)
    if m_top:
        try:
            max_results = max(1, min(50, int(m_top.group(2))))
        except ValueError:
            pass
        spans_to_strip.append(m_top.span())

    # If user asks for "each/all" (common for jobs), raise default cap.
    if _RE_EACH.search(original) and max_results < 50:
        max_results = 50

    # between/from ... and/to ...
    m_between = _RE_BETWEEN_AND.search(original)
    if m_between:
        s1 = _parse_any_iso_date(m_between.group(2))
        s2 = _parse_any_iso_date(m_between.group(4))
        if s1 and s2:
            start_date, end_date = (min(s1, s2), max(s1, s2))
            spans_to_strip.append(m_between.span())

    # since/after ...
    m_since = _RE_SINCE.search(original)
    if m_since:
        sd = _parse_any_iso_date(m_since.group(2))
        if sd:
            start_date = sd
            spans_to_strip.append(m_since.span())

    # until/before ...
    m_until = _RE_UNTIL.search(original)
    if m_until:
        ed = _parse_any_iso_date(m_until.group(2))
        if ed:
            end_date = ed
            spans_to_strip.append(m_until.span())

    # last N units
    m_last = _RE_LAST_N_UNITS.search(original)
    if m_last:
        n = int(m_last.group(2))
        unit = m_last.group(3).lower()
        today = _today_utc()
        if unit.startswith("day"):
            start_date = today - timedelta(days=n)
            end_date = today
        elif unit.startswith("week"):
            start_date = today - timedelta(days=7 * n)
            end_date = today
        elif unit.startswith("month"):
            # Lightweight approximation: 30 days per month.
            start_date = today - timedelta(days=30 * n)
            end_date = today
        spans_to_strip.append(m_last.span())

    sites = tuple(sorted(set(m.group(1).strip() for m in _RE_SITE.finditer(original) if m.group(1).strip())))
    excludes_found: list[str] = []
    for m in _RE_EXCLUDE.finditer(original):
        val = (m.group(1) or m.group(2) or "").strip()
        if val:
            excludes_found.append(val)
    excludes = tuple(sorted(set(excludes_found)))

    # Strip only "directive" spans (date/count/depth). Keep site: and exclude terms in the query
    # because Tavily often benefits from receiving them directly.
    query = _strip_spans(original, spans_to_strip)
    query = query.strip()

    # Brand-specific convenience: "CCCIS ... careers/jobs" implies their official site.
    if not sites and _RE_CCCIS.search(query) and _RE_JOBS.search(query):
        # CCCIS careers page points to Workday; target the jobs board directly for listings.
        sites = ("cccis.wd1.myworkdayjobs.com",)
        if max_results < 50:
            max_results = 50
        search_depth = "advanced"
        query = f"site:cccis.wd1.myworkdayjobs.com (broadbean_external OR job OR jobs OR requisition) {query}".strip()

    # If user says "at <domain>", default to site:<domain> unless they already used site:
    if not sites:
        m_at = _RE_AT_DOMAIN.search(query)
        if m_at:
            domain = m_at.group(1).lower().strip().rstrip(".,;:")
            if domain:
                sites = (domain,)
                query = f"site:{domain} {query}".strip()

    # Default to official sources for pricing queries unless user specified a site.
    # Heuristic: if it looks like "iphone ... price", constrain to apple.com.
    if not sites and _RE_IPHONE.search(query) and _RE_PRICEY.search(query):
        sites = ("apple.com",)
        query = f"site:apple.com {query}".strip()

    return ParsedInstruction(
        original=original,
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        start_date=start_date,
        end_date=end_date,
        sites=sites,
        excludes=excludes,
    )


def _parse_published_date(value: object) -> Optional[date]:
    """
    Tavily may return published_date in different formats depending on source.
    We support ISO-ish strings; everything else is treated as unknown.
    """
    if not value:
        return None
    if isinstance(value, (int, float)):
        # Some systems provide epoch seconds. Best-effort.
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).date()
        except Exception:
            return None
    if not isinstance(value, str):
        return None
    s = value.strip()
    # Common: "2025-03-01", "2025-03-01T12:34:56Z"
    iso_prefix = s[:10]
    d = _parse_any_iso_date(iso_prefix)
    return d


def _in_range(d: date, start: Optional[date], end: Optional[date]) -> bool:
    if start and d < start:
        return False
    if end and d > end:
        return False
    return True


def search_with_instruction(instruction: str) -> dict:
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY not set in environment (.env).")

    parsed = parse_instruction(instruction)
    client = TavilyClient(api_key=TAVILY_API_KEY)

    t0 = time.perf_counter()
    response = client.search(
        query=parsed.query,
        max_results=parsed.max_results,
        search_depth=parsed.search_depth,
        include_answer=False,
        include_raw_content=False,
    )
    elapsed = round(time.perf_counter() - t0, 3)

    results = response.get("results", []) if isinstance(response, dict) else []

    filtered = []
    undated = []
    if parsed.start_date or parsed.end_date:
        for r in results:
            pd = _parse_published_date(r.get("published_date"))
            if pd is None:
                undated.append(r)
                continue
            if _in_range(pd, parsed.start_date, parsed.end_date):
                filtered.append(r)
        # Keep undated results at the end (best-effort behavior).
        final_results = filtered + undated
    else:
        final_results = results

    # We always keep titles when available, to support "state the results with job titles"
    # and other "exact words" style outputs. We still do not include snippets/descriptions.
    minimal_results = []
    for r in final_results:
        url = (r.get("url") or "").strip() or None
        is_official = bool(url and ("//apple.com/" in url or url.startswith("https://apple.com/") or url.startswith("http://apple.com/")))
        item = {
            "url": url,
            "published_date": r.get("published_date") or None,
            "score": r.get("score") if "score" in r else None,
            "source": "official" if is_official else "other",
        }
        item["title"] = (r.get("title") or "").strip() or None
        minimal_results.append(item)

    return {
        "instruction": parsed.original,
        "parsed": {
            "query": parsed.query,
            "max_results": parsed.max_results,
            "search_depth": parsed.search_depth,
            "start_date": parsed.start_date.isoformat() if parsed.start_date else None,
            "end_date": parsed.end_date.isoformat() if parsed.end_date else None,
            "sites": list(parsed.sites),
            "excludes": list(parsed.excludes),
        },
        "response_time_s": elapsed,
        "result_count": len(minimal_results),
        "dated_in_range_count": len(filtered) if (parsed.start_date or parsed.end_date) else None,
        "undated_count": len(undated) if (parsed.start_date or parsed.end_date) else None,
        "results": minimal_results,
    }


def _print_summary(payload: dict) -> None:
    parsed = payload["parsed"]
    start = parsed["start_date"]
    end = parsed["end_date"]

    print("\n" + "═" * 80)
    print("  🔎  NATURAL-LANGUAGE TAVILY SEARCH")
    print("═" * 80)
    print(f"  Instruction   : {payload['instruction']}")
    print(f"  Query         : {parsed['query']}")
    print(f"  Depth         : {parsed['search_depth']}")
    print(f"  Max results   : {parsed['max_results']}")
    if start or end:
        print(f"  Date range    : {start or '…'} → {end or '…'}  (best-effort)")
        print(f"  In-range dated: {payload['dated_in_range_count']}")
        print(f"  Undated kept  : {payload['undated_count']}")
    print(f"  Response time : {payload['response_time_s']}s")
    print("─" * 80)

    results = payload.get("results", [])
    if not results:
        print("  (no results)")
        print("═" * 80)
        return

    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "").strip() if isinstance(r, dict) else ""
        url = (r.get("url") or "").strip() or ""
        pd = r.get("published_date")
        pd_str = str(pd).strip() if pd else "—"
        if title:
            if url:
                print(f"  {i:>2}. {title} — {url}")
            else:
                print(f"  {i:>2}. {title}")
        else:
            print(f"  {i:>2}. {url or '(no title/url)'}")
        # Keep date when present, but don't force it as primary.
        if pd_str != "—":
            print(f"      date: {pd_str}")
        print()

    print("═" * 80)


def main() -> None:
    if len(sys.argv) > 1:
        instruction = " ".join(sys.argv[1:]).strip()
    else:
        instruction = input("\nEnter instruction: ").strip()

    payload = search_with_instruction(instruction)
    _print_summary(payload)


if __name__ == "__main__":
    main()

