"""
Natural language entry: convert a user prompt (e.g. "visit example.com and summarize")
into a single URL to browse. Used by --prompt and by run-hitl when the user
types a sentence instead of a raw URL.
"""
import re
from typing import Optional

from .config import ANTHROPIC_MODEL, require_api_key

# Simple check: does the string look like a URL we can use as-is?
URL_PATTERN = re.compile(
    r"^https?://[^\s]+",
    re.IGNORECASE,
)


def looks_like_url(s: str) -> bool:
    """True if s is or starts with http(s):// (possibly with leading/trailing spaces)."""
    t = (s or "").strip()
    return bool(URL_PATTERN.match(t)) or t.startswith("http://") or t.startswith("https://")


def extract_url_from_text(s: str) -> Optional[str]:
    """If s contains something that looks like a URL, return it (trimmed)."""
    if not s:
        return None
    match = re.search(r"https?://[^\s\)\]\"]+", s, re.IGNORECASE)
    if match:
        return match.group(0).rstrip(".,;:)")
    return None


def parse_prompt_to_url(prompt: str) -> str:
    """
    Use LLM to interpret natural language and return a single URL to browse.
    E.g. "visit example website" -> "https://example.com"
    Raises if no URL can be determined.
    """
    require_api_key()
    from anthropic import Anthropic

    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Prompt is empty")

    # If it already looks like a URL, return normalized
    if looks_like_url(prompt):
        url = extract_url_from_text(prompt) or prompt.strip()
        if not url.startswith("http"):
            url = "https://" + url.lstrip("/")
        return url

    client = Anthropic()
    system = (
        "You extract exactly one URL from the user's message. "
        "The user wants to browse a webpage. Return only the URL, nothing else. "
        "If they mention a site name (e.g. example, wikipedia, stanley), use the common official URL (e.g. https://example.com, https://www.wikipedia.org). "
        "If the message contains a URL, return that URL. No explanation, no markdown, no quotes."
    )
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=200,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = []
    for block in msg.content:
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
    raw = (" ".join(parts)).strip()
    # Clean common LLM artifacts
    raw = raw.strip('"\'`').split("\n")[0].strip()
    if not raw:
        raise ValueError("Could not extract URL from prompt")
    if not raw.startswith("http"):
        raw = "https://" + raw.lstrip("/")
    return raw
