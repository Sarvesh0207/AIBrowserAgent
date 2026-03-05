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
        "You extract exactly one URL from the user's message. Return ONLY the URL, nothing else. No explanation, no markdown, no quotes, no sentences like 'I cannot extract'. "
        "If the user provides a full URL, return it as-is. "
        "Otherwise return the most likely website URL. For abbreviations or acronyms (e.g. CCCIS, UIC): use .edu for schools/universities, .org for organizations, .com otherwise. "
        "Examples: Stanley -> https://www.stanley1913.com ; Stanley Tools -> https://www.stanley.com ; "
        "Apple -> https://www.apple.com ; IKEA -> https://www.ikea.com ; "
        "UIC (university) -> https://www.uic.edu ; CCCIS (if school) -> https://www.cccis.edu or https://cccis.edu ; "
        "Unknown acronym -> https://www.<lowercase>.com . Never output an error message; always output a valid URL."
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
        raw = ""

    # Reject LLM error messages (e.g. "I cannot extract a URL from...") being used as URL
    def _looks_like_valid_domain(s: str) -> bool:
        if not s or len(s) > 250 or " " in s:
            return False
        s_lower = s.lower()
        if "cannot" in s_lower or "please" in s_lower or "provide" in s_lower or "extract" in s_lower:
            return False
        # Should look like a host: has a dot or is a known short TLD path
        if s.startswith("http"):
            return True
        return "." in s or len(s) <= 15

    if raw and _looks_like_valid_domain(raw):
        if not raw.startswith("http"):
            raw = "https://" + raw.lstrip("/")
        return raw

    # Fallback: use the user's prompt as a domain (e.g. "cccis" -> https://www.cccis.com)
    slug = re.sub(r"[^\w.-]", "", (prompt or "").strip().lower())
    slug = slug[:64].strip(".-")
    if slug:
        return f"https://www.{slug}.com"
    raise ValueError("Could not extract URL from prompt. Try providing a full URL or a clearer site name.")
