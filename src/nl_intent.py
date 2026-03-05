"""
Parse natural language into a structured intent for the browser agent.
Supports: navigate (go to URL), search (on current page), click, exit.
"""
from dataclasses import dataclass
from typing import Optional

from .config import ANTHROPIC_MODEL, require_api_key
from .prompt_to_url import extract_url_from_text, looks_like_url, parse_prompt_to_url

# Known site configs for "search on Google" style (open that site then search)
SITE_SEARCH_CONFIG = {
    "google": {"url": "https://www.google.com", "search_selector": "textarea[name=q], input[name=q]"},
    "bing": {"url": "https://www.bing.com", "search_selector": "input[name=q]"},
    "duckduckgo": {"url": "https://duckduckgo.com", "search_selector": "input[name=q]"},
}


@dataclass
class SearchIntent:
    """Open a search engine and search (used when we don't have a persistent page)."""
    url: str
    fill_selector: str
    query: str
    submit: bool = True


@dataclass
class NavigateIntent:
    url: str


@dataclass
class SearchOnPageIntent:
    """Search in the search box on the current page."""
    query: str


@dataclass
class ClickIntent:
    """Click: selector, nth_link, nth_result, link text, or description (match visible label on page)."""
    selector: Optional[str] = None
    nth_link: Optional[int] = None
    nth_result: Optional[int] = None
    link_text: Optional[str] = None
    description: Optional[str] = None  # match to element label (e.g. "Google Search", "Gmail") - generic, any site


@dataclass
class CookieConsentIntent:
    """Accept or decline cookie banner. accept=True -> Accept All; False -> Decline."""
    accept: bool


@dataclass
class ClosePopupIntent:
    """User wants to close the topmost modal/popup/dialog (e.g. close the popup)."""


@dataclass
class CloseSearchIntent:
    """User wants to close/dismiss the search box overlay (e.g. close search)."""


@dataclass
class HoverIntent:
    """User wants to HOVER over an element to open a dropdown/menu (e.g. hover Solutions)."""
    description: str


# Only nav items that open a DROPDOWN on hover (not direct links). News & Insights, Support = direct links → click.
NAV_MENU_HOVER_NAMES = frozenset({
    "solutions", "technology", "our company", "estimating",
    "esyimating",  # typo
})


@dataclass
class GoBackIntent:
    """User wants to go back to the previous page (go back)."""


@dataclass
class ExitIntent:
    pass


def _is_exit(msg: str) -> bool:
    t = (msg or "").strip().lower()
    return t in ("exit", "quit", "q", "bye")


def _is_go_back(msg: str) -> bool:
    t = (msg or "").strip().lower()
    return t in ("back", "go back")


def parse_nl_intent(user_message: str) -> Optional[object]:
    """
    Parse one user message into an intent for the chat loop.
    Returns: NavigateIntent | SearchOnPageIntent | ClickIntent | ExitIntent | None.
    - "go to Stanley" / "go to example.com" -> NavigateIntent(url)  (URL from parse_prompt_to_url)
    - "search owala" / "search for X" (on current page) -> SearchOnPageIntent(query)
    - "click the first link" -> ClickIntent
    - "exit" -> ExitIntent
    """
    msg = (user_message or "").strip()
    if not msg:
        return None
    if _is_exit(msg):
        return ExitIntent()
    if _is_go_back(msg):
        return GoBackIntent()
    # If user typed only a URL, go there directly (no LLM needed)
    if looks_like_url(msg):
        url = extract_url_from_text(msg) or msg
        if not url.startswith("http"):
            url = "https://" + url.lstrip("/")
        return NavigateIntent(url=url)

    require_api_key()
    from anthropic import Anthropic

    client = Anthropic()
    prompt = f"""You classify the user's intent for a browser agent. Reply with ONLY one line in this exact format:

action=ACTION param=value

Allowed actions:
- navigate: user wants to GO TO a website. param=the_site_or_url_description.
- search: user wants to SEARCH on the current page. param=the search query only.
- go_back: user wants to GO BACK to previous page (e.g. go back, back). param=go_back.
- close_popup: user wants to CLOSE the topmost modal/popup (e.g. close the popup). param=close_popup.
- close_search: user wants to CLOSE the search box/overlay (e.g. close search). param=close_search.
- cookie_accept: user wants to ACCEPT cookies. param=accept.
- cookie_decline: user wants to DECLINE cookies. param=decline.
- hover: user wants to HOVER over an element to open a dropdown/menu without clicking (e.g. hover Solutions). param=the element label (e.g. Solutions).
- click: user wants to CLICK something. param= one of:
  - first_link or nth_link:0 (first link on page)
  - nth_result:0 or nth_result:2 (1st or 3rd search result; 0-based)
  - text:Exact Product Name (e.g. text:All Day Slim Bottle)
  - OR a short label of the button/link (e.g. Google Search, Gmail, Repair, For Repairers).

User message: {msg}

Reply with exactly one line, e.g.:
action=navigate param=Stanley
action=search param=owala
action=close_popup param=close_popup
action=close_search param=close_search
action=cookie_accept param=accept
action=cookie_decline param=decline
action=click param=first_link
action=click param=nth_result:2
action=click param=text:All Day Slim Bottle
action=click param=Google Search
action=hover param=Solutions
action=go_back param=go_back
"""

    reply = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=150,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = ""
    for block in reply.content:
        if getattr(block, "type", "") == "text":
            text += block.text
    text = text.strip().split("\n")[0].strip()

    action = ""
    param = ""
    for part in text.split():
        if part.startswith("action="):
            action = part.split("=", 1)[1].strip().lower()
        elif part.startswith("param="):
            idx = text.find("param=")
            param = text[idx + 6 :].strip() if idx != -1 else ""
            break

    if action == "navigate" and param:
        try:
            if looks_like_url(param):
                url = extract_url_from_text(param) or param.strip()
            else:
                url = parse_prompt_to_url(param)
            if not url:
                return None
            if not url.startswith("http"):
                url = "https://" + url.lstrip("/")
            return NavigateIntent(url=url)
        except Exception:
            return None
    if action == "search" and param:
        return SearchOnPageIntent(query=param)
    if action == "close_popup":
        return ClosePopupIntent()
    if action == "close_search":
        return CloseSearchIntent()
    if action == "cookie_accept":
        return CookieConsentIntent(accept=True)
    if action == "cookie_decline":
        return CookieConsentIntent(accept=False)
    if action == "go_back":
        return GoBackIntent()
    if action == "hover" and param:
        return HoverIntent(description=param)
    if action == "click" and param:
        if param == "first_link" or param == "first":
            return ClickIntent(nth_link=0)
        if param.startswith("nth_link:"):
            try:
                n = int(param.split(":")[1].strip())
                return ClickIntent(nth_link=n)
            except Exception:
                pass
        if param.startswith("nth_result:"):
            try:
                n = int(param.split(":")[1].strip())
                return ClickIntent(nth_result=n)
            except Exception:
                pass
        if param.startswith("text:"):
            return ClickIntent(link_text=param[5:].strip())
        # Description: match to visible label. If it's a known nav menu that opens on hover, use hover instead.
        desc_normalized = param.strip().lower()
        if desc_normalized in NAV_MENU_HOVER_NAMES:
            return HoverIntent(description=param)
        return ClickIntent(description=param)
    return None


def parse_search_intent(user_message: str) -> Optional[SearchIntent]:
    """
    Legacy: parse "search X on google" into SearchIntent (open search engine + query).
    Used when run-nl is used without chat loop.
    """
    require_api_key()
    from anthropic import Anthropic

    client = Anthropic()
    prompt = f"""You are a parser. The user will say what they want to do on a search engine. Extract:
1. site: one of google, bing, duckduckgo (lowercase)
2. query: the exact search query string they want to type

User message: {user_message}

Reply with ONLY a single line in this exact format, no other text:
site=google query=the search words here
"""

    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=100,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = ""
    for block in msg.content:
        if getattr(block, "type", "") == "text":
            text += block.text
    text = text.strip()

    site = "google"
    query = ""
    idx = text.find("query=")
    if idx != -1:
        query = text[idx + 6 :].strip()
    for part in text.split():
        if part.startswith("site="):
            site = part.split("=", 1)[1].strip().lower()
            break

    if not query:
        return None
    if site not in SITE_SEARCH_CONFIG:
        site = "google"
    config = SITE_SEARCH_CONFIG[site]
    url = config["url"]
    sel = config["search_selector"].strip()
    fill_selector = sel.split(",")[0].strip() if "," in sel else sel
    return SearchIntent(url=url, fill_selector=fill_selector, query=query, submit=True)
