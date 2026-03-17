"""
Runtime configuration for the SerpAPI agents.

This mirrors just enough of Tavily-agent's config to support:
- HEADLESS / headed browser mode
- browser timeouts
- output directories for screenshots / logs / reports
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# API keys (kept here for convenience; agent/nl_agent also read from env)
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Browser / Playwright settings
HEADLESS = os.getenv("HEADLESS", "false").lower() in {"1", "true", "yes"}
BROWSER_TIMEOUT_MS = int(os.getenv("BROWSER_TIMEOUT_MS", "30000"))

# Output directories
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
LOG_DIR = OUTPUT_DIR / "logs"
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"
REPORT_DIR = OUTPUT_DIR / "reports"


def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def require_api_key(kind: str = "SERPAPI") -> None:
    """
    Fail fast with a clear message if the requested API key is missing.

    kind: "SERPAPI" or "ANTHROPIC"
    """
    if kind.upper() == "SERPAPI" and not SERPAPI_API_KEY:
        raise RuntimeError(
            "Missing SERPAPI_API_KEY. Put it in .env or environment variables."
        )
    if kind.upper() == "ANTHROPIC" and not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "Missing ANTHROPIC_API_KEY. Put it in .env or environment variables."
        )

