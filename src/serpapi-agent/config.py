"""
Runtime configuration for the SerpAPI agents.

This module is intentionally minimal: it only centralizes API key helpers.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# API keys (kept here for convenience; agent/nl_agent also read from env)
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


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

