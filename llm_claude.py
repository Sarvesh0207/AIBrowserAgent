from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from anthropic import Anthropic


@dataclass(frozen=True)
class ClaudeConfig:
    model: str = "claude-3-5-sonnet-latest"
    max_tokens: int = 800


class ClaudeJSON:
    """
    Minimal JSON-only Claude helper.
    - Returns dict parsed from assistant output
    - No tool use, no streaming (kept simple for demos)
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[ClaudeConfig] = None):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self._client = Anthropic(api_key=api_key)
        self._config = config or ClaudeConfig()

    def complete_json(self, *, system: str, user: str) -> Dict[str, Any]:
        msg = self._client.messages.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        # Anthropic SDK returns content blocks; we expect a single text block.
        text_parts = []
        for block in getattr(msg, "content", []) or []:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", "") or "")
        text = ("\n".join(text_parts)).strip()

        # Try direct JSON parse; if the model wrapped it, extract the first JSON object.
        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise

