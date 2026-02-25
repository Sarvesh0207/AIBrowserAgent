import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .config import LOG_DIR, ensure_dirs

# Max characters to store in log for long text (visual inputs / text outputs)
LOG_TEXT_MAX_LEN = 2000


def truncate_for_log(s: str | None, max_len: int = LOG_TEXT_MAX_LEN) -> str:
    """
    Truncate string for logging so log files stay manageable.
    Returns a copy with "..." suffix if truncated.
    """
    if s is None:
        return ""
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def utc_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def jsonl_path(prefix: str = "run") -> Path:
    ensure_dirs()
    return LOG_DIR / f"{prefix}_{utc_ts()}.jsonl"


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
